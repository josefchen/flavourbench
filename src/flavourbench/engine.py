from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, or_, select, text
from sqlalchemy.orm import Session

from .account_authority import (
    account_authorization,
    account_authorization_chain_valid,
)
from .budget_integrity import (
    BudgetIntegrityError,
    assert_budget_integrity,
    decrement_reservation,
)
from .budget_policy import (
    provider_account_hard_cap_micros,
    provider_account_scope_sha256,
)
from .commercial_authority import active_spend_authorization
from .config import budget_authorization_verification_keyring, get_settings
from .database import session_scope
from .endpoint_contract import (
    DECODING_PARAMETERS,
    REQUIRED_ENDPOINT_PARAMETERS,
    UNFROZEN_VALUES,
    endpoint_contract_sha256,
)
from .mcp_client import attest_epicure_runtime
from .models import (
    TOOL_CALL_REDACTION_JSON,
    TOOL_CALL_REDACTION_SENTINEL,
    Battle,
    CatalogModel,
    ControlledRun,
    ControlledRunAssignment,
    CostEvent,
    EvaluationOrder,
    GenerationAttempt,
    GovernanceAcceptance,
    Incident,
    Job,
    LeaderboardSnapshot,
    ModelRouteRevision,
    ModelSubmission,
    ProviderAccountAuthorization,
    ProviderAccountBudget,
    ResponseArm,
    RunEvent,
    Season,
    SeasonModel,
    SeasonProviderBudget,
    ToolCall,
    ValidatorResult,
)
from .protocol_contract import build_protocol_bundle
from .provider import (
    FINAL_SCHEMA_SHA256,
    GenerationFailureResult,
    GenerationResult,
    GenerationSpec,
    ProviderAttemptEvent,
    ProviderError,
    ToolTrace,
    UncertainDeliveryError,
    get_provider,
    system_prompt_sha256,
)
from .security import contains_identity_leak
from .task_evidence import TaskEvidenceError
from .task_evidence_registry import (
    TASK_SPECIFIC_VALIDATOR_NAME,
    TASK_SPECIFIC_VALIDATOR_VERSION,
    task_validator_receipt_for_battle,
    verified_task_evidence_for_battle,
)
from .validators import VALIDATOR_VERSION, validate_output

_SAFE_NO_COST_ATTEMPT_EVENTS = frozenset(
    {"pre_send_failure", "request_rejected", "retry_scheduled"}
)
_COMPLETE_FINISH_REASONS = frozenset(
    {
        "completed",  # Bedrock Responses/Mantle normalization
        "end_turn",  # Bedrock Converse
        "stop",  # OpenAI-compatible providers
        "stop_sequence",  # Bedrock Converse with an explicit stop sequence
    }
)


def _assert_commercial_external_work_authorized(
    session: Session,
    battle: Battle,
) -> GovernanceAcceptance | None:
    """Lock and revalidate the exact customer spend authority before network I/O."""

    if battle.controlled_run_id is None:
        return None
    season = session.scalar(select(Season).where(Season.id == battle.season_id).with_for_update())
    run = session.scalar(
        select(ControlledRun).where(ControlledRun.id == battle.controlled_run_id).with_for_update()
    )
    if season is None or run is None or run.status != "active":
        raise ProviderError("controlled run was revoked before external work was authorized")
    if run.evaluation_order_id is None:
        return None
    order = session.scalar(
        select(EvaluationOrder)
        .where(EvaluationOrder.id == run.evaluation_order_id)
        .with_for_update()
    )
    if (
        order is None
        or order.status not in {"ready", "running"}
        or order.billing_status != "authorized"
        or order.organization_id != run.organization_id
        or order.season_id != season.id
        or order.route_revision_id != run.route_revision_id
        or run.spend_authorization_id is None
        or run.spend_authorization_binding_sha256 is None
    ):
        raise ProviderError("commercial run no longer has an executable order contract")
    acceptance = active_spend_authorization(
        session,
        order=order,
        acceptance_id=run.spend_authorization_id,
        binding_sha256=run.spend_authorization_binding_sha256,
        lock=True,
    )
    if acceptance is None:
        raise ProviderError("commercial spend authorization is inactive or superseded")
    return acceptance


def _is_mcp_attempt(event: GenerationAttempt) -> bool:
    """MCP network events do not represent paid model-generation exposure."""

    return event.event_type.startswith("mcp_")


def has_unresolved_paid_attempt(events: list[GenerationAttempt]) -> bool:
    """Return whether any latest paid-provider attempt lacks a terminal cost state."""

    latest_by_attempt: dict[tuple[str, str], GenerationAttempt] = {}
    for event in events:
        latest_by_attempt[(event.arm_id, event.attempt_id)] = event
    return any(
        not _is_mcp_attempt(event)
        and event.event_type not in {*_SAFE_NO_COST_ATTEMPT_EVENTS, "accounting_reconciled"}
        for event in latest_by_attempt.values()
    )


def is_complete_finish_reason(reason: str | None) -> bool:
    """Accept only provider stop states that represent a complete final response."""

    return bool(reason and reason.strip().lower() in _COMPLETE_FINISH_REASONS)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _uses_postgresql_budget_authority(session: Session) -> bool:
    return bool(
        get_settings().execution_mode == "live" and session.get_bind().dialect.name == "postgresql"
    )


class LostJobLease(ProviderError):
    """The worker no longer owns the durable job generation."""


def _withdraw_snapshots_for_cost_halt(session: Session, season_id: str) -> list[str]:
    """Fail closed if cost reconciliation halts a season with published results."""

    snapshots = session.scalars(
        select(LeaderboardSnapshot)
        .where(
            LeaderboardSnapshot.season_id == season_id,
            LeaderboardSnapshot.publication_status == "published",
        )
        .order_by(LeaderboardSnapshot.id)
        .with_for_update()
    ).all()
    for snapshot in snapshots:
        snapshot.publication_status = "withdrawn"
        session.add(
            RunEvent(
                entity_type="leaderboard_snapshot",
                entity_id=snapshot.id,
                event_type="leaderboard_snapshot_automatically_withdrawn",
                payload_json={"reason_code": "season_cost_halted"},
            )
        )
    return [snapshot.id for snapshot in snapshots]


def _current_job_lease(
    session: Session,
    job_id: str,
    claimed_by: str,
    claim_attempt: int,
    *,
    lock: bool = False,
) -> Job | None:
    statement = select(Job).where(
        Job.id == job_id,
        Job.status == "running",
        Job.claimed_by == claimed_by,
        Job.attempts == claim_attempt,
    )
    if lock:
        statement = statement.with_for_update()
    return session.scalar(statement)


def _claim_fingerprint(claimed_by: str) -> str:
    return hashlib.sha256(claimed_by.encode()).hexdigest()


def _worker_id() -> str:
    return f"{socket.gethostname()}:{id(asyncio.current_task())}"


def recover_stale_jobs(session: Session) -> int:
    settings = get_settings()
    cutoff = datetime.now(UTC) - timedelta(seconds=settings.worker_claim_timeout_seconds)
    jobs = session.scalars(
        select(Job)
        .where(Job.status == "running", Job.claimed_at < cutoff)
        .with_for_update(skip_locked=True)
    ).all()
    for job in jobs:
        battle = session.get(Battle, job.battle_id) if job.battle_id else None
        incomplete_arm_ids: list[str] = []
        if battle is not None:
            incomplete_arm_ids = list(
                session.scalars(
                    select(ResponseArm.id).where(
                        ResponseArm.battle_id == battle.id,
                        ResponseArm.status != "complete",
                    )
                ).all()
            )
        attempt_events = (
            session.scalars(
                select(GenerationAttempt)
                .where(GenerationAttempt.arm_id.in_(incomplete_arm_ids))
                .order_by(GenerationAttempt.created_at, GenerationAttempt.id)
            ).all()
            if incomplete_arm_ids
            else []
        )
        latest_by_attempt: dict[tuple[str, str], GenerationAttempt] = {}
        for event in attempt_events:
            latest_by_attempt[(event.arm_id, event.attempt_id)] = event
        accepted_and_accounted = [
            event
            for event in latest_by_attempt.values()
            if event.event_type == "accounting_reconciled"
        ]
        uncertain = [
            event
            for event in latest_by_attempt.values()
            if not _is_mcp_attempt(event)
            and event.event_type not in {*_SAFE_NO_COST_ATTEMPT_EVENTS, "accounting_reconciled"}
        ]
        if uncertain or accepted_and_accounted:
            job.status = "uncertain" if uncertain else "failed"
            job.completed_at = datetime.now(UTC)
            job.claimed_by = None
            job.claimed_at = None
            job.last_error = (
                "stale worker had one or more provider requests with uncertain delivery; "
                "automatic replay refused"
                if uncertain
                else "stale worker completed paid provider work before arm commit; replay refused"
            )
            if battle is not None:
                for arm_id in incomplete_arm_ids:
                    arm = session.get(ResponseArm, arm_id)
                    if arm is None:
                        continue
                    arm_events = [event for event in attempt_events if event.arm_id == arm.id]
                    arm_latest: dict[tuple[str, str], GenerationAttempt] = {}
                    for event in arm_events:
                        arm_latest[(event.arm_id, event.attempt_id)] = event
                    arm_uncertain = has_unresolved_paid_attempt(arm_events)
                    accounting_events = [
                        event
                        for event in arm_latest.values()
                        if event.event_type == "accounting_reconciled"
                    ]
                    if arm_uncertain:
                        arm.status = "uncertain"
                        arm.error_code = "UncertainDelivery"
                        arm.error_detail = job.last_error
                    elif accounting_events:
                        by_generation = {
                            event.generation_id: event
                            for event in accounting_events
                            if event.generation_id
                        }
                        arm.provider_generation_ids_json = sorted(by_generation)
                        arm.generation_id = sorted(by_generation)[-1] if by_generation else None
                        arm.cost_micros = sum(
                            int(event.metadata_json.get("cost_micros") or 0)
                            for event in by_generation.values()
                        )
                        arm.cost_reconciled = all(
                            event.metadata_json.get("reconciled") is True
                            for event in by_generation.values()
                        )
                        arm.cost_accounting_basis = "recovered_from_attempt_journal"
                        arm.billing_reconciliation_status = (
                            "complete" if arm.cost_reconciled else "unresolved"
                        )
                        arm.status = "failed" if arm.cost_reconciled else "uncertain"
                        arm.error_code = (
                            "PaidResponseLostBeforeArmCommit"
                            if arm.cost_reconciled
                            else "UncertainDelivery"
                        )
                        arm.error_detail = job.last_error
                        if by_generation:
                            last = by_generation[sorted(by_generation)[-1]]
                            arm.actual_model_id = str(
                                last.metadata_json.get("model") or arm.actual_model_id or "unknown"
                            )
                            arm.actual_provider_slug = str(
                                last.metadata_json.get("provider")
                                or arm.actual_provider_slug
                                or "unknown"
                            )
                        session.add(
                            CostEvent(
                                season_id=battle.season_id,
                                battle_id=battle.id,
                                arm_id=arm.id,
                                kind="actual",
                                amount_micros=arm.cost_micros,
                                provider=arm.actual_provider_slug or arm.provider_slug,
                                generation_id=arm.generation_id,
                                accounting_json={
                                    "basis": "recovered_from_attempt_journal",
                                    "generation_ids": sorted(by_generation),
                                    "reconciled": arm.cost_reconciled,
                                    "cost_accounting_basis": arm.cost_accounting_basis,
                                    "billing_reconciliation_status": (
                                        arm.billing_reconciliation_status
                                    ),
                                },
                            )
                        )
                    else:
                        arm.status = "failed"
                        arm.error_code = "PeerArmRecoveryAborted"
                        arm.error_detail = job.last_error
                        _record_known_zero_cost(
                            session,
                            battle,
                            arm,
                            reason="peer_arm_paid_or_uncertain_after_worker_loss",
                        )
                    arm.completed_at = datetime.now(UTC)
                terminal_arms = session.scalars(
                    select(ResponseArm).where(ResponseArm.battle_id == battle.id)
                ).all()
                _terminalize_battle_after_arms(
                    session,
                    battle,
                    terminal_arms,
                    status="failed",
                )
                session.add(
                    Incident(
                        severity="critical",
                        code="stale_generation_delivery_uncertain",
                        detail=(
                            "Automatic retry was blocked because a durable pre-send provider "
                            "attempt exists without a provably safe terminal state. Reserved "
                            "budget remains held pending reconciliation."
                        ),
                        battle_id=battle.id,
                    )
                )
                session.add(
                    RunEvent(
                        entity_type="battle",
                        entity_id=battle.id,
                        event_type="generation_replay_blocked",
                        payload_json={
                            "attempt_ids": sorted(
                                event.attempt_id for event in [*uncertain, *accepted_and_accounted]
                            ),
                            "generation_ids": sorted(
                                {
                                    event.generation_id
                                    for event in [*uncertain, *accepted_and_accounted]
                                    if event.generation_id
                                }
                            ),
                        },
                    )
                )
                if not uncertain:
                    reconcile_battle_cost(session, battle)
            continue
        if job.attempts >= job.max_attempts:
            job.status = "failed"
            job.claimed_by = None
            job.claimed_at = None
            job.completed_at = datetime.now(UTC)
            job.last_error = "worker recovery attempt limit exhausted before provider dispatch"
            if battle is not None:
                for arm_id in incomplete_arm_ids:
                    arm = session.get(ResponseArm, arm_id)
                    if arm is None:
                        continue
                    arm.status = "failed"
                    arm.error_code = "WorkerRecoveryLimit"
                    arm.error_detail = job.last_error
                    arm.completed_at = datetime.now(UTC)
                    _record_known_zero_cost(
                        session,
                        battle,
                        arm,
                        reason="safe_worker_recovery_attempt_limit_exhausted",
                    )
                terminal_arms = session.scalars(
                    select(ResponseArm).where(ResponseArm.battle_id == battle.id)
                ).all()
                _terminalize_battle_after_arms(
                    session,
                    battle,
                    terminal_arms,
                    status="failed",
                )
                reconcile_battle_cost(session, battle)
            continue
        job.status = "queued"
        job.claimed_by = None
        job.claimed_at = None
        job.available_at = datetime.now(UTC)
        job.last_error = "stale worker claim recovered"
    return len(jobs)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _terminalize_battle_after_arms(
    session: Session,
    battle: Battle,
    arms: list[ResponseArm],
    *,
    status: str,
) -> None:
    """Persist both terminal arms before sealing their parent battle."""

    if status not in {"complete", "failed"}:
        raise ValueError("terminal battle status must be complete or failed")
    if len(arms) != 2 or {arm.side for arm in arms} != {"left", "right"}:
        raise RuntimeError("a terminal battle requires exactly one arm per side")
    allowed = {"complete"} if status == "complete" else {"complete", "failed", "uncertain"}
    if any(arm.status not in allowed or arm.completed_at is None for arm in arms):
        raise RuntimeError("battle arms must be terminal before the battle is sealed")
    if _uses_postgresql_budget_authority(session) and battle.controlled_run_id:
        # Controlled-run lifecycle endpoints use Season -> ControlledRun. Take
        # the same locks before the 0014 battle trigger touches the run row.
        season = session.scalar(
            select(Season).where(Season.id == battle.season_id).with_for_update()
        )
        controlled_run = session.scalar(
            select(ControlledRun)
            .where(
                ControlledRun.id == battle.controlled_run_id,
                ControlledRun.season_id == battle.season_id,
            )
            .with_for_update()
        )
        if season is None or controlled_run is None:
            raise RuntimeError("controlled battle lost its same-season budget authority")
    session.flush(arms)
    completed_at = max(datetime.now(UTC), *(_as_utc(arm.completed_at) for arm in arms))
    battle.status = status
    battle.completed_at = completed_at
    session.flush([battle])


def _assert_paid_request_authorized(
    session: Session,
    arm: ResponseArm,
    battle: Battle,
) -> ProviderAccountAuthorization | None:
    """Revalidate the complete paid boundary immediately before every request."""

    settings = get_settings()
    if settings.execution_mode != "live":
        return None
    backend = arm.execution_backend
    if backend not in {
        "bedrock",
        "kimi_direct",
        "openrouter",
        "qwencloud_direct",
    }:
        raise ProviderError("live provider request has an unsupported execution backend")
    season = session.scalar(select(Season).where(Season.id == battle.season_id).with_for_update())
    if (
        season is None
        or season.status not in {"pilot", "active"}
        or season.manifest_sha256 in UNFROZEN_VALUES
        or arm.battle_id != battle.id
    ):
        raise ProviderError("season no longer authorizes paid provider dispatch")
    controlled_run = (
        session.scalar(
            select(ControlledRun)
            .where(ControlledRun.id == battle.controlled_run_id)
            .with_for_update()
        )
        if battle.controlled_run_id
        else None
    )
    if battle.controlled_run_id and controlled_run is None:
        raise ProviderError("controlled run disappeared before paid dispatch")
    provider_reservations = battle.provider_reservations_json
    if not isinstance(provider_reservations, dict):
        raise ProviderError("battle provider reservation contract is malformed")
    provider_reservation = provider_reservations.get(backend)
    if (
        not isinstance(provider_reservation, int)
        or isinstance(provider_reservation, bool)
        or provider_reservation <= 0
        or arm.execution_backend != backend
    ):
        raise ProviderError("battle has no paid reservation for this response-arm backend")
    account_scope = provider_account_scope_sha256(backend)
    reservation_events = session.scalars(
        select(CostEvent).where(
            CostEvent.battle_id == battle.id,
            CostEvent.provider == backend,
            CostEvent.kind.in_({"provider_reserve", "provider_account_reserve"}),
        )
    ).all()
    by_kind = {event.kind: event for event in reservation_events}
    account_reserve = by_kind.get("provider_account_reserve")
    if (
        len(reservation_events) != 2
        or set(by_kind) != {"provider_reserve", "provider_account_reserve"}
        or by_kind["provider_reserve"].amount_micros != provider_reservation
        or account_reserve is None
        or account_reserve.amount_micros != provider_reservation
        or account_reserve.accounting_json.get("account_scope_sha256") != account_scope
    ):
        raise ProviderError("battle does not own matching provider reservation evidence")
    provider_budget = session.scalar(
        select(SeasonProviderBudget)
        .where(
            SeasonProviderBudget.season_id == season.id,
            SeasonProviderBudget.execution_backend == backend,
        )
        .with_for_update()
    )
    account_budget = session.scalar(
        select(ProviderAccountBudget)
        .where(
            ProviderAccountBudget.execution_backend == backend,
            ProviderAccountBudget.account_scope_sha256 == account_scope,
        )
        .with_for_update()
    )
    now = datetime.now(UTC)
    if account_budget is None or provider_budget is None:
        raise ProviderError("paid provider authorization disappeared before dispatch")
    try:
        assert_budget_integrity(session, season.id, lock_aggregates=True)
    except BudgetIntegrityError as exc:
        raise ProviderError("budget reservation evidence is inconsistent") from exc
    authorization = account_authorization(
        session,
        account_budget,
        for_update=True,
    )
    if (
        account_budget.status != "active"
        or account_budget.revoked_at is not None
        or account_budget.currency != "USD"
        or account_budget.budget_cap_micros != provider_account_hard_cap_micros(backend)
        or account_budget.budget_used_micros < account_budget.opening_used_micros
        or account_budget.budget_reserved_micros < 0
        or account_budget.budget_used_micros + account_budget.budget_reserved_micros
        > account_budget.budget_cap_micros
        or account_budget.budget_reserved_micros < provider_reservation
        or not account_authorization_chain_valid(
            session,
            account_budget,
            authorization,
            root_envelope_sha256=(provider_budget.account_authorization_envelope_sha256),
            signing_secret=settings.budget_authorization_signing_secret,
            verification_keys=budget_authorization_verification_keyring(settings),
            now=now,
        )
        or _canonical_sha256(account_budget.opening_balance_json)
        != account_budget.opening_balance_sha256
    ):
        raise ProviderError("account-wide provider authorization is invalid or exhausted")
    provider_envelope = provider_budget.authorization_envelope_json
    if (
        provider_budget.currency != "USD"
        or provider_budget.account_scope_sha256 != account_scope
        or provider_budget.budget_cap_micros <= 0
        or provider_budget.budget_cap_micros > account_budget.budget_cap_micros
        or provider_budget.budget_used_micros + provider_budget.budget_reserved_micros
        > provider_budget.budget_cap_micros
        or provider_budget.budget_reserved_micros < provider_reservation
        or _as_utc(provider_budget.valid_until) <= now
        or _canonical_sha256(provider_envelope) != provider_budget.authorization_envelope_sha256
        or provider_envelope.get("season_slug") != season.slug
        or provider_envelope.get("execution_backend") != backend
        or provider_envelope.get("account_scope_sha256") != account_scope
        or provider_envelope.get("account_authorization_envelope_sha256")
        != provider_budget.account_authorization_envelope_sha256
    ):
        raise ProviderError("season provider authorization is invalid or exhausted")
    if (
        season.budget_used_micros + season.budget_reserved_micros > season.budget_cap_micros
        or season.budget_reserved_micros < battle.reserved_cost_micros
        or battle.reserved_cost_micros <= 0
    ):
        raise ProviderError("season or battle reservation is invalid before dispatch")
    if controlled_run is not None:
        if (
            controlled_run.status != "active"
            or controlled_run.budget_reserved_micros < battle.reserved_cost_micros
            or controlled_run.budget_used_micros + controlled_run.budget_reserved_micros
            > controlled_run.budget_cap_micros
        ):
            raise ProviderError("controlled run no longer authorizes paid dispatch")
    assert authorization is not None
    return authorization


def _persist_provider_attempt(
    event: ProviderAttemptEvent,
    *,
    job_id: str,
    claimed_by: str,
    claim_attempt: int,
) -> None:
    """Commit provider lifecycle evidence in its own transaction before network I/O."""

    if not event.arm_id:
        raise ProviderError("provider attempt is missing its response-arm identity")
    with session_scope() as session:
        arm = session.get(ResponseArm, event.arm_id)
        if arm is None:
            raise ProviderError("provider attempt references an unknown response arm")
        starts_external_work = event.event_type in {
            "request_started",
            "mcp_session_started",
            "mcp_call_started",
        }
        battle = session.get(Battle, arm.battle_id)
        if battle is None:
            raise ProviderError("provider attempt references an orphaned response arm")
        verified_commercial_spend = (
            _assert_commercial_external_work_authorized(session, battle)
            if starts_external_work
            else None
        )
        verified_authorization: ProviderAccountAuthorization | None = None
        if event.event_type == "request_started":
            verified_authorization = _assert_paid_request_authorized(
                session,
                arm,
                battle,
            )
            if verified_authorization is not None and (
                event.metadata.get("provider_account_authorization_envelope_sha256")
                != verified_authorization.authorization_envelope_sha256
                or event.metadata.get("provider_credential_binding_sha256")
                != verified_authorization.credential_binding_sha256
            ):
                raise ProviderError(
                    "provider request does not name the active governed credential epoch"
                )
        lease = _current_job_lease(
            session,
            job_id,
            claimed_by,
            claim_attempt,
            lock=True,
        )
        claim_metadata = {
            "job_id": job_id,
            "claim_attempt": claim_attempt,
            "claim_fingerprint": _claim_fingerprint(claimed_by),
        }
        if starts_external_work:
            prior_dispatches = session.scalars(
                select(GenerationAttempt).where(
                    GenerationAttempt.arm_id == event.arm_id,
                    GenerationAttempt.request_key_sha256 == event.request_key_sha256,
                    GenerationAttempt.attempt_index == event.attempt_index,
                    GenerationAttempt.event_type == event.event_type,
                )
            ).all()
            if any(
                prior.metadata_json.get("job_id") == job_id
                and prior.metadata_json.get("claim_attempt") == claim_attempt
                for prior in prior_dispatches
            ):
                raise ProviderError(
                    "duplicate external-work dispatch was blocked for the active job lease"
                )
        prior_start = session.scalar(
            select(GenerationAttempt).where(
                GenerationAttempt.attempt_id == event.attempt_id,
                GenerationAttempt.event_type.in_(
                    {"request_started", "mcp_session_started", "mcp_call_started"}
                ),
            )
        )
        if lease is None and (
            starts_external_work
            or prior_start is None
            or any(
                prior_start.metadata_json.get(field) != value
                for field, value in claim_metadata.items()
            )
        ):
            raise LostJobLease("worker lease was superseded before external work")
        if event.event_type == "mcp_session_attested":
            attestation = event.metadata.get("attestation")
            attestation_sha256 = event.metadata.get("attestation_sha256")
            if (
                not isinstance(attestation, dict)
                or not isinstance(attestation_sha256, str)
                or _canonical_sha256(attestation) != attestation_sha256
            ):
                raise ProviderError("MCP session attestation evidence is malformed")
            arm.epicure_attestation_json = attestation
            arm.epicure_attestation_sha256 = attestation_sha256
        session.add(
            GenerationAttempt(
                attempt_id=event.attempt_id,
                arm_id=event.arm_id,
                request_key_sha256=event.request_key_sha256,
                phase=event.phase,
                attempt_index=event.attempt_index,
                event_type=event.event_type,
                generation_id=event.generation_id or None,
                http_status=event.http_status,
                error_type=event.error_type or None,
                payload_sha256=event.payload_sha256,
                metadata_json={
                    **event.metadata,
                    **(
                        {
                            "verified_provider_account_authorization_id": (
                                verified_authorization.id
                            ),
                            "verified_provider_account_authorization_envelope_sha256": (
                                verified_authorization.authorization_envelope_sha256
                            ),
                        }
                        if verified_authorization is not None
                        else {}
                    ),
                    **(
                        {
                            "verified_spend_authorization_id": (verified_commercial_spend.id),
                            "verified_spend_authorization_binding_sha256": (
                                verified_commercial_spend.binding_sha256
                            ),
                        }
                        if verified_commercial_spend is not None
                        else {}
                    ),
                    **claim_metadata,
                },
            )
        )


def _persist_tool_trace(
    arm_id: str,
    trace: ToolTrace,
    *,
    job_id: str,
    claimed_by: str,
    claim_attempt: int,
) -> None:
    """Commit each complete MCP call before the model can request another round."""

    with session_scope() as session:
        if (
            _current_job_lease(
                session,
                job_id,
                claimed_by,
                claim_attempt,
                lock=True,
            )
            is None
        ):
            raise LostJobLease("worker lease was superseded before MCP trace commit")
        if session.get(ResponseArm, arm_id) is None:
            raise ProviderError("tool trace references an unknown response arm")
        prior = session.scalar(
            select(ToolCall).where(
                ToolCall.arm_id == arm_id,
                ToolCall.round_index == trace.round_index,
                ToolCall.call_index == trace.call_index,
            )
        )
        if prior is not None:
            raise ProviderError("tool trace position already exists for this response arm")
        session.add(
            ToolCall(
                arm_id=arm_id,
                round_index=trace.round_index,
                call_index=trace.call_index,
                tool_call_id=trace.tool_call_id or None,
                tool_name=trace.name,
                arguments_json=trace.arguments,
                result_text=trace.result,
                structured_content_json=trace.structured_content,
                result_sha256=hashlib.sha256(trace.result.encode()).hexdigest(),
                latency_ms=trace.latency_ms,
                is_error=trace.is_error,
            )
        )


def claim_job(session: Session, worker_id: str) -> tuple[str, str, int] | None:
    now = datetime.now(UTC)
    job = session.scalar(
        select(Job)
        .where(
            Job.status == "queued",
            Job.available_at <= now,
            Job.attempts < Job.max_attempts,
        )
        .order_by(Job.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    if job is None:
        return None
    job.status = "running"
    job.attempts += 1
    job.claimed_by = worker_id
    job.claimed_at = now
    return job.id, worker_id, job.attempts


def _endpoint_contract(session: Session, battle: Battle, arm: ResponseArm) -> SeasonModel:
    slot = session.scalar(
        select(SeasonModel).where(
            SeasonModel.season_id == battle.season_id,
            SeasonModel.model_id == arm.model_id,
        )
    )
    if slot is None:
        raise ProviderError("response arm has no per-season endpoint contract")
    if arm.provider_slug != slot.provider_slug:
        raise ProviderError("response arm endpoint tag differs from the season manifest")
    controlled_run = (
        session.get(ControlledRun, battle.controlled_run_id)
        if battle.controlled_run_id is not None
        else None
    )
    commercial_candidate = bool(
        controlled_run is not None
        and controlled_run.evaluation_order_id is not None
        and arm.model_id == controlled_run.submitted_endpoint_model_id
    )
    if commercial_candidate:
        order = session.get(EvaluationOrder, controlled_run.evaluation_order_id)
        route = session.get(ModelRouteRevision, controlled_run.route_revision_id)
        submission = (
            session.get(ModelSubmission, order.model_submission_id) if order is not None else None
        )
        if (
            order is None
            or route is None
            or submission is None
            or order.status not in {"ready", "running"}
            or order.organization_id != controlled_run.organization_id
            or order.route_revision_id != route.id
            or order.season_id != battle.season_id
            or route.status != "approved"
            or route.model_submission_id != submission.id
            or route.descriptor_sha256 != controlled_run.endpoint_descriptor_sha256
            or route.approved_season_id != battle.season_id
            or route.approved_season_manifest_sha256 != battle.manifest_sha256
            or route.approved_endpoint_contract_sha256 != slot.endpoint_contract_sha256
            or submission.status != "approved"
            or submission.catalog_model_id != arm.model_id
            or route.execution_backend != slot.execution_backend
            or route.expected_actual_model_id != slot.expected_actual_model_id
            or route.expected_actual_provider_slug != slot.expected_actual_provider_slug
            or route.endpoint_document_sha256 != slot.endpoint_document_sha256
            or sorted(route.supported_parameters_json) != sorted(slot.supported_parameters_json)
            or route.rate_card_json != slot.rate_card_json
            or route.decoding_bounds_json.get("maxTokens") != slot.endpoint_max_completion_tokens
            or route.decoding_bounds_json.get("temperatureMaximum")
            != slot.decoding_json.get("temperature")
            or route.data_policy_json.get("training") != "deny"
            or route.data_policy_json.get("retention") != "deny"
            or slot.backend_contract_json.get("data_collection") != "deny"
            or arm.route_revision_id != route.id
            or arm.endpoint_descriptor_sha256 != route.descriptor_sha256
        ):
            raise ProviderError(
                "commercial response arm is not bound to its approved frozen endpoint"
            )
    elif arm.route_revision_id is not None or arm.endpoint_descriptor_sha256 is not None:
        raise ProviderError("response arm claims a route outside its commercial binding")
    supported = slot.supported_parameters_json
    decoding = slot.decoding_json
    if (
        not isinstance(supported, list)
        or not supported
        or any(not isinstance(item, str) for item in supported)
        or not isinstance(decoding, dict)
    ):
        raise ProviderError("season endpoint parameter contract is malformed")
    missing = REQUIRED_ENDPOINT_PARAMETERS - set(supported)
    if missing:
        raise ProviderError(
            "season endpoint contract is missing required parameters: " + ", ".join(sorted(missing))
        )
    if set(decoding) - DECODING_PARAMETERS or set(decoding) - set(supported):
        raise ProviderError("season decoding contract contains unsupported parameters")
    max_tokens = decoding.get("max_tokens")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
        or (
            slot.endpoint_max_completion_tokens > 0
            and max_tokens > slot.endpoint_max_completion_tokens
        )
    ):
        raise ProviderError("season decoding contract has an invalid output-token bound")

    settings = get_settings()
    if settings.execution_mode == "live":
        season = session.get(Season, battle.season_id)
        if (
            season is None
            or battle.manifest_sha256 in UNFROZEN_VALUES
            or battle.manifest_sha256 != season.manifest_sha256
            or slot.manifest_sha256 != battle.manifest_sha256
            or slot.execution_backend != arm.execution_backend
            or slot.expected_actual_model_id in UNFROZEN_VALUES
            or slot.expected_actual_provider_slug in UNFROZEN_VALUES
            or slot.endpoint_document_sha256 in UNFROZEN_VALUES
            or slot.endpoint_contract_sha256 in UNFROZEN_VALUES
            or (
                slot.execution_backend in {"bedrock", "kimi_direct", "qwencloud_direct"}
                and (
                    slot.backend_contract_sha256 in UNFROZEN_VALUES
                    or _canonical_sha256(slot.backend_contract_json) != slot.backend_contract_sha256
                )
            )
        ):
            raise ProviderError("live battle is not bound to a complete frozen endpoint contract")
        protocol_bundle, protocol_bundle_sha256 = build_protocol_bundle(
            tool_registry_sha256=season.tool_registry_sha256,
            epicure_release_id=season.epicure_release_id,
            epicure_bundle_sha256=season.epicure_bundle_sha256,
            epicure_application_sha256=season.epicure_application_sha256,
            analysis_plan_sha256=season.analysis_plan_sha256,
            model_smoke_registry_sha256=str(
                season.protocol_bundle_json.get("model_smoke_registry_sha256", "unfrozen")
            ),
        )
        expected_arm_decoding = {
            **{
                name: decoding.get(name, "provider_fixed_unsupported")
                for name in sorted(DECODING_PARAMETERS)
            },
            "structured_output": True,
            "max_tool_rounds": settings.max_tool_rounds,
        }
        if (
            season.protocol_bundle_sha256 in UNFROZEN_VALUES
            or season.protocol_bundle_json != protocol_bundle
            or season.protocol_bundle_sha256 != protocol_bundle_sha256
            or battle.protocol_bundle_sha256 != protocol_bundle_sha256
            or arm.protocol_bundle_sha256 != protocol_bundle_sha256
            or arm.system_prompt_sha256 != system_prompt_sha256(arm.condition)
            or arm.schema_sha256 != FINAL_SCHEMA_SHA256
            or arm.tool_schema_sha256 != season.tool_registry_sha256
            or arm.decoding_json != expected_arm_decoding
            or arm.epicure_release_id != season.epicure_release_id
            or arm.epicure_bundle_sha256 != season.epicure_bundle_sha256
            or arm.epicure_application_sha256 != season.epicure_application_sha256
        ):
            raise ProviderError("live battle execution protocol differs from its frozen bundle")
        computed = endpoint_contract_sha256(
            model_id=slot.model_id,
            provider_slug=slot.provider_slug,
            expected_actual_model_id=slot.expected_actual_model_id,
            expected_actual_provider_slug=slot.expected_actual_provider_slug,
            supported_parameters=supported,
            decoding=decoding,
            endpoint_max_completion_tokens=slot.endpoint_max_completion_tokens,
            endpoint_document_sha256=slot.endpoint_document_sha256,
        )
        if computed != slot.endpoint_contract_sha256:
            raise ProviderError("frozen endpoint contract digest does not match its contents")
    return slot


def _generation_spec(session: Session, battle: Battle, arm: ResponseArm) -> GenerationSpec:
    model = session.get(CatalogModel, arm.model_id)
    if model is None or battle.prompt is None:
        raise ProviderError("battle input or model manifest is unavailable")
    endpoint = _endpoint_contract(session, battle, arm)
    season = session.get(Season, battle.season_id)
    provider_budget = session.scalar(
        select(SeasonProviderBudget).where(
            SeasonProviderBudget.season_id == battle.season_id,
            SeasonProviderBudget.execution_backend == arm.execution_backend,
        )
    )
    account_budget = (
        session.scalar(
            select(ProviderAccountBudget).where(
                ProviderAccountBudget.execution_backend == arm.execution_backend,
                ProviderAccountBudget.account_scope_sha256 == provider_budget.account_scope_sha256,
            )
        )
        if provider_budget is not None
        and arm.execution_backend
        in {"bedrock", "kimi_direct", "openrouter", "qwencloud_direct"}
        else None
    )
    authorization = (
        account_authorization(
            session,
            account_budget,
        )
        if account_budget is not None
        else None
    )
    return GenerationSpec(
        arm_id=arm.id,
        battle_id=battle.id,
        prompt=battle.prompt,
        category=battle.category,
        model_id=arm.model_id,
        model_name=model.name,
        provider_slug=arm.provider_slug,
        condition=arm.condition,
        idempotency_key=f"flavourbench:{arm.id}",
        execution_backend=arm.execution_backend,
        rate_card_json=dict(endpoint.rate_card_json),
        backend_contract_json=dict(endpoint.backend_contract_json),
        supported_parameters=frozenset(endpoint.supported_parameters_json),
        decoding_parameters=dict(endpoint.decoding_json),
        expected_actual_model_id=endpoint.expected_actual_model_id,
        expected_actual_provider_slug=endpoint.expected_actual_provider_slug,
        endpoint_contract_sha256=endpoint.endpoint_contract_sha256,
        protocol_bundle_sha256=arm.protocol_bundle_sha256,
        expected_epicure_release_id=arm.epicure_release_id,
        expected_epicure_bundle_sha256=arm.epicure_bundle_sha256,
        expected_epicure_application_sha256=arm.epicure_application_sha256,
        expected_epicure_tool_schema_sha256=arm.tool_schema_sha256,
        provider_budget_cap_micros=(
            provider_budget.budget_cap_micros if provider_budget is not None else 0
        ),
        provider_account_budget_cap_micros=(
            account_budget.budget_cap_micros if account_budget is not None else 0
        ),
        provider_account_scope_sha256=(
            provider_budget.account_scope_sha256 if provider_budget is not None else "unresolved"
        ),
        provider_authorization_envelope_sha256=(
            provider_budget.authorization_envelope_sha256
            if provider_budget is not None
            else "unresolved"
        ),
        provider_account_authorization_envelope_sha256=(
            authorization.authorization_envelope_sha256
            if authorization is not None
            else "unresolved"
        ),
        provider_credential_binding_sha256=(
            authorization.credential_binding_sha256 if authorization is not None else "unresolved"
        ),
        provider_credential_scope_sha256=(
            str(authorization.credential_binding_json.get("credential_scope_sha256", "unresolved"))
            if authorization is not None
            else "unresolved"
        ),
        contract_smoke_registry_sha256=(
            str(season.protocol_bundle_json.get("model_smoke_registry_sha256", "unresolved"))
            if season is not None
            else "unresolved"
        ),
    )


async def _generate_arm(
    battle_id: str,
    arm_id: str,
    provider: object,
    *,
    job_id: str,
    claimed_by: str,
    claim_attempt: int,
) -> tuple[str, GenerationResult | GenerationFailureResult | Exception]:
    with session_scope() as session:
        if (
            _current_job_lease(
                session,
                job_id,
                claimed_by,
                claim_attempt,
                lock=True,
            )
            is None
        ):
            return arm_id, LostJobLease("worker lease was superseded before arm dispatch")
        battle = session.get(Battle, battle_id)
        arm = session.get(ResponseArm, arm_id)
        if battle is None or arm is None:
            return arm_id, ProviderError("battle arm no longer exists")
        if arm.status == "complete":
            return arm_id, ProviderError("already_complete")
        arm.status = "running"
        spec = _generation_spec(session, battle, arm)
    heartbeat_stop = asyncio.Event()
    heartbeat = asyncio.create_task(
        _heartbeat_generation_job(
            job_id,
            claimed_by,
            claim_attempt,
            heartbeat_stop,
        )
    )
    try:
        result = await provider.generate(spec)  # type: ignore[attr-defined]
        return arm_id, result
    except Exception as exc:  # worker boundary records the typed failure
        reconcile_failure = getattr(provider, "reconcile_failure", None)
        if isinstance(exc, ProviderError) and callable(reconcile_failure):
            try:
                failure = await reconcile_failure(spec, exc)
            except Exception as reconciliation_error:
                return arm_id, UncertainDeliveryError(
                    "accepted generation could not be reconciled after failure: "
                    f"{type(reconciliation_error).__name__}"
                )
            if failure is not None:
                return arm_id, failure
        return arm_id, exc
    finally:
        heartbeat_stop.set()
        await heartbeat


async def _heartbeat_generation_job(
    job_id: str,
    claimed_by: str,
    claim_attempt: int,
    stop: asyncio.Event,
) -> None:
    """Renew the PostgreSQL job lease while a provider/tool loop is active."""

    interval = max(
        5.0,
        min(30.0, get_settings().worker_claim_timeout_seconds / 3),
    )
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            with session_scope() as session:
                job = _current_job_lease(
                    session,
                    job_id,
                    claimed_by,
                    claim_attempt,
                    lock=True,
                )
                if job is None:
                    return
                job.claimed_at = datetime.now(UTC)


def _record_result(
    session: Session, battle: Battle, arm: ResponseArm, result: GenerationResult
) -> None:
    # Provider accounting is operational evidence, even when the answer later
    # fails identity, decoding, anonymity, or scientific eligibility checks.
    arm.actual_model_id = result.actual_model_id
    arm.actual_provider_slug = result.provider_slug
    arm.generation_id = result.generation_id or None
    arm.provider_generation_ids_json = result.generation_ids
    arm.prompt_tokens = result.prompt_tokens
    arm.completion_tokens = result.completion_tokens
    arm.reasoning_tokens = result.reasoning_tokens
    arm.observed_decoding_json = result.decoding_json
    if result.epicure_attestation:
        arm.epicure_attestation_json = result.epicure_attestation
        arm.epicure_attestation_sha256 = _canonical_sha256(result.epicure_attestation)
    arm.cost_micros = max(0, result.cost_micros)
    arm.cost_reconciled = result.cost_reconciled
    arm.cost_accounting_basis = result.cost_accounting_basis
    arm.billing_reconciliation_status = result.billing_reconciliation_status
    arm.backend_response_schema_sha256 = result.backend_response_schema_sha256
    arm.backend_tool_schema_sha256 = result.backend_tool_schema_sha256
    arm.latency_ms = result.latency_ms
    arm.retries = result.retries
    arm.finish_reason = result.finish_reason
    if (
        session.scalar(
            select(CostEvent.id).where(
                CostEvent.arm_id == arm.id,
                CostEvent.kind == "actual",
            )
        )
        is None
    ):
        session.add(
            CostEvent(
                season_id=battle.season_id,
                battle_id=battle.id,
                arm_id=arm.id,
                kind="actual",
                amount_micros=arm.cost_micros,
                provider=result.provider_slug,
                generation_id=result.generation_id or None,
                accounting_json={
                    "generation_ids": result.generation_ids,
                    "generation_metadata": result.generation_metadata,
                    "reconciled": result.cost_reconciled,
                    "cost_accounting_basis": result.cost_accounting_basis,
                    "billing_reconciliation_status": (result.billing_reconciliation_status),
                },
            )
        )
    model = session.get(CatalogModel, arm.model_id)
    if model is None:
        raise ProviderError("model manifest disappeared during generation")
    endpoint = _endpoint_contract(session, battle, arm)
    expected_model = endpoint.expected_actual_model_id
    expected_provider = endpoint.expected_actual_provider_slug
    if result.actual_model_id != expected_model:
        session.add(
            Incident(
                severity="high",
                code="provider_model_substitution",
                detail=(
                    f"requested alias {arm.model_id}; expected frozen actual model "
                    f"{expected_model}; returned {result.actual_model_id}"
                ),
                battle_id=battle.id,
            )
        )
        raise ProviderError("provider returned a model outside the frozen endpoint contract")
    if result.provider_slug != expected_provider:
        session.add(
            Incident(
                severity="high",
                code="provider_endpoint_substitution",
                detail=(
                    f"requested endpoint {arm.provider_slug}; expected frozen actual provider "
                    f"{expected_provider}; returned {result.provider_slug}"
                ),
                battle_id=battle.id,
            )
        )
        raise ProviderError("provider returned a provider outside the frozen endpoint contract")
    if get_settings().execution_mode == "live":
        if any(
            value in UNFROZEN_VALUES or len(value) != 64
            for value in (
                result.backend_response_schema_sha256,
                result.backend_tool_schema_sha256,
            )
        ):
            raise ProviderError("provider did not retain its exact sent schema hashes")
        metadata_generation_ids = [
            str(item.get("generation_id") or "") for item in result.generation_metadata
        ]
        if (
            len(result.generation_metadata) != len(result.generation_ids)
            or not result.generation_id
            or result.generation_id not in result.generation_ids
            or len(set(result.generation_ids)) != len(result.generation_ids)
            or sorted(metadata_generation_ids) != sorted(result.generation_ids)
        ):
            raise ProviderError("provider accounting did not identify every generation request")
        for item in result.generation_metadata:
            if item.get("model") != expected_model or item.get("provider") != expected_provider:
                session.add(
                    Incident(
                        severity="high",
                        code="provider_subrequest_substitution",
                        detail=(
                            "one or more tool/final requests differed from the frozen actual "
                            f"identity {expected_model}@{expected_provider}"
                        ),
                        battle_id=battle.id,
                    )
                )
                raise ProviderError(
                    "provider subrequest identity is outside the frozen endpoint contract"
                )
    if get_settings().execution_mode == "live" and not result.cost_reconciled:
        session.add(
            Incident(
                severity="high",
                code="cost_reconciliation_failed",
                detail=f"generation accounting unavailable for {result.generation_id}",
                battle_id=battle.id,
            )
        )
        raise ProviderError("provider generation cost could not be reconciled")
    if not is_complete_finish_reason(result.finish_reason):
        session.add(
            Incident(
                severity="high",
                code="incomplete_generation",
                detail=(
                    "The final provider response ended before normal completion: "
                    f"{result.finish_reason or 'missing'}"
                ),
                battle_id=battle.id,
            )
        )
        raise ProviderError("provider final response did not complete normally")
    expected_decoding = {
        name: endpoint.decoding_json.get(name, "provider_fixed_unsupported")
        for name in sorted(DECODING_PARAMETERS)
    }
    if result.decoding_json != expected_decoding:
        session.add(
            Incident(
                severity="high",
                code="provider_decoding_contract_mismatch",
                detail="effective decoding does not match the frozen season contract",
                battle_id=battle.id,
            )
        )
        raise ProviderError("provider decoding differs from the frozen endpoint contract")
    if get_settings().execution_mode == "live" and arm.condition == "epicure_on":
        expected_epicure = {
            "release_id": arm.epicure_release_id,
            "bundle_sha256": arm.epicure_bundle_sha256,
            "application_sha256": arm.epicure_application_sha256,
            "tool_schema_sha256": arm.tool_schema_sha256,
        }
        if (
            not result.epicure_attestation
            or any(
                result.epicure_attestation.get(field) != value
                for field, value in expected_epicure.items()
            )
            or arm.epicure_attestation_sha256 != _canonical_sha256(result.epicure_attestation)
        ):
            raise ProviderError("Epicure session identity differs from the frozen intervention")
    elif get_settings().execution_mode == "live" and result.epicure_attestation:
        raise ProviderError("Epicure-off arm unexpectedly received Epicure runtime access")
    if contains_identity_leak(
        result.answer_markdown,
        model.name,
        battle.prompt or "",
    ):
        session.add(
            Incident(
                severity="medium",
                code="identity_leak",
                detail="response contained a provider or model identity marker",
                battle_id=battle.id,
            )
        )
        raise ProviderError("response failed anonymous-display validation")

    try:
        task_validation = task_validator_receipt_for_battle(
            session,
            battle,
            result.answer_markdown,
            expected_container_image_digest=get_settings().build_image_digest,
        )
    except TaskEvidenceError as exc:
        session.add(
            Incident(
                severity="high",
                code="task_evidence_runtime_failure",
                detail=f"Frozen task evidence did not reproduce: {exc}",
                battle_id=battle.id,
            )
        )
        raise ProviderError("frozen task evidence failed at response scoring") from exc

    arm.status = "complete"
    arm.answer_markdown = result.answer_markdown
    arm.answer_markdown_sha256 = hashlib.sha256(result.answer_markdown.encode()).hexdigest()
    arm.output_json = result.output_json
    arm.output_json_sha256 = _canonical_sha256(result.output_json)
    arm.system_prompt_sha256 = system_prompt_sha256(arm.condition)
    arm.schema_sha256 = FINAL_SCHEMA_SHA256
    arm.error_code = None
    arm.error_detail = None
    arm.completed_at = datetime.now(UTC)

    for trace in result.tool_traces:
        persisted = session.scalar(
            select(ToolCall).where(
                ToolCall.arm_id == arm.id,
                ToolCall.round_index == trace.round_index,
                ToolCall.call_index == trace.call_index,
            )
        )
        if persisted is None:
            session.add(
                ToolCall(
                    arm_id=arm.id,
                    round_index=trace.round_index,
                    call_index=trace.call_index,
                    tool_call_id=trace.tool_call_id or None,
                    tool_name=trace.name,
                    arguments_json=trace.arguments,
                    result_text=trace.result,
                    structured_content_json=trace.structured_content,
                    result_sha256=hashlib.sha256(trace.result.encode()).hexdigest(),
                    latency_ms=trace.latency_ms,
                    is_error=trace.is_error,
                )
            )
    validations = validate_output(
        prompt=battle.prompt or "",
        output=result.output_json,
        answer=result.answer_markdown,
        model_name=model.name,
        tool_errors=sum(trace.is_error for trace in result.tool_traces),
        tool_calls=len(result.tool_traces),
        finish_reason=result.finish_reason,
    )
    for validation in validations:
        session.add(
            ValidatorResult(
                arm_id=arm.id,
                validator_name=validation.name,
                validator_version=VALIDATOR_VERSION,
                status=validation.status,
                score_milli=validation.score_milli,
                detail_json=validation.detail,
            )
        )
    if task_validation is not None:
        session.add(
            ValidatorResult(
                arm_id=arm.id,
                validator_name=TASK_SPECIFIC_VALIDATOR_NAME,
                validator_version=TASK_SPECIFIC_VALIDATOR_VERSION,
                status=str(task_validation["status"]),
                score_milli=(
                    int(task_validation["score_milli"])
                    if task_validation["score_milli"] is not None
                    else None
                ),
                detail_json=task_validation,
            )
        )


def _record_failure_result(
    session: Session,
    battle: Battle,
    arm: ResponseArm,
    result: GenerationFailureResult,
) -> None:
    """Persist cost and identity for a paid but scientifically invalid arm."""

    arm.actual_model_id = result.actual_model_id
    arm.actual_provider_slug = result.provider_slug
    arm.generation_id = result.generation_id or None
    arm.provider_generation_ids_json = result.generation_ids
    arm.prompt_tokens = result.prompt_tokens
    arm.completion_tokens = result.completion_tokens
    arm.observed_decoding_json = result.decoding_json
    arm.cost_micros = max(0, result.cost_micros)
    arm.cost_reconciled = result.cost_reconciled
    arm.cost_accounting_basis = result.cost_accounting_basis
    arm.billing_reconciliation_status = result.billing_reconciliation_status
    arm.backend_response_schema_sha256 = result.backend_response_schema_sha256
    arm.backend_tool_schema_sha256 = result.backend_tool_schema_sha256
    arm.retries = result.retries
    arm.finish_reason = "error"
    arm.status = "failed" if result.cost_reconciled else "uncertain"
    arm.error_code = type(result.error).__name__
    arm.error_detail = str(result.error)
    arm.completed_at = datetime.now(UTC)
    if (
        session.scalar(
            select(CostEvent.id).where(
                CostEvent.arm_id == arm.id,
                CostEvent.kind == "actual",
            )
        )
        is None
    ):
        session.add(
            CostEvent(
                season_id=battle.season_id,
                battle_id=battle.id,
                arm_id=arm.id,
                kind="actual",
                amount_micros=arm.cost_micros,
                provider=result.provider_slug,
                generation_id=result.generation_id or None,
                accounting_json={
                    "basis": "accepted_response_failed_scientific_validation",
                    "generation_ids": result.generation_ids,
                    "generation_metadata": result.generation_metadata,
                    "reconciled": result.cost_reconciled,
                    "cost_accounting_basis": result.cost_accounting_basis,
                    "billing_reconciliation_status": (result.billing_reconciliation_status),
                    "failure_type": type(result.error).__name__,
                },
            )
        )
    if not result.cost_reconciled:
        session.add(
            Incident(
                severity="high",
                code="failed_arm_cost_reconciliation_incomplete",
                detail=(
                    "One or more accepted provider responses failed scientific processing "
                    "and could not be fully reconciled."
                ),
                battle_id=battle.id,
            )
        )
    session.add(
        RunEvent(
            entity_type="response_arm",
            entity_id=arm.id,
            event_type="generation_completed",
            payload_json={
                "generation_id": result.generation_id,
                "actual_model_id": result.actual_model_id,
                "actual_provider": result.provider_slug,
                "cost_micros": result.cost_micros,
                "latency_ms": result.latency_ms,
                "tool_calls": len(result.tool_traces),
            },
        )
    )


def reconcile_battle_cost(session: Session, battle: Battle) -> None:
    if _uses_postgresql_budget_authority(session):
        # Arm receipts and the terminal parent record must be visible to the
        # owner function. PostgreSQL then owns every mutable aggregate and
        # governed settlement event in one transaction.
        session.flush()
        result = (
            session.execute(
                text(
                    "SELECT released_micros, actual_micros, cost_halted, "
                    "halt_reasons, idempotent "
                    "FROM public.flavourbench_settle_battle_budget(:battle_id)"
                ),
                {"battle_id": battle.id},
            )
            .mappings()
            .one()
        )
        session.expire_all()
        battle = session.get(Battle, battle.id)
        if battle is None:
            raise RuntimeError("settled battle disappeared")
        season = session.get(Season, battle.season_id)
        if season is None:
            raise RuntimeError("settled battle season disappeared")
        halt_reasons = [str(reason) for reason in (result["halt_reasons"] or [])]
        if bool(result["cost_halted"]) and not bool(result["idempotent"]):
            session.add(
                Incident(
                    severity="critical",
                    code=(
                        "actual_cost_exceeded_frozen_reservation"
                        if any("actual_exceeds_reservation" in reason for reason in halt_reasons)
                        else "governed_cost_cap_exceeded"
                    ),
                    detail=(
                        "PostgreSQL budget authority halted the season after settlement: "
                        + ", ".join(halt_reasons)
                    ),
                    battle_id=battle.id,
                )
            )
            withdrawn_snapshot_ids = _withdraw_snapshots_for_cost_halt(
                session,
                season.id,
            )
            session.add(
                RunEvent(
                    entity_type="season",
                    entity_id=season.id,
                    event_type="cost_halt_snapshot_withdrawal",
                    payload_json={"withdrawn_snapshot_ids": withdrawn_snapshot_ids},
                )
            )
        session.flush()
        assert_budget_integrity(session, season.id, lock_aggregates=True)
        return

    season = session.scalar(select(Season).where(Season.id == battle.season_id).with_for_update())
    if season is None:
        return
    controlled_run = (
        session.scalar(
            select(ControlledRun)
            .where(ControlledRun.id == battle.controlled_run_id)
            .with_for_update()
        )
        if battle.controlled_run_id
        else None
    )
    live_budget_controls = get_settings().execution_mode == "live"
    if live_budget_controls:
        assert_budget_integrity(session, season.id, lock_aggregates=True)
    if session.scalar(
        select(CostEvent.id).where(
            CostEvent.battle_id == battle.id,
            CostEvent.kind == "reconcile",
        )
    ):
        return
    arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
    actual = sum(arm.cost_micros for arm in arms)
    actual_by_backend: dict[str, int] = {}
    for arm in arms:
        backend = arm.execution_backend or "openrouter"
        actual_by_backend[backend] = actual_by_backend.get(backend, 0) + arm.cost_micros
    reserved = battle.reserved_cost_micros
    if actual > reserved:
        season.status = "cost_halted"
        session.add(
            Incident(
                severity="critical",
                code="actual_cost_exceeded_frozen_reservation",
                detail=(
                    f"Actual cost {actual} micros exceeded the frozen reservation "
                    f"{reserved}; further season admission was halted."
                ),
                battle_id=battle.id,
            )
        )
    season.budget_reserved_micros = decrement_reservation(
        season.budget_reserved_micros,
        reserved,
        label=f"season {season.id}",
    )
    season.budget_used_micros += actual
    provider_reservations = {
        str(backend): int(amount)
        for backend, amount in (battle.provider_reservations_json or {}).items()
    }
    provider_budgets = session.scalars(
        select(SeasonProviderBudget)
        .where(SeasonProviderBudget.season_id == season.id)
        .order_by(SeasonProviderBudget.execution_backend)
        .with_for_update()
    ).all()
    if provider_budgets:
        budgets_by_backend = {row.execution_backend: row for row in provider_budgets}
        account_budget_rows = session.scalars(
            select(ProviderAccountBudget)
            .where(
                ProviderAccountBudget.execution_backend.in_(sorted(budgets_by_backend)),
                ProviderAccountBudget.account_scope_sha256.in_(
                    sorted(row.account_scope_sha256 for row in provider_budgets)
                ),
            )
            .order_by(
                ProviderAccountBudget.execution_backend,
                ProviderAccountBudget.account_scope_sha256,
            )
            .with_for_update()
        ).all()
        account_budgets = {
            (row.execution_backend, row.account_scope_sha256): row for row in account_budget_rows
        }
        budget_keys = set(provider_reservations) | set(actual_by_backend)
        if (
            not budget_keys.issubset(budgets_by_backend)
            or sum(provider_reservations.values()) != reserved
        ):
            season.status = "cost_halted"
            session.add(
                Incident(
                    severity="critical",
                    code="provider_budget_reconciliation_contract_mismatch",
                    detail=(
                        "The battle provider reservation did not match the authorized "
                        "provider ledger; further season admission was halted."
                    ),
                    battle_id=battle.id,
                )
            )
        for backend in sorted(set(budgets_by_backend) & budget_keys):
            provider_budget = budgets_by_backend[backend]
            account_budget = account_budgets.get((backend, provider_budget.account_scope_sha256))
            provider_reserved = provider_reservations.get(backend, 0)
            provider_actual = actual_by_backend.get(backend, 0)
            if account_budget is None:
                season.status = "cost_halted"
                session.add(
                    Incident(
                        severity="critical",
                        code="provider_account_budget_reconciliation_missing",
                        detail=(
                            f"{backend} account-wide budget authorization was missing "
                            "during reconciliation."
                        ),
                        battle_id=battle.id,
                    )
                )
            if provider_actual > provider_reserved:
                season.status = "cost_halted"
                session.add(
                    Incident(
                        severity="critical",
                        code="provider_actual_cost_exceeded_frozen_reservation",
                        detail=(
                            f"{backend} actual cost {provider_actual} micros exceeded "
                            f"its frozen reservation {provider_reserved}."
                        ),
                        battle_id=battle.id,
                    )
                )
            provider_budget.budget_reserved_micros = decrement_reservation(
                provider_budget.budget_reserved_micros,
                provider_reserved,
                label=f"season provider {backend}",
            )
            provider_budget.budget_used_micros += provider_actual
            if account_budget is not None:
                if provider_actual > provider_reserved:
                    season.status = "cost_halted"
                account_budget.budget_reserved_micros = decrement_reservation(
                    account_budget.budget_reserved_micros,
                    provider_reserved,
                    label=f"provider account {backend}",
                )
                account_budget.budget_used_micros += provider_actual
                if account_budget.budget_used_micros > account_budget.budget_cap_micros:
                    season.status = "cost_halted"
            if provider_budget.budget_used_micros > provider_budget.budget_cap_micros:
                season.status = "cost_halted"
            if provider_reserved:
                session.add(
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="provider_release",
                        amount_micros=-provider_reserved,
                        provider=backend,
                        accounting_json={"budget_scope": "provider"},
                    )
                )
            session.add(
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_reconcile",
                    amount_micros=provider_actual,
                    provider=backend,
                    accounting_json={"budget_scope": "provider"},
                )
            )
            if account_budget is not None:
                if provider_reserved:
                    session.add(
                        CostEvent(
                            season_id=season.id,
                            battle_id=battle.id,
                            kind="provider_account_release",
                            amount_micros=-provider_reserved,
                            provider=backend,
                            accounting_json={
                                "budget_scope": "provider_account",
                                "account_scope_sha256": (provider_budget.account_scope_sha256),
                            },
                        )
                    )
                session.add(
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="provider_account_reconcile",
                        amount_micros=provider_actual,
                        provider=backend,
                        accounting_json={
                            "budget_scope": "provider_account",
                            "account_scope_sha256": (provider_budget.account_scope_sha256),
                        },
                    )
                )
    if controlled_run is not None:
        controlled_run.budget_reserved_micros = decrement_reservation(
            controlled_run.budget_reserved_micros,
            reserved,
            label=f"controlled run {controlled_run.id}",
        )
        controlled_run.budget_used_micros += actual
    battle.reserved_cost_micros = 0
    if reserved:
        session.add(
            CostEvent(
                season_id=season.id,
                battle_id=battle.id,
                kind="release",
                amount_micros=-reserved,
                provider="governor",
            )
        )
    session.add(
        CostEvent(
            season_id=season.id,
            battle_id=battle.id,
            kind="reconcile",
            amount_micros=actual,
            provider="governor",
        )
    )
    if season.status == "cost_halted":
        withdrawn_snapshot_ids = _withdraw_snapshots_for_cost_halt(
            session,
            season.id,
        )
        session.add(
            RunEvent(
                entity_type="season",
                entity_id=season.id,
                event_type="cost_halt_snapshot_withdrawal",
                payload_json={"withdrawn_snapshot_ids": withdrawn_snapshot_ids},
            )
        )
    if live_budget_controls:
        session.flush()
        assert_budget_integrity(session, season.id, lock_aggregates=True)


def _record_known_zero_cost(
    session: Session,
    battle: Battle,
    arm: ResponseArm,
    *,
    reason: str,
) -> None:
    """Close a provably unbilled arm without conflating it with uncertain delivery."""

    if arm.cost_reconciled:
        return
    arm.cost_micros = 0
    arm.cost_reconciled = True
    arm.cost_accounting_basis = "known_zero_no_provider_acceptance"
    arm.billing_reconciliation_status = "known_zero_no_provider_acceptance"
    session.add(
        CostEvent(
            season_id=battle.season_id,
            battle_id=battle.id,
            arm_id=arm.id,
            kind="actual",
            amount_micros=0,
            provider=arm.provider_slug,
            accounting_json={
                "basis": "known_zero_no_provider_acceptance",
                "reason": reason,
                "reconciled": True,
                "cost_accounting_basis": "known_zero_no_provider_acceptance",
                "billing_reconciliation_status": "known_zero_no_provider_acceptance",
            },
        )
    )


def cancel_unstarted_controlled_jobs(session: Session, controlled_run: ControlledRun) -> int:
    """Cancel queued work and release its reservations during run revocation."""

    battles = session.scalars(
        select(Battle)
        .where(
            Battle.controlled_run_id == controlled_run.id,
            Battle.status == "queued",
        )
        .with_for_update(skip_locked=True)
    ).all()
    cancelled = 0
    for battle in battles:
        job = session.scalar(
            select(Job)
            .where(Job.battle_id == battle.id, Job.status == "queued")
            .with_for_update(skip_locked=True)
        )
        if job is None:
            continue
        job.status = "failed"
        job.last_error = "controlled run was revoked before generation"
        job.completed_at = datetime.now(UTC)
        arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
        for arm in arms:
            if arm.status != "complete":
                arm.status = "failed"
                arm.error_code = "ControlledRunRevoked"
                arm.error_detail = job.last_error
                arm.completed_at = datetime.now(UTC)
                _record_known_zero_cost(
                    session,
                    battle,
                    arm,
                    reason="controlled_run_revoked_before_generation",
                )
        _terminalize_battle_after_arms(session, battle, arms, status="failed")
        assignment = session.scalar(
            select(ControlledRunAssignment).where(
                ControlledRunAssignment.controlled_run_id == controlled_run.id,
                ControlledRunAssignment.battle_id == battle.id,
            )
        )
        if assignment is not None:
            assignment.status = "cancelled"
        reconcile_battle_cost(session, battle)
        session.add(
            RunEvent(
                entity_type="battle",
                entity_id=battle.id,
                event_type="generation_cancelled_before_start",
                payload_json={"controlled_run_id": controlled_run.id},
            )
        )
        cancelled += 1
    pending = session.scalars(
        select(ControlledRunAssignment)
        .where(
            ControlledRunAssignment.controlled_run_id == controlled_run.id,
            ControlledRunAssignment.status == "pending",
        )
        .with_for_update(skip_locked=True)
    ).all()
    for assignment in pending:
        assignment.status = "cancelled"
    return cancelled


async def process_generation_job(
    job_id: str,
    claimed_by: str,
    claim_attempt: int,
) -> None:
    with session_scope() as session:
        job = _current_job_lease(
            session,
            job_id,
            claimed_by,
            claim_attempt,
            lock=True,
        )
        if job is None:
            return
        battle = session.get(Battle, job.battle_id) if job and job.battle_id else None
        if job is None or battle is None:
            raise RuntimeError("generation job has no battle")
        season = session.get(Season, battle.season_id)
        controlled_run = (
            session.get(ControlledRun, battle.controlled_run_id)
            if battle.controlled_run_id
            else None
        )
        settings = get_settings()
        if controlled_run is not None and controlled_run.status not in {
            "active",
            "collection_complete",
        }:
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.last_error = "controlled run no longer permits generation"
            arms = session.scalars(
                select(ResponseArm).where(ResponseArm.battle_id == battle.id)
            ).all()
            for arm in arms:
                if arm.status != "complete":
                    arm.status = "failed"
                    arm.error_code = "ControlledRunInactive"
                    arm.error_detail = job.last_error
                    arm.completed_at = datetime.now(UTC)
                    _record_known_zero_cost(
                        session,
                        battle,
                        arm,
                        reason="controlled_run_inactive_before_generation",
                    )
            _terminalize_battle_after_arms(session, battle, arms, status="failed")
            reconcile_battle_cost(session, battle)
            return
        if (
            settings.execution_mode == "live"
            and season
            and season.budget_cap_micros > 0
            and season.budget_used_micros >= season.budget_cap_micros
        ):
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.last_error = "hard budget stop reached before generation"
            arms = session.scalars(
                select(ResponseArm).where(ResponseArm.battle_id == battle.id)
            ).all()
            for arm in arms:
                if arm.status != "complete":
                    arm.status = "failed"
                    arm.error_code = "BudgetHardStop"
                    arm.error_detail = job.last_error
                    arm.completed_at = datetime.now(UTC)
                    _record_known_zero_cost(
                        session, battle, arm, reason="season_budget_stop_before_generation"
                    )
            _terminalize_battle_after_arms(session, battle, arms, status="failed")
            session.add(
                Incident(
                    severity="critical",
                    code="budget_hard_stop",
                    detail="No generation started because the season reached 100% of its cap.",
                    battle_id=battle.id,
                )
            )
            reconcile_battle_cost(session, battle)
            return
        if (
            settings.execution_mode == "live"
            and controlled_run
            and controlled_run.budget_cap_micros > 0
            and controlled_run.budget_used_micros >= controlled_run.budget_cap_micros
        ):
            job.status = "failed"
            job.completed_at = datetime.now(UTC)
            job.last_error = "controlled-run hard budget stop reached before generation"
            arms = session.scalars(
                select(ResponseArm).where(ResponseArm.battle_id == battle.id)
            ).all()
            for arm in arms:
                if arm.status != "complete":
                    arm.status = "failed"
                    arm.error_code = "ControlledRunBudgetHardStop"
                    arm.error_detail = job.last_error
                    arm.completed_at = datetime.now(UTC)
                    _record_known_zero_cost(
                        session,
                        battle,
                        arm,
                        reason="controlled_run_budget_stop_before_generation",
                    )
            _terminalize_battle_after_arms(session, battle, arms, status="failed")
            session.add(
                Incident(
                    severity="critical",
                    code="controlled_run_budget_hard_stop",
                    detail=(
                        "No generation started because the controlled run reached 100% of its cap."
                    ),
                    battle_id=battle.id,
                )
            )
            reconcile_battle_cost(session, battle)
            return
        if (
            settings.execution_mode == "live"
            and season
            and season.budget_cap_micros > 0
            and (season.budget_used_micros + season.budget_reserved_micros) * 10_000
            >= season.budget_cap_micros * 9_500
        ):
            session.add(
                RunEvent(
                    entity_type="season",
                    entity_id=season.id,
                    event_type="budget_drain_mode",
                    payload_json={"threshold_basis_points": 9500},
                )
            )
        if (
            settings.execution_mode == "live"
            and controlled_run
            and controlled_run.budget_cap_micros > 0
            and (controlled_run.budget_used_micros + controlled_run.budget_reserved_micros) * 10_000
            >= controlled_run.budget_cap_micros * 9_500
        ):
            session.add(
                RunEvent(
                    entity_type="controlled_run",
                    entity_id=controlled_run.id,
                    event_type="budget_drain_mode",
                    payload_json={"threshold_basis_points": 9500},
                )
            )
        battle.status = "running"
        arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
        pending_ids = [arm.id for arm in arms if arm.status != "complete"]
        battle_id = battle.id

    if get_settings().execution_mode == "live":
        try:
            with session_scope() as preflight_session:
                if (
                    _current_job_lease(
                        preflight_session,
                        job_id,
                        claimed_by,
                        claim_attempt,
                        lock=True,
                    )
                    is None
                ):
                    return
                preflight_battle = preflight_session.get(Battle, battle_id)
                preflight_season = (
                    preflight_session.get(Season, preflight_battle.season_id)
                    if preflight_battle is not None
                    else None
                )
                if preflight_battle is None or preflight_season is None:
                    raise ProviderError("battle season disappeared before provider dispatch")
                preflight_arms = preflight_session.scalars(
                    select(ResponseArm).where(ResponseArm.battle_id == preflight_battle.id)
                ).all()
                for preflight_arm in preflight_arms:
                    _endpoint_contract(
                        preflight_session,
                        preflight_battle,
                        preflight_arm,
                    )
                verified_task_evidence_for_battle(
                    preflight_session,
                    preflight_battle,
                    expected_container_image_digest=settings.build_image_digest,
                )
                frozen_epicure_identity = {
                    "release_id": preflight_season.epicure_release_id,
                    "bundle_sha256": preflight_season.epicure_bundle_sha256,
                    "application_sha256": preflight_season.epicure_application_sha256,
                    "tool_schema_sha256": preflight_season.tool_registry_sha256,
                }
            epicure_attestation = await attest_epicure_runtime(frozen_epicure_identity)
        except Exception as exc:
            task_evidence_failure = isinstance(exc, TaskEvidenceError)
            preflight_error_code = (
                "TaskEvidencePreflightError"
                if task_evidence_failure
                else "EpicureAttestationError"
            )
            preflight_reason = (
                "task_evidence_preflight_failed_before_provider_dispatch"
                if task_evidence_failure
                else "epicure_preflight_failed_before_provider_dispatch"
            )
            with session_scope() as session:
                job = _current_job_lease(
                    session,
                    job_id,
                    claimed_by,
                    claim_attempt,
                    lock=True,
                )
                battle = session.get(Battle, battle_id)
                if job is None:
                    return
                if battle is None:
                    raise RuntimeError("generation battle disappeared during preflight") from exc
                arms = session.scalars(
                    select(ResponseArm).where(ResponseArm.battle_id == battle.id)
                ).all()
                for arm in arms:
                    if arm.status != "complete":
                        arm.status = "failed"
                        arm.error_code = preflight_error_code
                        arm.error_detail = str(exc)
                        arm.completed_at = datetime.now(UTC)
                        _record_known_zero_cost(
                            session,
                            battle,
                            arm,
                            reason=preflight_reason,
                        )
                job.status = "failed"
                job.completed_at = datetime.now(UTC)
                job.last_error = (
                    f"{'Task evidence' if task_evidence_failure else 'Epicure identity'} "
                    f"preflight failed: {exc}"
                )[:2000]
                _terminalize_battle_after_arms(session, battle, arms, status="failed")
                session.add(
                    Incident(
                        severity="critical",
                        code=(
                            "task_evidence_preflight_failed"
                            if task_evidence_failure
                            else "epicure_identity_preflight_failed"
                        ),
                        detail=f"{type(exc).__name__}: {exc}"[:2000],
                        battle_id=battle.id,
                    )
                )
                reconcile_battle_cost(session, battle)
            return
        with session_scope() as session:
            if (
                _current_job_lease(
                    session,
                    job_id,
                    claimed_by,
                    claim_attempt,
                    lock=True,
                )
                is None
            ):
                return
            session.add(
                RunEvent(
                    entity_type="battle",
                    entity_id=battle_id,
                    event_type="epicure_runtime_attested",
                    payload_json=epicure_attestation,
                )
            )

    with session_scope() as session:
        job = _current_job_lease(
            session,
            job_id,
            claimed_by,
            claim_attempt,
            lock=True,
        )
        if job is None:
            return
        battle = session.get(Battle, battle_id)
        controlled_run = (
            session.get(ControlledRun, battle.controlled_run_id)
            if battle is not None and battle.controlled_run_id
            else None
        )
        if (
            job is None
            or battle is None
            or (
                controlled_run is not None
                and controlled_run.status not in {"active", "collection_complete"}
            )
        ):
            if job is not None and battle is not None:
                job.status = "failed"
                job.completed_at = datetime.now(UTC)
                job.last_error = "controlled run changed state before provider dispatch"
                arms = session.scalars(
                    select(ResponseArm).where(ResponseArm.battle_id == battle.id)
                ).all()
                for arm in arms:
                    if arm.status != "complete":
                        arm.status = "failed"
                        arm.error_code = "ControlledRunInactive"
                        arm.error_detail = job.last_error
                        arm.completed_at = datetime.now(UTC)
                        _record_known_zero_cost(
                            session,
                            battle,
                            arm,
                            reason="controlled_run_changed_before_provider_dispatch",
                        )
                _terminalize_battle_after_arms(session, battle, arms, status="failed")
                reconcile_battle_cost(session, battle)
            return

    provider = get_provider(
        attempt_sink=lambda event: _persist_provider_attempt(
            event,
            job_id=job_id,
            claimed_by=claimed_by,
            claim_attempt=claim_attempt,
        ),
        tool_sink=lambda arm_id, trace: _persist_tool_trace(
            arm_id,
            trace,
            job_id=job_id,
            claimed_by=claimed_by,
            claim_attempt=claim_attempt,
        ),
    )
    try:
        outcomes = await asyncio.gather(
            *(
                _generate_arm(
                    battle_id,
                    arm_id,
                    provider,
                    job_id=job_id,
                    claimed_by=claimed_by,
                    claim_attempt=claim_attempt,
                )
                for arm_id in pending_ids
            )
        )
    finally:
        await provider.aclose()
    errors: list[str] = []
    with session_scope() as session:
        job = _current_job_lease(
            session,
            job_id,
            claimed_by,
            claim_attempt,
            lock=True,
        )
        battle = session.get(Battle, battle_id)
        if job is None:
            if battle is not None:
                session.add(
                    Incident(
                        severity="high",
                        code="superseded_worker_outcome_discarded",
                        detail=(
                            "A stale worker returned after its job lease was superseded; "
                            "scientific and operational state was not mutated."
                        ),
                        battle_id=battle.id,
                    )
                )
            return
        if battle is None:
            raise RuntimeError("generation battle disappeared")
        for arm_id, outcome in outcomes:
            arm = session.get(ResponseArm, arm_id)
            if arm is None:
                errors.append("missing arm")
                continue
            if isinstance(outcome, GenerationResult):
                try:
                    _record_result(session, battle, arm, outcome)
                except Exception as exc:
                    if not outcome.cost_reconciled:
                        arm.status = "uncertain"
                        arm.actual_model_id = outcome.actual_model_id
                        arm.actual_provider_slug = outcome.provider_slug
                        arm.generation_id = outcome.generation_id or None
                        arm.provider_generation_ids_json = outcome.generation_ids
                        arm.cost_micros = max(0, outcome.cost_micros)
                        arm.cost_reconciled = False
                    else:
                        arm.status = "failed"
                    arm.error_code = type(exc).__name__
                    arm.error_detail = str(exc)
                    arm.completed_at = arm.completed_at or datetime.now(UTC)
                    errors.append(str(exc))
            elif isinstance(outcome, GenerationFailureResult):
                _record_failure_result(session, battle, arm, outcome)
                errors.append(str(outcome.error))
            elif str(outcome) != "already_complete":
                arm.status = (
                    "uncertain" if isinstance(outcome, UncertainDeliveryError) else "failed"
                )
                arm.error_code = type(outcome).__name__
                arm.error_detail = str(outcome)
                arm.completed_at = arm.completed_at or datetime.now(UTC)
                errors.append(str(outcome))

        all_arms = session.scalars(
            select(ResponseArm).where(ResponseArm.battle_id == battle.id)
        ).all()
        for arm in all_arms:
            if arm.status != "failed" or arm.generation_id is not None or arm.cost_reconciled:
                continue
            attempt_events = session.scalars(
                select(GenerationAttempt)
                .where(GenerationAttempt.arm_id == arm.id)
                .order_by(GenerationAttempt.created_at, GenerationAttempt.id)
            ).all()
            latest_by_attempt: dict[tuple[str, str], GenerationAttempt] = {}
            for event in attempt_events:
                latest_by_attempt[(event.arm_id, event.attempt_id)] = event
            paid_latest = [
                event for event in latest_by_attempt.values() if not _is_mcp_attempt(event)
            ]
            if not paid_latest or all(
                event.event_type in _SAFE_NO_COST_ATTEMPT_EVENTS for event in paid_latest
            ):
                _record_known_zero_cost(
                    session,
                    battle,
                    arm,
                    reason="provider_request_conclusively_not_accepted",
                )
        unreconciled_exposure = any(
            arm.status == "uncertain" or (arm.generation_id is not None and not arm.cost_reconciled)
            for arm in all_arms
        )
        if not unreconciled_exposure and all_arms:
            attempt_events = session.scalars(
                select(GenerationAttempt)
                .where(GenerationAttempt.arm_id.in_([arm.id for arm in all_arms]))
                .order_by(GenerationAttempt.created_at, GenerationAttempt.id)
            ).all()
            unreconciled_exposure = has_unresolved_paid_attempt(attempt_events)
        if errors or any(arm.status != "complete" for arm in all_arms):
            _terminalize_battle_after_arms(
                session,
                battle,
                all_arms,
                status="failed",
            )
            job.status = "uncertain" if unreconciled_exposure else "failed"
            job.completed_at = datetime.now(UTC)
            job.last_error = "; ".join(errors)[:2000] or "one or more response arms failed"
            event_type = (
                "battle_cost_reconciliation_held" if unreconciled_exposure else "battle_failed"
            )
            if unreconciled_exposure:
                session.add(
                    Incident(
                        severity="critical",
                        code="generation_cost_exposure_unreconciled",
                        detail=(
                            "Reserved budget remains held because provider delivery or "
                            "generation accounting is uncertain."
                        ),
                        battle_id=battle.id,
                    )
                )
        else:
            _terminalize_battle_after_arms(
                session,
                battle,
                all_arms,
                status="complete",
            )
            job.status = "complete"
            job.completed_at = datetime.now(UTC)
            event_type = "battle_completed"
        if not unreconciled_exposure:
            reconcile_battle_cost(session, battle)
        session.add(
            RunEvent(
                entity_type="battle",
                entity_id=battle.id,
                event_type=event_type,
                payload_json={"errors": errors},
            )
        )


def redact_expired(session: Session) -> int:
    now = datetime.now(UTC)
    battles = session.scalars(
        select(Battle).where(
            Battle.research_consent.is_(False),
            Battle.retention_basis.in_(
                (
                    "public_nonconsented",
                    "commercial_private",
                    "controlled_development",
                    "development_research",
                )
            ),
            Battle.prompt_redacted.is_(False),
            Battle.retention_until <= now,
        )
    ).all()
    for battle in battles:
        battle.prompt = None
        battle.prompt_redacted = True
        session.flush([battle])
        arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
        arm_ids = [arm.id for arm in arms]
        for arm in arms:
            arm.answer_markdown = None
            arm.output_json = {"redacted": True}
            arm.error_detail = None
        if arm_ids:
            calls = session.scalars(select(ToolCall).where(ToolCall.arm_id.in_(arm_ids))).all()
            for call in calls:
                call.arguments_json = TOOL_CALL_REDACTION_JSON
                call.result_text = TOOL_CALL_REDACTION_SENTINEL
                call.structured_content_json = TOOL_CALL_REDACTION_JSON
            validations = session.scalars(
                select(ValidatorResult).where(ValidatorResult.arm_id.in_(arm_ids))
            ).all()
            for validation in validations:
                validation.detail_json = TOOL_CALL_REDACTION_JSON
        jobs = session.scalars(select(Job).where(Job.battle_id == battle.id)).all()
        for job in jobs:
            if job.last_error:
                job.last_error = "[REDACTED AFTER OPERATIONAL RETENTION]"
        incidents = session.scalars(select(Incident).where(Incident.battle_id == battle.id)).all()
        for incident in incidents:
            incident.detail = "[REDACTED AFTER OPERATIONAL RETENTION]"
        events = session.scalars(
            select(RunEvent).where(
                or_(
                    and_(
                        RunEvent.entity_type == "battle",
                        RunEvent.entity_id == battle.id,
                    ),
                    and_(
                        RunEvent.entity_type == "response_arm",
                        RunEvent.entity_id.in_(arm_ids),
                    ),
                )
            )
        ).all()
        for event in events:
            event.payload_json = {"redacted": True}
        session.add(
            RunEvent(
                entity_type="battle",
                entity_id=battle.id,
                event_type="content_redacted",
                payload_json={
                    "retention_basis": battle.retention_basis,
                    "retention_deadline": battle.retention_until.isoformat(),
                },
            )
        )
    return len(battles)


async def process_job(job_id: str, claimed_by: str, claim_attempt: int) -> None:
    with session_scope() as session:
        job = _current_job_lease(
            session,
            job_id,
            claimed_by,
            claim_attempt,
        )
        kind = job.kind if job else "missing"
    if kind == "generate_battle":
        await process_generation_job(job_id, claimed_by, claim_attempt)
        return
    if kind == "leaderboard_snapshot":
        from .snapshot_worker import process_leaderboard_snapshot_job

        await asyncio.to_thread(
            process_leaderboard_snapshot_job,
            job_id,
            claimed_by,
            claim_attempt,
        )
        return
    with session_scope() as session:
        job = _current_job_lease(
            session,
            job_id,
            claimed_by,
            claim_attempt,
            lock=True,
        )
        if job:
            job.status = "failed"
            job.last_error = f"unsupported job kind: {kind}"
            job.completed_at = datetime.now(UTC)


async def run_worker_once(worker_id: str | None = None) -> bool:
    identity = worker_id or _worker_id()
    with session_scope() as session:
        lease = claim_job(session, identity)
    if lease is None:
        return False
    await process_job(*lease)
    return True
