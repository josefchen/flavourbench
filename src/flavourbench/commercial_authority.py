"""Exact, revocation-aware authority checks for commercial execution and release."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ControlledRun, EvaluationOrder, GovernanceAcceptance


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def spend_authorization_binding(order: EvaluationOrder) -> dict[str, object] | None:
    if order.quote_reference_sha256 is None:
        return None
    return {
        "orderCardSha256": order.order_card_sha256,
        "budgetCapMicros": order.budget_cap_micros,
        "currency": order.currency,
        "forecastCostMicros": order.forecast_cost_micros,
        "routeRevisionId": order.route_revision_id,
        "seasonId": order.season_id,
        "quoteReferenceSha256": order.quote_reference_sha256,
    }


def publication_authorization_binding(
    order: EvaluationOrder,
    run: ControlledRun,
) -> dict[str, object]:
    return {
        "evaluationOrderId": order.id,
        "organizationId": order.organization_id,
        "orderCardSha256": order.order_card_sha256,
        "publicationScope": "controlled_run_results_and_evidence",
        "requestedVisibility": order.requested_visibility,
        "runCardSha256": run.run_card_sha256,
        "seasonId": order.season_id,
    }


def exact_active_acceptance(
    session: Session,
    *,
    acceptance_id: str,
    organization_id: str,
    evaluation_order_id: str,
    agreement_type: str,
    expected_binding: dict[str, object],
    now: datetime | None = None,
    lock: bool = False,
) -> GovernanceAcceptance | None:
    """Return one exact authority only while its append-only chain remains active."""

    instant = _utc(now or datetime.now(UTC))
    statement = select(GovernanceAcceptance).where(GovernanceAcceptance.id == acceptance_id)
    if lock:
        statement = statement.with_for_update()
    acceptance = session.scalar(statement)
    if acceptance is None:
        return None
    superseding_statement = (
        select(GovernanceAcceptance.id)
        .where(
            GovernanceAcceptance.supersedes_acceptance_id == acceptance.id,
            GovernanceAcceptance.status == "active",
            GovernanceAcceptance.accepted_at <= instant,
        )
        .limit(1)
    )
    if lock:
        superseding_statement = superseding_statement.with_for_update()
    superseding = session.scalar(superseding_statement)
    expected_sha256 = canonical_sha256(expected_binding)
    if (
        acceptance.organization_id != organization_id
        or acceptance.evaluation_order_id != evaluation_order_id
        or acceptance.model_submission_id is not None
        or acceptance.route_revision_id is not None
        or acceptance.agreement_type != agreement_type
        or acceptance.status != "active"
        or _utc(acceptance.accepted_at) > instant
        or (acceptance.expires_at is not None and _utc(acceptance.expires_at) <= instant)
        or superseding is not None
        or acceptance.binding_json != expected_binding
        or acceptance.binding_sha256 != expected_sha256
    ):
        return None
    return acceptance


def active_spend_authorization(
    session: Session,
    *,
    order: EvaluationOrder,
    acceptance_id: str,
    binding_sha256: str,
    lock: bool = False,
) -> GovernanceAcceptance | None:
    binding = spend_authorization_binding(order)
    if binding is None or canonical_sha256(binding) != binding_sha256:
        return None
    return exact_active_acceptance(
        session,
        acceptance_id=acceptance_id,
        organization_id=order.organization_id,
        evaluation_order_id=order.id,
        agreement_type="spend_authorization",
        expected_binding=binding,
        lock=lock,
    )


def active_publication_authorization(
    session: Session,
    *,
    order: EvaluationOrder,
    run: ControlledRun,
    acceptance_id: str,
    binding_sha256: str,
    lock: bool = False,
) -> GovernanceAcceptance | None:
    binding = publication_authorization_binding(order, run)
    if canonical_sha256(binding) != binding_sha256:
        return None
    return exact_active_acceptance(
        session,
        acceptance_id=acceptance_id,
        organization_id=order.organization_id,
        evaluation_order_id=order.id,
        agreement_type="publication_authorization",
        expected_binding=binding,
        lock=lock,
    )
