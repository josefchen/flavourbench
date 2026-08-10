"""Fail-closed reconciliation of governed cost counters and append-only evidence.

The mutable reservation and used-spend counters are operational caches. The
CostEvent journal, response-arm receipts, billing crosschecks, and each battle's
immutable provider reservation contract are the evidence of record. Paid
dispatch, settlement, billing reconciliation, and publication call this module
before trusting the cached counters.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .account_authority import account_authorization_chain_valid
from .config import budget_authorization_verification_keyring, get_settings
from .models import (
    Battle,
    BedrockBillingCrosscheck,
    BedrockBillingCrosscheckArm,
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

_GOVERNOR_KINDS = frozenset({"reserve", "release", "reconcile"})
_PROVIDER_KINDS = frozenset({"provider_reserve", "provider_release", "provider_reconcile"})
_ACCOUNT_KINDS = frozenset(
    {
        "provider_account_reserve",
        "provider_account_release",
        "provider_account_reconcile",
    }
)
_RESERVATION_KINDS = _GOVERNOR_KINDS | _PROVIDER_KINDS | _ACCOUNT_KINDS
_RECEIPT_KINDS = frozenset({"actual", "actual_settlement"})
_AUDITED_COST_KINDS = (
    _RESERVATION_KINDS | _RECEIPT_KINDS | frozenset({"bedrock_billing_adjustment"})
)


@dataclass(frozen=True)
class BudgetIntegrityReport:
    """Structured result retained on a raised :class:`BudgetIntegrityError`."""

    season_id: str
    battles_checked: int
    season_reserved_micros: int
    battle_reserved_micros: int
    governor_event_net_micros: int
    season_used_micros: int
    governor_reconciled_micros: int
    provider_reserved_micros: dict[str, int]
    provider_event_net_micros: dict[str, int]
    provider_used_micros: dict[str, int]
    provider_governed_event_micros: dict[str, int]
    account_reserved_micros: dict[str, int]
    account_event_net_micros: dict[str, int]
    account_used_micros: dict[str, int]
    account_governed_event_micros: dict[str, int]
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


class BudgetIntegrityError(RuntimeError):
    """Mutable governed cost state disagrees with append-only evidence."""

    def __init__(self, report: BudgetIntegrityReport):
        self.report = report
        preview = "; ".join(report.violations[:4])
        if len(report.violations) > 4:
            preview += f"; plus {len(report.violations) - 4} more violation(s)"
        super().__init__(f"governed budget integrity failed: {preview}")


def _exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _event_amounts(
    events: list[CostEvent],
    *,
    battle_id: str,
    kind: str,
    provider: str | None = None,
) -> list[int]:
    return [
        event.amount_micros
        for event in events
        if event.battle_id == battle_id
        and event.kind == kind
        and (provider is None or event.provider == provider)
    ]


def _expect_single(
    violations: list[str],
    *,
    values: list[int],
    expected: int,
    label: str,
) -> None:
    if values != [expected]:
        violations.append(f"{label} must be exactly [{expected}], observed {values}")


def _expect_absent(
    violations: list[str],
    *,
    values: list[int],
    label: str,
) -> None:
    if values:
        violations.append(f"{label} must be absent, observed {values}")


def _contract(
    battle: Battle,
    violations: list[str],
) -> dict[str, int]:
    value = battle.provider_reservations_json
    if value is None:
        value = {}
    if not isinstance(value, dict):
        violations.append(f"battle {battle.id} provider reservation contract is not an object")
        return {}
    contract: dict[str, int] = {}
    for raw_backend, raw_amount in sorted(value.items(), key=lambda item: str(item[0])):
        backend = str(raw_backend)
        if not backend or not _exact_int(raw_amount) or raw_amount <= 0:
            violations.append(
                f"battle {battle.id} has an invalid provider reservation for {backend!r}"
            )
            continue
        contract[backend] = raw_amount
    return contract


def _account_key(backend: str, scope: str) -> str:
    return f"{backend}:{scope}"


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _crosscheck_datetime_matches(value: object, expected: datetime) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and _utc(parsed) == _utc(expected)


def _validate_endpoint_receipts(
    session: Session,
    *,
    season_id: str,
    battles_by_id: dict[str, Battle],
    arms: list[ResponseArm],
    events: list[CostEvent],
    settled_battle_ids: set[str],
    violations: list[str],
) -> dict[str, int]:
    """Bind every settled arm cost to one immutable endpoint receipt."""

    arms_by_id = {arm.id: arm for arm in arms}
    receipts_by_arm: dict[str, list[CostEvent]] = defaultdict(list)
    for event in events:
        if event.kind not in _RECEIPT_KINDS:
            continue
        arm = arms_by_id.get(event.arm_id or "")
        if arm is None:
            violations.append(
                f"endpoint receipt {event.id} does not name an arm in season {season_id}"
            )
            continue
        if event.season_id != season_id or event.battle_id != arm.battle_id:
            violations.append(f"endpoint receipt {event.id} disagrees with arm {arm.id} ownership")
        receipts_by_arm[arm.id].append(event)

    settled_cost_by_battle: dict[str, int] = defaultdict(int)
    arms_by_battle: dict[str, list[ResponseArm]] = defaultdict(list)
    settlement_receipts_by_battle: dict[str, list[CostEvent]] = defaultdict(list)
    for arm in arms:
        arms_by_battle[arm.battle_id].append(arm)
        receipts = receipts_by_arm.get(arm.id, [])
        actual_receipts = [event for event in receipts if event.kind == "actual"]
        settlement_receipts = [event for event in receipts if event.kind == "actual_settlement"]
        if len(actual_receipts) > 1 or len(settlement_receipts) > 1:
            violations.append(f"arm {arm.id} has duplicate endpoint-generation cost receipts")
        final_receipt = (
            settlement_receipts[0]
            if len(settlement_receipts) == 1
            else actual_receipts[0]
            if len(actual_receipts) == 1
            else None
        )
        is_settled = arm.battle_id in settled_battle_ids
        requires_receipt = (
            is_settled
            or arm.status not in {"queued", "running"}
            or arm.completed_at is not None
            or arm.cost_reconciled
            or arm.cost_micros != 0
        )
        if final_receipt is None:
            if requires_receipt:
                violations.append(
                    f"terminal, reconciled, or nonzero-cost arm {arm.id} lacks an "
                    "endpoint-generation cost receipt"
                )
            continue

        expected_provider = arm.actual_provider_slug or arm.provider_slug
        if (
            final_receipt.amount_micros != arm.cost_micros
            or final_receipt.provider != expected_provider
            or final_receipt.generation_id != arm.generation_id
        ):
            violations.append(
                f"endpoint receipt {final_receipt.id} disagrees with arm {arm.id} cost or identity"
            )
        if (
            arm.status in {"queued", "running"}
            or arm.completed_at is None
            or not arm.cost_reconciled
            or arm.cost_accounting_basis == "unrecorded"
            or arm.billing_reconciliation_status == "unrecorded"
        ):
            violations.append(f"arm {arm.id} has a receipt without a terminal reconciled lifecycle")
        accounting = final_receipt.accounting_json
        if not isinstance(accounting, dict):
            violations.append(f"endpoint receipt {final_receipt.id} accounting is not an object")
        elif final_receipt.kind == "actual_settlement":
            settlement_receipts_by_battle[arm.battle_id].append(final_receipt)
            prior_actual = actual_receipts[0] if len(actual_receipts) == 1 else None
            if (
                arm.cost_accounting_basis != "manual_authorized_settlement"
                or arm.billing_reconciliation_status != "manual_authorized_settlement"
                or accounting.get("settlement") != "manual_authorized"
                or not isinstance(accounting.get("authorization_reference_sha256"), str)
                or len(str(accounting.get("authorization_reference_sha256"))) != 64
                or accounting.get("supersedes_cost_event_id")
                != (prior_actual.id if prior_actual is not None else None)
                or accounting.get("prior_cost_state") != "unresolved_attempt_journal"
            ):
                violations.append(
                    f"manual endpoint receipt {final_receipt.id} lacks its authorization binding"
                )
        else:
            recorded_status = accounting.get("billing_reconciliation_status")
            recorded_basis = accounting.get("cost_accounting_basis", accounting.get("basis"))
            recorded_reconciled = accounting.get("reconciled")
            if recorded_status is not None and recorded_status != arm.billing_reconciliation_status:
                violations.append(
                    f"endpoint receipt {final_receipt.id} billing status disagrees "
                    f"with arm {arm.id}"
                )
            if recorded_basis is not None and recorded_basis != arm.cost_accounting_basis:
                violations.append(
                    f"endpoint receipt {final_receipt.id} accounting basis disagrees "
                    f"with arm {arm.id}"
                )
            if recorded_reconciled is not None and recorded_reconciled is not True:
                violations.append(f"endpoint receipt {final_receipt.id} is not marked reconciled")
        if is_settled:
            settled_cost_by_battle[arm.battle_id] += arm.cost_micros

    for battle_id in sorted(settled_battle_ids):
        if not arms_by_battle.get(battle_id):
            violations.append(f"settled battle {battle_id} has no response arms")

    settlement_battle_ids = set(settlement_receipts_by_battle)
    settlement_events = list(
        session.scalars(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "battle",
                RunEvent.entity_id.in_(sorted(settlement_battle_ids)),
                RunEvent.event_type == "generation_cost_exposure_settled",
            )
            .order_by(RunEvent.created_at, RunEvent.id)
        ).all()
        if settlement_battle_ids
        else []
    )
    events_by_battle: dict[str, list[RunEvent]] = defaultdict(list)
    for event in settlement_events:
        events_by_battle[event.entity_id].append(event)
    settlement_arm_ids = sorted(
        receipt.arm_id
        for receipts in settlement_receipts_by_battle.values()
        for receipt in receipts
        if receipt.arm_id is not None
    )
    attempts = list(
        session.scalars(
            select(GenerationAttempt)
            .where(GenerationAttempt.arm_id.in_(settlement_arm_ids))
            .order_by(GenerationAttempt.created_at, GenerationAttempt.id)
        ).all()
        if settlement_arm_ids
        else []
    )
    attempts_by_arm: dict[str, list[GenerationAttempt]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_arm[attempt.arm_id].append(attempt)

    for battle_id, receipts in sorted(settlement_receipts_by_battle.items()):
        battle = battles_by_id.get(battle_id)
        authorization_events = events_by_battle.get(battle_id, [])
        if battle is None or battle.controlled_run_id is None:
            violations.append(
                f"manual settlement receipts for battle {battle_id} lack a controlled run"
            )
            continue
        if len(authorization_events) != 1:
            violations.append(
                f"manual settlement for battle {battle_id} must own exactly one authorization event"
            )
        authorization_event = authorization_events[0] if len(authorization_events) == 1 else None
        references = {
            str(receipt.accounting_json.get("authorization_reference_sha256", ""))
            for receipt in receipts
        }
        settled_arm_costs = {str(receipt.arm_id): receipt.amount_micros for receipt in receipts}
        expected_payload = {
            "controlled_run_id": battle.controlled_run_id,
            "authorization_reference_sha256": next(iter(references), ""),
            "arm_costs_micros": settled_arm_costs,
        }
        if (
            len(references) != 1
            or authorization_event is None
            or authorization_event.payload_json != expected_payload
        ):
            violations.append(
                f"manual settlement for battle {battle_id} disagrees with its exact "
                "authorization event"
            )
        for receipt in receipts:
            if receipt.arm_id is None:
                continue
            latest_by_attempt: dict[tuple[str, str], GenerationAttempt] = {}
            for attempt in attempts_by_arm.get(receipt.arm_id, []):
                latest_by_attempt[(attempt.arm_id, attempt.attempt_id)] = attempt
            unresolved = any(
                not attempt.event_type.startswith("mcp_")
                and attempt.event_type
                not in {"pre_send_failure", "request_rejected", "accounting_reconciled"}
                for attempt in latest_by_attempt.values()
            )
            prior_actual = next(
                (
                    event
                    for event in receipts_by_arm.get(receipt.arm_id, [])
                    if event.kind == "actual"
                ),
                None,
            )
            prior_receipt_unresolved = (
                prior_actual is not None
                and isinstance(prior_actual.accounting_json, dict)
                and prior_actual.accounting_json.get("reconciled") is False
            )
            if not unresolved and not prior_receipt_unresolved:
                violations.append(
                    f"manual settlement receipt {receipt.id} lacks prior unresolved "
                    "provider-cost evidence"
                )
    return dict(settled_cost_by_battle)


def _validate_bedrock_crosscheck(
    session: Session,
    *,
    season: Season,
    crosscheck: BedrockBillingCrosscheck,
    arms_by_id: dict[str, ResponseArm],
    violations: list[str],
) -> None:
    """Reconstruct one aggregate billing record from relational evidence."""

    label = f"Bedrock crosscheck {crosscheck.id}"
    evidence = crosscheck.evidence_json
    if not isinstance(evidence, dict):
        violations.append(f"{label} evidence is not an object")
        return

    account = session.get(ProviderAccountBudget, crosscheck.provider_account_budget_id)
    if account is None or account.execution_backend != "bedrock":
        violations.append(f"{label} does not resolve to a Bedrock account ledger")
        return
    provider_budget = session.scalar(
        select(SeasonProviderBudget).where(
            SeasonProviderBudget.season_id == season.id,
            SeasonProviderBudget.execution_backend == "bedrock",
            SeasonProviderBudget.account_scope_sha256 == account.account_scope_sha256,
        )
    )
    if provider_budget is None:
        violations.append(f"{label} does not resolve to its season Bedrock budget")
        return

    memberships = list(
        session.scalars(
            select(BedrockBillingCrosscheckArm)
            .where(BedrockBillingCrosscheckArm.crosscheck_id == crosscheck.id)
            .order_by(BedrockBillingCrosscheckArm.arm_id)
        ).all()
    )
    arm_ids = [membership.arm_id for membership in memberships]
    member_arms = [arms_by_id.get(arm_id) for arm_id in arm_ids]
    if not memberships or any(arm is None for arm in member_arms):
        violations.append(f"{label} membership does not resolve to its exact season arm set")
        return

    resolved_arms = [arm for arm in member_arms if arm is not None]
    if any(arm.execution_backend != "bedrock" for arm in resolved_arms):
        violations.append(f"{label} includes a non-Bedrock response arm")
    covered_battle_ids = {arm.battle_id for arm in resolved_arms}
    receipt_events = list(
        session.scalars(
            select(CostEvent).where(
                CostEvent.arm_id.in_(arm_ids),
                CostEvent.kind.in_(sorted(_RECEIPT_KINDS)),
            )
        ).all()
    )
    reconciled_battle_ids = set(
        session.scalars(
            select(CostEvent.battle_id).where(
                CostEvent.battle_id.in_(sorted(covered_battle_ids)),
                CostEvent.kind == "reconcile",
            )
        ).all()
    )
    if reconciled_battle_ids != covered_battle_ids:
        violations.append(f"{label} includes an unreconciled battle")
    _validate_endpoint_receipts(
        session,
        season_id=season.id,
        battles_by_id={arm.battle_id: session.get(Battle, arm.battle_id) for arm in resolved_arms},
        arms=resolved_arms,
        events=receipt_events,
        settled_battle_ids=covered_battle_ids,
        violations=violations,
    )
    expected_arm_set_sha256 = _canonical_sha256({"arm_ids": arm_ids})
    expected_rate_card = sum(arm.cost_micros for arm in resolved_arms)

    attempts = list(
        session.scalars(
            select(GenerationAttempt).where(
                GenerationAttempt.arm_id.in_(arm_ids),
                GenerationAttempt.event_type == "request_started",
            )
        ).all()
    )
    attempts_by_arm: dict[str, list[GenerationAttempt]] = defaultdict(list)
    for attempt in attempts:
        attempts_by_arm[attempt.arm_id].append(attempt)

    generation_map: list[dict[str, object]] = []
    authorization_envelope_sha256s: set[str] = set()
    membership_by_arm = {membership.arm_id: membership for membership in memberships}
    for arm in sorted(resolved_arms, key=lambda item: item.id):
        generation_ids = sorted(set(arm.provider_generation_ids_json or []))
        generation_set_sha256 = _canonical_sha256({"generation_ids": generation_ids})
        authorization_hashes = {
            str(
                attempt.metadata_json.get(
                    "verified_provider_account_authorization_envelope_sha256",
                    "",
                )
            )
            for attempt in attempts_by_arm.get(arm.id, [])
        }
        authorization_hashes.discard("")
        if not generation_ids or len(authorization_hashes) != 1:
            violations.append(
                f"{label} arm {arm.id} lacks an exact generation set or credential epoch"
            )
            continue
        authorization_sha256 = next(iter(authorization_hashes))
        authorization_envelope_sha256s.add(authorization_sha256)
        membership = membership_by_arm[arm.id]
        if membership.generation_set_sha256 != generation_set_sha256:
            violations.append(f"{label} arm {arm.id} generation membership hash is invalid")
        generation_map.append(
            {
                "arm_id": arm.id,
                "generation_ids": generation_ids,
                "account_authorization_envelope_sha256": authorization_sha256,
                "generation_set_sha256": generation_set_sha256,
            }
        )

    expected_generation_map_sha256 = _canonical_sha256({"arms": generation_map})
    expected_difference = crosscheck.billed_usage_micros - expected_rate_card
    predecessor = (
        session.get(BedrockBillingCrosscheck, crosscheck.supersedes_crosscheck_id)
        if crosscheck.supersedes_crosscheck_id is not None
        else None
    )
    predecessor_valid = predecessor is not None
    if predecessor is not None:
        predecessor_arm_ids = sorted(
            session.scalars(
                select(BedrockBillingCrosscheckArm.arm_id).where(
                    BedrockBillingCrosscheckArm.crosscheck_id == predecessor.id
                )
            ).all()
        )
        predecessor_valid = (
            predecessor.season_id == season.id
            and predecessor.provider_account_budget_id == account.id
            and predecessor_arm_ids == arm_ids
        )
    if crosscheck.supersedes_crosscheck_id is not None and not predecessor_valid:
        violations.append(f"{label} has an invalid or nonmatching predecessor arm set")
    predecessor_difference = (
        predecessor.billing_difference_micros
        if predecessor is not None and predecessor_valid
        else 0
    )
    expected_ledger_delta = expected_difference - predecessor_difference
    expected_tolerance = max(10_000, (expected_rate_card + 49) // 50)
    expected_status = "accepted" if abs(expected_difference) <= expected_tolerance else "discrepant"
    expected_governed_delta = max(0, expected_ledger_delta)

    expected_evidence_keys = {
        "schema_version",
        "season_slug",
        "account_scope_sha256",
        "source_kind",
        "source_artifact_uri",
        "source_artifact_sha256",
        "statement_sha256",
        "coverage_start",
        "coverage_end",
        "arm_set_sha256",
        "generation_request_map_sha256",
        "rate_card_estimated_micros",
        "billed_usage_micros",
        "billing_difference_micros",
        "crosscheck_status",
        "supersedes_crosscheck_id",
        "tolerance_micros",
        "credits_policy",
        "authorization_reference_sha256",
        "ledger_delta_micros",
        "governed_budget_delta_micros",
    }
    semantic_fields = {
        "schema_version": "flavourbench-bedrock-billing-crosscheck-v1",
        "season_slug": season.slug,
        "account_scope_sha256": account.account_scope_sha256,
        "source_kind": crosscheck.source_kind,
        "source_artifact_uri": crosscheck.source_artifact_uri,
        "source_artifact_sha256": crosscheck.source_artifact_sha256,
        "statement_sha256": crosscheck.statement_sha256,
        "arm_set_sha256": expected_arm_set_sha256,
        "generation_request_map_sha256": expected_generation_map_sha256,
        "rate_card_estimated_micros": expected_rate_card,
        "billed_usage_micros": crosscheck.billed_usage_micros,
        "billing_difference_micros": expected_difference,
        "crosscheck_status": expected_status,
        "supersedes_crosscheck_id": crosscheck.supersedes_crosscheck_id,
        "tolerance_micros": expected_tolerance,
        "credits_policy": crosscheck.credits_policy,
        "authorization_reference_sha256": crosscheck.authorization_reference_sha256,
        "ledger_delta_micros": expected_ledger_delta,
        "governed_budget_delta_micros": expected_governed_delta,
    }
    row_semantics_valid = (
        crosscheck.season_id == season.id
        and crosscheck.arm_set_sha256 == expected_arm_set_sha256
        and crosscheck.generation_request_map_sha256 == expected_generation_map_sha256
        and crosscheck.rate_card_estimated_micros == expected_rate_card
        and crosscheck.billing_difference_micros == expected_difference
        and crosscheck.ledger_delta_micros == expected_ledger_delta
        and crosscheck.tolerance_micros == expected_tolerance
        and crosscheck.status == expected_status
        and crosscheck.coverage_end > crosscheck.coverage_start
        and all(
            arm.completed_at is not None
            and _utc(crosscheck.coverage_start)
            <= _utc(arm.completed_at)
            <= _utc(crosscheck.coverage_end)
            for arm in resolved_arms
        )
        and crosscheck.credits_policy == "gross_usage_before_credits_excluding_tax"
    )
    evidence_semantics_valid = (
        set(evidence) == expected_evidence_keys
        and all(evidence.get(key) == value for key, value in semantic_fields.items())
        and _crosscheck_datetime_matches(evidence.get("coverage_start"), crosscheck.coverage_start)
        and _crosscheck_datetime_matches(evidence.get("coverage_end"), crosscheck.coverage_end)
        and _canonical_sha256(evidence) == crosscheck.evidence_sha256
    )
    if not row_semantics_valid or not evidence_semantics_valid:
        violations.append(f"{label} cannot be reconstructed from relational cost evidence")

    if len(authorization_envelope_sha256s) != 1:
        violations.append(f"{label} spans zero or multiple credential epochs")
    else:
        authorization_sha256 = next(iter(authorization_envelope_sha256s))
        authorization = session.scalar(
            select(ProviderAccountAuthorization).where(
                ProviderAccountAuthorization.provider_account_budget_id == account.id,
                ProviderAccountAuthorization.authorization_envelope_sha256 == authorization_sha256,
            )
        )
        request_attempts = [
            attempt for arm_attempts in attempts_by_arm.values() for attempt in arm_attempts
        ]
        interval_valid = authorization is not None and all(
            _utc(authorization.created_at) <= _utc(attempt.created_at)
            and _utc(attempt.created_at) < _utc(authorization.valid_until)
            and (
                authorization.revoked_at is None
                or _utc(attempt.created_at) < _utc(authorization.revoked_at)
            )
            for attempt in request_attempts
        )
        if (
            authorization is None
            or authorization.authorization_reference_sha256
            != crosscheck.authorization_reference_sha256
            or not interval_valid
            or not account_authorization_chain_valid(
                session,
                account,
                authorization,
                root_envelope_sha256=(provider_budget.account_authorization_envelope_sha256),
                signing_secret=get_settings().budget_authorization_signing_secret,
                verification_keys=budget_authorization_verification_keyring(get_settings()),
                require_head_active=False,
            )
        ):
            violations.append(f"{label} credential epoch is not independently verifiable")


def assert_budget_integrity(
    session: Session,
    season_id: str,
    *,
    lock_aggregates: bool = False,
) -> BudgetIntegrityReport:
    """Verify one season and its shared provider-account governed balances.

    ``lock_aggregates`` is used at transactional authority boundaries.  The
    season lock serializes admissions and settlements within the season; the
    provider/account locks also protect shared account exposure across seasons.
    SQLite ignores ``FOR UPDATE`` but exercises identical evidence semantics.
    """

    session.flush()
    season_statement = select(Season).where(Season.id == season_id)
    if lock_aggregates:
        season_statement = season_statement.with_for_update()
    season = session.scalar(season_statement)
    if season is None:
        empty = BudgetIntegrityReport(
            season_id=season_id,
            battles_checked=0,
            season_reserved_micros=0,
            battle_reserved_micros=0,
            governor_event_net_micros=0,
            season_used_micros=0,
            governor_reconciled_micros=0,
            provider_reserved_micros={},
            provider_event_net_micros={},
            provider_used_micros={},
            provider_governed_event_micros={},
            account_reserved_micros={},
            account_event_net_micros={},
            account_used_micros={},
            account_governed_event_micros={},
            violations=(f"season {season_id} does not exist",),
        )
        raise BudgetIntegrityError(empty)

    provider_statement = (
        select(SeasonProviderBudget)
        .where(SeasonProviderBudget.season_id == season_id)
        .order_by(SeasonProviderBudget.execution_backend)
    )
    run_statement = (
        select(ControlledRun).where(ControlledRun.season_id == season_id).order_by(ControlledRun.id)
    )
    if lock_aggregates:
        provider_statement = provider_statement.with_for_update()
        run_statement = run_statement.with_for_update()

    provider_rows = list(session.scalars(provider_statement).all())
    controlled_runs = list(session.scalars(run_statement).all())
    battles = list(
        session.scalars(
            select(Battle).where(Battle.season_id == season_id).order_by(Battle.id)
        ).all()
    )
    arms = list(
        session.scalars(
            select(ResponseArm)
            .where(ResponseArm.battle_id.in_([battle.id for battle in battles]))
            .order_by(ResponseArm.battle_id, ResponseArm.side, ResponseArm.id)
        ).all()
        if battles
        else []
    )
    events = list(
        session.scalars(
            select(CostEvent)
            .where(
                CostEvent.season_id == season_id,
                CostEvent.kind.in_(sorted(_AUDITED_COST_KINDS)),
            )
            .order_by(CostEvent.created_at, CostEvent.id)
        ).all()
    )
    violations: list[str] = []

    battle_by_id = {battle.id: battle for battle in battles}
    for event in events:
        if event.kind != "bedrock_billing_adjustment" and (
            event.battle_id is None or event.battle_id not in battle_by_id
        ):
            violations.append(
                f"{event.kind} event {event.id} does not name a battle in season {season_id}"
            )

    settled_battle_ids = {
        event.battle_id
        for event in events
        if event.kind == "reconcile" and event.battle_id is not None
    }
    receipt_cost_by_battle = _validate_endpoint_receipts(
        session,
        season_id=season_id,
        battles_by_id=battle_by_id,
        arms=arms,
        events=events,
        settled_battle_ids=settled_battle_ids,
        violations=violations,
    )

    active_provider_contracts: dict[str, int] = defaultdict(int)
    battle_contracts: dict[str, dict[str, int]] = {}
    arms_by_battle: dict[str, list[ResponseArm]] = defaultdict(list)
    arms_by_id = {arm.id: arm for arm in arms}
    for arm in arms:
        arms_by_battle[arm.battle_id].append(arm)
    governor_net = 0
    for battle in battles:
        contract = _contract(battle, violations)
        battle_contracts[battle.id] = contract
        contract_total = sum(contract.values())
        battle_arms = arms_by_battle.get(battle.id, [])
        arm_cost_total = sum(arm.cost_micros for arm in battle_arms)
        if (
            battle.id in settled_battle_ids
            and receipt_cost_by_battle.get(battle.id, 0) != arm_cost_total
        ):
            violations.append(
                f"battle {battle.id} endpoint receipts do not sum to its reconciled arm cost"
            )
        arm_cost_by_backend: dict[str, int] = defaultdict(int)
        for arm in battle_arms:
            arm_cost_by_backend[arm.execution_backend or "openrouter"] += arm.cost_micros
        current = battle.reserved_cost_micros
        if not _exact_int(current) or current < 0:
            violations.append(f"battle {battle.id} has an invalid current reservation")
            current = 0
        if contract_total == 0:
            if current != 0:
                violations.append(
                    f"battle {battle.id} reserves {current} micros without a provider contract"
                )
            if arm_cost_total != 0:
                violations.append(
                    f"battle {battle.id} records {arm_cost_total} micros of response-arm cost "
                    "without a frozen provider reservation contract"
                )
        elif current not in {0, contract_total}:
            violations.append(
                f"battle {battle.id} current reservation {current} is neither zero nor "
                f"its frozen contract {contract_total}"
            )

        governor_reserve = _event_amounts(
            events, battle_id=battle.id, kind="reserve", provider="governor"
        )
        governor_release = _event_amounts(
            events, battle_id=battle.id, kind="release", provider="governor"
        )
        governor_reconcile = _event_amounts(
            events, battle_id=battle.id, kind="reconcile", provider="governor"
        )
        if contract_total:
            _expect_single(
                violations,
                values=governor_reserve,
                expected=contract_total,
                label=f"battle {battle.id} governor reserve",
            )
            if current == contract_total:
                _expect_absent(
                    violations,
                    values=governor_release,
                    label=f"battle {battle.id} governor release before settlement",
                )
                _expect_absent(
                    violations,
                    values=governor_reconcile,
                    label=f"battle {battle.id} governor reconciliation before settlement",
                )
            elif current == 0:
                _expect_single(
                    violations,
                    values=governor_release,
                    expected=-contract_total,
                    label=f"battle {battle.id} governor release",
                )
                if governor_reconcile != [arm_cost_total]:
                    violations.append(
                        f"battle {battle.id} governor reconciliation {governor_reconcile} "
                        f"does not match response-arm cost {arm_cost_total}"
                    )
        else:
            _expect_absent(
                violations,
                values=governor_reserve,
                label=f"battle {battle.id} zero-contract governor reserve",
            )
            _expect_absent(
                violations,
                values=governor_release,
                label=f"battle {battle.id} zero-contract governor release",
            )
            if len(governor_reconcile) > 1 or (
                governor_reconcile and governor_reconcile != [arm_cost_total]
            ):
                violations.append(
                    f"battle {battle.id} zero-contract reconciliation {governor_reconcile} "
                    f"does not match response-arm cost {arm_cost_total}"
                )

        battle_governor_net = sum(governor_reserve) + sum(governor_release)
        governor_net += battle_governor_net
        if battle_governor_net != current:
            violations.append(
                f"battle {battle.id} counter {current} disagrees with governor event net "
                f"{battle_governor_net}"
            )

        event_backends = {
            event.provider
            for event in events
            if event.battle_id == battle.id and event.kind in (_PROVIDER_KINDS | _ACCOUNT_KINDS)
        }
        if event_backends != set(contract):
            violations.append(
                f"battle {battle.id} provider event backends {sorted(event_backends)} "
                f"do not match its contract {sorted(contract)}"
            )
        for backend, amount in contract.items():
            provider_reserve = _event_amounts(
                events,
                battle_id=battle.id,
                kind="provider_reserve",
                provider=backend,
            )
            provider_release = _event_amounts(
                events,
                battle_id=battle.id,
                kind="provider_release",
                provider=backend,
            )
            provider_reconcile = _event_amounts(
                events,
                battle_id=battle.id,
                kind="provider_reconcile",
                provider=backend,
            )
            account_reserve = _event_amounts(
                events,
                battle_id=battle.id,
                kind="provider_account_reserve",
                provider=backend,
            )
            account_release = _event_amounts(
                events,
                battle_id=battle.id,
                kind="provider_account_release",
                provider=backend,
            )
            account_reconcile = _event_amounts(
                events,
                battle_id=battle.id,
                kind="provider_account_reconcile",
                provider=backend,
            )
            _expect_single(
                violations,
                values=provider_reserve,
                expected=amount,
                label=f"battle {battle.id} {backend} provider reserve",
            )
            _expect_single(
                violations,
                values=account_reserve,
                expected=amount,
                label=f"battle {battle.id} {backend} account reserve",
            )
            if current == contract_total:
                active_provider_contracts[backend] += amount
                for values, label in (
                    (provider_release, "provider release"),
                    (provider_reconcile, "provider reconciliation"),
                    (account_release, "account release"),
                    (account_reconcile, "account reconciliation"),
                ):
                    _expect_absent(
                        violations,
                        values=values,
                        label=f"battle {battle.id} {backend} {label} before settlement",
                    )
            elif current == 0:
                _expect_single(
                    violations,
                    values=provider_release,
                    expected=-amount,
                    label=f"battle {battle.id} {backend} provider release",
                )
                _expect_single(
                    violations,
                    values=account_release,
                    expected=-amount,
                    label=f"battle {battle.id} {backend} account release",
                )
                expected_provider_actual = arm_cost_by_backend.get(backend, 0)
                if provider_reconcile != [expected_provider_actual]:
                    violations.append(
                        f"battle {battle.id} {backend} provider reconciliation "
                        f"{provider_reconcile} does not match response-arm cost "
                        f"{expected_provider_actual}"
                    )
                if account_reconcile != provider_reconcile:
                    violations.append(
                        f"battle {battle.id} {backend} provider/account reconciliations differ"
                    )

    battle_reserved = sum(max(0, battle.reserved_cost_micros) for battle in battles)
    if season.budget_reserved_micros != battle_reserved:
        violations.append(
            f"season counter {season.budget_reserved_micros} disagrees with battle net "
            f"{battle_reserved}"
        )
    if governor_net != battle_reserved:
        violations.append(
            f"season governor event net {governor_net} disagrees with battle net {battle_reserved}"
        )

    billing_adjustments = [event for event in events if event.kind == "bedrock_billing_adjustment"]
    crosschecks = list(
        session.scalars(
            select(BedrockBillingCrosscheck)
            .where(BedrockBillingCrosscheck.season_id == season_id)
            .order_by(BedrockBillingCrosscheck.created_at, BedrockBillingCrosscheck.id)
        ).all()
    )
    crosschecks_by_id = {crosscheck.id: crosscheck for crosscheck in crosschecks}
    adjustments_by_crosscheck: dict[str, list[CostEvent]] = defaultdict(list)
    for event in billing_adjustments:
        crosscheck_id = (event.accounting_json or {}).get("crosscheck_id")
        if isinstance(crosscheck_id, str):
            adjustments_by_crosscheck[crosscheck_id].append(event)
    for crosscheck in crosschecks:
        linked = adjustments_by_crosscheck.get(crosscheck.id, [])
        if len(linked) != 1:
            violations.append(
                f"Bedrock crosscheck {crosscheck.id} must own exactly one billing adjustment"
            )
        _validate_bedrock_crosscheck(
            session,
            season=season,
            crosscheck=crosscheck,
            arms_by_id=arms_by_id,
            violations=violations,
        )
    governed_billing_by_backend: dict[str, int] = defaultdict(int)
    for event in billing_adjustments:
        if event.provider != "bedrock" or event.battle_id is not None or event.arm_id is not None:
            violations.append(f"billing adjustment {event.id} is not a season-level Bedrock event")
        accounting = event.accounting_json or {}
        crosscheck_id = accounting.get("crosscheck_id")
        crosscheck = (
            crosschecks_by_id.get(crosscheck_id) if isinstance(crosscheck_id, str) else None
        )
        if crosscheck is None:
            violations.append(f"billing adjustment {event.id} has no matching crosscheck")
            continue
        governed_delta = max(0, crosscheck.ledger_delta_micros)
        if (
            event.amount_micros != crosscheck.ledger_delta_micros
            or accounting.get("governed_budget_delta_micros") != governed_delta
            or accounting.get("evidence_sha256") != crosscheck.evidence_sha256
            or _canonical_sha256(crosscheck.evidence_json) != crosscheck.evidence_sha256
            or crosscheck.evidence_json.get("ledger_delta_micros") != crosscheck.ledger_delta_micros
            or crosscheck.evidence_json.get("governed_budget_delta_micros") != governed_delta
        ):
            violations.append(
                f"billing adjustment {event.id} disagrees with immutable crosscheck {crosscheck.id}"
            )
        governed_billing_by_backend[event.provider] += governed_delta
    governor_reconciled = sum(
        event.amount_micros for event in events if event.kind == "reconcile"
    ) + sum(governed_billing_by_backend.values())
    if season.budget_used_micros != governor_reconciled:
        violations.append(
            f"season used counter {season.budget_used_micros} disagrees with governed "
            f"cost evidence {governor_reconciled}"
        )

    run_by_id = {run.id: run for run in controlled_runs}
    run_expected: dict[str, int] = defaultdict(int)
    run_used_expected: dict[str, int] = defaultdict(int)
    for battle in battles:
        if battle.controlled_run_id is not None:
            if battle.controlled_run_id not in run_by_id:
                violations.append(
                    f"battle {battle.id} references missing controlled run "
                    f"{battle.controlled_run_id}"
                )
            run_expected[battle.controlled_run_id] += max(0, battle.reserved_cost_micros)
            run_used_expected[battle.controlled_run_id] += receipt_cost_by_battle.get(battle.id, 0)
    for run in controlled_runs:
        expected = run_expected.get(run.id, 0)
        if run.budget_reserved_micros != expected:
            violations.append(
                f"controlled run {run.id} counter {run.budget_reserved_micros} "
                f"disagrees with battle net {expected}"
            )
        used_expected = run_used_expected.get(run.id, 0)
        if run.budget_used_micros != used_expected:
            violations.append(
                f"controlled run {run.id} used counter {run.budget_used_micros} "
                f"disagrees with battle reconciliation {used_expected}"
            )

    providers = {row.execution_backend: row for row in provider_rows}
    provider_event_net: dict[str, int] = defaultdict(int)
    provider_governed_used: dict[str, int] = defaultdict(int)
    for event in events:
        if event.kind in {"provider_reserve", "provider_release"}:
            provider_event_net[event.provider] += event.amount_micros
        elif event.kind == "provider_reconcile":
            provider_governed_used[event.provider] += event.amount_micros
    for backend, amount in governed_billing_by_backend.items():
        provider_governed_used[backend] += amount
    provider_keys = (
        set(providers)
        | set(active_provider_contracts)
        | set(provider_event_net)
        | set(provider_governed_used)
    )
    for backend in sorted(provider_keys):
        row = providers.get(backend)
        counter = row.budget_reserved_micros if row is not None else None
        contract_net = active_provider_contracts.get(backend, 0)
        event_net = provider_event_net.get(backend, 0)
        if row is None:
            violations.append(f"season lacks a provider budget row for {backend}")
            continue
        if counter != contract_net or counter != event_net:
            violations.append(
                f"season provider {backend} counter {counter}, active contract "
                f"{contract_net}, and event net {event_net} disagree"
            )
        governed_used = provider_governed_used.get(backend, 0)
        if row.budget_used_micros != governed_used:
            violations.append(
                f"season provider {backend} used counter {row.budget_used_micros} "
                f"disagrees with governed cost evidence {governed_used}"
            )

    account_rows: list[ProviderAccountBudget] = []
    if provider_rows:
        account_statement = (
            select(ProviderAccountBudget)
            .where(
                ProviderAccountBudget.execution_backend.in_(sorted(providers)),
                ProviderAccountBudget.account_scope_sha256.in_(
                    sorted(row.account_scope_sha256 for row in provider_rows)
                ),
            )
            .order_by(
                ProviderAccountBudget.execution_backend,
                ProviderAccountBudget.account_scope_sha256,
            )
        )
        if lock_aggregates:
            account_statement = account_statement.with_for_update()
        account_rows = list(session.scalars(account_statement).all())
    accounts = {(row.execution_backend, row.account_scope_sha256): row for row in account_rows}
    account_counters: dict[str, int] = {}
    account_event_nets: dict[str, int] = {}
    account_used_counters: dict[str, int] = {}
    account_used_evidence: dict[str, int] = {}
    for backend, provider in sorted(providers.items()):
        scope = provider.account_scope_sha256
        key = _account_key(backend, scope)
        account = accounts.get((backend, scope))
        if account is None:
            violations.append(f"season provider {backend} lacks account budget {scope}")
            continue
        authorizations = list(
            session.scalars(
                select(ProviderAccountAuthorization)
                .where(ProviderAccountAuthorization.provider_account_budget_id == account.id)
                .order_by(
                    ProviderAccountAuthorization.created_at,
                    ProviderAccountAuthorization.id,
                )
            ).all()
        )
        superseded_ids = {
            authorization.supersedes_authorization_id
            for authorization in authorizations
            if authorization.supersedes_authorization_id is not None
        }
        heads = [
            authorization
            for authorization in authorizations
            if authorization.id not in superseded_ids
        ]
        checkpoint = heads[0] if len(heads) == 1 else None
        if checkpoint is None:
            violations.append(
                f"provider account {key} must have exactly one authorization-chain head"
            )
            checkpoint_reserved = account.opening_reserved_micros
            checkpoint_used = account.opening_used_micros
        else:
            checkpoint_reserved = checkpoint.authorized_reserved_micros
            checkpoint_used = checkpoint.authorized_used_micros
            if (
                checkpoint.execution_backend != backend
                or checkpoint.account_scope_sha256 != scope
                or checkpoint.authorized_reserved_micros < 0
                or checkpoint.authorized_used_micros < 0
            ):
                violations.append(
                    f"provider account {key} authorization-head checkpoint is invalid"
                )
            if not account_authorization_chain_valid(
                session,
                account,
                checkpoint,
                root_envelope_sha256=provider.account_authorization_envelope_sha256,
                signing_secret=get_settings().budget_authorization_signing_secret,
                verification_keys=budget_authorization_verification_keyring(get_settings()),
                require_head_active=False,
            ):
                violations.append(
                    f"provider account {key} authorization chain is not independently verifiable"
                )
        account_events = list(
            session.scalars(
                select(CostEvent)
                .where(
                    CostEvent.provider == backend,
                    CostEvent.kind.in_(
                        [
                            "provider_account_reserve",
                            "provider_account_release",
                            "provider_account_reconcile",
                            "bedrock_billing_adjustment",
                        ]
                    ),
                )
                .order_by(CostEvent.created_at, CostEvent.id)
            ).all()
        )
        account_crosschecks = list(
            session.scalars(
                select(BedrockBillingCrosscheck)
                .where(BedrockBillingCrosscheck.provider_account_budget_id == account.id)
                .order_by(
                    BedrockBillingCrosscheck.created_at,
                    BedrockBillingCrosscheck.id,
                )
            ).all()
        )
        account_crosschecks_by_id = {
            crosscheck.id: crosscheck for crosscheck in account_crosschecks
        }
        account_adjustments_by_crosscheck: dict[str, list[CostEvent]] = defaultdict(list)
        for event in account_events:
            if event.kind != "bedrock_billing_adjustment":
                continue
            crosscheck_id = (event.accounting_json or {}).get("crosscheck_id")
            if isinstance(crosscheck_id, str):
                account_adjustments_by_crosscheck[crosscheck_id].append(event)
        for crosscheck in account_crosschecks:
            owner_season = session.get(Season, crosscheck.season_id)
            membership_arm_ids = list(
                session.scalars(
                    select(BedrockBillingCrosscheckArm.arm_id).where(
                        BedrockBillingCrosscheckArm.crosscheck_id == crosscheck.id
                    )
                ).all()
            )
            owner_arms = (
                list(
                    session.scalars(
                        select(ResponseArm).where(ResponseArm.id.in_(membership_arm_ids))
                    ).all()
                )
                if membership_arm_ids
                else []
            )
            if owner_season is None:
                violations.append(f"Bedrock crosscheck {crosscheck.id} has no owning season")
            else:
                _validate_bedrock_crosscheck(
                    session,
                    season=owner_season,
                    crosscheck=crosscheck,
                    arms_by_id={arm.id: arm for arm in owner_arms},
                    violations=violations,
                )
            linked_adjustments = account_adjustments_by_crosscheck.get(crosscheck.id, [])
            if len(linked_adjustments) != 1:
                violations.append(
                    f"Bedrock crosscheck {crosscheck.id} must own exactly one account adjustment"
                )
        matching_net = 0
        matching_used = 0
        for event in account_events:
            accounting = event.accounting_json or {}
            event_scope = accounting.get("account_scope_sha256")
            governed_billing_delta: int | None = None
            if event.kind == "bedrock_billing_adjustment":
                crosscheck_id = accounting.get("crosscheck_id")
                crosscheck = (
                    account_crosschecks_by_id.get(crosscheck_id)
                    if isinstance(crosscheck_id, str)
                    else None
                )
                if crosscheck is None:
                    if event_scope == scope:
                        violations.append(
                            f"billing adjustment {event.id} has no provider-account crosscheck"
                        )
                    continue
                if event.season_id != crosscheck.season_id:
                    violations.append(
                        f"billing adjustment {event.id} and crosscheck {crosscheck.id} "
                        "name different seasons"
                    )
                if crosscheck.provider_account_budget_id != account.id:
                    if event_scope == scope:
                        violations.append(
                            f"billing adjustment {event.id} falsely names provider account "
                            f"scope {scope}"
                        )
                    continue
                if event_scope != scope:
                    violations.append(
                        f"billing adjustment {event.id} names scope {event_scope!r}, "
                        f"but crosscheck {crosscheck.id} belongs to {scope}"
                    )
                governed_billing_delta = max(0, crosscheck.ledger_delta_micros)
                if (
                    event.amount_micros != crosscheck.ledger_delta_micros
                    or accounting.get("governed_budget_delta_micros") != governed_billing_delta
                    or accounting.get("evidence_sha256") != crosscheck.evidence_sha256
                    or _canonical_sha256(crosscheck.evidence_json) != crosscheck.evidence_sha256
                ):
                    violations.append(
                        f"billing adjustment {event.id} disagrees with account crosscheck "
                        f"{crosscheck.id}"
                    )
            elif event_scope != scope:
                continue
            if checkpoint is not None:
                event_created = _utc(event.created_at)
                checkpoint_created = _utc(checkpoint.created_at)
                if event_created < checkpoint_created:
                    continue
                if event_created == checkpoint_created:
                    violations.append(
                        f"provider account {key} has an event on the authorization "
                        "checkpoint boundary"
                    )
                    continue
            if event.kind in {"provider_account_reserve", "provider_account_release"}:
                matching_net += event.amount_micros
            elif event.kind == "provider_account_reconcile":
                matching_used += event.amount_micros
            elif governed_billing_delta is not None:
                matching_used += governed_billing_delta
        expected = checkpoint_reserved + matching_net
        used_expected = checkpoint_used + matching_used
        account_counters[key] = account.budget_reserved_micros
        account_event_nets[key] = expected
        account_used_counters[key] = account.budget_used_micros
        account_used_evidence[key] = used_expected
        if expected < 0 or account.budget_reserved_micros != expected:
            violations.append(
                f"provider account {key} counter {account.budget_reserved_micros} "
                f"disagrees with checkpoint-plus-event net {expected}"
            )
        if account.budget_used_micros != used_expected:
            violations.append(
                f"provider account {key} used counter {account.budget_used_micros} "
                f"disagrees with checkpoint-plus-event evidence {used_expected}"
            )
        for battle_id, contract in battle_contracts.items():
            if backend not in contract:
                continue
            reserves = [
                event
                for event in events
                if event.battle_id == battle_id
                and event.provider == backend
                and event.kind in _ACCOUNT_KINDS
            ]
            for event in reserves:
                event_scope = (event.accounting_json or {}).get("account_scope_sha256")
                if event_scope != scope:
                    violations.append(
                        f"battle {battle_id} {backend} account event {event.id} "
                        f"names scope {event_scope!r}, expected {scope}"
                    )

    report = BudgetIntegrityReport(
        season_id=season_id,
        battles_checked=len(battles),
        season_reserved_micros=season.budget_reserved_micros,
        battle_reserved_micros=battle_reserved,
        governor_event_net_micros=governor_net,
        season_used_micros=season.budget_used_micros,
        governor_reconciled_micros=governor_reconciled,
        provider_reserved_micros={
            backend: row.budget_reserved_micros for backend, row in sorted(providers.items())
        },
        provider_event_net_micros=dict(sorted(provider_event_net.items())),
        provider_used_micros={
            backend: row.budget_used_micros for backend, row in sorted(providers.items())
        },
        provider_governed_event_micros=dict(sorted(provider_governed_used.items())),
        account_reserved_micros=dict(sorted(account_counters.items())),
        account_event_net_micros=dict(sorted(account_event_nets.items())),
        account_used_micros=dict(sorted(account_used_counters.items())),
        account_governed_event_micros=dict(sorted(account_used_evidence.items())),
        violations=tuple(violations),
    )
    if not report.ok:
        raise BudgetIntegrityError(report)
    return report


def decrement_reservation(current: int, release: int, *, label: str) -> int:
    """Subtract an authorized release without masking evidence corruption."""

    if not _exact_int(current) or current < 0:
        raise ValueError(f"{label} current reservation is invalid")
    if not _exact_int(release) or release < 0:
        raise ValueError(f"{label} release must be a nonnegative integer")
    if current < release:
        raise ValueError(f"{label} reservation underflow: current {current}, release {release}")
    return current - release
