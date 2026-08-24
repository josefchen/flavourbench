from __future__ import annotations

import hashlib
import math
import secrets
from datetime import UTC, datetime, timedelta
from fractions import Fraction

from fastapi import HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from .account_authority import (
    account_authorization,
    account_authorization_chain_valid,
)
from .budget_integrity import BudgetIntegrityError, assert_budget_integrity
from .budget_policy import (
    provider_account_hard_cap_micros,
    provider_account_scope_sha256,
)
from .config import budget_authorization_verification_keyring, get_settings
from .endpoint_contract import DECODING_PARAMETERS
from .models import (
    PERMANENT_RESEARCH_RETENTION_UNTIL,
    AdmissionEvent,
    Battle,
    CatalogModel,
    ControlledRun,
    ControlledRunAssignment,
    Job,
    Organization,
    ProviderAccountBudget,
    ResponseArm,
    RunEvent,
    Season,
    SeasonModel,
    SeasonProviderBudget,
    Task,
)
from .provider import FINAL_SCHEMA_SHA256, system_prompt_sha256
from .schemas import BattleCreate, ControlledBattleCreate
from .task_lifecycle import TaskLifecycleError, record_task_first_use

SCHEDULER_VERSION = "coverage-balanced-server-random-v1"
CONTROLLED_SCHEDULER_VERSION = "controlled-frozen-schedule-v1"


def _fraction_text(numerator: int, denominator: int) -> str:
    value = Fraction(numerator, denominator)
    return f"{value.numerator}/{value.denominator}"


def _seeded_index(seed: str, label: str, upper_bound: int) -> int:
    """Draw an auditable unbiased index from server-generated entropy."""

    if upper_bound <= 0:
        raise ValueError("upper_bound must be positive")
    space = 1 << 256
    acceptance_limit = space - (space % upper_bound)
    counter = 0
    while True:
        value = int(
            hashlib.sha256(f"{seed}:{label}:{counter}".encode()).hexdigest(),
            16,
        )
        if value < acceptance_limit:
            return value % upper_bound
        counter += 1


def _seeded_shuffle(
    rows: list[tuple[CatalogModel, SeasonModel]], seed: str, label: str
) -> list[tuple[CatalogModel, SeasonModel]]:
    shuffled = sorted(rows, key=lambda row: row[0].model_id)
    for index in range(len(shuffled) - 1, 0, -1):
        replacement = _seeded_index(seed, f"{label}:{index}", index + 1)
        shuffled[index], shuffled[replacement] = shuffled[replacement], shuffled[index]
    return shuffled


def controlled_side_is_reversed(assignment_seed: str) -> bool:
    """Apply the published deterministic side rule to a committed secret seed."""

    return _seeded_index(assignment_seed, "side", 2) == 1


def _run_class(season: Season) -> str:
    settings = get_settings()
    if settings.execution_mode == "mock":
        return "mock"
    if season.status == "active" and season.official:
        return "official"
    if season.status == "pilot":
        return "pilot"
    return "exploratory"


def _admit(
    session: Session,
    pseudonym: str,
    *,
    action: str = "create_battle",
    limit: int | None = None,
) -> None:
    settings = get_settings()
    if session.bind and session.bind.dialect.name == "postgresql":
        lock_key = int(hashlib.sha256(pseudonym.encode()).hexdigest()[:16], 16)
        if lock_key >= 2**63:
            lock_key -= 2**64
        session.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": lock_key})
    cutoff = datetime.now(UTC) - timedelta(seconds=settings.admission_window_seconds)
    recent = (
        session.scalar(
            select(func.count(AdmissionEvent.id)).where(
                AdmissionEvent.pseudonym == pseudonym,
                AdmissionEvent.action == action,
                AdmissionEvent.admitted.is_(True),
                AdmissionEvent.created_at >= cutoff,
            )
        )
        or 0
    )
    admitted = recent < (limit or settings.admission_max_battles)
    session.add(
        AdmissionEvent(
            pseudonym=pseudonym,
            action=action,
            admitted=admitted,
            reason="within_limit" if admitted else "rate_limited",
        )
    )
    if not admitted:
        raise HTTPException(status_code=429, detail="battle admission limit reached")


def _select_models(
    session: Session,
    season: Season,
    track: str,
    selector: str,
    run_class: str,
    data_stratum: str,
    controlled_run_id: str | None,
) -> tuple[list[tuple[CatalogModel, SeasonModel]], str]:
    rows = session.execute(
        select(CatalogModel, SeasonModel)
        .join(SeasonModel, SeasonModel.model_id == CatalogModel.model_id)
        .where(SeasonModel.season_id == season.id, SeasonModel.eligible.is_(True))
    ).all()
    if run_class == "official":
        rows = [
            row
            for row in rows
            if not row[0].model_id.startswith("flavourbench/mock-")
            and row[1].provider_slug != "mock"
        ]
    if len(rows) < 2 and track == "model_arena":
        raise HTTPException(status_code=503, detail="season needs at least two eligible models")
    if not rows:
        raise HTTPException(status_code=503, detail="season has no eligible models")

    exposure_query = (
        select(ResponseArm.model_id, func.count(ResponseArm.id))
        .join(Battle, Battle.id == ResponseArm.battle_id)
        .where(
            Battle.season_id == season.id,
            Battle.track == track,
            Battle.run_class == run_class,
            Battle.manifest_sha256 == season.manifest_sha256,
            Battle.data_stratum == data_stratum,
        )
        .group_by(ResponseArm.model_id)
    )
    if data_stratum == "controlled":
        if controlled_run_id is None:
            raise HTTPException(status_code=409, detail="controlled run identity is required")
        exposure_query = exposure_query.where(Battle.controlled_run_id == controlled_run_id)
    else:
        exposure_query = exposure_query.where(Battle.controlled_run_id.is_(None))
    if run_class == "official":
        exposure_query = exposure_query.where(Battle.rank_eligible.is_(True))
    counts = dict(session.execute(exposure_query).all())
    ordered = []
    for exposure in sorted({counts.get(row[0].model_id, 0) for row in rows}):
        group = [row for row in rows if counts.get(row[0].model_id, 0) == exposure]
        ordered.extend(_seeded_shuffle(group, selector, f"exposure:{exposure}"))
    minimum_count = counts.get(ordered[0][0].model_id, 0)
    minimum_group = [row for row in ordered if counts.get(row[0].model_id, 0) == minimum_count]
    if track == "epicure_uplift":
        return ordered[:1], _fraction_text(1, len(minimum_group))

    if len(minimum_group) >= 2:
        denominator = math.comb(len(minimum_group), 2)
    else:
        next_count = min(
            counts.get(row[0].model_id, 0)
            for row in ordered
            if counts.get(row[0].model_id, 0) > minimum_count
        )
        next_group_size = sum(counts.get(row[0].model_id, 0) == next_count for row in ordered)
        denominator = next_group_size
    return ordered[:2], _fraction_text(1, denominator)


def _controlled_models(
    session: Session,
    season: Season,
    assignment: ControlledRunAssignment,
) -> list[tuple[CatalogModel, SeasonModel]]:
    model_ids = list(assignment.model_ids_json)
    rows = session.execute(
        select(CatalogModel, SeasonModel)
        .join(SeasonModel, SeasonModel.model_id == CatalogModel.model_id)
        .where(
            SeasonModel.season_id == season.id,
            SeasonModel.model_id.in_(model_ids),
        )
    ).all()
    by_id = {catalog.model_id: (catalog, slot) for catalog, slot in rows}
    if set(by_id) != set(model_ids):
        raise HTTPException(status_code=409, detail="controlled assignment model is unavailable")
    selected = [by_id[model_id] for model_id in model_ids]
    if any(not slot.eligible for _, slot in selected):
        raise HTTPException(status_code=409, detail="controlled assignment model is ineligible")
    if assignment.track == "model_arena":
        if len(selected) != 2 or selected[0][0].model_id == selected[1][0].model_id:
            raise HTTPException(
                status_code=409,
                detail="controlled model-arena assignment requires two distinct models",
            )
    elif assignment.track == "epicure_uplift":
        if len(selected) != 1:
            raise HTTPException(
                status_code=409,
                detail="controlled uplift assignment requires exactly one model",
            )
    else:
        raise HTTPException(status_code=409, detail="controlled assignment track is invalid")
    return selected


def _provider_reservation_amounts(models: list[SeasonModel]) -> dict[str, int]:
    amounts: dict[str, int] = {}
    for model in models:
        backend = model.execution_backend or "openrouter"
        amounts[backend] = amounts.get(backend, 0) + model.worst_case_cost_micros
    return amounts


def _uses_postgresql_budget_authority(session: Session | None) -> bool:
    return bool(
        session is not None
        and get_settings().execution_mode == "live"
        and session.get_bind().dialect.name == "postgresql"
    )


def _reserve_budget(
    session: Session | None,
    season: Season,
    models: list[SeasonModel],
    controlled_run: ControlledRun | None = None,
) -> int:
    settings = get_settings()
    provider_amounts = _provider_reservation_amounts(models)
    amount = sum(provider_amounts.values())
    if settings.execution_mode == "mock":
        return 0
    if season.status not in {"pilot", "active"} or season.manifest_sha256 == "unfrozen":
        raise HTTPException(status_code=503, detail="live season manifest is not frozen")
    if (
        season.epicure_bundle_sha256 in {"", "unresolved"}
        or season.epicure_application_sha256 in {"", "unresolved"}
        or season.tool_registry_sha256 in {"", "unfrozen", "unresolved"}
    ):
        raise HTTPException(status_code=503, detail="Epicure bundle lineage is unresolved")
    cap = season.budget_cap_micros
    projected = (season.budget_used_micros or 0) + (season.budget_reserved_micros or 0) + amount
    if cap <= 0 or projected * 10_000 >= cap * 8_500:
        raise HTTPException(status_code=503, detail="season budget admission is closed")
    if controlled_run is not None:
        run_cap = controlled_run.budget_cap_micros
        run_projected = (
            (controlled_run.budget_used_micros or 0)
            + (controlled_run.budget_reserved_micros or 0)
            + amount
        )
        if run_cap <= 0 or run_projected * 10_000 >= run_cap * 8_500:
            raise HTTPException(
                status_code=503,
                detail="controlled-run budget admission is closed",
            )
    if session is not None:
        provider_rows = session.scalars(
            select(SeasonProviderBudget)
            .where(
                SeasonProviderBudget.season_id == season.id,
                SeasonProviderBudget.execution_backend.in_(sorted(provider_amounts)),
            )
            .order_by(SeasonProviderBudget.execution_backend)
            .with_for_update()
        ).all()
        providers = {row.execution_backend: row for row in provider_rows}
        if set(providers) != set(provider_amounts):
            raise HTTPException(
                status_code=503,
                detail="provider budget authorization is missing",
            )
        now = datetime.now(UTC)
        account_rows = session.scalars(
            select(ProviderAccountBudget)
            .where(
                ProviderAccountBudget.execution_backend.in_(sorted(provider_amounts)),
                ProviderAccountBudget.account_scope_sha256.in_(
                    sorted(row.account_scope_sha256 for row in provider_rows)
                ),
            )
            .order_by(
                ProviderAccountBudget.execution_backend,
                ProviderAccountBudget.account_scope_sha256,
            )
            .with_for_update()
        ).all()
        account_budgets = {
            (row.execution_backend, row.account_scope_sha256): row for row in account_rows
        }
        try:
            assert_budget_integrity(
                session,
                season.id,
                lock_aggregates=True,
            )
        except BudgetIntegrityError as exc:
            raise HTTPException(
                status_code=503,
                detail="budget reservation evidence is inconsistent",
            ) from exc
        for backend, provider_amount in provider_amounts.items():
            provider_budget = providers[backend]
            account_budget = account_budgets.get((backend, provider_budget.account_scope_sha256))
            if account_budget is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"{backend} account-wide provider authorization is missing",
                )
            authorization = account_authorization(
                session,
                account_budget,
                for_update=True,
            )
            if (
                provider_budget.account_scope_sha256 != provider_account_scope_sha256(backend)
                or account_budget.account_scope_sha256 != provider_account_scope_sha256(backend)
                or account_budget.status != "active"
                or account_budget.budget_cap_micros != provider_account_hard_cap_micros(backend)
                or not account_authorization_chain_valid(
                    session,
                    account_budget,
                    authorization,
                    root_envelope_sha256=(provider_budget.account_authorization_envelope_sha256),
                    signing_secret=settings.budget_authorization_signing_secret,
                    verification_keys=budget_authorization_verification_keyring(settings),
                    now=now,
                )
            ):
                raise HTTPException(
                    status_code=503,
                    detail=f"{backend} provider account authorization is invalid",
                )
            valid_until = provider_budget.valid_until
            if valid_until.tzinfo is None:
                valid_until = valid_until.replace(tzinfo=UTC)
            provider_projected = (
                (provider_budget.budget_used_micros or 0)
                + (provider_budget.budget_reserved_micros or 0)
                + provider_amount
            )
            assert authorization is not None
            account_valid_until = authorization.valid_until
            if account_valid_until.tzinfo is None:
                account_valid_until = account_valid_until.replace(tzinfo=UTC)
            account_projected = (
                (account_budget.budget_used_micros or 0)
                + (account_budget.budget_reserved_micros or 0)
                + provider_amount
            )
            if (
                valid_until <= now
                or provider_budget.budget_cap_micros <= 0
                or provider_projected * 10_000 >= provider_budget.budget_cap_micros * 8_500
            ):
                raise HTTPException(
                    status_code=503,
                    detail=f"{backend} provider budget admission is closed",
                )
            if (
                account_valid_until <= now
                or account_budget.budget_cap_micros <= 0
                or account_projected * 10_000 >= account_budget.budget_cap_micros * 8_500
            ):
                raise HTTPException(
                    status_code=503,
                    detail=f"{backend} account-wide budget admission is closed",
                )
        if not _uses_postgresql_budget_authority(session):
            for backend, provider_amount in provider_amounts.items():
                provider_budget = providers[backend]
                provider_budget.budget_reserved_micros += provider_amount
                account_budgets[
                    (backend, provider_budget.account_scope_sha256)
                ].budget_reserved_micros += provider_amount
    if not _uses_postgresql_budget_authority(session):
        season.budget_reserved_micros = (season.budget_reserved_micros or 0) + amount
        if controlled_run is not None:
            controlled_run.budget_reserved_micros = (
                controlled_run.budget_reserved_micros or 0
            ) + amount
    return amount


def create_battle(
    session: Session,
    request: BattleCreate | ControlledBattleCreate,
    pseudonym: str,
    *,
    task: Task | None = None,
    controlled_run: ControlledRun | None = None,
    season_row: Season | None = None,
    admission_pseudonyms: list[tuple[str, str]] | None = None,
) -> Battle:
    settings = get_settings()
    season_selector = (
        select(Season).where(Season.id == season_row.id)
        if season_row is not None
        else select(Season).where(Season.slug == settings.default_season_slug)
    )
    season = session.scalar(season_selector.with_for_update())
    if season is None:
        raise HTTPException(status_code=503, detail="Season 0 is not initialized")
    if controlled_run is None and not isinstance(request, BattleCreate):
        raise HTTPException(status_code=409, detail="public battle request is invalid")
    if controlled_run is not None and not isinstance(request, ControlledBattleCreate):
        raise HTTPException(status_code=409, detail="controlled battle request is invalid")
    assignment: ControlledRunAssignment | None = None
    if controlled_run is not None:
        controlled_run = session.scalar(
            select(ControlledRun).where(ControlledRun.id == controlled_run.id).with_for_update()
        )
        if controlled_run is None or (
            controlled_run.season_id != season.id or controlled_run.status != "active"
        ):
            raise HTTPException(
                status_code=409, detail="controlled run is not active for this season"
            )
    nonce_sha = hashlib.sha256(request.client_nonce.encode()).hexdigest()
    existing = session.scalar(
        select(Battle).where(
            Battle.season_id == season.id,
            Battle.requester_pseudonym == pseudonym,
            Battle.client_nonce_sha256 == nonce_sha,
        )
    )
    if existing:
        if controlled_run is not None:
            existing_assignment = session.scalar(
                select(ControlledRunAssignment).where(
                    ControlledRunAssignment.controlled_run_id == controlled_run.id,
                    ControlledRunAssignment.battle_id == existing.id,
                )
            )
            if existing_assignment is None:
                raise HTTPException(
                    status_code=409,
                    detail="idempotent battle is not bound to the frozen schedule",
                )
            if (
                request.task_public_id is not None
                and request.task_public_id != existing_assignment.task_public_id
            ) or (
                request.expected_assignment_ordinal is not None
                and request.expected_assignment_ordinal != existing_assignment.ordinal
            ):
                raise HTTPException(
                    status_code=409,
                    detail="idempotency key is already bound to a different assignment",
                )
        if settings.execution_mode == "live":
            try:
                assert_budget_integrity(session, season.id, lock_aggregates=True)
            except BudgetIntegrityError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="budget reservation evidence is inconsistent",
                ) from exc
        return existing
    if controlled_run is not None:
        assignment = session.scalar(
            select(ControlledRunAssignment)
            .where(
                ControlledRunAssignment.controlled_run_id == controlled_run.id,
                ControlledRunAssignment.status == "pending",
            )
            .order_by(ControlledRunAssignment.ordinal)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if assignment is None:
            raise HTTPException(status_code=409, detail="controlled-run schedule is exhausted")
        if (
            request.task_public_id is not None
            and request.task_public_id != assignment.task_public_id
        ) or (
            request.expected_assignment_ordinal is not None
            and request.expected_assignment_ordinal != assignment.ordinal
        ):
            raise HTTPException(
                status_code=409,
                detail="request does not match the next frozen assignment",
            )
        scheduled_task = session.get(Task, assignment.task_id)
        if scheduled_task is None:
            raise HTTPException(status_code=409, detail="controlled assignment task is unavailable")
        if task is not None and task.id != scheduled_task.id:
            raise HTTPException(status_code=409, detail="controlled task assertion is invalid")
        task = scheduled_task
        if (
            task.season_id != season.id
            or task.public_id != assignment.task_public_id
            or task.revision != assignment.task_revision
            or task.prompt_sha256 != assignment.task_prompt_sha256
            or task.family != assignment.task_family
        ):
            raise HTTPException(status_code=409, detail="controlled assignment task has drifted")
    admission_keys = [("create_battle", pseudonym)]
    admission_keys.extend(
        (f"create_battle:{label}", value) for label, value in (admission_pseudonyms or [])
    )
    for action, value in sorted(admission_keys):
        limit = (
            settings.admission_max_battles * settings.admission_network_multiplier
            if action == "create_battle:network"
            else settings.admission_max_battles
        )
        _admit(session, value, action=action, limit=limit)

    run_class = _run_class(season)
    base_rank_eligible = (
        run_class == "official"
        and season.manifest_sha256 not in {"", "unfrozen", "unresolved"}
        and season.protocol_bundle_sha256 not in {"", "unfrozen", "unresolved"}
    )
    if assignment is not None:
        assignment_seed = assignment.assignment_seed
        track = assignment.track
        track_probability_text = "1/1"
        model_probability = "1/1"
        scheduler_version = CONTROLLED_SCHEDULER_VERSION
        selected = _controlled_models(session, season, assignment)
        prompt = task.prompt
        category = task.family
        research_consent = False
        if controlled_run is None:
            raise HTTPException(status_code=409, detail="controlled run is unavailable")
        if controlled_run.evaluation_order_id is None and run_class == "official":
            retention_basis = "official_research"
            retention_until = PERMANENT_RESEARCH_RETENTION_UNTIL
        elif controlled_run.evaluation_order_id is None:
            retention_basis = "controlled_development"
            retention_until = datetime.now(UTC) + timedelta(days=settings.retention_days)
        else:
            organization = session.get(Organization, controlled_run.organization_id)
            private_days = (
                organization.retention_policy_json.get("privateEvidenceDays")
                if organization is not None
                else None
            )
            if (
                isinstance(private_days, bool)
                or not isinstance(private_days, int)
                or not 1 <= private_days <= 3650
            ):
                raise HTTPException(
                    status_code=409,
                    detail="commercial run lacks a valid sealed private-evidence retention term",
                )
            retention_basis = "commercial_private"
            retention_until = datetime.now(UTC) + timedelta(days=private_days)
    else:
        assignment_seed = secrets.token_hex(32)
        track_value = _seeded_index(assignment_seed, "track", 100)
        track = "model_arena" if track_value < settings.model_track_percent else "epicure_uplift"
        track_probability = (
            settings.model_track_percent
            if track == "model_arena"
            else 100 - settings.model_track_percent
        )
        track_probability_text = _fraction_text(track_probability, 100)
        model_probability = ""
        scheduler_version = SCHEDULER_VERSION
        prompt = request.prompt
        category = request.category.value
        research_consent = request.research_consent
        retention_basis = "public_consented" if research_consent else "public_nonconsented"
        retention_until = datetime.now(UTC) + timedelta(days=settings.retention_days)
    data_stratum = "controlled" if assignment is not None else "public_freeform"
    # Free-form prompts stay outside the general leaderboard until a separately
    # versioned scope review creates a battle_general_track_scope_admitted event.
    rank_eligible = base_rank_eligible and assignment is not None
    if assignment is None:
        selected, model_probability = _select_models(
            session,
            season,
            track,
            f"{assignment_seed}:models",
            run_class,
            data_stratum,
            None,
        )
    reservation_models = [row[1] for row in selected]
    if track == "epicure_uplift":
        reservation_models = [selected[0][1], selected[0][1]]
    reserved = _reserve_budget(session, season, reservation_models, controlled_run)
    provider_reservations = _provider_reservation_amounts(reservation_models) if reserved else {}

    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    battle = Battle(
        season_id=season.id,
        run_class=run_class,
        rank_eligible=rank_eligible,
        data_stratum=data_stratum,
        task_id=task.id if task is not None else None,
        task_revision=task.revision if task is not None else None,
        controlled_run_id=controlled_run.id if controlled_run is not None else None,
        manifest_sha256=season.manifest_sha256,
        protocol_bundle_sha256=season.protocol_bundle_sha256,
        scheduler_version=scheduler_version,
        assignment_seed=assignment_seed,
        track_assignment_probability=track_probability_text,
        model_assignment_probability=model_probability,
        side_assignment_probability="1/2",
        track=track,
        category=category,
        prompt=prompt,
        prompt_sha256=prompt_sha,
        client_nonce_sha256=nonce_sha,
        research_consent=research_consent,
        retention_basis=retention_basis,
        release_review_status="pending" if research_consent else "not_requested",
        requester_pseudonym=pseudonym,
        status="queued",
        reserved_cost_micros=reserved,
        provider_reservations_json=provider_reservations,
        retention_until=retention_until,
    )
    session.add(battle)
    session.flush()
    if (
        task is not None
        and isinstance(task.provenance_json, dict)
        and (task.provenance_json.get("confirmatory_eligible") is True)
    ):
        try:
            record_task_first_use(session, task=task, battle=battle)
        except TaskLifecycleError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"confirmatory task lifecycle is invalid: {exc}",
            ) from exc
    if assignment is not None:
        assignment.status = "queued"
        assignment.battle_id = battle.id
    postgresql_budget_authority = _uses_postgresql_budget_authority(session)
    if reserved and not postgresql_budget_authority:
        from .models import CostEvent

        session.add(
            CostEvent(
                season_id=season.id,
                battle_id=battle.id,
                kind="reserve",
                amount_micros=reserved,
                provider="governor",
            )
        )
        for backend, backend_reserved in sorted(provider_reservations.items()):
            session.add(
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_reserve",
                    amount_micros=backend_reserved,
                    provider=backend,
                    accounting_json={"budget_scope": "provider"},
                )
            )
            session.add(
                CostEvent(
                    season_id=season.id,
                    battle_id=battle.id,
                    kind="provider_account_reserve",
                    amount_micros=backend_reserved,
                    provider=backend,
                    accounting_json={
                        "budget_scope": "provider_account",
                        "account_scope_sha256": provider_account_scope_sha256(backend),
                    },
                )
            )

    if track == "model_arena":
        specifications = [
            (selected[0][0], selected[0][1], "epicure_on"),
            (selected[1][0], selected[1][1], "epicure_on"),
        ]
    else:
        specifications = [
            (selected[0][0], selected[0][1], "epicure_on"),
            (selected[0][0], selected[0][1], "epicure_off"),
        ]

    reverse = controlled_side_is_reversed(assignment_seed)
    if reverse:
        specifications.reverse()
    arms: list[ResponseArm] = []
    for side, (model, season_model, condition) in zip(
        ("left", "right"), specifications, strict=True
    ):
        is_submitted_commercial_arm = bool(
            controlled_run is not None
            and controlled_run.evaluation_order_id is not None
            and model.model_id == controlled_run.submitted_endpoint_model_id
        )
        arm = ResponseArm(
            battle_id=battle.id,
            side=side,
            condition=condition,
            model_id=model.model_id,
            execution_backend=season_model.execution_backend,
            route_revision_id=(
                controlled_run.route_revision_id if is_submitted_commercial_arm else None
            ),
            endpoint_descriptor_sha256=(
                controlled_run.endpoint_descriptor_sha256 if is_submitted_commercial_arm else None
            ),
            provider_slug=season_model.provider_slug,
            prompt_sha256=prompt_sha,
            system_prompt_sha256=system_prompt_sha256(condition),
            schema_sha256=FINAL_SCHEMA_SHA256,
            tool_schema_sha256=season.tool_registry_sha256,
            decoding_json={
                **{
                    name: season_model.decoding_json.get(name, "provider_fixed_unsupported")
                    for name in sorted(DECODING_PARAMETERS)
                },
                "structured_output": True,
                "max_tool_rounds": settings.max_tool_rounds,
            },
            protocol_bundle_sha256=season.protocol_bundle_sha256,
            epicure_release_id=season.epicure_release_id,
            epicure_bundle_sha256=season.epicure_bundle_sha256,
            epicure_application_sha256=season.epicure_application_sha256,
        )
        session.add(arm)
        arms.append(arm)
    session.flush()
    battle.left_arm_id = arms[0].id
    battle.right_arm_id = arms[1].id
    session.flush()
    if reserved and postgresql_budget_authority:
        authority_result = (
            session.execute(
                text(
                    "SELECT reserved_cost_micros, provider_reservations, idempotent "
                    "FROM public.flavourbench_reserve_battle_budget(:battle_id)"
                ),
                {"battle_id": battle.id},
            )
            .mappings()
            .one()
        )
        observed_provider_reservations = {
            str(backend): int(amount)
            for backend, amount in dict(authority_result["provider_reservations"] or {}).items()
        }
        if (
            int(authority_result["reserved_cost_micros"]) != reserved
            or observed_provider_reservations != provider_reservations
            or bool(authority_result["idempotent"])
        ):
            raise HTTPException(
                status_code=503,
                detail="database budget authority returned inconsistent reservation evidence",
            )
        session.expire_all()
        battle = session.get(Battle, battle.id)
        if battle is None:
            raise HTTPException(status_code=503, detail="reserved battle disappeared")
    session.add(
        Job(
            kind="generate_battle",
            battle_id=battle.id,
            payload_json={"season": season.slug},
            max_attempts=3,
        )
    )
    session.add(
        RunEvent(
            entity_type="battle",
            entity_id=battle.id,
            event_type="battle_queued",
            payload_json={
                "track": track,
                "category": category,
                "research_consent": research_consent,
                "reserved_cost_micros": reserved,
                "run_class": run_class,
                "rank_eligible": rank_eligible,
                "rank_eligibility_boundary": (
                    "frozen_controlled_task"
                    if assignment is not None
                    else "withheld_pending_general_track_scope_admission"
                ),
                "data_stratum": data_stratum,
                "task_id": task.id if task is not None else None,
                "task_revision": task.revision if task is not None else None,
                "controlled_run_id": controlled_run.id if controlled_run is not None else None,
                "manifest_sha256": season.manifest_sha256,
                "protocol_bundle_sha256": season.protocol_bundle_sha256,
                "scheduler_version": scheduler_version,
                "controlled_assignment_id": assignment.id if assignment else None,
                "controlled_assignment_ordinal": assignment.ordinal if assignment else None,
                "controlled_assignment_sha256": (
                    assignment.assignment_sha256 if assignment else None
                ),
                "track_assignment_probability": battle.track_assignment_probability,
                "model_assignment_probability": model_probability,
                "side_assignment_probability": "1/2",
            },
        )
    )
    if settings.execution_mode == "live":
        session.flush()
        try:
            assert_budget_integrity(session, season.id, lock_aggregates=True)
        except BudgetIntegrityError as exc:
            raise HTTPException(
                status_code=503,
                detail="budget reservation evidence is inconsistent",
            ) from exc
    return battle
