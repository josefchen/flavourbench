from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import uuid
from collections import Counter
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from .account_authority import (
    account_authorization,
    account_authorization_chain_valid,
    as_utc,
)
from .anonymous_reviewer_control import (
    RECONSENT_STATEMENT,
    RECONSENT_STATEMENT_SHA256,
    admission_activation_sha256,
    anonymous_pool_reconsented,
    append_pool_reconsent,
    reviewer_control_lock,
)
from .arena import _admit as _postgres_admit
from .arena import controlled_side_is_reversed, create_battle
from .bedrock_contract import parse_bedrock_endpoint_contract
from .budget_integrity import BudgetIntegrityError, assert_budget_integrity
from .budget_policy import (
    provider_account_hard_cap_micros,
    provider_account_scope_sha256,
)
from .catalog import fetch_openrouter_catalog, sync_catalog
from .commercial import _spend_authorization_for_order
from .commercial import router as commercial_router
from .commercial_authority import (
    active_publication_authorization,
    active_spend_authorization,
    publication_authorization_binding,
)
from .commercial_authority import (
    canonical_sha256 as commercial_binding_sha256,
)
from .config import (
    budget_authorization_verification_keyring,
    get_settings,
    run_card_verification_keyring,
)
from .consent_documents import resolve_expert_consent_document
from .construct_blueprint import (
    BLUEPRINT_SHA256,
    CONSTRUCT_CELLS,
    DIFFICULTY_TIERS,
    ConstructBlueprintError,
    validate_candidate_binding,
    validate_task_binding,
)
from .contamination_calibration import (
    ContaminationCalibrationArtifact,
    ContaminationCalibrationError,
    load_contamination_calibration,
)
from .controlled_integrity import (
    ControlledRunIntegrityError,
    verify_controlled_run_bijection,
)
from .database import SessionLocal, database_readiness, get_db, init_database
from .development_task_statistics import (
    DevelopmentTaskStatisticsError,
    summarize_task_validation,
)
from .development_task_validation import (
    REQUIRED_INDEPENDENT_REVIEWERS,
    DevelopmentTaskValidationError,
    verify_validation_packet,
)
from .endpoint_contract import (
    UNFROZEN_VALUES,
    endpoint_contract_payload,
    endpoint_contract_sha256,
)
from .engine import (
    cancel_unstarted_controlled_jobs,
    has_unresolved_paid_attempt,
    is_complete_finish_reason,
    reconcile_battle_cost,
    redact_expired,
)
from .expert_review import (
    PROTOCOL_SHA256 as EXPERT_PROTOCOL_SHA256,
)
from .expert_review import (
    PROTOCOL_VERSION as EXPERT_PROTOCOL_VERSION,
)
from .expert_review import (
    RELIABILITY_REPEAT_INTERVAL,
    WORKLOAD_TARGET,
    author_evaluator_workload_cell_targets,
    canonical_sha256,
    isolated_uplift_workload_cell_targets,
    normalize_choice,
    normalize_rubric,
    presentation_sha256,
    protocol_payload,
    reliability_summary,
    workload_cell_targets,
)
from .models import (
    TOOL_CALL_REDACTION_JSON,
    TOOL_CALL_REDACTION_SENTINEL,
    Battle,
    BedrockBillingCrosscheck,
    BedrockBillingCrosscheckArm,
    CatalogModel,
    ControlledRun,
    ControlledRunAssignment,
    ControlledRunReviewer,
    CostEvent,
    EpicureRelease,
    EvaluationOrder,
    ExpertReviewer,
    GenerationAttempt,
    Incident,
    Job,
    LeaderboardSnapshot,
    ModelRouteRevision,
    ModelSubmission,
    ProviderAccountAuthorization,
    ProviderAccountBudget,
    ResponseArm,
    ReviewerCalibrationBallot,
    ReviewerCalibrationSet,
    ReviewerIdentityBinding,
    ReviewerQualificationEvidence,
    RunEvent,
    Season,
    SeasonModel,
    SeasonProviderBudget,
    Task,
    TaskEvidenceArtifact,
    ToolCall,
    ValidatorResult,
    Vote,
)
from .participant_lifecycle import (
    ParticipantLifecycleError,
    accept_participant_consent,
    create_retention_schedule,
    enroll_participant_identity,
    enrollment_consent_view,
    execute_participant_private_payload_deletion,
    execute_private_payload_deletion,
    issue_enrollment_offer,
    privacy_safe_participant_status,
    require_active_participant_authority,
    withdraw_participant,
)
from .protocol_contract import build_protocol_bundle
from .provider import FINAL_SCHEMA_SHA256
from .reviewer_admission import (
    calibrated_expert_admission_active as _calibrated_expert_admission_active,
)
from .reviewer_admission import calibrated_expert_admission_event
from .reviewer_identity import (
    ReviewerIdentityError,
    apply_verified_vote_provenance,
    bind_reviewer_identity,
    consume_reviewer_credential,
    derive_family_admission,
    exchange_enrollment_credential,
    freeze_calibration_set,
    issue_reviewer_credential,
    privacy_safe_vote_release,
    record_calibration_ballot,
    record_qualification_evidence,
    resolve_verified_vote_admission,
    reviewer_rater_pseudonym,
)
from .schemas import (
    AnonymousExternalAdmissionCreate,
    AnonymousExternalReconsentCreate,
    AuthorEvaluatorAdmissionCreate,
    BattleCreate,
    BedrockBillingCrosscheckCreate,
    ConfirmatoryTaskBankCreate,
    ControlledBattleCreate,
    ControlledReviewerAuthorizationCreate,
    ControlledRunCreate,
    ControlledRunLifecycleCreate,
    ControlledRunReleaseCreate,
    ControlledTokenRotationCreate,
    CostSettlementCreate,
    DevelopmentTaskAdjudicationCreate,
    DevelopmentTaskBlindValidityCreate,
    DevelopmentTaskCriteriaCreate,
    EpicureReleaseRegisterCreate,
    EvaluationOrderProvisionCreate,
    ExpertAdmissionCreate,
    ExpertCalibrationCandidateRegister,
    ExpertInviteCreate,
    ExpertReviewCreate,
    ExpertSessionCreate,
    ExpertTaskAssessmentCreate,
    ModelSmokeCreate,
    ParticipantConsentAcceptanceCreate,
    ParticipantEnrollmentOfferCreate,
    ParticipantIdentityEnrollmentCreate,
    ParticipantWithdrawalCreate,
    ProviderAccountAuthorizationCreate,
    ProviderAccountAuthorizationRevokeCreate,
    ReleaseReviewCreate,
    ReviewerCalibrationBallotCreate,
    ReviewerCalibrationSetCreate,
    ReviewerFamilyAdmissionCreate,
    ReviewerIdentityBindingCreate,
    ReviewerPrivatePayloadDeletionCreate,
    ReviewerQualificationEvidenceCreate,
    ReviewerRetentionScheduleCreate,
    Season1ArenaMethodValidationCreate,
    Season1PostcollectionItemAuditCreate,
    SeasonFreezeCreate,
    SeasonOfficializeCreate,
    SeasonProvisionCreate,
    SnapshotPublishCreate,
    TaskCandidateAdjudicationCreate,
    TaskCandidateBlindValidityCreate,
    TaskCandidateReconciliationCreate,
    TaskCandidateReviewCreate,
    TaskChallengeAdjudicationCreate,
    TaskChallengeCreate,
    TaskContaminationAuditReviewCreate,
    TaskContributionCreate,
    TaskContributionWithdrawalCreate,
    TaskContributorInviteCreate,
    TaskContributorProtocolAcceptanceCreate,
    TaskRegistryFreezeCreate,
    TaskValidatorContractReviewCreate,
    TaskValidatorInviteCreate,
    VoteCreate,
)
from .season1_arena_acceptance import (
    DEFAULT_POLICY_PATH,
    STUDY_DESIGN_SHA256,
    ArenaInferenceAcceptanceError,
    publication_acceptance_deficits,
)
from .season1_arena_monte_carlo import verify_production_result
from .season1_readiness import valid_post_collection_item_audit
from .season_design import (
    CONFIRMATORY_TASK_COUNT,
    CONFIRMATORY_TASKS_PER_FAMILY,
    SEASON_MODEL_COUNT,
    SEASON_SLOT_ROLE_COUNTS,
    SEASON_TASK_SPLIT_COUNTS,
    SEASON_TASK_SPLIT_COUNTS_PER_FAMILY,
)
from .security import require_admin_token, require_service_token, sanitize_for_release
from .seed import seed_database
from .service_ranking import (
    InProcessSnapshotAnalysisForbidden,
    analysis_battle_eligibility,
    analysis_vote_eligibility,
    assert_api_analysis_runtime_clean,
    model_leaderboard,
    require_snapshot_analysis_process,
    snapshot_hash,
    uplift_leaderboard,
)
from .task_contributor_protocol import (
    PROTOCOL_SCOPE as TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
)
from .task_contributor_protocol import (
    PROTOCOL_SHA256 as TASK_CONTRIBUTOR_PROTOCOL_SHA256,
)
from .task_contributor_protocol import (
    PROTOCOL_VERSION as TASK_CONTRIBUTOR_PROTOCOL_VERSION,
)
from .task_contributor_protocol import (
    TaskContributorProtocolError,
    protocol_binding_active,
)
from .task_contributor_protocol import (
    protocol_text as task_contributor_protocol_text,
)
from .task_evidence import (
    ContaminationScanBundle,
    TaskEvidenceError,
    load_contamination_scan_bundle,
    task_evidence_review_sha256,
    task_evidence_root_sha256,
    verify_contamination_audit,
    verify_validator_contract,
)
from .task_evidence import (
    canonical_sha256 as task_evidence_sha256,
)
from .task_evidence_registry import verify_task_evidence_registry
from .task_lifecycle import (
    TaskLifecycleError,
    task_lifecycle_seal_sha256,
    verify_task_lifecycle,
)
from .task_validation_runtime import (
    router as task_validation_router,
)
from .task_validation_runtime import (
    verify_task_validation_runtime_evidence,
)
from .validator_calibration import (
    ValidatorCalibrationArtifact,
    load_validator_calibration,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    assert_api_analysis_runtime_clean()
    if get_settings().task_validation_campaign_enabled:
        verify_task_validation_runtime_evidence()
    init_database()
    seed_database()
    with SessionLocal() as session:
        database_readiness(session, expected_role="flavourbench_api")
    assert_api_analysis_runtime_clean()
    yield


app = FastAPI(
    title="Epicure FlavourBench API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
router = APIRouter(dependencies=[Depends(require_service_token)])
Db = Annotated[Session, Depends(get_db)]

CONTROLLED_RUN_CARD_SCHEMA_VERSION = "flavourbench-controlled-run-card-v7"
SUPPORTED_CONTROLLED_RUN_CARD_SCHEMA_VERSIONS = frozenset(
    {
        "flavourbench-controlled-run-card-v4",
        "flavourbench-controlled-run-card-v5",
        "flavourbench-controlled-run-card-v6",
        CONTROLLED_RUN_CARD_SCHEMA_VERSION,
    }
)
CONTROLLED_RUN_COST_ACCOUNTING_POLICY = {
    "controlled_run_used_basis": "endpoint_generation_receipts",
    "aggregate_invoice_variance_scope": "season_and_provider_account_only",
    "credits_restore_spend_authority": False,
}


def _pseudonym(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HTTPException(status_code=401, detail="request pseudonym is missing")
    return value


def _participant_bearer(authorization: str) -> str:
    scheme, separator, token = authorization.partition(" ")
    if scheme != "Bearer" or not separator or not token or token != token.strip():
        raise HTTPException(status_code=401, detail="participant credential is required")
    return token


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _utc_iso(value: datetime) -> str:
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return normalized.astimezone(UTC).isoformat()


def _uses_postgresql_budget_authority(session: Session) -> bool:
    return bool(
        get_settings().execution_mode == "live" and session.get_bind().dialect.name == "postgresql"
    )


def _require_budget_integrity(
    session: Session,
    season_id: str,
    *,
    lock_aggregates: bool = True,
) -> None:
    try:
        assert_budget_integrity(
            session,
            season_id,
            lock_aggregates=lock_aggregates,
        )
    except BudgetIntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="budget reservation evidence is inconsistent",
        ) from exc


class _SnapshotVerificationError(RuntimeError):
    """A stored snapshot or its live evidence no longer matches its content address."""

    def __init__(self, code: str, detail: str, diagnostics: dict[str, str | None]):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.diagnostics = diagnostics


_NONOFFICIAL_LINEAGE_MARKERS = (
    "exploratory",
    "unmatched",
    "unresolved",
    "development",
    "draft",
    "mock",
)


def _lineage_release_is_named_for_official_use(release_id: str) -> bool:
    normalized = release_id.casefold()
    return not any(marker in normalized for marker in _NONOFFICIAL_LINEAGE_MARKERS)


def _verified_official_epicure_release(
    session: Session,
    season: Season,
) -> EpicureRelease:
    release = session.get(EpicureRelease, season.epicure_release_id)
    if (
        release is None
        or not release.official_eligible
        or not release.public_release_match
        or not release.redistribution_rights_cleared
        or not release.reproducibility_verified
        or not _lineage_release_is_named_for_official_use(release.release_id)
        or release.bundle_sha256 != season.epicure_bundle_sha256
        or release.application_sha256 != season.epicure_application_sha256
        or _canonical_sha256(release.lineage_manifest_json) != release.lineage_manifest_sha256
    ):
        raise HTTPException(
            status_code=409,
            detail="officialization requires a matching eligible public Epicure release",
        )
    return release


def _verify_season_protocol(season: Season) -> None:
    """Fail closed when executable or analytical code drifts after season freeze."""

    bundle, digest = build_protocol_bundle(
        tool_registry_sha256=season.tool_registry_sha256,
        epicure_release_id=season.epicure_release_id,
        epicure_bundle_sha256=season.epicure_bundle_sha256,
        epicure_application_sha256=season.epicure_application_sha256,
        analysis_plan_sha256=season.analysis_plan_sha256,
        model_smoke_registry_sha256=str(
            season.protocol_bundle_json.get("model_smoke_registry_sha256", "unfrozen")
        ),
    )
    if (
        season.protocol_bundle_sha256 in UNFROZEN_VALUES
        or season.protocol_bundle_json != bundle
        or not hmac.compare_digest(season.protocol_bundle_sha256, digest)
    ):
        raise HTTPException(
            status_code=503,
            detail="frozen season execution or analysis protocol has drifted",
        )


def _task_registry_payload(tasks: list[Task]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: item.public_id):
        record: dict[str, Any] = {
            "public_id": task.public_id,
            "family": task.family,
            "prompt_sha256": task.prompt_sha256,
            "revision": task.revision,
            "split": task.split,
        }
        provenance = task.provenance_json if isinstance(task.provenance_json, dict) else {}
        if provenance.get("confirmatory_eligible") is True:
            validator_review = provenance.get("validator_contract_review")
            contamination_review = provenance.get("contamination_audit_review")
            record["confirmatory_evidence"] = {
                "candidate_record_sha256": provenance.get("candidate_record_sha256"),
                "human_author_person_commitment_sha256": provenance.get(
                    "human_author_person_commitment_sha256"
                ),
                "review_history_sha256": provenance.get("review_history_sha256"),
                "validator_contract_sha256": provenance.get("validator_contract_sha256"),
                "validator_contract_review_event_sha256": (
                    validator_review.get("review_event_sha256")
                    if isinstance(validator_review, dict)
                    else None
                ),
                "validator_calibration_artifact_sha256": provenance.get(
                    "validator_calibration_artifact_sha256"
                ),
                "validator_calibration_receipt_sha256": provenance.get(
                    "validator_calibration_receipt_sha256"
                ),
                "contamination_audit_sha256": provenance.get("contamination_audit_sha256"),
                "contamination_audit_review_event_sha256": (
                    contamination_review.get("review_event_sha256")
                    if isinstance(contamination_review, dict)
                    else None
                ),
                "contamination_calibration_artifact_sha256": provenance.get(
                    "contamination_calibration_artifact_sha256"
                ),
                "contamination_calibration_receipt_sha256": provenance.get(
                    "contamination_calibration_receipt_sha256"
                ),
                "task_record_sha256": provenance.get("task_record_sha256"),
                "task_evidence_root_sha256": provenance.get("task_evidence_root_sha256"),
            }
        payload.append(record)
    return payload


def _task_registry_sha256(tasks: list[Task]) -> str:
    return _canonical_sha256({"tasks": _task_registry_payload(tasks)})


def _verified_task_evidence_registry(session: Session, task: Task) -> bool:
    try:
        verify_task_evidence_registry(
            session,
            task,
            expected_container_image_digest=get_settings().build_image_digest,
            contamination_scan_bundle=_contamination_scan_bundle(),
            validator_calibration=_validator_calibration(),
            contamination_calibration=_contamination_calibration(),
        )
    except TaskEvidenceError:
        return False
    return True


def _assignment_payload(
    *,
    ordinal: int,
    task_public_id: str,
    task_revision: int,
    task_prompt_sha256: str,
    task_family: str,
    track: str,
    model_ids: list[str],
    repetition_index: int,
    side_seed_commitment_sha256: str,
) -> dict[str, Any]:
    canonical_models = sorted(model_ids) if track == "model_arena" else list(model_ids)
    return {
        "ordinal": ordinal,
        "task_public_id": task_public_id,
        "task_revision": task_revision,
        "task_prompt_sha256": task_prompt_sha256,
        "task_family": task_family,
        "track": track,
        "model_ids": canonical_models,
        "repetition_index": repetition_index,
        "side_seed_commitment_sha256": side_seed_commitment_sha256,
    }


def _schedule_sha256(payloads: list[dict[str, Any]]) -> str:
    return _canonical_sha256({"assignments": payloads})


def _rate_card_contract(
    model: CatalogModel,
    *,
    max_completion_tokens: int,
    endpoint_rate_card: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, int]:
    """Freeze a conservative provider price envelope for one response arm."""

    endpoint_rate_card = endpoint_rate_card or {}
    pricing = model.pricing_json if isinstance(model.pricing_json, dict) else {}
    try:
        prompt_price = Decimal(
            str(endpoint_rate_card.get("prompt_price_per_token", pricing.get("prompt", "")))
        )
        completion_price = Decimal(
            str(endpoint_rate_card.get("completion_price_per_token", pricing.get("completion", "")))
        )
        request_price = Decimal(
            str(endpoint_rate_card.get("request_price", pricing.get("request", "0") or "0"))
        )
        reasoning_price = Decimal(
            str(
                endpoint_rate_card.get(
                    "internal_reasoning_price_per_token",
                    pricing.get("internal_reasoning", "0") or "0",
                )
            )
        )
        cache_read_price = Decimal(
            str(
                endpoint_rate_card.get(
                    "input_cache_read_price_per_token",
                    pricing.get("input_cache_read", "0") or "0",
                )
            )
        )
        cache_write_price = Decimal(
            str(
                endpoint_rate_card.get(
                    "input_cache_write_price_per_token",
                    pricing.get("input_cache_write", "0") or "0",
                )
            )
        )
        cache_write_1h_price = Decimal(
            str(
                endpoint_rate_card.get(
                    "input_cache_write_1h_price_per_token",
                    pricing.get("input_cache_write_1h", "0") or "0",
                )
            )
        )
        image_price = Decimal(
            str(endpoint_rate_card.get("image_price_per_unit", pricing.get("image", "0") or "0"))
        )
        web_search_price = Decimal(
            str(
                endpoint_rate_card.get(
                    "web_search_price_per_request",
                    pricing.get("web_search", "0") or "0",
                )
            )
        )
    except (InvalidOperation, ValueError) as exc:
        raise HTTPException(status_code=409, detail="endpoint pricing metadata is invalid") from exc
    if (
        not prompt_price.is_finite()
        or not completion_price.is_finite()
        or not request_price.is_finite()
        or any(
            not price.is_finite() or price < 0
            for price in (
                prompt_price,
                completion_price,
                request_price,
                reasoning_price,
                cache_read_price,
                cache_write_price,
                cache_write_1h_price,
                image_price,
                web_search_price,
            )
        )
    ):
        raise HTTPException(status_code=409, detail="endpoint pricing metadata is invalid")
    settings = get_settings()
    maximum_requests = settings.max_tool_rounds + 1
    context_length = int(endpoint_rate_card.get("context_length") or model.context_length or 0)
    if settings.execution_mode != "mock" and (
        context_length <= 0
        or not endpoint_rate_card
        or not endpoint_rate_card.get("pricing_source_uri")
        or not endpoint_rate_card.get("pricing_source_document_sha256")
        or not endpoint_rate_card.get("pricing_observed_at")
        or any(
            field not in endpoint_rate_card
            for field in (
                "internal_reasoning_price_per_token",
                "input_cache_read_price_per_token",
                "input_cache_write_price_per_token",
                "input_cache_write_1h_price_per_token",
                "image_price_per_unit",
                "web_search_price_per_request",
            )
        )
    ):
        raise HTTPException(
            status_code=409,
            detail="live endpoint lacks an endpoint-specific price and source envelope",
        )
    rate_card = {
        "schema_version": "flavourbench-endpoint-rate-card-v3",
        "currency": "USD",
        "unit": "per_token_unless_request",
        "prompt_price_per_token": str(prompt_price),
        "completion_price_per_token": str(completion_price),
        "request_price": str(request_price),
        "internal_reasoning_price_per_token": str(reasoning_price),
        "input_cache_read_price_per_token": str(cache_read_price),
        "input_cache_write_price_per_token": str(cache_write_price),
        "input_cache_write_1h_price_per_token": str(cache_write_1h_price),
        "image_price_per_unit": str(image_price),
        "web_search_price_per_request": str(web_search_price),
        "context_length": context_length,
        "pricing_source_uri": endpoint_rate_card.get("pricing_source_uri", "mock://catalog"),
        "pricing_source_document_sha256": endpoint_rate_card.get(
            "pricing_source_document_sha256", "0" * 64
        ),
        "pricing_observed_at": endpoint_rate_card.get("pricing_observed_at", "mock"),
        "maximum_provider_requests_per_arm": maximum_requests,
        "maximum_completion_tokens_per_request": max_completion_tokens,
        "maximum_images_per_request": 0,
        "maximum_web_searches_per_request": 0,
        "calculation": (
            "full_context_at_maximum_input_or_cache_write_rate_plus_"
            "max_completion_and_reasoning_each_request"
        ),
    }
    digest = _canonical_sha256(rate_card)
    input_price = max(
        prompt_price + max(cache_write_price, cache_write_1h_price),
        cache_read_price,
    )
    dollars = maximum_requests * (
        context_length * input_price
        + max_completion_tokens * (completion_price + reasoning_price)
        + request_price
    )
    micros = int((dollars * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))
    return rate_card, digest, micros


def _snapshot_evidence_manifest(
    session: Session,
    *,
    season: Season,
    track: str,
    cohort: str,
    category: str,
    data_stratum: str,
    controlled_run_id: str | None,
    evidence_cutoff_at: datetime,
) -> dict[str, Any]:
    battle_query = select(Battle).where(
        Battle.season_id == season.id,
        Battle.track == track,
        Battle.data_stratum == data_stratum,
        Battle.completed_at.is_not(None),
        Battle.completed_at <= evidence_cutoff_at,
    )
    if category != "all":
        battle_query = battle_query.where(Battle.category == category)
    if controlled_run_id is None:
        battle_query = battle_query.where(Battle.controlled_run_id.is_(None))
    else:
        battle_query = battle_query.where(Battle.controlled_run_id == controlled_run_id)
    battles = session.scalars(battle_query.order_by(Battle.id)).all()
    battles_by_id = {battle.id: battle for battle in battles}
    battle_ids = [battle.id for battle in battles]
    all_arms = (
        session.scalars(
            select(ResponseArm)
            .where(
                ResponseArm.battle_id.in_(battle_ids),
                ResponseArm.created_at <= evidence_cutoff_at,
            )
            .order_by(ResponseArm.battle_id, ResponseArm.side, ResponseArm.id)
        ).all()
        if battle_ids
        else []
    )
    battle_selection = analysis_battle_eligibility(
        session,
        season,
        battles,
        track=track,
        data_stratum=data_stratum,
        controlled_run_id=controlled_run_id,
        evidence_cutoff_at=evidence_cutoff_at,
    )
    operational_battle_ids = {
        battle_id
        for battle_id, selection in battle_selection.items()
        if selection["operational_included"]
    }
    arms = [
        arm
        for arm in all_arms
        if arm.battle_id in operational_battle_ids
        and not arm.model_id.startswith("flavourbench/mock-")
        and arm.provider_slug != "mock"
    ]
    arms_by_id = {arm.id: arm for arm in arms}
    arm_ids = [arm.id for arm in arms]
    model_ids = sorted({arm.model_id for arm in arms})
    season_models = (
        session.scalars(
            select(SeasonModel)
            .where(
                SeasonModel.season_id == season.id,
                SeasonModel.model_id.in_(model_ids),
                SeasonModel.created_at <= evidence_cutoff_at,
            )
            .order_by(SeasonModel.model_id, SeasonModel.id)
        ).all()
        if model_ids
        else []
    )
    votes = (
        session.scalars(
            select(Vote)
            .where(Vote.battle_id.in_(battle_ids))
            .where(Vote.cohort == cohort if cohort != "combined" else text("1 = 1"))
            .where(Vote.created_at <= evidence_cutoff_at)
            .order_by(Vote.battle_id, Vote.cohort, Vote.id)
        ).all()
        if battle_ids
        else []
    )
    vote_selection = {item.id: analysis_vote_eligibility(item, battle_selection) for item in votes}
    validators = (
        session.scalars(
            select(ValidatorResult)
            .where(
                ValidatorResult.arm_id.in_(arm_ids),
                ValidatorResult.created_at <= evidence_cutoff_at,
            )
            .order_by(ValidatorResult.arm_id, ValidatorResult.validator_name, ValidatorResult.id)
        ).all()
        if arm_ids
        else []
    )
    tools = (
        session.scalars(
            select(ToolCall)
            .where(
                ToolCall.arm_id.in_(arm_ids),
                ToolCall.created_at <= evidence_cutoff_at,
            )
            .order_by(
                ToolCall.arm_id,
                ToolCall.round_index,
                ToolCall.call_index,
                ToolCall.id,
            )
        ).all()
        if arm_ids
        else []
    )
    attempts = (
        session.scalars(
            select(GenerationAttempt)
            .where(
                GenerationAttempt.arm_id.in_(arm_ids),
                GenerationAttempt.created_at <= evidence_cutoff_at,
            )
            .order_by(GenerationAttempt.arm_id, GenerationAttempt.created_at, GenerationAttempt.id)
        ).all()
        if arm_ids
        else []
    )
    jobs = (
        session.scalars(
            select(Job)
            .where(
                Job.battle_id.in_(operational_battle_ids),
                Job.created_at <= evidence_cutoff_at,
            )
            .order_by(Job.battle_id, Job.created_at, Job.id)
        ).all()
        if operational_battle_ids
        else []
    )
    costs = (
        session.scalars(
            select(CostEvent)
            .where(
                CostEvent.battle_id.in_(operational_battle_ids),
                CostEvent.created_at <= evidence_cutoff_at,
            )
            .order_by(CostEvent.battle_id, CostEvent.arm_id, CostEvent.created_at, CostEvent.id)
        ).all()
        if battle_ids
        else []
    )
    billing_memberships = (
        session.scalars(
            select(BedrockBillingCrosscheckArm)
            .where(
                BedrockBillingCrosscheckArm.arm_id.in_(arm_ids),
                BedrockBillingCrosscheckArm.created_at <= evidence_cutoff_at,
            )
            .order_by(
                BedrockBillingCrosscheckArm.crosscheck_id,
                BedrockBillingCrosscheckArm.arm_id,
            )
        ).all()
        if arm_ids
        else []
    )
    candidate_crosscheck_ids = sorted({item.crosscheck_id for item in billing_memberships})
    # A provider invoice export may legitimately reconcile arms from several
    # controlled runs.  Such an account-level crosscheck must never be copied
    # wholesale into one customer's evidence document.  Include a crosscheck
    # only when its complete membership at the snapshot cutoff is contained in
    # this exact snapshot arm set.
    complete_candidate_memberships = (
        session.scalars(
            select(BedrockBillingCrosscheckArm)
            .where(
                BedrockBillingCrosscheckArm.crosscheck_id.in_(candidate_crosscheck_ids),
                BedrockBillingCrosscheckArm.created_at <= evidence_cutoff_at,
            )
            .order_by(
                BedrockBillingCrosscheckArm.crosscheck_id,
                BedrockBillingCrosscheckArm.arm_id,
            )
        ).all()
        if candidate_crosscheck_ids
        else []
    )
    membership_arm_ids: dict[str, set[str]] = {}
    for membership in complete_candidate_memberships:
        membership_arm_ids.setdefault(membership.crosscheck_id, set()).add(membership.arm_id)
    snapshot_arm_ids = set(arm_ids)
    billing_crosscheck_ids = sorted(
        crosscheck_id
        for crosscheck_id, member_ids in membership_arm_ids.items()
        if member_ids and member_ids <= snapshot_arm_ids
    )
    safe_crosscheck_ids = set(billing_crosscheck_ids)
    billing_memberships = [
        item for item in complete_candidate_memberships if item.crosscheck_id in safe_crosscheck_ids
    ]
    billing_crosschecks = (
        session.scalars(
            select(BedrockBillingCrosscheck)
            .where(
                BedrockBillingCrosscheck.id.in_(billing_crosscheck_ids),
                BedrockBillingCrosscheck.created_at <= evidence_cutoff_at,
            )
            .order_by(BedrockBillingCrosscheck.created_at, BedrockBillingCrosscheck.id)
        ).all()
        if billing_crosscheck_ids
        else []
    )
    billing_adjustments = (
        session.scalars(
            select(CostEvent)
            .where(
                CostEvent.season_id == season.id,
                CostEvent.kind == "bedrock_billing_adjustment",
                CostEvent.created_at <= evidence_cutoff_at,
            )
            .order_by(CostEvent.created_at, CostEvent.id)
        ).all()
        if billing_crosscheck_ids
        else []
    )
    relevant_crosschecks = set(billing_crosscheck_ids)
    costs = [
        *costs,
        *[
            item
            for item in billing_adjustments
            if item.accounting_json.get("crosscheck_id") in relevant_crosschecks
        ],
    ]
    assignments = (
        session.scalars(
            select(ControlledRunAssignment)
            .where(
                ControlledRunAssignment.controlled_run_id == controlled_run_id,
                ControlledRunAssignment.created_at <= evidence_cutoff_at,
            )
            .order_by(ControlledRunAssignment.ordinal)
        ).all()
        if controlled_run_id
        else []
    )
    postcollection_item_audit_events = (
        session.scalars(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "controlled_run",
                RunEvent.entity_id == controlled_run_id,
                RunEvent.event_type == "season1_post_collection_item_audit_verified",
                RunEvent.created_at <= evidence_cutoff_at,
            )
            .order_by(RunEvent.created_at, RunEvent.id)
        ).all()
        if controlled_run_id
        else []
    )
    arena_method_validation_events = (
        session.scalars(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "controlled_run",
                RunEvent.entity_id == controlled_run_id,
                RunEvent.event_type == "season1_arena_monte_carlo_validation_verified",
                RunEvent.created_at <= evidence_cutoff_at,
            )
            .order_by(RunEvent.created_at, RunEvent.id)
        ).all()
        if controlled_run_id
        else []
    )
    arm_output_digests: dict[str, tuple[str | None, str]] = {}
    for arm in arms:
        battle = battles_by_id[arm.battle_id]
        stored_answer_sha256 = arm.answer_markdown_sha256
        stored_output_sha256 = arm.output_json_sha256
        if arm.status == "complete" and not stored_answer_sha256:
            raise _SnapshotVerificationError(
                "response_arm_answer_digest_missing",
                "a completed response arm has no immutable normalized-answer digest",
                {"arm_id": arm.id},
            )
        if not stored_output_sha256:
            raise _SnapshotVerificationError(
                "response_arm_output_digest_missing",
                "a response arm has no immutable normalized-output digest",
                {"arm_id": arm.id},
            )
        if not battle.prompt_redacted:
            observed_answer_sha256 = (
                hashlib.sha256(arm.answer_markdown.encode()).hexdigest()
                if arm.answer_markdown is not None
                else None
            )
            observed_output_sha256 = _canonical_sha256(arm.output_json or {})
            if observed_answer_sha256 != stored_answer_sha256:
                raise _SnapshotVerificationError(
                    "response_arm_answer_digest_mismatch",
                    "normalized answer content differs from its write-once digest",
                    {"arm_id": arm.id},
                )
            if observed_output_sha256 != stored_output_sha256:
                raise _SnapshotVerificationError(
                    "response_arm_output_digest_mismatch",
                    "normalized output content differs from its write-once digest",
                    {"arm_id": arm.id},
                )
        arm_output_digests[arm.id] = (stored_answer_sha256, stored_output_sha256)
    validator_detail_digests: dict[str, str] = {}
    for result in validators:
        stored_detail_sha256 = result.detail_sha256
        if not stored_detail_sha256:
            raise _SnapshotVerificationError(
                "validator_detail_digest_missing",
                "a validator result has no immutable detail digest",
                {"validator_result_id": result.id},
            )
        observed_detail_sha256 = _canonical_sha256({"detail": result.detail_json})
        battle = battles_by_id[arms_by_id[result.arm_id].battle_id]
        detail_is_redacted = (
            battle.prompt_redacted and result.detail_json == TOOL_CALL_REDACTION_JSON
        )
        if not detail_is_redacted and observed_detail_sha256 != stored_detail_sha256:
            raise _SnapshotVerificationError(
                "validator_detail_digest_mismatch",
                "a validator result differs from its write-once detail digest",
                {"validator_result_id": result.id},
            )
        validator_detail_digests[result.id] = stored_detail_sha256
    tool_content_digests: dict[str, tuple[str, str, str]] = {}
    for call in tools:
        stored_arguments_sha256 = call.arguments_sha256
        stored_result_sha256 = call.result_sha256
        stored_structured_sha256 = call.structured_content_sha256
        if not stored_arguments_sha256 or not stored_result_sha256 or not stored_structured_sha256:
            raise _SnapshotVerificationError(
                "tool_call_content_digest_missing",
                "an Epicure tool-call trace has incomplete immutable content digests",
                {"tool_call_id": call.id},
            )
        observed_arguments_sha256 = _canonical_sha256({"arguments": call.arguments_json})
        observed_result_sha256 = hashlib.sha256(call.result_text.encode()).hexdigest()
        observed_structured_sha256 = _canonical_sha256({"structured": call.structured_content_json})
        battle = battles_by_id[arms_by_id[call.arm_id].battle_id]
        content_is_redacted = (
            battle.prompt_redacted
            and call.arguments_json == TOOL_CALL_REDACTION_JSON
            and call.result_text == TOOL_CALL_REDACTION_SENTINEL
            and call.structured_content_json == TOOL_CALL_REDACTION_JSON
        )
        if not content_is_redacted and (
            observed_arguments_sha256 != stored_arguments_sha256
            or observed_result_sha256 != stored_result_sha256
            or observed_structured_sha256 != stored_structured_sha256
        ):
            raise _SnapshotVerificationError(
                "tool_call_content_digest_mismatch",
                "an Epicure tool-call trace differs from its write-once content digests",
                {"tool_call_id": call.id},
            )
        tool_content_digests[call.id] = (
            stored_arguments_sha256,
            stored_result_sha256,
            stored_structured_sha256,
        )
    ranking_source = Path(__file__).with_name("service_ranking.py").read_bytes()
    season1_statistics_source = Path(__file__).with_name("season1_statistics.py").read_bytes()
    arena_acceptance_source = Path(__file__).with_name("season1_arena_acceptance.py").read_bytes()
    arena_acceptance_policy = DEFAULT_POLICY_PATH.read_bytes()
    eligible_judgment_ids = sorted(
        item.id for item in votes if vote_selection[item.id]["preference_role"] != "excluded"
    )
    preference_observation_ids = sorted(
        item.id for item in votes if vote_selection[item.id]["preference_role"] == "included"
    )
    return {
        "schema_version": "flavourbench-snapshot-evidence-v7",
        "evidence_scope": "exact_analysis_inputs_with_exclusion_log",
        "analysis_source_sha256": hashlib.sha256(ranking_source).hexdigest(),
        "season1_statistics_source_sha256": hashlib.sha256(season1_statistics_source).hexdigest(),
        "arena_acceptance_source_sha256": hashlib.sha256(arena_acceptance_source).hexdigest(),
        "arena_acceptance_policy_file_sha256": hashlib.sha256(arena_acceptance_policy).hexdigest(),
        "analysis_dependency": "arena-rank==0.1.1",
        "scope": {
            "season_id": season.id,
            "season_manifest_sha256": season.manifest_sha256,
            "analysis_plan_sha256": season.analysis_plan_sha256,
            "protocol_bundle_sha256": season.protocol_bundle_sha256,
            "track": track,
            "cohort": cohort,
            "category": category,
            "data_stratum": data_stratum,
            "controlled_run_id": controlled_run_id,
            "evidence_cutoff_at": _utc_iso(evidence_cutoff_at),
        },
        "analysis_observations": {
            "eligible_judgment_ids": eligible_judgment_ids,
            "preference_observation_ids": preference_observation_ids,
            "preference_observation_sha256": _canonical_sha256(
                {"vote_ids": preference_observation_ids}
            ),
        },
        "assignments": [
            {
                "id": item.id,
                "ordinal": item.ordinal,
                "assignment_sha256": item.assignment_sha256,
                "battle_id": item.battle_id,
                "status": item.status,
                "created_at": _utc_iso(item.created_at),
            }
            for item in assignments
        ],
        "postcollection_item_audit_events": [
            {
                "id": item.id,
                "artifact_sha256": item.payload_json.get("artifact_sha256"),
                "verification_status": item.payload_json.get("verification_status"),
                "verifier": item.payload_json.get("verifier"),
                "supersedes_event_id": item.payload_json.get("supersedes_event_id"),
                "payload_sha256": _canonical_sha256(item.payload_json),
                "created_at": _utc_iso(item.created_at),
            }
            for item in postcollection_item_audit_events
        ],
        "arena_method_validation_events": [
            {
                "id": item.id,
                "artifact_sha256": item.payload_json.get("artifact_sha256"),
                "verification_status": item.payload_json.get("verification_status"),
                "verifier": item.payload_json.get("verifier"),
                "payload_sha256": _canonical_sha256(item.payload_json),
                "created_at": _utc_iso(item.created_at),
            }
            for item in arena_method_validation_events
        ],
        "battles": [
            {
                "id": item.id,
                "status": item.status,
                "run_class": item.run_class,
                "rank_eligible": item.rank_eligible,
                "manifest_sha256": item.manifest_sha256,
                "protocol_bundle_sha256": item.protocol_bundle_sha256,
                "track": item.track,
                "category": item.category,
                "data_stratum": item.data_stratum,
                "controlled_run_id": item.controlled_run_id,
                "task_id": item.task_id,
                "task_revision": item.task_revision,
                "prompt_sha256": item.prompt_sha256,
                "left_arm_id": item.left_arm_id,
                "right_arm_id": item.right_arm_id,
                "scheduler_version": item.scheduler_version,
                "assignment_seed_sha256": hashlib.sha256(item.assignment_seed.encode()).hexdigest(),
                "created_at": _utc_iso(item.created_at),
                "completed_at": _utc_iso(item.completed_at),
                **battle_selection[item.id],
            }
            for item in battles
        ],
        "arms": [
            {
                "id": item.id,
                "battle_id": item.battle_id,
                "side": item.side,
                "condition": item.condition,
                "model_id": item.model_id,
                "execution_backend": item.execution_backend,
                "provider_slug": item.provider_slug,
                "actual_model_id": item.actual_model_id,
                "actual_provider_slug": item.actual_provider_slug,
                "generation_id": item.generation_id,
                "provider_generation_ids_sha256": _canonical_sha256(
                    {"generation_ids": item.provider_generation_ids_json}
                ),
                "status": item.status,
                "cost_micros": item.cost_micros,
                "cost_reconciled": item.cost_reconciled,
                "cost_accounting_basis": item.cost_accounting_basis,
                "billing_reconciliation_status": item.billing_reconciliation_status,
                "latency_ms": item.latency_ms,
                "prompt_sha256": item.prompt_sha256,
                "system_prompt_sha256": item.system_prompt_sha256,
                "schema_sha256": item.schema_sha256,
                "tool_schema_sha256": item.tool_schema_sha256,
                "decoding_sha256": _canonical_sha256({"decoding": item.decoding_json}),
                "observed_decoding_sha256": _canonical_sha256(
                    {"observed_decoding": item.observed_decoding_json}
                ),
                "protocol_bundle_sha256": item.protocol_bundle_sha256,
                "epicure_release_id": item.epicure_release_id,
                "epicure_bundle_sha256": item.epicure_bundle_sha256,
                "epicure_application_sha256": item.epicure_application_sha256,
                "epicure_attestation_payload_sha256": _canonical_sha256(
                    {"attestation": item.epicure_attestation_json}
                ),
                "epicure_attestation_sha256": item.epicure_attestation_sha256,
                "answer_markdown_sha256": arm_output_digests[item.id][0],
                "output_json_sha256": arm_output_digests[item.id][1],
                "created_at": _utc_iso(item.created_at),
                "completed_at": _utc_iso(item.completed_at),
                "operational_metric_included": True,
            }
            for item in arms
        ],
        "season_models": [
            {
                "id": item.id,
                "model_id": item.model_id,
                "slot_role": item.slot_role,
                "execution_backend": item.execution_backend,
                "provider_slug": item.provider_slug,
                "expected_actual_model_id": item.expected_actual_model_id,
                "expected_actual_provider_slug": item.expected_actual_provider_slug,
                "supported_parameters_sha256": _canonical_sha256(
                    {"supported_parameters": item.supported_parameters_json}
                ),
                "decoding_sha256": _canonical_sha256({"decoding": item.decoding_json}),
                "endpoint_max_completion_tokens": item.endpoint_max_completion_tokens,
                "endpoint_document_sha256": item.endpoint_document_sha256,
                "endpoint_contract_sha256": item.endpoint_contract_sha256,
                "backend_contract_sha256": item.backend_contract_sha256,
                "backend_contract_payload_sha256": _canonical_sha256(
                    {"backend_contract": item.backend_contract_json}
                ),
                "rate_card_sha256": item.rate_card_sha256,
                "rate_card_payload_sha256": _canonical_sha256({"rate_card": item.rate_card_json}),
                "worst_case_cost_micros": item.worst_case_cost_micros,
                "manifest_sha256": item.manifest_sha256,
                "eligible": item.eligible,
                "created_at": _utc_iso(item.created_at),
            }
            for item in season_models
        ],
        "votes": [
            {
                "id": item.id,
                "battle_id": item.battle_id,
                "cohort": item.cohort,
                "choice": item.choice,
                "reason_tags": item.reason_tags_json,
                "rubric_sha256": _canonical_sha256({"rubric": item.rubric_json}),
                **vote_selection[item.id],
                "created_at": _utc_iso(item.created_at),
            }
            for item in votes
        ],
        "validators": [
            {
                "id": item.id,
                "arm_id": item.arm_id,
                "name": item.validator_name,
                "version": item.validator_version,
                "status": item.status,
                "score_milli": item.score_milli,
                "detail_sha256": validator_detail_digests[item.id],
                "created_at": _utc_iso(item.created_at),
            }
            for item in validators
        ],
        "tool_calls": [
            {
                "id": item.id,
                "arm_id": item.arm_id,
                "round_index": item.round_index,
                "call_index": item.call_index,
                "tool_call_id": item.tool_call_id,
                "tool_name": item.tool_name,
                "arguments_sha256": tool_content_digests[item.id][0],
                "result_sha256": tool_content_digests[item.id][1],
                "structured_content_sha256": tool_content_digests[item.id][2],
                "latency_ms": item.latency_ms,
                "is_error": item.is_error,
                "created_at": _utc_iso(item.created_at),
            }
            for item in tools
        ],
        "generation_attempts": [
            {
                "id": item.id,
                "attempt_id": item.attempt_id,
                "arm_id": item.arm_id,
                "phase": item.phase,
                "attempt_index": item.attempt_index,
                "event_type": item.event_type,
                "generation_id": item.generation_id,
                "payload_sha256": item.payload_sha256,
                "metadata_sha256": _canonical_sha256({"metadata": item.metadata_json}),
                "created_at": _utc_iso(item.created_at),
            }
            for item in attempts
        ],
        "jobs": [
            {
                "id": item.id,
                "battle_id": item.battle_id,
                "kind": item.kind,
                "payload_sha256": _canonical_sha256({"payload": item.payload_json}),
                "status": item.status,
                "attempts": item.attempts,
                "max_attempts": item.max_attempts,
                "created_at": _utc_iso(item.created_at),
                "completed_at": (
                    _utc_iso(item.completed_at) if item.completed_at is not None else None
                ),
            }
            for item in jobs
        ],
        "bedrock_billing_crosschecks": [
            {
                "id": item.id,
                "status": item.status,
                "supersedes_crosscheck_id": item.supersedes_crosscheck_id,
                "source_artifact_sha256": item.source_artifact_sha256,
                "statement_sha256": item.statement_sha256,
                "arm_set_sha256": item.arm_set_sha256,
                "generation_request_map_sha256": (item.generation_request_map_sha256),
                "rate_card_estimated_micros": item.rate_card_estimated_micros,
                "billed_usage_micros": item.billed_usage_micros,
                "billing_difference_micros": item.billing_difference_micros,
                "ledger_delta_micros": item.ledger_delta_micros,
                "evidence_sha256": item.evidence_sha256,
                "created_at": _utc_iso(item.created_at),
            }
            for item in billing_crosschecks
        ],
        "bedrock_billing_memberships": [
            {
                "crosscheck_id": item.crosscheck_id,
                "arm_id": item.arm_id,
                "generation_set_sha256": item.generation_set_sha256,
                "created_at": _utc_iso(item.created_at),
            }
            for item in billing_memberships
        ],
        "cost_events": [
            {
                "id": item.id,
                "battle_id": item.battle_id,
                "arm_id": item.arm_id,
                "kind": item.kind,
                "amount_micros": item.amount_micros,
                "provider": item.provider,
                "generation_id": item.generation_id,
                "accounting_sha256": _canonical_sha256({"accounting": item.accounting_json}),
                "created_at": _utc_iso(item.created_at),
            }
            for item in costs
        ],
    }


def _run_model_roster(
    session: Session,
    season: Season,
    model_ids: list[str],
    submitted_endpoint_model_id: str,
) -> list[dict[str, Any]]:
    rows = session.execute(
        select(CatalogModel, SeasonModel)
        .join(SeasonModel, SeasonModel.model_id == CatalogModel.model_id)
        .where(
            SeasonModel.season_id == season.id,
            CatalogModel.model_id.in_(model_ids),
        )
    ).all()
    by_id = {catalog.model_id: (catalog, slot) for catalog, slot in rows}
    missing = sorted(set(model_ids) - set(by_id))
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"controlled-run roster contains unavailable models: {', '.join(missing)}",
        )
    roster: list[dict[str, Any]] = []
    for model_id in sorted(model_ids):
        _, slot = by_id[model_id]
        expected_contract = endpoint_contract_sha256(
            model_id=slot.model_id,
            provider_slug=slot.provider_slug,
            expected_actual_model_id=slot.expected_actual_model_id,
            expected_actual_provider_slug=slot.expected_actual_provider_slug,
            supported_parameters=slot.supported_parameters_json,
            decoding=slot.decoding_json,
            endpoint_max_completion_tokens=slot.endpoint_max_completion_tokens,
            endpoint_document_sha256=slot.endpoint_document_sha256,
        )
        if (
            not slot.eligible
            or slot.endpoint_contract_sha256 in UNFROZEN_VALUES
            or expected_contract != slot.endpoint_contract_sha256
            or slot.backend_contract_sha256 in UNFROZEN_VALUES
            or _canonical_sha256(slot.backend_contract_json) != slot.backend_contract_sha256
            or slot.rate_card_sha256 in UNFROZEN_VALUES
            or _canonical_sha256(slot.rate_card_json) != slot.rate_card_sha256
            or (get_settings().execution_mode != "mock" and slot.worst_case_cost_micros <= 0)
            or (
                get_settings().execution_mode != "mock"
                and slot.manifest_sha256 != season.manifest_sha256
            )
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{model_id} lacks an intact eligible endpoint contract",
            )
        roster.append(
            {
                "model_id": model_id,
                "execution_backend": slot.execution_backend,
                "provider_slug": slot.provider_slug,
                "expected_actual_model_id": slot.expected_actual_model_id,
                "expected_actual_provider_slug": slot.expected_actual_provider_slug,
                "endpoint_contract_sha256": slot.endpoint_contract_sha256,
                "endpoint_document_sha256": slot.endpoint_document_sha256,
                "backend_contract_sha256": slot.backend_contract_sha256,
                "rate_card_sha256": slot.rate_card_sha256,
                "worst_case_cost_micros": slot.worst_case_cost_micros,
                "manifest_sha256": slot.manifest_sha256,
                "submitted_endpoint": model_id == submitted_endpoint_model_id,
                "routing_policy": {
                    "allow_fallbacks": False,
                    "require_parameters": True,
                    "data_collection": "deny",
                },
            }
        )
    return roster


def _verify_commercial_run_binding(session: Session, run: ControlledRun) -> None:
    binding = (
        run.organization_id,
        run.evaluation_order_id,
        run.route_revision_id,
        run.endpoint_descriptor_sha256,
        run.spend_authorization_id,
        run.spend_authorization_binding_sha256,
    )
    if all(value is None for value in binding):
        return
    if any(value is None for value in binding):
        raise HTTPException(status_code=503, detail="commercial run binding is incomplete")
    order = session.get(EvaluationOrder, run.evaluation_order_id)
    route = session.get(ModelRouteRevision, run.route_revision_id)
    submission = session.get(ModelSubmission, order.model_submission_id) if order else None
    season = session.get(Season, run.season_id)
    if order is None or route is None or submission is None or season is None:
        raise HTTPException(status_code=503, detail="commercial run binding is unavailable")
    spend_authorization = active_spend_authorization(
        session,
        order=order,
        acceptance_id=str(run.spend_authorization_id),
        binding_sha256=str(run.spend_authorization_binding_sha256),
    )
    plan = order.comparison_plan_json
    order_card = order.order_card_json
    if not isinstance(plan, dict) or not isinstance(order_card, dict):
        raise HTTPException(status_code=503, detail="commercial order contract is unavailable")
    signing = order_card.get("signing")
    signing_key_id = (
        str(signing.get("key_id"))
        if isinstance(signing, dict) and signing.get("key_id")
        else get_settings().run_card_signing_key_id
    )
    verification_secret = run_card_verification_keyring().get(signing_key_id)
    expected_order_signature = (
        hmac.new(
            verification_secret.encode(),
            f"flavourbench-evaluation-order-card-v2:{order.order_card_sha256}".encode(),
            hashlib.sha256,
        ).hexdigest()
        if verification_secret is not None
        else None
    )
    expected_card_binding = {
        "evaluation_order_id": order.id,
        "organization_id": order.organization_id,
        "model_submission_id": order.model_submission_id,
        "route_revision_id": order.route_revision_id,
        "season_id": order.season_id,
        "submitted_endpoint_model_id": submission.catalog_model_id,
    }
    if (
        order.status
        not in {
            "provisioning",
            "ready",
            "running",
            "collection_complete",
            "analysis_complete",
            "delivered",
        }
        or order.organization_id != run.organization_id
        or order.route_revision_id != route.id
        or order.season_id != run.season_id
        or order.rater_plan_sha256 != run.rater_plan_sha256
        or order.analysis_plan_sha256 != run.analysis_plan_sha256
        or order.budget_cap_micros != run.budget_cap_micros
        or route.model_submission_id != submission.id
        or route.status != "approved"
        or submission.status != "approved"
        or route.approved_season_id != season.id
        or route.approved_season_manifest_sha256 != season.manifest_sha256
        or route.descriptor_sha256 != run.endpoint_descriptor_sha256
        or submission.catalog_model_id != run.submitted_endpoint_model_id
        or submission.model_card_sha256 != run.submitted_model_card_sha256
        or route.data_policy_sha256 != run.data_policy_sha256
        or order.comparison_plan_sha256 != _canonical_sha256(plan)
        or order.order_card_sha256 != _canonical_sha256(order_card)
        or expected_order_signature is None
        or not hmac.compare_digest(expected_order_signature, order.order_card_signature)
        or any(order_card.get(key) != value for key, value in expected_card_binding.items())
        or plan.get("submitted_endpoint_model_id") != run.submitted_endpoint_model_id
        or plan.get("model_ids") != [str(item.get("model_id")) for item in run.model_roster_json]
        or spend_authorization is None
    ):
        raise HTTPException(status_code=503, detail="commercial run contract drifted")


def _verify_controlled_run_contract(session: Session, run: ControlledRun) -> Season:
    """Verify the signed card and every frozen dependency before serving a run."""

    season = session.get(Season, run.season_id)
    card = run.run_card_json
    if season is None or not isinstance(card, dict):
        raise HTTPException(status_code=503, detail="controlled-run contract is unavailable")
    digest = _canonical_sha256(card)
    signing = card.get("signing")
    signing_key_id = (
        str(signing.get("key_id"))
        if isinstance(signing, dict) and signing.get("key_id")
        else get_settings().run_card_signing_key_id
    )
    verification_secret = run_card_verification_keyring().get(signing_key_id)
    if verification_secret is None:
        raise HTTPException(
            status_code=503,
            detail="controlled-run signing key is unavailable",
        )
    expected_signature = hmac.new(
        verification_secret.encode(),
        digest.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not (
        hmac.compare_digest(digest, run.run_card_sha256)
        and hmac.compare_digest(expected_signature, run.run_card_signature)
    ):
        raise HTTPException(status_code=503, detail="controlled-run signature verification failed")
    expected_season = {
        "slug": season.slug,
        "manifest_sha256": season.manifest_sha256,
        "task_registry_sha256": season.prompt_registry_sha256,
        "tool_registry_sha256": season.tool_registry_sha256,
        "epicure_release_id": season.epicure_release_id,
        "epicure_bundle_sha256": season.epicure_bundle_sha256,
        "epicure_application_sha256": season.epicure_application_sha256,
        "analysis_plan_sha256": season.analysis_plan_sha256,
        "protocol_bundle_sha256": season.protocol_bundle_sha256,
    }
    if card.get("season") != expected_season:
        raise HTTPException(status_code=503, detail="controlled-run season contract drifted")
    schema_version = card.get("schema_version")
    expected_commercial_binding: dict[str, str | None] = {
        "organization_id": run.organization_id,
        "evaluation_order_id": run.evaluation_order_id,
        "route_revision_id": run.route_revision_id,
        "endpoint_descriptor_sha256": run.endpoint_descriptor_sha256,
    }
    if schema_version == CONTROLLED_RUN_CARD_SCHEMA_VERSION:
        expected_commercial_binding.update(
            {
                "spend_authorization_id": run.spend_authorization_id,
                "spend_authorization_binding_sha256": (run.spend_authorization_binding_sha256),
            }
        )
    if (
        card.get("schema_version") not in SUPPORTED_CONTROLLED_RUN_CARD_SCHEMA_VERSIONS
        or card.get("run_id") != run.id
        or card.get("data_stratum") != "controlled"
        or card.get("organization_reference_sha256") != run.organization_reference_sha256
        or card.get("cost_accounting_policy") != CONTROLLED_RUN_COST_ACCOUNTING_POLICY
        or not isinstance(run.model_roster_json, list)
        or not run.model_roster_json
        or any(not isinstance(item, dict) for item in run.model_roster_json)
    ):
        raise HTTPException(status_code=503, detail="controlled-run card identity failed")
    if (
        schema_version == CONTROLLED_RUN_CARD_SCHEMA_VERSION
        and card.get("commercial_binding") != expected_commercial_binding
    ) or (
        schema_version != CONTROLLED_RUN_CARD_SCHEMA_VERSION
        and any(value is not None for value in expected_commercial_binding.values())
    ):
        raise HTTPException(status_code=503, detail="controlled-run commercial binding failed")
    _verify_commercial_run_binding(session, run)
    roster = _run_model_roster(
        session,
        season,
        [str(item.get("model_id")) for item in run.model_roster_json],
        run.submitted_endpoint_model_id,
    )
    roster_sha256 = _canonical_sha256({"models": roster})
    assignments = session.scalars(
        select(ControlledRunAssignment)
        .where(ControlledRunAssignment.controlled_run_id == run.id)
        .order_by(ControlledRunAssignment.ordinal)
    ).all()
    task_rows = session.scalars(
        select(Task).where(Task.id.in_([assignment.task_id for assignment in assignments]))
    ).all()
    tasks = {task.id: task for task in task_rows}
    schedule_payloads: list[dict[str, Any]] = []
    for assignment in assignments:
        task = tasks.get(assignment.task_id)
        if task is None or (
            task.public_id != assignment.task_public_id
            or task.revision != assignment.task_revision
            or task.prompt_sha256 != assignment.task_prompt_sha256
            or task.family != assignment.task_family
        ):
            raise HTTPException(status_code=503, detail="controlled-run task contract drifted")
        payload = _assignment_payload(
            ordinal=assignment.ordinal,
            task_public_id=assignment.task_public_id,
            task_revision=assignment.task_revision,
            task_prompt_sha256=assignment.task_prompt_sha256,
            task_family=assignment.task_family,
            track=assignment.track,
            model_ids=assignment.model_ids_json,
            repetition_index=assignment.repetition_index,
            side_seed_commitment_sha256=hashlib.sha256(
                assignment.assignment_seed.encode()
            ).hexdigest(),
        )
        if not hmac.compare_digest(_canonical_sha256(payload), assignment.assignment_sha256):
            raise HTTPException(
                status_code=503, detail="controlled-run assignment integrity failed"
            )
        schedule_payloads.append(payload)
    schedule_sha256 = _schedule_sha256(schedule_payloads)
    if not (
        roster == run.model_roster_json
        and hmac.compare_digest(roster_sha256, run.model_roster_sha256)
        and hmac.compare_digest(schedule_sha256, run.task_schedule_sha256)
        and card.get("model_roster") == roster
        and card.get("model_roster_sha256") == roster_sha256
        and card.get("task_schedule") == schedule_payloads
        and card.get("task_schedule_sha256") == schedule_sha256
        and card.get("submitted_endpoint_model_id") == run.submitted_endpoint_model_id
        and card.get("submitted_model_card_sha256") == run.submitted_model_card_sha256
        and card.get("data_policy_sha256") == run.data_policy_sha256
        and card.get("protocol_version") == run.protocol_version
        and card.get("rater_plan_sha256") == run.rater_plan_sha256
        and card.get("analysis_plan_sha256") == run.analysis_plan_sha256
        and card.get("budget_cap_micros") == run.budget_cap_micros
    ):
        raise HTTPException(status_code=503, detail="controlled-run contract integrity failed")
    return season


def _prepare_snapshot_transaction(session: Session) -> None:
    """Use one PostgreSQL evidence view for every multi-query snapshot operation."""

    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        session.connection(execution_options={"isolation_level": "REPEATABLE READ"})


def _verified_snapshot_payload(
    snapshot: LeaderboardSnapshot, session: Session | None = None
) -> dict[str, Any]:
    """Verify both stored content addresses without mutating the caller transaction."""

    del session  # Retained for compatibility with internal callers predating v3.
    payload = dict(snapshot.payload_json)
    payload_digest = snapshot_hash(payload)
    evidence_digest = (
        _canonical_sha256(snapshot.input_evidence_json)
        if isinstance(snapshot.input_evidence_json, dict)
        else None
    )
    payload_valid = bool(snapshot.payload_sha256) and hmac.compare_digest(
        payload_digest,
        snapshot.payload_sha256 or "",
    )
    input_valid = bool(snapshot.input_sha256) and hmac.compare_digest(
        payload_digest,
        snapshot.input_sha256,
    )
    evidence_valid = bool(
        evidence_digest and snapshot.input_evidence_sha256
    ) and hmac.compare_digest(
        evidence_digest or "",
        snapshot.input_evidence_sha256 or "",
    )
    if not (payload_valid and input_valid and evidence_valid):
        raise _SnapshotVerificationError(
            "stored_snapshot_content_address_mismatch",
            "stored leaderboard payload or evidence manifest failed content-address verification",
            {
                "stored_input_sha256": snapshot.input_sha256,
                "stored_payload_sha256": snapshot.payload_sha256,
                "computed_payload_sha256": payload_digest,
                "stored_evidence_sha256": snapshot.input_evidence_sha256,
                "computed_evidence_sha256": evidence_digest,
            },
        )
    return payload


def _verify_snapshot_observation_alignment(
    payload: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    """Require the fitted vote IDs to equal the sealed evidence IDs byte-for-byte."""

    payload_ids = payload.get("preference_observation_ids")
    evidence_observations = evidence.get("analysis_observations")
    evidence_ids = (
        evidence_observations.get("preference_observation_ids")
        if isinstance(evidence_observations, dict)
        else None
    )
    if not (
        isinstance(payload_ids, list)
        and isinstance(evidence_ids, list)
        and payload_ids == sorted(set(payload_ids))
        and evidence_ids == sorted(set(evidence_ids))
        and payload_ids == evidence_ids
    ):
        raise _SnapshotVerificationError(
            "snapshot_analysis_observation_mismatch",
            "leaderboard payload and evidence manifest name different preference observations",
            {
                "payload_observation_sha256": _canonical_sha256({"vote_ids": payload_ids}),
                "evidence_observation_sha256": _canonical_sha256({"vote_ids": evidence_ids}),
            },
        )
    expected = _canonical_sha256({"vote_ids": payload_ids})
    if not (
        payload.get("preference_observation_sha256") == expected
        and evidence_observations.get("preference_observation_sha256") == expected
    ):
        raise _SnapshotVerificationError(
            "snapshot_analysis_observation_digest_mismatch",
            "leaderboard observation IDs failed their content-address binding",
            {"expected_sha256": expected},
        )


def _current_snapshot_evidence(
    session: Session,
    *,
    season: Season,
    snapshot: LeaderboardSnapshot,
) -> dict[str, Any]:
    """Rebuild the exact evidence head without repeating a costly bootstrap fit."""

    if snapshot.season_id != season.id:
        raise _SnapshotVerificationError(
            "snapshot_season_scope_mismatch",
            "leaderboard snapshot is attached to a different season",
            {
                "snapshot_season_id": snapshot.season_id,
                "current_season_id": season.id,
            },
        )
    if snapshot.track not in {"model_arena", "epicure_uplift"}:
        raise _SnapshotVerificationError(
            "snapshot_track_scope_invalid",
            "leaderboard snapshot has an unsupported track",
            {"track": snapshot.track},
        )
    if snapshot.data_stratum == "public_freeform":
        if snapshot.controlled_run_id is not None:
            raise _SnapshotVerificationError(
                "snapshot_public_scope_invalid",
                "public leaderboard snapshot names a controlled run",
                {"controlled_run_id": snapshot.controlled_run_id},
            )
    elif snapshot.data_stratum == "controlled":
        if snapshot.controlled_run_id is None:
            raise _SnapshotVerificationError(
                "snapshot_controlled_scope_invalid",
                "controlled leaderboard snapshot lacks a controlled run",
                {"controlled_run_id": None},
            )
    else:
        raise _SnapshotVerificationError(
            "snapshot_data_stratum_invalid",
            "leaderboard snapshot has an unsupported data stratum",
            {"data_stratum": snapshot.data_stratum},
        )
    if snapshot.evidence_cutoff_at is None:
        raise _SnapshotVerificationError(
            "snapshot_evidence_cutoff_missing",
            "leaderboard snapshot has no immutable observation cutoff",
            {"snapshot_id": snapshot.id},
        )

    evidence = _snapshot_evidence_manifest(
        session,
        season=season,
        track=snapshot.track,
        cohort=snapshot.cohort,
        category=snapshot.category,
        data_stratum=snapshot.data_stratum,
        controlled_run_id=snapshot.controlled_run_id,
        evidence_cutoff_at=snapshot.evidence_cutoff_at,
    )
    return evidence


def _verified_current_snapshot_payload(
    session: Session,
    *,
    season: Season,
    snapshot: LeaderboardSnapshot,
) -> dict[str, Any]:
    """Require a snapshot to match both its stored bytes and the current evidence head."""

    payload = _verified_snapshot_payload(snapshot)
    current_evidence = _current_snapshot_evidence(
        session,
        season=season,
        snapshot=snapshot,
    )
    _verify_snapshot_observation_alignment(payload, current_evidence)
    current_evidence_sha256 = _canonical_sha256(current_evidence)
    if not (
        bool(snapshot.input_evidence_sha256)
        and hmac.compare_digest(
            current_evidence_sha256,
            snapshot.input_evidence_sha256 or "",
        )
    ):
        raise _SnapshotVerificationError(
            "leaderboard_snapshot_evidence_stale",
            "leaderboard snapshot no longer matches the current evidence head",
            {
                "stored_payload_sha256": snapshot.payload_sha256,
                "current_payload_sha256": snapshot_hash(payload),
                "stored_evidence_sha256": snapshot.input_evidence_sha256,
                "current_evidence_sha256": current_evidence_sha256,
            },
        )
    return payload


def _require_season1_statistical_acceptance(
    season: Season,
    snapshot: LeaderboardSnapshot,
    payload: dict[str, Any],
) -> None:
    """Block Season 1 publication until the prospective inference contract passes."""

    if season.slug != "season-1":
        return
    acceptance = payload.get("statistical_acceptance")
    rows = payload.get("rows")
    if snapshot.cohort not in {"public", "expert_independent"}:
        raise HTTPException(
            status_code=409,
            detail="Season 1 primary snapshots require a public or independent-expert cohort",
        )
    if snapshot.category != "all":
        raise HTTPException(
            status_code=409,
            detail="Season 1 primary publication is restricted to the prespecified all-family view",
        )
    if not (
        isinstance(acceptance, dict)
        and acceptance.get("status") == "pass"
        and payload.get("ranking_status") == "estimated"
        and payload.get("bootstrap_replicates") == 5_000
        and isinstance(rows, list)
        and rows
        and all(isinstance(row, dict) and row.get("provisional") is False for row in rows)
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Season 1 publication requires an accepted 5,000-replicate analysis "
                "with no provisional model rows"
            ),
        )
    if getattr(snapshot, "track", "model_arena") == "model_arena":
        try:
            arena_deficits = publication_acceptance_deficits(payload, view="all")
        except ArenaInferenceAcceptanceError as exc:
            raise HTTPException(
                status_code=409,
                detail="Season 1 arena acceptance policy could not be verified",
            ) from exc
        if arena_deficits:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Season 1 model-arena publication failed the frozen inference gate: "
                    + ", ".join(arena_deficits)
                ),
            )
        method_validation = payload.get("production_layout_method_validation")
        if not isinstance(method_validation, Mapping) or not verify_production_result(
            method_validation
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Season 1 model-arena publication requires a complete passing "
                    "production-layout Monte Carlo validation"
                ),
            )
    if snapshot.cohort == "expert_independent":
        coverage = payload.get("rater_coverage")
        if not (
            isinstance(coverage, dict)
            and int(coverage.get("unique_comparisons", 0)) >= 800
            and int(coverage.get("minimum_distinct_raters_per_comparison", 0)) >= 2
            and int(coverage.get("comparisons_with_two_or_more_distinct_raters", 0)) >= 800
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Season 1 expert publication requires 800 unique comparisons "
                    "with at least two distinct admitted raters each"
                ),
            )


def _withdraw_snapshot_after_verification_failure(
    session: Session,
    *,
    snapshot: LeaderboardSnapshot,
    failure: _SnapshotVerificationError,
) -> None:
    """Persist one fail-closed withdrawal record without committing unrelated work."""

    prior_event = session.scalar(
        select(RunEvent.id).where(
            RunEvent.entity_type == "leaderboard_snapshot",
            RunEvent.entity_id == snapshot.id,
            RunEvent.event_type == "leaderboard_snapshot_integrity_withdrawn",
        )
    )
    snapshot.publication_status = "withdrawn"
    if prior_event is not None:
        return
    session.add(
        Incident(
            severity="critical",
            code=failure.code,
            detail=(f"Leaderboard snapshot was withdrawn after verification failed: {snapshot.id}"),
        )
    )
    session.add(
        RunEvent(
            entity_type="leaderboard_snapshot",
            entity_id=snapshot.id,
            event_type="leaderboard_snapshot_integrity_withdrawn",
            payload_json={
                "reason_code": failure.code,
                "diagnostics": failure.diagnostics,
            },
        )
    )


def _withdraw_published_snapshots(
    session: Session,
    *,
    season_id: str,
    reason_code: str,
    controlled_run_id: str | None = None,
    replacement_snapshot_id: str | None = None,
) -> list[str]:
    """Withdraw published heads transactionally after evidence or authority changes."""

    statement = select(LeaderboardSnapshot).where(
        LeaderboardSnapshot.season_id == season_id,
        LeaderboardSnapshot.publication_status == "published",
    )
    if controlled_run_id is not None:
        statement = statement.where(LeaderboardSnapshot.controlled_run_id == controlled_run_id)
    snapshots = session.scalars(statement.order_by(LeaderboardSnapshot.id).with_for_update()).all()
    for item in snapshots:
        item.publication_status = "withdrawn"
        session.add(
            RunEvent(
                entity_type="leaderboard_snapshot",
                entity_id=item.id,
                event_type="leaderboard_snapshot_automatically_withdrawn",
                payload_json={
                    "reason_code": reason_code,
                    "controlled_run_id": item.controlled_run_id,
                    "replacement_snapshot_id": replacement_snapshot_id,
                },
            )
        )
    return [item.id for item in snapshots]


def _withdraw_published_scope_predecessors(
    session: Session,
    *,
    snapshot: LeaderboardSnapshot,
) -> list[str]:
    """Retire the prior published head before advancing one exact scope."""

    statement = select(LeaderboardSnapshot).where(
        LeaderboardSnapshot.id != snapshot.id,
        LeaderboardSnapshot.season_id == snapshot.season_id,
        LeaderboardSnapshot.track == snapshot.track,
        LeaderboardSnapshot.cohort == snapshot.cohort,
        LeaderboardSnapshot.category == snapshot.category,
        LeaderboardSnapshot.data_stratum == snapshot.data_stratum,
        LeaderboardSnapshot.publication_status == "published",
    )
    if snapshot.controlled_run_id is None:
        statement = statement.where(LeaderboardSnapshot.controlled_run_id.is_(None))
    else:
        statement = statement.where(
            LeaderboardSnapshot.controlled_run_id == snapshot.controlled_run_id
        )
    predecessors = session.scalars(
        statement.order_by(LeaderboardSnapshot.id).with_for_update()
    ).all()
    current_head_id = predecessors[0].id if len(predecessors) == 1 else None
    if len(predecessors) > 1 or current_head_id != snapshot.supersedes_snapshot_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "published leaderboard head changed after this draft was sealed; "
                "create a new draft against the current head"
            ),
        )
    for item in predecessors:
        item.publication_status = "withdrawn"
        session.add(
            RunEvent(
                entity_type="leaderboard_snapshot",
                entity_id=item.id,
                event_type="leaderboard_snapshot_superseded",
                payload_json={"replacement_snapshot_id": snapshot.id},
            )
        )
    if predecessors:
        session.flush()
    return [item.id for item in predecessors]


def _controlled_run_identity(
    session: Session, run_id: str, authorization: str
) -> tuple[ControlledRun, str]:
    supplied = authorization.removeprefix("Bearer ").strip()
    if not supplied:
        raise HTTPException(status_code=401, detail="controlled-run credential is required")
    token_sha256 = hashlib.sha256(supplied.encode()).hexdigest()
    run = session.scalar(
        select(ControlledRun).where(
            ControlledRun.id == run_id,
            ControlledRun.access_token_sha256 == token_sha256,
        )
    )
    if run is None or run.status == "revoked":
        raise HTTPException(status_code=401, detail="controlled-run credential is invalid")
    _verify_controlled_run_contract(session, run)
    return run, token_sha256


def _locked_controlled_run(
    session: Session,
    run_id: str,
    *,
    read_only: bool = False,
) -> tuple[Season, ControlledRun]:
    """Acquire commercial lifecycle locks in the global Season → Run order."""

    probe = session.get(ControlledRun, run_id)
    if probe is None:
        raise HTTPException(status_code=404, detail="controlled run not found")
    season = session.scalar(
        select(Season).where(Season.id == probe.season_id).with_for_update(read=read_only)
    )
    run = session.scalar(
        select(ControlledRun).where(ControlledRun.id == run_id).with_for_update(read=read_only)
    )
    if season is None or run is None:
        raise HTTPException(status_code=404, detail="controlled run not found")
    _verify_controlled_run_contract(session, run)
    return season, run


def _active_controlled_release_authorization(
    session: Session,
    run: ControlledRun,
    *,
    lock: bool = False,
) -> bool:
    if (
        not run.release_authorized
        or run.release_authorization_reference_sha256 is None
        or run.release_authorized_at is None
    ):
        return False
    if run.evaluation_order_id is None:
        return True
    if (
        run.publication_authorization_id is None
        or run.publication_authorization_binding_sha256 is None
    ):
        return False
    statement = select(EvaluationOrder).where(EvaluationOrder.id == run.evaluation_order_id)
    if lock:
        statement = statement.with_for_update()
    order = session.scalar(statement)
    if order is None or order.publication_status not in {"authorized", "published"}:
        return False
    return (
        active_publication_authorization(
            session,
            order=order,
            run=run,
            acceptance_id=run.publication_authorization_id,
            binding_sha256=run.publication_authorization_binding_sha256,
            lock=lock,
        )
        is not None
    )


def _signed_run_card(
    *,
    run_id: str,
    season: Season,
    organization_reference_sha256: str,
    protocol_version: str,
    rater_plan_sha256: str,
    analysis_plan_sha256: str,
    submitted_endpoint_model_id: str,
    submitted_model_card_sha256: str,
    data_policy_sha256: str,
    model_roster: list[dict[str, Any]],
    model_roster_sha256: str,
    task_schedule: list[dict[str, Any]],
    task_schedule_sha256: str,
    forecast_worst_case_cost_micros: int,
    budget_cap_micros: int,
    issued_at: datetime,
    organization_id: str | None = None,
    evaluation_order_id: str | None = None,
    route_revision_id: str | None = None,
    endpoint_descriptor_sha256: str | None = None,
    spend_authorization_id: str | None = None,
    spend_authorization_binding_sha256: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    settings = get_settings()
    card = {
        "schema_version": CONTROLLED_RUN_CARD_SCHEMA_VERSION,
        "signing": {
            "algorithm": "HMAC-SHA256",
            "key_id": settings.run_card_signing_key_id,
            "verification_scope": "FlavourBench service-held key",
        },
        "run_id": run_id,
        "issued_at": issued_at.isoformat(),
        "data_stratum": "controlled",
        "organization_reference_sha256": organization_reference_sha256,
        "commercial_binding": {
            "organization_id": organization_id,
            "evaluation_order_id": evaluation_order_id,
            "route_revision_id": route_revision_id,
            "endpoint_descriptor_sha256": endpoint_descriptor_sha256,
            "spend_authorization_id": spend_authorization_id,
            "spend_authorization_binding_sha256": (spend_authorization_binding_sha256),
        },
        "protocol_version": protocol_version,
        "rater_plan_sha256": rater_plan_sha256,
        "analysis_plan_sha256": analysis_plan_sha256,
        "submitted_endpoint_model_id": submitted_endpoint_model_id,
        "submitted_model_card_sha256": submitted_model_card_sha256,
        "data_policy_sha256": data_policy_sha256,
        "model_roster": model_roster,
        "model_roster_sha256": model_roster_sha256,
        "task_schedule": task_schedule,
        "task_schedule_sha256": task_schedule_sha256,
        "forecast_worst_case_cost_micros": forecast_worst_case_cost_micros,
        "assignment_policy": {
            "scheduler_version": "controlled-frozen-schedule-v1",
            "track_probability": "1/1",
            "model_probability": "1/1",
            "side_probability": "1/2",
            "client_selectable": False,
        },
        "budget_cap_micros": budget_cap_micros,
        "budget_governor": {
            "admission_stop_basis_points": 8500,
            "drain_basis_points": 9500,
            "hard_stop_basis_points": 10000,
        },
        "cost_accounting_policy": CONTROLLED_RUN_COST_ACCOUNTING_POLICY,
        "publication_default": "private",
        "season": {
            "slug": season.slug,
            "manifest_sha256": season.manifest_sha256,
            "task_registry_sha256": season.prompt_registry_sha256,
            "tool_registry_sha256": season.tool_registry_sha256,
            "epicure_release_id": season.epicure_release_id,
            "epicure_bundle_sha256": season.epicure_bundle_sha256,
            "epicure_application_sha256": season.epicure_application_sha256,
            "analysis_plan_sha256": season.analysis_plan_sha256,
            "protocol_bundle_sha256": season.protocol_bundle_sha256,
        },
    }
    digest = _canonical_sha256(card)
    signature = hmac.new(
        settings.run_card_signing_secret.encode(),
        digest.encode(),
        hashlib.sha256,
    ).hexdigest()
    return card, digest, signature


def _battle_for_rater(
    session: Session, battle_id: str, pseudonym: str, *, expert: bool = False
) -> Battle:
    battle = session.get(Battle, battle_id)
    if battle is None:
        raise HTTPException(status_code=404, detail="battle not found")
    if not expert and battle.requester_pseudonym != pseudonym:
        raise HTTPException(status_code=404, detail="battle not found")
    return battle


def _reveal(session: Session, battle: Battle) -> dict[str, Any]:
    arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
    return {
        "track": battle.track,
        "arms": [
            {
                "side": arm.side,
                "modelId": arm.model_id,
                "actualModelId": arm.actual_model_id,
                "provider": arm.actual_provider_slug,
                "condition": arm.condition,
                "costMicros": arm.cost_micros,
                "latencyMs": arm.latency_ms,
            }
            for arm in sorted(arms, key=lambda item: item.side)
        ],
    }


def _public_battle(
    session: Session, battle: Battle, pseudonym: str, cohort: str = "public"
) -> dict:
    answers: list[dict] = []
    arms: list[ResponseArm] = []
    if battle.status == "complete":
        arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
        if len(arms) == 2 and all(
            arm.status == "complete" and is_complete_finish_reason(arm.finish_reason)
            for arm in arms
        ):
            answers = [
                {"side": arm.side, "answerMarkdown": arm.answer_markdown or ""}
                for arm in sorted(arms, key=lambda item: item.side)
            ]
    response_failed = battle.status == "failed" or (battle.status == "complete" and not answers)
    vote = session.scalar(
        select(Vote).where(
            Vote.battle_id == battle.id,
            Vote.rater_pseudonym == pseudonym,
            Vote.cohort == cohort,
        )
    )
    return {
        "battleId": battle.id,
        "status": "failed" if response_failed else battle.status,
        "category": battle.category,
        "prompt": battle.prompt,
        "answers": answers,
        "reveal": _reveal(session, battle) if vote else None,
        "error": "One or both answers failed; this battle cannot be voted on."
        if response_failed
        else None,
    }


def _record_vote(
    session: Session,
    *,
    battle: Battle,
    rater: str,
    cohort: str,
    request: VoteCreate,
    idempotency_key: str,
    reviewer: ExpertReviewer | None = None,
    reviewer_binding: ReviewerIdentityBinding | None = None,
    reviewer_admission: Any | None = None,
    commit: bool = True,
) -> Vote:
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    season_id = session.scalar(
        select(Season.id).where(Season.id == battle.season_id).with_for_update()
    )
    if season_id is None:
        raise HTTPException(status_code=409, detail="battle season is unavailable")
    lock_material = f"vote:{battle.id}:{rater}:{cohort}"
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        lock_key = int(hashlib.sha256(lock_material.encode()).hexdigest()[:16], 16)
        if lock_key >= 2**63:
            lock_key -= 2**64
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )
    else:
        session.scalar(select(Battle.id).where(Battle.id == battle.id).with_for_update())
    prior_key = session.scalar(select(Vote).where(Vote.idempotency_key == idempotency_key))
    if prior_key:
        if prior_key.battle_id != battle.id or prior_key.rater_pseudonym != rater:
            raise HTTPException(status_code=409, detail="idempotency key was already used")
        requested_rubric = request.model_dump(mode="json")["rubric"]
        if (
            prior_key.cohort != cohort
            or prior_key.choice != request.choice.value
            or prior_key.reason_tags_json != request.reason_tags
            or prior_key.rubric_json != requested_rubric
        ):
            raise HTTPException(
                status_code=409,
                detail="idempotency key payload does not match the recorded vote",
            )
        return prior_key
    prior_vote = session.scalar(
        select(Vote).where(
            Vote.battle_id == battle.id,
            Vote.rater_pseudonym == rater,
            Vote.cohort == cohort,
        )
    )
    if prior_vote:
        raise HTTPException(status_code=409, detail="this battle already has a vote")
    if battle.status != "complete":
        raise HTTPException(status_code=409, detail="battle is not voteable")
    arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
    if len(arms) != 2 or any(
        arm.status != "complete" or not is_complete_finish_reason(arm.finish_reason) for arm in arms
    ):
        raise HTTPException(
            status_code=409,
            detail="both response arms require normal final completion before voting",
        )
    vote = Vote(
        battle_id=battle.id,
        rater_pseudonym=rater,
        cohort=cohort,
        choice=request.choice.value,
        reason_tags_json=request.reason_tags,
        rubric_json=request.model_dump(mode="json")["rubric"],
        idempotency_key=idempotency_key,
        provenance_status=("public_pseudonymous" if cohort == "public" else "legacy_unverified"),
    )
    if any(value is not None for value in (reviewer, reviewer_binding, reviewer_admission)):
        if reviewer is None or reviewer_binding is None or reviewer_admission is None:
            raise HTTPException(status_code=409, detail="expert vote provenance is incomplete")
        apply_verified_vote_provenance(
            vote,
            reviewer=reviewer,
            binding=reviewer_binding,
            admission=reviewer_admission,
        )
    session.add(vote)
    session.flush()
    session.add(
        RunEvent(
            entity_type="vote",
            entity_id=vote.id,
            event_type="vote_recorded",
            payload_json={"battle_id": battle.id, "cohort": cohort, "choice": request.choice.value},
        )
    )
    if commit:
        session.commit()
        session.refresh(vote)
    else:
        session.flush()
    return vote


@app.get("/health")
def health(session: Db) -> dict[str, str]:
    try:
        readiness = database_readiness(session, expected_role="flavourbench_api")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database is not ready") from exc
    return {
        "status": "ready",
        "executionMode": get_settings().execution_mode,
        **readiness,
    }


@router.post("/battles", status_code=202)
def post_battle(
    request: BattleCreate,
    session: Db,
    x_flavourbench_pseudonym: Annotated[str, Header()] = "",
    x_flavourbench_browser_pseudonym: Annotated[str, Header()] = "",
    x_flavourbench_network_pseudonym: Annotated[str, Header()] = "",
) -> dict:
    if not get_settings().public_arena_enabled:
        raise HTTPException(status_code=503, detail="public arena admission is disabled")
    pseudonym = _pseudonym(x_flavourbench_pseudonym)
    if get_settings().execution_mode == "mock" and not (
        x_flavourbench_browser_pseudonym and x_flavourbench_network_pseudonym
    ):
        browser_pseudonym = pseudonym
        network_pseudonym = pseudonym
    else:
        browser_pseudonym = _pseudonym(x_flavourbench_browser_pseudonym)
        network_pseudonym = _pseudonym(x_flavourbench_network_pseudonym)
    battle = create_battle(
        session,
        request,
        pseudonym,
        admission_pseudonyms=[
            ("browser", browser_pseudonym),
            ("network", network_pseudonym),
        ],
    )
    session.commit()
    return {"battleId": battle.id, "status": battle.status}


@router.get("/battles/{battle_id}")
def get_battle(
    battle_id: str,
    session: Db,
    x_flavourbench_pseudonym: Annotated[str, Header()] = "",
) -> dict:
    pseudonym = _pseudonym(x_flavourbench_pseudonym)
    battle = _battle_for_rater(session, battle_id, pseudonym)
    return _public_battle(session, battle, pseudonym)


@router.post("/battles/{battle_id}/votes")
def post_vote(
    battle_id: str,
    request: VoteCreate,
    session: Db,
    idempotency_key: Annotated[str, Header()] = "",
    x_flavourbench_pseudonym: Annotated[str, Header()] = "",
) -> dict:
    pseudonym = _pseudonym(x_flavourbench_pseudonym)
    battle = _battle_for_rater(session, battle_id, pseudonym)
    vote = _record_vote(
        session,
        battle=battle,
        rater=pseudonym,
        cohort="public",
        request=request,
        idempotency_key=idempotency_key,
    )
    return {"voteId": vote.id, "reveal": _reveal(session, battle)}


@router.post("/tasks/{public_id}/challenges", status_code=202)
def submit_task_challenge(
    public_id: str,
    request: TaskChallengeCreate,
    session: Db,
    idempotency_key: Annotated[str, Header()] = "",
    x_flavourbench_pseudonym: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    pseudonym = _pseudonym(x_flavourbench_pseudonym)
    _postgres_admit(
        session,
        pseudonym,
        action="task_challenge",
        limit=max(2, get_settings().admission_max_battles // 2),
    )
    task = session.scalar(
        select(Task)
        .join(Season, Season.id == Task.season_id)
        .where(
            Task.public_id == public_id,
            Season.official.is_(True),
        )
        .order_by(Task.created_at.desc())
    )
    if (
        task is None
        or not isinstance(task.provenance_json, dict)
        or task.provenance_json.get("confirmatory_eligible") is not True
    ):
        raise HTTPException(status_code=404, detail="challengeable task not found")
    request_payload = request.model_dump(mode="json", exclude={"client_nonce"})
    request_payload["description"] = sanitize_for_release(request.description)
    if request.evidence_reference is not None:
        request_payload["evidence_reference"] = sanitize_for_release(request.evidence_reference)
    nonce_sha256 = hashlib.sha256(request.client_nonce.encode()).hexdigest()
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    challenge_sha256 = _canonical_sha256(
        {
            "schema_version": "flavourbench-task-challenge-v1",
            "task_id": task.id,
            "task_public_id": task.public_id,
            "task_revision": task.revision,
            "task_record_sha256": task.provenance_json.get("task_record_sha256"),
            "requester_pseudonym_sha256": hashlib.sha256(pseudonym.encode()).hexdigest(),
            "request": request_payload,
        }
    )
    existing = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "task_challenge",
            RunEvent.event_type == "task_challenge_submitted",
            RunEvent.payload_json["task_id"].as_string() == task.id,
            RunEvent.payload_json["requester_pseudonym_sha256"].as_string()
            == hashlib.sha256(pseudonym.encode()).hexdigest(),
            RunEvent.payload_json["client_nonce_sha256"].as_string() == nonce_sha256,
        )
    )
    if existing is not None:
        if (
            existing.payload_json.get("idempotency_key_sha256") == idempotency_sha256
            and existing.payload_json.get("challenge_sha256") == challenge_sha256
        ):
            return {
                "challengeId": existing.entity_id,
                "challengeSha256": challenge_sha256,
                "status": "sealed_pending_independent_adjudication",
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail="client nonce already identifies a challenge")
    challenge_id = str(uuid.uuid4())
    event = RunEvent(
        entity_type="task_challenge",
        entity_id=challenge_id,
        event_type="task_challenge_submitted",
        payload_json={
            **request_payload,
            "task_id": task.id,
            "task_public_id": task.public_id,
            "task_revision": task.revision,
            "task_record_sha256": task.provenance_json.get("task_record_sha256"),
            "requester_pseudonym_sha256": hashlib.sha256(pseudonym.encode()).hexdigest(),
            "client_nonce_sha256": nonce_sha256,
            "idempotency_key_sha256": idempotency_sha256,
            "challenge_sha256": challenge_sha256,
            "status": "sealed_pending_independent_adjudication",
        },
    )
    session.add(event)
    session.commit()
    return {
        "challengeId": challenge_id,
        "challengeSha256": challenge_sha256,
        "status": "sealed_pending_independent_adjudication",
        "idempotent": False,
    }


@router.post(
    "/admin/task-challenges/{challenge_id}/adjudications",
    dependencies=[Depends(require_admin_token)],
)
def adjudicate_task_challenge(
    challenge_id: str,
    request: TaskChallengeAdjudicationCreate,
    session: Db,
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    submitted = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "task_challenge",
            RunEvent.entity_id == challenge_id,
            RunEvent.event_type == "task_challenge_submitted",
        )
    )
    if submitted is None:
        raise HTTPException(status_code=404, detail="task challenge not found")
    task = session.scalar(
        select(Task).where(Task.id == submitted.payload_json.get("task_id")).with_for_update()
    )
    if task is None or not isinstance(task.provenance_json, dict):
        raise HTTPException(status_code=409, detail="challenged task is unavailable")
    prior_events = list(
        session.scalars(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "task_challenge",
                RunEvent.entity_id == challenge_id,
                RunEvent.event_type == "task_challenge_adjudicated",
            )
            .order_by(RunEvent.created_at, RunEvent.id)
        ).all()
    )
    final_prior = next(
        (
            event
            for event in reversed(prior_events)
            if event.payload_json.get("decision") in {"confirmed", "dismissed"}
        ),
        None,
    )
    reviewer_ids = set(request.adjudicator_reviewer_ids)
    reviewers = {
        reviewer.id: reviewer
        for reviewer in session.scalars(
            select(ExpertReviewer).where(ExpertReviewer.id.in_(reviewer_ids))
        ).all()
    }
    if set(reviewers) != reviewer_ids or any(
        not reviewer.active
        or not reviewer.qualification_verified
        or task.family not in reviewer.qualification_json
        or not _verified_independent_task_validator(reviewer)
        for reviewer in reviewers.values()
    ):
        raise HTTPException(
            status_code=409,
            detail="challenge adjudication requires two qualified independent reviewers",
        )
    provenance = task.provenance_json
    original_role_ids = {str(provenance.get("human_author_id", ""))}
    original_role_ids.update(
        str(review.get("reviewer_id", ""))
        for review in provenance.get("independent_reviews") or []
        if isinstance(review, dict)
    )
    adjudication = provenance.get("adjudication")
    if isinstance(adjudication, dict):
        original_role_ids.add(str(adjudication.get("adjudicator_reviewer_id", "")))
    for key in ("validator_contract_review", "contamination_audit_review"):
        binding = provenance.get(key)
        if isinstance(binding, dict):
            original_role_ids.add(str(binding.get("reviewer_id", "")))
    original_people = {str(provenance.get("human_author_person_commitment_sha256", ""))}
    original_people.update(
        str(reviewer.profile_json.get("person_uniqueness_commitment_sha256", ""))
        for reviewer in session.scalars(
            select(ExpertReviewer).where(ExpertReviewer.id.in_(original_role_ids))
        ).all()
    )
    adjudicator_people = {
        str(reviewer.profile_json.get("person_uniqueness_commitment_sha256", ""))
        for reviewer in reviewers.values()
    }
    if (
        reviewer_ids & original_role_ids
        or "" in adjudicator_people
        or len(adjudicator_people) != len(reviewer_ids)
        or adjudicator_people & original_people
    ):
        raise HTTPException(
            status_code=409,
            detail="challenge adjudicators are not person-independent of the original task",
        )
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    request_payload = request.model_dump(mode="json")
    prior_event_id = (
        final_prior.payload_json.get("supersedes_adjudication_event_id")
        if final_prior is not None
        else prior_events[-1].id
        if prior_events
        else None
    )
    adjudication_sha256 = _canonical_sha256(
        {
            "schema_version": "flavourbench-task-challenge-adjudication-v1",
            "challenge_id": challenge_id,
            "challenge_sha256": submitted.payload_json.get("challenge_sha256"),
            "task_id": task.id,
            "task_record_sha256": provenance.get("task_record_sha256"),
            "supersedes_adjudication_event_id": prior_event_id,
            "adjudication": request_payload,
        }
    )
    if final_prior is not None:
        if (
            final_prior.payload_json.get("idempotency_key_sha256") == idempotency_sha256
            and final_prior.payload_json.get("adjudication_sha256") == adjudication_sha256
        ):
            return {
                "challengeId": challenge_id,
                "decision": final_prior.payload_json["decision"],
                "adjudicationSha256": adjudication_sha256,
                "taskRetired": final_prior.payload_json["decision"] == "confirmed",
                "leaderboardRecomputationRequired": (
                    final_prior.payload_json["decision"] == "confirmed"
                ),
                "withdrawnSnapshotIds": final_prior.payload_json.get("withdrawn_snapshot_ids", []),
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail="task challenge already has a final decision")
    now = datetime.now(UTC)
    event = RunEvent(
        entity_type="task_challenge",
        entity_id=challenge_id,
        event_type="task_challenge_adjudicated",
        payload_json={
            **request_payload,
            "challenge_sha256": submitted.payload_json.get("challenge_sha256"),
            "task_id": task.id,
            "task_public_id": task.public_id,
            "task_record_sha256": provenance.get("task_record_sha256"),
            "supersedes_adjudication_event_id": prior_event_id,
            "adjudication_sha256": adjudication_sha256,
            "idempotency_key_sha256": idempotency_sha256,
            "adjudicator_person_commitment_sha256s": sorted(adjudicator_people),
            "decided_at": now.isoformat(),
        },
        created_at=now,
    )
    session.add(event)
    task_retired = request.decision == "confirmed"
    withdrawn_snapshot_ids: list[str] = []
    if task_retired:
        if (
            session.scalar(
                select(RunEvent.id).where(
                    RunEvent.entity_type == "task",
                    RunEvent.entity_id == task.id,
                    RunEvent.event_type == "confirmatory_task_retired",
                )
            )
            is not None
        ):
            raise HTTPException(status_code=409, detail="task is already retired")
        session.add(
            RunEvent(
                entity_type="task",
                entity_id=task.id,
                event_type="confirmatory_task_retired",
                payload_json={
                    "task_public_id": task.public_id,
                    "task_revision": task.revision,
                    "task_record_sha256": provenance.get("task_record_sha256"),
                    "task_lifecycle_seal_sha256": provenance.get("task_lifecycle_seal_sha256"),
                    "challenge_id": challenge_id,
                    "challenge_sha256": submitted.payload_json.get("challenge_sha256"),
                    "adjudication_sha256": adjudication_sha256,
                    "retired_at": now.isoformat(),
                    "ranking_use": False,
                    "recomputation_required": True,
                    "correction_reference_sha256": hashlib.sha256(
                        str(request.correction_reference).encode()
                    ).hexdigest(),
                },
                created_at=now,
            )
        )
        withdrawn_snapshot_ids = _withdraw_published_snapshots(
            session,
            season_id=task.season_id,
            reason_code="confirmatory_task_retired_after_confirmed_challenge",
        )
        event.payload_json = {
            **event.payload_json,
            "withdrawn_snapshot_ids": withdrawn_snapshot_ids,
        }
    session.commit()
    return {
        "challengeId": challenge_id,
        "decision": request.decision,
        "adjudicationSha256": adjudication_sha256,
        "taskRetired": task_retired,
        "leaderboardRecomputationRequired": task_retired,
        "withdrawnSnapshotIds": withdrawn_snapshot_ids,
        "idempotent": False,
    }


def _create_controlled_run(
    request: ControlledRunCreate,
    session: Session,
    *,
    organization_id: str | None = None,
    evaluation_order_id: str | None = None,
    route_revision_id: str | None = None,
    endpoint_descriptor_sha256: str | None = None,
    spend_authorization_id: str | None = None,
    spend_authorization_binding_sha256: str | None = None,
) -> tuple[ControlledRun, str, dict[str, Any]]:
    binding_values = (
        organization_id,
        evaluation_order_id,
        route_revision_id,
        endpoint_descriptor_sha256,
        spend_authorization_id,
        spend_authorization_binding_sha256,
    )
    if any(value is not None for value in binding_values) and any(
        value is None for value in binding_values
    ):
        raise HTTPException(status_code=409, detail="commercial run binding must be complete")
    season = session.scalar(select(Season).where(Season.slug == request.season))
    if season is None:
        raise HTTPException(status_code=404, detail="season not found")
    if get_settings().execution_mode != "mock" and (
        not season.official
        or season.status != "active"
        or season.manifest_sha256 in UNFROZEN_VALUES
    ):
        raise HTTPException(
            status_code=409,
            detail="controlled runs require an active frozen season",
        )
    if get_settings().execution_mode != "mock":
        _verify_season_protocol(season)
        if request.analysis_plan_sha256 != season.analysis_plan_sha256:
            raise HTTPException(
                status_code=409,
                detail="controlled-run analysis must match the frozen season plan",
            )

    roster = _run_model_roster(
        session,
        season,
        request.model_ids,
        request.submitted_endpoint_model_id,
    )
    roster_sha256 = _canonical_sha256({"models": roster})
    task_public_ids = sorted({entry.task_public_id for entry in request.task_schedule})
    task_rows = session.scalars(
        select(Task).where(
            Task.season_id == season.id,
            Task.public_id.in_(task_public_ids),
        )
    ).all()
    tasks = {task.public_id: task for task in task_rows}
    missing_tasks = sorted(set(task_public_ids) - set(tasks))
    if missing_tasks:
        raise HTTPException(
            status_code=409,
            detail=f"controlled-run schedule contains unknown tasks: {', '.join(missing_tasks)}",
        )
    if get_settings().execution_mode != "mock" and any(
        task.review_status != "frozen"
        or task.split != "scored"
        or not isinstance(task.provenance_json, dict)
        or task.provenance_json.get("confirmatory_eligible") is not True
        for task in task_rows
    ):
        raise HTTPException(
            status_code=409,
            detail="controlled-run schedule contains a task outside the frozen held-out bank",
        )
    if get_settings().execution_mode != "mock":
        try:
            lifecycles = [verify_task_lifecycle(session, task) for task in task_rows]
        except TaskLifecycleError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"controlled-run schedule contains invalid task lifecycle: {exc}",
            ) from exc
        if any(lifecycle.retired_at is not None for lifecycle in lifecycles):
            raise HTTPException(
                status_code=409,
                detail="controlled-run schedule contains a retired task",
            )
    schedule_payloads: list[dict[str, Any]] = []
    assignment_seeds: dict[int, str] = {}
    for ordinal, entry in enumerate(request.task_schedule):
        task = tasks[entry.task_public_id]
        assignment_seed = secrets.token_hex(32)
        assignment_seeds[ordinal] = assignment_seed
        schedule_payloads.append(
            _assignment_payload(
                ordinal=ordinal,
                task_public_id=task.public_id,
                task_revision=task.revision,
                task_prompt_sha256=task.prompt_sha256,
                task_family=task.family,
                track=entry.track.value,
                model_ids=entry.model_ids,
                repetition_index=entry.repetition_index,
                side_seed_commitment_sha256=hashlib.sha256(assignment_seed.encode()).hexdigest(),
            )
        )
    schedule_sha256 = _schedule_sha256(schedule_payloads)
    slots = {
        slot.model_id: slot
        for slot in session.scalars(
            select(SeasonModel).where(
                SeasonModel.season_id == season.id,
                SeasonModel.model_id.in_(request.model_ids),
            )
        ).all()
    }
    forecast_worst_case_cost_micros = 0
    for payload in schedule_payloads:
        assignment_cost = sum(
            slots[model_id].worst_case_cost_micros for model_id in payload["model_ids"]
        )
        if payload["track"] == "epicure_uplift":
            assignment_cost *= 2
        forecast_worst_case_cost_micros += assignment_cost
    if (
        get_settings().execution_mode != "mock"
        and forecast_worst_case_cost_micros * 10_000 >= request.budget_cap_micros * 8_500
    ):
        raise HTTPException(
            status_code=409,
            detail="controlled-run budget cap cannot admit the frozen worst-case schedule",
        )

    issued_at = datetime.now(UTC)
    run_id = str(uuid.uuid4())
    access_token = secrets.token_urlsafe(32)
    organization_reference_sha256 = hashlib.sha256(
        request.organization_reference.encode()
    ).hexdigest()
    card, card_sha256, signature = _signed_run_card(
        run_id=run_id,
        season=season,
        organization_reference_sha256=organization_reference_sha256,
        protocol_version=request.protocol_version,
        rater_plan_sha256=request.rater_plan_sha256,
        analysis_plan_sha256=request.analysis_plan_sha256,
        submitted_endpoint_model_id=request.submitted_endpoint_model_id,
        submitted_model_card_sha256=request.submitted_model_card_sha256,
        data_policy_sha256=request.data_policy_sha256,
        model_roster=roster,
        model_roster_sha256=roster_sha256,
        task_schedule=schedule_payloads,
        task_schedule_sha256=schedule_sha256,
        forecast_worst_case_cost_micros=forecast_worst_case_cost_micros,
        budget_cap_micros=request.budget_cap_micros,
        issued_at=issued_at,
        organization_id=organization_id,
        evaluation_order_id=evaluation_order_id,
        route_revision_id=route_revision_id,
        endpoint_descriptor_sha256=endpoint_descriptor_sha256,
        spend_authorization_id=spend_authorization_id,
        spend_authorization_binding_sha256=spend_authorization_binding_sha256,
    )
    run = ControlledRun(
        id=run_id,
        season_id=season.id,
        organization_id=organization_id,
        evaluation_order_id=evaluation_order_id,
        route_revision_id=route_revision_id,
        endpoint_descriptor_sha256=endpoint_descriptor_sha256,
        spend_authorization_id=spend_authorization_id,
        spend_authorization_binding_sha256=spend_authorization_binding_sha256,
        organization_reference_sha256=organization_reference_sha256,
        access_token_sha256=hashlib.sha256(access_token.encode()).hexdigest(),
        status="active",
        protocol_version=request.protocol_version,
        rater_plan_sha256=request.rater_plan_sha256,
        analysis_plan_sha256=request.analysis_plan_sha256,
        submitted_endpoint_model_id=request.submitted_endpoint_model_id,
        submitted_model_card_sha256=request.submitted_model_card_sha256,
        data_policy_sha256=request.data_policy_sha256,
        model_roster_json=roster,
        model_roster_sha256=roster_sha256,
        task_schedule_sha256=schedule_sha256,
        budget_cap_micros=request.budget_cap_micros,
        run_card_json=card,
        run_card_sha256=card_sha256,
        run_card_signature=signature,
        created_at=issued_at,
    )
    session.add(run)
    session.flush()
    for payload in schedule_payloads:
        session.add(
            ControlledRunAssignment(
                controlled_run_id=run.id,
                ordinal=payload["ordinal"],
                task_id=tasks[payload["task_public_id"]].id,
                task_public_id=payload["task_public_id"],
                task_revision=payload["task_revision"],
                task_prompt_sha256=payload["task_prompt_sha256"],
                task_family=payload["task_family"],
                track=payload["track"],
                model_ids_json=payload["model_ids"],
                repetition_index=payload["repetition_index"],
                assignment_sha256=_canonical_sha256(payload),
                assignment_seed=assignment_seeds[payload["ordinal"]],
            )
        )
    session.add(
        RunEvent(
            entity_type="controlled_run",
            entity_id=run.id,
            event_type="controlled_run_created",
            payload_json={
                "run_card_sha256": card_sha256,
                "organization_reference_sha256": organization_reference_sha256,
                "season": season.slug,
                "model_roster_sha256": roster_sha256,
                "task_schedule_sha256": schedule_sha256,
                "assignments": len(schedule_payloads),
                "forecast_worst_case_cost_micros": forecast_worst_case_cost_micros,
                "evaluation_order_id": evaluation_order_id,
                "route_revision_id": route_revision_id,
                "endpoint_descriptor_sha256": endpoint_descriptor_sha256,
            },
        )
    )
    response = {
        "runId": run.id,
        "accessToken": access_token,
        "runCard": card,
        "runCardSha256": card_sha256,
        "runCardSignature": signature,
        "notice": "The controlled-run credential is returned once and is not stored in plaintext.",
    }
    return run, access_token, response


@router.post(
    "/admin/controlled-runs",
    dependencies=[Depends(require_admin_token)],
    status_code=201,
)
def admin_create_controlled_run(request: ControlledRunCreate, session: Db) -> dict:
    _run, _access_token, response = _create_controlled_run(request, session)
    session.commit()
    return response


@router.post(
    "/admin/evaluation-orders/{order_id}/provision",
    dependencies=[Depends(require_admin_token)],
    status_code=201,
)
def admin_provision_evaluation_order(
    order_id: str,
    request: EvaluationOrderProvisionCreate,
    session: Db,
) -> dict[str, Any]:
    order = session.scalar(
        select(EvaluationOrder).where(EvaluationOrder.id == order_id).with_for_update()
    )
    if order is None:
        raise HTTPException(status_code=404, detail="evaluation order not found")
    existing_run = session.scalar(
        select(ControlledRun).where(ControlledRun.evaluation_order_id == order.id)
    )
    if existing_run is not None:
        raise HTTPException(
            status_code=409,
            detail="evaluation order is already provisioned; its credential cannot be reissued",
        )
    if (
        order.status != "approved"
        or order.billing_status != "authorized"
        or order.quote_reference_sha256 is None
    ):
        raise HTTPException(status_code=409, detail="evaluation order is not provisionable")
    spend_authorization = _spend_authorization_for_order(
        session,
        order,
        order.quote_reference_sha256,
    )
    if spend_authorization is None:
        raise HTTPException(
            status_code=409,
            detail="order-specific spend authorization is no longer active",
        )
    season = session.get(Season, order.season_id)
    route = session.get(ModelRouteRevision, order.route_revision_id)
    submission = session.get(ModelSubmission, order.model_submission_id)
    if season is None or route is None or submission is None:
        raise HTTPException(status_code=409, detail="evaluation order contract is unavailable")
    if not (
        season.status == "active"
        and season.official
        and season.frozen_at is not None
        and route.status == "approved"
        and route.model_submission_id == submission.id
        and route.approved_season_id == season.id
        and route.approved_season_manifest_sha256 == season.manifest_sha256
        and submission.status == "approved"
        and submission.organization_id == order.organization_id
        and submission.catalog_model_id is not None
    ):
        raise HTTPException(status_code=409, detail="evaluation order dependencies drifted")
    order_card = order.order_card_json
    plan = order.comparison_plan_json
    if not isinstance(order_card, dict) or not isinstance(plan, dict):
        raise HTTPException(status_code=409, detail="evaluation order card is unavailable")
    signing = order_card.get("signing")
    signing_key_id = (
        str(signing.get("key_id"))
        if isinstance(signing, dict) and signing.get("key_id")
        else get_settings().run_card_signing_key_id
    )
    verification_secret = run_card_verification_keyring().get(signing_key_id)
    expected_signature = (
        hmac.new(
            verification_secret.encode(),
            f"flavourbench-evaluation-order-card-v2:{order.order_card_sha256}".encode(),
            hashlib.sha256,
        ).hexdigest()
        if verification_secret is not None
        else None
    )
    if (
        order_card.get("schema_version") != "flavourbench-evaluation-order-card-v2"
        or order_card.get("evaluation_order_id") != order.id
        or order.order_card_sha256 != _canonical_sha256(order_card)
        or order.comparison_plan_sha256 != _canonical_sha256(plan)
        or expected_signature is None
        or not hmac.compare_digest(expected_signature, order.order_card_signature)
    ):
        raise HTTPException(status_code=409, detail="evaluation order signature failed")
    try:
        controlled_request = ControlledRunCreate.model_validate(
            {
                "season": season.slug,
                "organizationReference": order.organization_id,
                "protocolVersion": "flavourbench-managed-evaluation-v1",
                "raterPlanSha256": order.rater_plan_sha256,
                "analysisPlanSha256": order.analysis_plan_sha256,
                "submittedEndpointModelId": submission.catalog_model_id,
                "submittedModelCardSha256": submission.model_card_sha256,
                "dataPolicySha256": route.data_policy_sha256,
                "modelIds": plan.get("model_ids"),
                "taskSchedule": plan.get("task_schedule"),
                "budgetCapMicros": order.budget_cap_micros,
            }
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail="evaluation order has an invalid frozen schedule",
        ) from exc
    run, _access_token, run_response = _create_controlled_run(
        controlled_request,
        session,
        organization_id=order.organization_id,
        evaluation_order_id=order.id,
        route_revision_id=route.id,
        endpoint_descriptor_sha256=route.descriptor_sha256,
        spend_authorization_id=spend_authorization.id,
        spend_authorization_binding_sha256=spend_authorization.binding_sha256,
    )
    session.flush()
    order.status = "provisioning"
    session.flush()
    order.status = "ready"
    session.flush()
    session.add(
        RunEvent(
            entity_type="evaluation_order",
            entity_id=order.id,
            event_type="evaluation_order_provisioned",
            payload_json={
                "organization_id": order.organization_id,
                "controlled_run_id": run.id,
                "run_card_sha256": run.run_card_sha256,
                "order_card_sha256": order.order_card_sha256,
                "route_revision_id": route.id,
                "endpoint_descriptor_sha256": route.descriptor_sha256,
                "spend_authorization_id": spend_authorization.id,
                "provision_reference_sha256": request.provision_reference_sha256,
            },
        )
    )
    session.commit()
    return {
        **run_response,
        "evaluationOrderId": order.id,
        "orderStatus": order.status,
        "commercialBinding": run.run_card_json["commercial_binding"],
    }


@router.get("/controlled/runs/{run_id}")
def get_controlled_run(
    run_id: str,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict:
    run, _ = _controlled_run_identity(session, run_id, authorization)
    battles = session.scalars(select(Battle).where(Battle.controlled_run_id == run.id)).all()
    status_counts: dict[str, int] = {}
    for battle in battles:
        status_counts[battle.status] = status_counts.get(battle.status, 0) + 1
    assignments = session.scalars(
        select(ControlledRunAssignment)
        .where(ControlledRunAssignment.controlled_run_id == run.id)
        .order_by(ControlledRunAssignment.ordinal)
    ).all()
    assignment_counts: dict[str, int] = {}
    for assignment in assignments:
        assignment_counts[assignment.status] = assignment_counts.get(assignment.status, 0) + 1
    return {
        "runId": run.id,
        "status": run.status,
        "dataStratum": "controlled",
        "releaseAuthorized": run.release_authorized,
        "budget": {
            "capMicros": run.budget_cap_micros,
            "usedMicros": run.budget_used_micros,
            "reservedMicros": run.budget_reserved_micros,
            "admissionThresholdBasisPoints": 8500,
            "drainThresholdBasisPoints": 9500,
            "hardStopBasisPoints": 10000,
            "accountingBasis": "endpoint_generation_receipts",
            "aggregateInvoiceVarianceScope": "season_and_provider_account_only",
            "creditsRestoreSpendAuthority": False,
        },
        "battleCounts": status_counts,
        "assignmentCounts": assignment_counts,
        "assignmentTotal": len(assignments),
        "submittedEndpointModelId": run.submitted_endpoint_model_id,
        "submittedModelCardSha256": run.submitted_model_card_sha256,
        "dataPolicySha256": run.data_policy_sha256,
        "modelRosterSha256": run.model_roster_sha256,
        "taskScheduleSha256": run.task_schedule_sha256,
        "tokenVersion": run.token_version,
        "sideRandomizationReveal": [
            {
                "ordinal": assignment.ordinal,
                "assignmentSha256": assignment.assignment_sha256,
                "assignmentSeed": assignment.assignment_seed,
                "sideSeedCommitmentSha256": hashlib.sha256(
                    assignment.assignment_seed.encode()
                ).hexdigest(),
                "reversed": controlled_side_is_reversed(assignment.assignment_seed),
            }
            for assignment in assignments
        ]
        if run.status == "closed"
        else None,
        "runCard": run.run_card_json,
        "runCardSha256": run.run_card_sha256,
        "runCardSignature": run.run_card_signature,
    }


@router.post("/controlled/runs/{run_id}/battles", status_code=202)
def post_controlled_battle(
    run_id: str,
    request: ControlledBattleCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict:
    run, _ = _controlled_run_identity(session, run_id, authorization)
    if run.status != "active":
        raise HTTPException(status_code=409, detail="controlled run is not accepting battles")
    season = session.get(Season, run.season_id)
    if season is None:
        raise HTTPException(status_code=409, detail="controlled run season is unavailable")
    _verify_season_protocol(season)
    run_pseudonym = hmac.new(
        get_settings().pseudonym_secret.encode(),
        f"controlled:{run.id}".encode(),
        hashlib.sha256,
    ).hexdigest()
    battle = create_battle(
        session,
        request,
        run_pseudonym,
        controlled_run=run,
        season_row=season,
    )
    if run.evaluation_order_id is not None:
        order = session.scalar(
            select(EvaluationOrder)
            .where(EvaluationOrder.id == run.evaluation_order_id)
            .with_for_update()
        )
        if order is None or order.status not in {"ready", "running"}:
            raise HTTPException(status_code=409, detail="evaluation order is not executable")
        if (
            active_spend_authorization(
                session,
                order=order,
                acceptance_id=str(run.spend_authorization_id),
                binding_sha256=str(run.spend_authorization_binding_sha256),
                lock=True,
            )
            is None
        ):
            raise HTTPException(
                status_code=409,
                detail="evaluation order spend authorization is no longer active",
            )
        if order.status == "ready":
            order.status = "running"
            order.started_at = datetime.now(UTC)
    session.commit()
    assignment = session.scalar(
        select(ControlledRunAssignment).where(
            ControlledRunAssignment.controlled_run_id == run.id,
            ControlledRunAssignment.battle_id == battle.id,
        )
    )
    return {
        "battleId": battle.id,
        "runId": run.id,
        "status": battle.status,
        "assignmentOrdinal": assignment.ordinal if assignment else None,
        "assignmentSha256": assignment.assignment_sha256 if assignment else None,
    }


@router.get("/controlled/runs/{run_id}/leaderboards")
def get_controlled_run_leaderboard(
    run_id: str,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    track: str = Query(default="model_arena", pattern="^(model_arena|epicure_uplift)$"),
    rater_cohort: str = Query(
        default="expert_independent",
        pattern=(
            "^(public|expert_independent|expert_product_affiliated|"
            "expert_provider_affiliated|combined)$"
        ),
    ),
    task_family: str = Query(default="all"),
) -> dict:
    _prepare_snapshot_transaction(session)
    _controlled_run_identity(session, run_id, authorization)
    season, run = _locked_controlled_run(session, run_id, read_only=True)
    if season.status != "active":
        raise HTTPException(status_code=409, detail="controlled-run season is not active")
    if run.status != "closed" or not _active_controlled_release_authorization(session, run):
        raise HTTPException(
            status_code=409,
            detail="controlled leaderboard requires a closed, release-authorized run",
        )
    _verify_season_protocol(season)
    snapshot = session.scalar(
        select(LeaderboardSnapshot)
        .where(
            LeaderboardSnapshot.controlled_run_id == run.id,
            LeaderboardSnapshot.track == track,
            LeaderboardSnapshot.cohort == rater_cohort,
            LeaderboardSnapshot.category == task_family,
            LeaderboardSnapshot.data_stratum == "controlled",
            LeaderboardSnapshot.publication_status == "published",
        )
        .order_by(
            LeaderboardSnapshot.published_at.desc(),
            LeaderboardSnapshot.created_at.desc(),
            LeaderboardSnapshot.id.desc(),
        )
        .limit(1)
        .with_for_update(read=True)
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="published controlled-run snapshot not found")
    try:
        payload = _verified_current_snapshot_payload(
            session,
            season=season,
            snapshot=snapshot,
        )
    except _SnapshotVerificationError as failure:
        _withdraw_snapshot_after_verification_failure(
            session,
            snapshot=snapshot,
            failure=failure,
        )
        session.commit()
        raise HTTPException(
            status_code=503,
            detail="controlled leaderboard snapshot failed current-evidence verification",
        ) from failure
    return {
        **payload,
        "snapshotId": snapshot.id,
        "publicationStatus": snapshot.publication_status,
        "releaseAuthorized": run.release_authorized,
        "runCardSha256": run.run_card_sha256,
        "runCardSignature": run.run_card_signature,
    }


@router.get("/controlled/runs/{run_id}/evidence")
def get_controlled_run_evidence(
    run_id: str,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    snapshot_id: str | None = Query(default=None, min_length=36, max_length=36),
) -> dict:
    """Return one deterministic, authenticated private result and evidence document."""

    _prepare_snapshot_transaction(session)
    _controlled_run_identity(session, run_id, authorization)
    season, run = _locked_controlled_run(session, run_id, read_only=True)
    if run.status != "closed":
        raise HTTPException(
            status_code=409,
            detail="controlled evidence requires a closed run",
        )
    if not _active_controlled_release_authorization(session, run):
        raise HTTPException(
            status_code=409,
            detail="controlled evidence requires an active attributable release authorization",
        )
    statement = select(LeaderboardSnapshot).where(
        LeaderboardSnapshot.controlled_run_id == run.id,
        LeaderboardSnapshot.data_stratum == "controlled",
        LeaderboardSnapshot.publication_status == "published",
    )
    if snapshot_id is not None:
        statement = statement.where(LeaderboardSnapshot.id == snapshot_id)
    snapshot = session.scalar(
        statement.order_by(
            LeaderboardSnapshot.published_at.desc(),
            LeaderboardSnapshot.created_at.desc(),
            LeaderboardSnapshot.id.desc(),
        )
        .limit(1)
        .with_for_update(read=True)
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="published controlled snapshot not found")
    try:
        payload = _verified_current_snapshot_payload(
            session,
            season=season,
            snapshot=snapshot,
        )
    except _SnapshotVerificationError as failure:
        _withdraw_snapshot_after_verification_failure(
            session,
            snapshot=snapshot,
            failure=failure,
        )
        session.commit()
        raise HTTPException(
            status_code=503,
            detail="controlled evidence failed current-evidence verification",
        ) from failure
    evidence_manifest = snapshot.input_evidence_json
    if not isinstance(evidence_manifest, dict) or not snapshot.input_evidence_sha256:
        raise HTTPException(
            status_code=503,
            detail="controlled snapshot lacks a sealed evidence manifest",
        )
    if snapshot.publication_reference_sha256 is None or snapshot.published_at is None:
        raise HTTPException(
            status_code=503,
            detail="controlled snapshot lacks publication provenance",
        )
    envelope = {
        "schema_version": "flavourbench-controlled-evidence-envelope-v1",
        "run_id": run.id,
        "run_card": run.run_card_json,
        "run_card_sha256": run.run_card_sha256,
        "run_card_signature": run.run_card_signature,
        "release_authorization": {
            "reference_sha256": run.release_authorization_reference_sha256,
            "authorized_at": _utc_iso(run.release_authorized_at),
            "publication_acceptance_id": run.publication_authorization_id,
            "publication_acceptance_binding_sha256": (run.publication_authorization_binding_sha256),
        },
        "snapshot": {
            "id": snapshot.id,
            "track": snapshot.track,
            "rater_cohort": snapshot.cohort,
            "task_family": snapshot.category,
            "evidence_cutoff_at": _utc_iso(snapshot.evidence_cutoff_at),
            "published_at": _utc_iso(snapshot.published_at),
            "publication_status": snapshot.publication_status,
            "publication_reference_sha256": snapshot.publication_reference_sha256,
            "supersedes_snapshot_id": snapshot.supersedes_snapshot_id,
            "input_sha256": snapshot.input_sha256,
            "input_evidence_sha256": snapshot.input_evidence_sha256,
            "payload_sha256": snapshot.payload_sha256,
        },
        "result_payload": payload,
        "evidence_manifest": evidence_manifest,
    }
    envelope_sha256 = _canonical_sha256(envelope)
    settings = get_settings()
    signature = hmac.new(
        settings.run_card_signing_secret.encode(),
        f"flavourbench-controlled-evidence-v1:{envelope_sha256}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return {
        "envelope": envelope,
        "envelopeSha256": envelope_sha256,
        "signature": signature,
        "signatureAlgorithm": "HMAC-SHA256",
        "signingKeyId": settings.run_card_signing_key_id,
        "verificationScope": "FlavourBench service-held key",
    }


@router.post(
    "/controlled/runs/{run_id}/release-authorization",
    dependencies=[Depends(require_admin_token)],
)
def set_controlled_run_release_authorization(
    run_id: str,
    request: ControlledRunReleaseCreate,
    session: Db,
) -> dict:
    season, run = _locked_controlled_run(session, run_id)
    acceptance = None
    order = None
    if run.evaluation_order_id is not None:
        order = session.scalar(
            select(EvaluationOrder)
            .where(EvaluationOrder.id == run.evaluation_order_id)
            .with_for_update()
        )
        if order is None:
            raise HTTPException(status_code=409, detail="commercial order is unavailable")
        if request.authorized:
            acceptance_id = str(request.publication_acceptance_id)
            expected_binding = publication_authorization_binding(order, run)
            expected_binding_sha256 = commercial_binding_sha256(expected_binding)
            acceptance = active_publication_authorization(
                session,
                order=order,
                run=run,
                acceptance_id=acceptance_id,
                binding_sha256=expected_binding_sha256,
                lock=True,
            )
            if acceptance is None:
                raise HTTPException(
                    status_code=409,
                    detail="publication authorization is inactive, superseded, or mismatched",
                )
            if order.publication_status != "private":
                raise HTTPException(
                    status_code=409,
                    detail="evaluation order cannot enter publication authorization",
                )
            order.publication_status = "authorized"
            run.publication_authorization_id = acceptance.id
            run.publication_authorization_binding_sha256 = acceptance.binding_sha256
        elif order.publication_status in {"authorized", "published"}:
            order.publication_status = "withdrawn"
    run.release_authorized = request.authorized
    run.release_authorization_reference_sha256 = (
        acceptance.external_envelope_reference_sha256
        if acceptance is not None
        else hashlib.sha256(request.authorization_reference.encode()).hexdigest()
    )
    run.release_authorized_at = datetime.now(UTC) if request.authorized else None
    session.add(
        RunEvent(
            entity_type="controlled_run",
            entity_id=run.id,
            event_type=(
                "controlled_run_release_authorized"
                if request.authorized
                else "controlled_run_release_authorization_revoked"
            ),
            payload_json={
                "authorization_reference_sha256": (run.release_authorization_reference_sha256),
                "publication_authorization_id": (acceptance.id if acceptance is not None else None),
                "publication_authorization_binding_sha256": (
                    acceptance.binding_sha256 if acceptance is not None else None
                ),
            },
        )
    )
    withdrawn_snapshot_ids = (
        []
        if request.authorized
        else _withdraw_published_snapshots(
            session,
            season_id=season.id,
            controlled_run_id=run.id,
            reason_code="controlled_run_release_authorization_revoked",
        )
    )
    session.commit()
    return {
        "runId": run.id,
        "releaseAuthorized": run.release_authorized,
        "withdrawnSnapshotIds": withdrawn_snapshot_ids,
    }


@router.post(
    "/admin/controlled-runs/{run_id}/lifecycle",
    dependencies=[Depends(require_admin_token)],
)
def admin_controlled_run_lifecycle(
    run_id: str,
    request: ControlledRunLifecycleCreate,
    session: Db,
) -> dict:
    season, run = _locked_controlled_run(session, run_id)
    now = datetime.now(UTC)
    action = request.action.value
    prior_status = run.status
    reference_sha256 = hashlib.sha256(request.authorization_reference.encode()).hexdigest()
    assignments = session.scalars(
        select(ControlledRunAssignment)
        .where(ControlledRunAssignment.controlled_run_id == run.id)
        .order_by(ControlledRunAssignment.ordinal)
        .with_for_update()
    ).all()
    battles = session.scalars(
        select(Battle).where(Battle.controlled_run_id == run.id).with_for_update()
    ).all()

    if action == "collection_complete":
        if run.status != "active":
            raise HTTPException(status_code=409, detail="invalid lifecycle transition")
        try:
            verify_controlled_run_bijection(
                session,
                assignments,
                battles,
                require_terminal=True,
            )
        except ControlledRunIntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"controlled-run collection integrity failed: {exc}",
            ) from exc
        run.status = "collection_complete"
        run.collection_completed_at = now
    elif action == "close":
        if run.status != "collection_complete":
            raise HTTPException(status_code=409, detail="invalid lifecycle transition")
        _require_budget_integrity(session, season.id)
        try:
            verify_controlled_run_bijection(
                session,
                assignments,
                battles,
                require_terminal=True,
            )
        except ControlledRunIntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"controlled-run collection integrity failed: {exc}",
            ) from exc
        arms = session.scalars(
            select(ResponseArm)
            .join(Battle, Battle.id == ResponseArm.battle_id)
            .where(Battle.controlled_run_id == run.id)
        ).all()
        jobs = session.scalars(
            select(Job)
            .join(Battle, Battle.id == Job.battle_id)
            .where(Battle.controlled_run_id == run.id)
        ).all()
        arm_ids = {arm.id for arm in arms}
        battle_ids = {battle.id for battle in battles}
        cost_events = session.scalars(
            select(CostEvent).where(CostEvent.battle_id.in_(battle_ids))
        ).all()
        jobs_by_battle = {
            battle_id: [job for job in jobs if job.battle_id == battle_id]
            for battle_id in battle_ids
        }
        arm_cost_evidence_complete = all(
            arm.cost_reconciled
            and arm.cost_accounting_basis != "unrecorded"
            and arm.billing_reconciliation_status != "unrecorded"
            and any(
                event.arm_id == arm.id
                and event.kind in {"actual", "actual_settlement"}
                and event.amount_micros == arm.cost_micros
                and event.generation_id == arm.generation_id
                for event in cost_events
            )
            for arm in arms
            if arm.id in arm_ids
        )
        battle_cost_evidence_complete = all(
            any(
                event.battle_id == battle.id
                and event.kind == "reconcile"
                and event.amount_micros
                == sum(arm.cost_micros for arm in arms if arm.battle_id == battle.id)
                for event in cost_events
            )
            for battle in battles
        )
        if (
            run.budget_reserved_micros != 0
            or any(battle.reserved_cost_micros != 0 for battle in battles)
            or any(
                len(jobs_by_battle[battle.id]) != 1
                or jobs_by_battle[battle.id][0].status not in {"complete", "failed"}
                or jobs_by_battle[battle.id][0].completed_at is None
                for battle in battles
            )
            or not arm_cost_evidence_complete
            or not battle_cost_evidence_complete
        ):
            raise HTTPException(
                status_code=409,
                detail="controlled run cannot close before cost and job reconciliation",
            )
        run.status = "closed"
        run.closed_at = now
    elif action == "revoke":
        if run.status == "revoked":
            raise HTTPException(status_code=409, detail="controlled run is already revoked")
        running_jobs = session.scalar(
            select(Job)
            .join(Battle, Battle.id == Job.battle_id)
            .where(Battle.controlled_run_id == run.id, Job.status == "running")
            .limit(1)
        )
        if running_jobs is not None:
            raise HTTPException(
                status_code=409,
                detail="controlled run must drain running generation jobs before revocation",
            )
        cancelled_jobs = cancel_unstarted_controlled_jobs(session, run)
        session.flush()
        run.status = "revoked"
        run.revoked_at = now
        run.release_authorized = False
        run.release_authorized_at = None
        run.access_token_sha256 = hashlib.sha256(
            f"revoked:{run.id}:{secrets.token_urlsafe(32)}".encode()
        ).hexdigest()
        run.token_version += 1
        for authorization in session.scalars(
            select(ControlledRunReviewer).where(
                ControlledRunReviewer.controlled_run_id == run.id,
                ControlledRunReviewer.active.is_(True),
            )
        ).all():
            authorization.active = False
        withdrawn_snapshot_ids = _withdraw_published_snapshots(
            session,
            season_id=run.season_id,
            controlled_run_id=run.id,
            reason_code="controlled_run_revoked",
        )
    else:  # pragma: no cover - Pydantic rejects unknown actions.
        raise HTTPException(status_code=422, detail="unsupported lifecycle action")

    event_payload: dict[str, Any] = {
        "authorization_reference_sha256": reference_sha256,
        "prior_status": prior_status,
        "new_status": run.status,
    }
    if action == "revoke":
        event_payload.update(
            {
                "cancelled_unstarted_jobs": cancelled_jobs,
                "running_job_present": running_jobs is not None,
                "token_version": run.token_version,
                "withdrawn_snapshot_ids": withdrawn_snapshot_ids,
            }
        )
    session.add(
        RunEvent(
            entity_type="controlled_run",
            entity_id=run.id,
            event_type=f"controlled_run_{action}",
            payload_json=event_payload,
        )
    )
    session.commit()
    return {
        "runId": run.id,
        "status": run.status,
        "collectionCompletedAt": run.collection_completed_at.isoformat()
        if run.collection_completed_at
        else None,
        "closedAt": run.closed_at.isoformat() if run.closed_at else None,
        "revokedAt": run.revoked_at.isoformat() if run.revoked_at else None,
    }


@router.post(
    "/admin/controlled-runs/{run_id}/rotate-token",
    dependencies=[Depends(require_admin_token)],
)
def admin_rotate_controlled_run_token(
    run_id: str,
    request: ControlledTokenRotationCreate,
    session: Db,
) -> dict:
    _, run = _locked_controlled_run(session, run_id)
    if run.status == "revoked":
        raise HTTPException(status_code=409, detail="revoked credentials cannot rotate")
    access_token = secrets.token_urlsafe(32)
    run.access_token_sha256 = hashlib.sha256(access_token.encode()).hexdigest()
    run.token_version += 1
    reference_sha256 = hashlib.sha256(request.authorization_reference.encode()).hexdigest()
    session.add(
        RunEvent(
            entity_type="controlled_run",
            entity_id=run.id,
            event_type="controlled_run_token_rotated",
            payload_json={
                "authorization_reference_sha256": reference_sha256,
                "token_version": run.token_version,
            },
        )
    )
    session.commit()
    return {
        "runId": run.id,
        "accessToken": access_token,
        "tokenVersion": run.token_version,
        "notice": (
            "The replacement credential is returned once and immediately "
            "invalidates the prior credential."
        ),
    }


@router.post(
    "/admin/controlled-runs/{run_id}/battles/{battle_id}/cost-settlement",
    dependencies=[Depends(require_admin_token)],
)
def admin_settle_uncertain_cost(
    run_id: str,
    battle_id: str,
    request: CostSettlementCreate,
    session: Db,
) -> dict:
    _, run = _locked_controlled_run(session, run_id)
    battle = session.scalar(
        select(Battle)
        .where(
            Battle.id == battle_id,
            Battle.controlled_run_id == run.id,
        )
        .with_for_update()
    )
    if battle is None:
        raise HTTPException(status_code=404, detail="controlled battle not found")
    job = session.scalar(select(Job).where(Job.battle_id == battle.id).with_for_update())
    if job is None or job.status != "uncertain":
        raise HTTPException(
            status_code=409,
            detail="battle has no unresolved provider-cost exposure",
        )
    arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
    arm_ids = [arm.id for arm in arms]
    events = session.scalars(
        select(GenerationAttempt)
        .where(GenerationAttempt.arm_id.in_(arm_ids))
        .order_by(GenerationAttempt.created_at, GenerationAttempt.id)
    ).all()
    unsafe_arm_ids = {
        arm.id
        for arm in arms
        if has_unresolved_paid_attempt([event for event in events if event.arm_id == arm.id])
    }
    unsafe_arm_ids.update(
        arm.id
        for arm in arms
        if arm.status == "uncertain" or (arm.generation_id is not None and not arm.cost_reconciled)
    )
    if set(request.arm_costs_micros) != unsafe_arm_ids:
        raise HTTPException(
            status_code=409,
            detail="settlement must cover every and only the unresolved response arms",
        )
    if (
        battle.status != "failed"
        or battle.completed_at is None
        or job.completed_at is None
        or any(arm.id in unsafe_arm_ids and arm.completed_at is None for arm in arms)
    ):
        raise HTTPException(
            status_code=409,
            detail="uncertain-cost settlement requires pre-existing terminal timestamps",
        )
    reference_sha256 = hashlib.sha256(request.authorization_reference.encode()).hexdigest()
    prior_actual_receipts = {
        event.arm_id: event
        for event in session.scalars(
            select(CostEvent).where(
                CostEvent.arm_id.in_(arm_ids),
                CostEvent.kind == "actual",
            )
        ).all()
        if event.arm_id is not None
    }
    for arm in arms:
        if arm.id not in unsafe_arm_ids:
            continue
        arm.cost_micros = request.arm_costs_micros[arm.id]
        arm.cost_reconciled = True
        arm.cost_accounting_basis = "manual_authorized_settlement"
        arm.billing_reconciliation_status = "manual_authorized_settlement"
        arm.status = "failed"
        arm.error_code = "CostExposureSettled"
        arm.error_detail = "Provider cost exposure was settled by an authorized record."
        session.add(
            CostEvent(
                season_id=battle.season_id,
                battle_id=battle.id,
                arm_id=arm.id,
                kind="actual_settlement",
                amount_micros=arm.cost_micros,
                provider=arm.actual_provider_slug or arm.provider_slug,
                generation_id=arm.generation_id,
                accounting_json={
                    "authorization_reference_sha256": reference_sha256,
                    "settlement": "manual_authorized",
                    "supersedes_cost_event_id": (
                        prior_actual_receipts[arm.id].id
                        if arm.id in prior_actual_receipts
                        else None
                    ),
                    "prior_cost_state": "unresolved_attempt_journal",
                },
            )
        )
    job.status = "failed"
    job.last_error = "uncertain provider-cost exposure was manually settled"
    session.add(
        RunEvent(
            entity_type="battle",
            entity_id=battle.id,
            event_type="generation_cost_exposure_settled",
            payload_json={
                "controlled_run_id": run.id,
                "authorization_reference_sha256": reference_sha256,
                "arm_costs_micros": request.arm_costs_micros,
            },
        )
    )
    session.flush(arms)
    reconcile_battle_cost(session, battle)
    session.commit()
    return {
        "runId": run.id,
        "battleId": battle.id,
        "jobStatus": job.status,
        "reservedCostMicros": battle.reserved_cost_micros,
        "settledCostMicros": sum(request.arm_costs_micros.values()),
    }


@router.get("/leaderboards")
def get_leaderboards(
    session: Db,
    season: str = Query(default="season-0"),
    track: str = Query(default="model_arena", pattern="^(model_arena|epicure_uplift)$"),
    rater_cohort: str = Query(
        default="public",
        pattern=(
            "^(public|expert_independent|expert_product_affiliated|"
            "expert_provider_affiliated|combined)$"
        ),
    ),
    task_family: str = Query(default="all"),
) -> dict:
    _prepare_snapshot_transaction(session)
    season_row = session.scalar(
        select(Season).where(Season.slug == season).with_for_update(read=True)
    )
    if season_row is None:
        raise HTTPException(status_code=404, detail="season not found")
    release_allowed = season_row.official and season_row.status == "active"
    if release_allowed:
        _verify_season_protocol(season_row)
    snapshot = (
        session.scalar(
            select(LeaderboardSnapshot)
            .where(
                LeaderboardSnapshot.season_id == season_row.id,
                LeaderboardSnapshot.track == track,
                LeaderboardSnapshot.cohort == rater_cohort,
                LeaderboardSnapshot.category == task_family,
                LeaderboardSnapshot.data_stratum == "public_freeform",
                LeaderboardSnapshot.controlled_run_id.is_(None),
                LeaderboardSnapshot.publication_status == "published",
            )
            .order_by(
                LeaderboardSnapshot.published_at.desc(),
                LeaderboardSnapshot.created_at.desc(),
                LeaderboardSnapshot.id.desc(),
            )
            .limit(1)
            .with_for_update(read=True)
        )
        if release_allowed
        else None
    )
    if snapshot is not None:
        try:
            payload = _verified_current_snapshot_payload(
                session,
                season=season_row,
                snapshot=snapshot,
            )
        except _SnapshotVerificationError as failure:
            _withdraw_snapshot_after_verification_failure(
                session,
                snapshot=snapshot,
                failure=failure,
            )
            session.commit()
            raise HTTPException(
                status_code=503,
                detail="public leaderboard snapshot failed current-evidence verification",
            ) from failure
    else:
        payload = {
            "track": track,
            "cohort": rater_cohort,
            "cohort_label": (
                "Combined · secondary" if rater_cohort == "combined" else rater_cohort.title()
            ),
            "category": task_family,
            "data_stratum": "public_freeform",
            "controlled_run_id": None,
            "rows": [],
            "method": None,
            "manifest_sha256": season_row.manifest_sha256,
            "eligibility_filter": {"publication_status": "no_published_snapshot"},
        }
    payload.update(
        {
            "season": season_row.slug,
            "seasonStatus": season_row.status,
            "official": bool(release_allowed and snapshot),
            "manifestSha256": season_row.manifest_sha256,
            "snapshotId": snapshot.id if snapshot else None,
            "snapshotPublishedAt": snapshot.published_at.isoformat()
            if snapshot and snapshot.published_at
            else None,
            "budget": {
                "capMicros": season_row.budget_cap_micros,
                "usedMicros": season_row.budget_used_micros,
                "reservedMicros": season_row.budget_reserved_micros,
                "admissionThresholdBasisPoints": 8500,
                "drainThresholdBasisPoints": 9500,
                "hardStopBasisPoints": 10000,
            },
            "sampleNotice": (
                None
                if release_allowed and snapshot
                else "No release-approved public leaderboard snapshot"
            ),
        }
    )
    return payload


@router.get("/models")
def get_models(session: Db, season: str = Query(default="season-0")) -> dict:
    season_row = session.scalar(select(Season).where(Season.slug == season))
    eligibility: dict[str, SeasonModel] = {}
    if season_row:
        eligibility = {
            item.model_id: item
            for item in session.scalars(
                select(SeasonModel).where(SeasonModel.season_id == season_row.id)
            ).all()
        }
    models = session.scalars(select(CatalogModel).order_by(CatalogModel.name)).all()
    return {
        "season": season,
        "catalogCount": len(models),
        "models": [
            {
                "id": model.model_id,
                "canonicalSlug": model.canonical_slug,
                "name": model.name,
                "family": model.family,
                "openWeight": model.open_weight,
                "status": "season_eligible"
                if model.model_id in eligibility and eligibility[model.model_id].eligible
                else "smoke_passed"
                if model.status == "season_eligible"
                else model.status,
                "supportsTools": model.supports_tools,
                "supportsStructuredOutput": model.supports_structured_outputs,
                "contextLength": model.context_length,
                "slotRole": eligibility[model.model_id].slot_role
                if model.model_id in eligibility
                else None,
            }
            for model in models
        ],
    }


def _invited_expert_identity(
    session: Session,
    authorization: str,
) -> tuple[str, ExpertReviewer]:
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="expert invitation is required")
    if token.startswith("fbrv1_"):
        try:
            credential = consume_reviewer_credential(
                session,
                token=token,
                required_scope="expert_review",
            )
            # Authentication use is a security event, not part of the route's
            # later business transaction. Persist it even when a read-only
            # request returns or a subsequent review operation fails.
            session.commit()
        except (ReviewerIdentityError, DBAPIError) as exc:
            session.rollback()
            raise HTTPException(
                status_code=401,
                detail="expert credential is invalid, expired, or exhausted",
            ) from exc
        reviewer = session.get(ExpertReviewer, credential.reviewer_id)
        binding = session.get(ReviewerIdentityBinding, credential.identity_binding_id)
        if (
            reviewer is None
            or binding is None
            or not reviewer.active
            or binding.reviewer_id != reviewer.id
            or binding.season_id != credential.season_id
            or binding.assurance_level != "server_verified"
        ):
            raise HTTPException(status_code=401, detail="expert credential binding is unavailable")
        reviewer._flavourbench_verified_credential_id = credential.id
        reviewer._flavourbench_identity_binding_id = binding.id
        return reviewer_rater_pseudonym(binding), reviewer
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    reviewer = session.scalar(
        select(ExpertReviewer).where(
            ExpertReviewer.invitation_sha256 == token_hash,
            ExpertReviewer.active.is_(True),
        )
    )
    if reviewer is None:
        raise HTTPException(status_code=401, detail="expert invitation is invalid or revoked")
    rater = hmac.new(token_hash.encode(), reviewer.id.encode(), hashlib.sha256).hexdigest()
    return rater, reviewer


def _task_contributor_identity(
    session: Session,
    authorization: str,
) -> ExpertReviewer:
    _, contributor = _invited_expert_identity(session, authorization)
    profile = contributor.profile_json
    if not (
        contributor.cohort == "expert_independent"
        and not contributor.qualification_verified
        and profile.get("admission_pathway") == "task_contributor"
        and profile.get("task_contributor_status") == "active"
        and profile.get("raw_identity_retention_prohibited") is True
        and profile.get("person_uniqueness_verified") is True
        and profile.get("person_uniqueness_method") == "admin-witnessed-season-hmac-v1"
        and isinstance(profile.get("person_uniqueness_commitment_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}",
            profile["person_uniqueness_commitment_sha256"],
        )
        and isinstance(profile.get("person_uniqueness_evidence_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}",
            profile["person_uniqueness_evidence_sha256"],
        )
        and _task_contributor_protocol_binding_active(session, contributor)
    ):
        raise HTTPException(status_code=403, detail="task-contributor admission is incomplete")
    return contributor


def _task_contributor_protocol_binding_active(
    session: Session,
    contributor: ExpertReviewer,
) -> bool:
    profile = contributor.profile_json
    if not protocol_binding_active(profile):
        return False
    event_id = str(profile["task_contributor_protocol_acceptance_event_id"])
    event = session.scalar(
        select(RunEvent).where(
            RunEvent.id == event_id,
            RunEvent.entity_type == "task_contributor",
            RunEvent.entity_id == contributor.id,
            RunEvent.event_type == "task_contributor_protocol_accepted",
        )
    )
    if event is None:
        return False
    payload = event.payload_json
    return bool(
        payload.get("protocol_version") == TASK_CONTRIBUTOR_PROTOCOL_VERSION
        and payload.get("protocol_sha256") == TASK_CONTRIBUTOR_PROTOCOL_SHA256
        and payload.get("protocol_scope") == TASK_CONTRIBUTOR_PROTOCOL_SCOPE
        and payload.get("voluntary_participation_accepted") is True
        and payload.get("task_contribution_agreement_accepted") is True
        and payload.get("human_only_methods_acknowledged") is True
    )


def _season_person_uniqueness_commitment(verified_identity_handle: str) -> str:
    normalized = " ".join(verified_identity_handle.strip().casefold().split())
    if not normalized:
        raise HTTPException(status_code=422, detail="verified identity handle is empty")
    message = f"flavourbench-season-person-uniqueness-v1\0{BLUEPRINT_SHA256}\0{normalized}"
    return hmac.new(
        get_settings().task_validator_identity_hmac_secret.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()


def _task_candidate_events(session: Session) -> list[RunEvent]:
    return list(
        session.scalars(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "task_candidate",
                RunEvent.event_type == "task_candidate_submitted",
            )
            .order_by(RunEvent.created_at, RunEvent.id)
        ).all()
    )


def _task_candidate_review_events(session: Session, candidate_id: str) -> list[RunEvent]:
    return list(
        session.scalars(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "task_candidate",
                RunEvent.entity_id == candidate_id,
                RunEvent.event_type.in_(
                    {
                        "task_candidate_review_recorded",
                        "task_candidate_blind_validity_recorded",
                        "task_candidate_reconciliation_recorded",
                        "task_candidate_adjudication_recorded",
                        "task_candidate_withdrawal_recorded",
                    }
                ),
            )
            .order_by(RunEvent.created_at, RunEvent.id)
        ).all()
    )


def _task_candidate_evidence_review_events(
    session: Session,
    candidate_id: str,
) -> list[RunEvent]:
    return list(
        session.scalars(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "task_candidate",
                RunEvent.entity_id == candidate_id,
                RunEvent.event_type.in_(
                    {
                        "task_candidate_validator_contract_verified",
                        "task_candidate_contamination_audit_verified",
                    }
                ),
            )
            .order_by(RunEvent.created_at, RunEvent.id)
        ).all()
    )


def _task_candidate_imported_task(session: Session, candidate_id: str) -> Task | None:
    """Resolve a candidate's immutable task-bank import without dialect-specific JSON SQL."""

    return next(
        (
            task
            for task in session.scalars(select(Task).order_by(Task.created_at, Task.id)).all()
            if task.provenance_json.get("source_candidate_id") == candidate_id
        ),
        None,
    )


def _task_candidate_status(review_events: list[RunEvent]) -> dict[str, Any]:
    blind_by_reviewer: dict[str, RunEvent] = {}
    reconciliation_by_reviewer: dict[str, RunEvent] = {}
    legacy_events: list[RunEvent] = []
    adjudication: RunEvent | None = None
    withdrawals: list[RunEvent] = []
    for event in review_events:
        reviewer_id = str(event.payload_json.get("reviewer_id", ""))
        if event.event_type == "task_candidate_review_recorded":
            legacy_events.append(event)
        elif event.event_type == "task_candidate_withdrawal_recorded":
            withdrawals.append(event)
        elif event.event_type == "task_candidate_adjudication_recorded":
            adjudication = event
        elif reviewer_id and event.event_type == "task_candidate_blind_validity_recorded":
            blind_by_reviewer[reviewer_id] = event
        elif reviewer_id and event.event_type == "task_candidate_reconciliation_recorded":
            reconciliation_by_reviewer[reviewer_id] = event

    complete_reviewers = {
        reviewer_id
        for reviewer_id, blind in blind_by_reviewer.items()
        if blind.payload_json.get("decision") != "valid"
        or reviewer_id in reconciliation_by_reviewer
    }
    source_decisions = Counter(
        str(
            reconciliation_by_reviewer.get(
                reviewer_id, blind_by_reviewer[reviewer_id]
            ).payload_json.get("decision")
        )
        for reviewer_id in complete_reviewers
    )
    reconciliation_approvals = sum(
        event.payload_json.get("decision") == "approve"
        for event in reconciliation_by_reviewer.values()
    )
    withdrawal = withdrawals[-1] if withdrawals else None
    if len(withdrawals) > 1:
        status = "invalid_duplicate_withdrawals"
    elif withdrawal is not None:
        status = "withdrawn"
    elif len(blind_by_reviewer) > 2:
        status = "invalid_excess_source_reviews"
    elif adjudication is not None:
        adjudication_decision = str(adjudication.payload_json.get("decision", "unknown"))
        status = {
            "approve": "approved_for_bank_assembly",
            "revise": "revision_requested",
            "reject": "rejected",
        }.get(adjudication_decision, "invalid_adjudication")
    elif len(complete_reviewers) == 2:
        status = "awaiting_independent_adjudication"
    elif len(blind_by_reviewer) == 2:
        status = "awaiting_independent_reconciliation"
    elif blind_by_reviewer:
        status = "source_review_in_progress"
    elif legacy_events:
        status = "legacy_review_quarantined"
    else:
        status = "awaiting_independent_review"
    return {
        "status": status,
        "reviews": len(complete_reviewers),
        "approvals": reconciliation_approvals,
        "revisionRequests": source_decisions["revise"],
        "rejections": source_decisions["exclude"] + source_decisions["reject"],
        "blindByReviewer": blind_by_reviewer,
        "reconciliationByReviewer": reconciliation_by_reviewer,
        "completeReviewers": complete_reviewers,
        "sourceDecisionCounts": dict(sorted(source_decisions.items())),
        "adjudication": adjudication,
        "withdrawal": withdrawal,
        "withdrawalCount": len(withdrawals),
        "legacyReviewCount": len(legacy_events),
    }


def _reviewer_task_candidate_events(
    review_events: list[RunEvent], reviewer_id: str
) -> dict[str, RunEvent]:
    return {
        event.event_type: event
        for event in review_events
        if event.payload_json.get("reviewer_id") == reviewer_id
    }


def _task_candidate_status_view(state: dict[str, Any]) -> dict[str, Any]:
    """Return only non-identifying, JSON-safe progress fields."""

    return {
        key: state[key]
        for key in (
            "status",
            "reviews",
            "approvals",
            "revisionRequests",
            "rejections",
            "sourceDecisionCounts",
            "withdrawalCount",
            "legacyReviewCount",
        )
    }


def _task_candidate_blind_view(candidate: RunEvent) -> dict[str, Any]:
    payload = candidate.payload_json
    return {
        "candidateId": candidate.entity_id,
        "phase": "blind_validity",
        "prompt": payload.get("prompt"),
        "promptSha256": payload.get("prompt_sha256"),
        "recordSha256": payload.get("candidate_record_sha256"),
        "authorIdentity": None,
        "authorPackVisible": False,
        "modelOutputsVisible": False,
    }


def _task_candidate_reconciliation_view(candidate: RunEvent) -> dict[str, Any]:
    payload = candidate.payload_json
    return {
        "candidateId": candidate.entity_id,
        "phase": "reconciliation",
        "family": payload.get("family"),
        "prompt": payload.get("prompt"),
        "promptSha256": payload.get("prompt_sha256"),
        "constructBlueprintSha256": payload.get("construct_blueprint_sha256"),
        "constructCellId": payload.get("construct_cell_id"),
        "difficultyTier": payload.get("difficulty_tier"),
        "subskills": payload.get("subskills"),
        "explicitConstraints": payload.get("explicit_constraints"),
        "unacceptableOutcomes": payload.get("unacceptable_outcomes"),
        "acceptableSolutionOutline": payload.get("acceptable_solution_outline"),
        "objectiveValidatorPossible": payload.get("objective_validator_possible"),
        "validatorNotes": payload.get("validator_notes"),
        "rightsBasis": payload.get("rights_basis"),
        "recordSha256": payload.get("candidate_record_sha256"),
        "authorIdentity": None,
        "authorPackVisible": True,
        "modelOutputsVisible": False,
    }


def _task_candidate_adjudication_view(
    candidate: RunEvent,
    state: dict[str, Any],
) -> dict[str, Any]:
    reviews: list[dict[str, Any]] = []
    for ordinal, reviewer_id in enumerate(sorted(state["completeReviewers"]), start=1):
        blind = state["blindByReviewer"][reviewer_id]
        reconciliation = state["reconciliationByReviewer"].get(reviewer_id)
        reviews.append(
            {
                "review": ordinal,
                "blindDecision": blind.payload_json.get("decision"),
                "blindChecks": {
                    key: blind.payload_json.get(key)
                    for key in (
                        "construct_fit",
                        "context_complete",
                        "coherent_question",
                        "general_track_scope",
                        "answer_leakage_absent",
                        "discrimination_value",
                    )
                },
                "classification": {
                    "family": blind.payload_json.get("family_classification"),
                    "constructCellId": blind.payload_json.get("construct_cell_classification"),
                    "difficultyTier": blind.payload_json.get("difficulty_tier_classification"),
                },
                "independentSolutionOutline": blind.payload_json.get(
                    "independent_solution_outline", ""
                ),
                "blindSuccessCriteria": blind.payload_json.get("success_criteria", []),
                "blindDisqualifyingErrors": blind.payload_json.get("disqualifying_errors", []),
                "blindIssueTags": blind.payload_json.get("issue_tags", []),
                "blindNote": blind.payload_json.get("note", ""),
                "blindReviewSha256": blind.payload_json.get("blind_review_sha256"),
                "reconciliation": (
                    {
                        "decision": reconciliation.payload_json.get("decision"),
                        "authorPackAdequacy": reconciliation.payload_json.get(
                            "author_pack_adequacy"
                        ),
                        "successCriteria": reconciliation.payload_json.get("success_criteria", []),
                        "permittedVariations": reconciliation.payload_json.get(
                            "permitted_variations", []
                        ),
                        "disqualifyingErrors": reconciliation.payload_json.get(
                            "disqualifying_errors", []
                        ),
                        "objectiveChecks": reconciliation.payload_json.get("objective_checks", []),
                        "issueTags": reconciliation.payload_json.get("issue_tags", []),
                        "note": reconciliation.payload_json.get("note", ""),
                        "reconciliationSha256": reconciliation.payload_json.get(
                            "reconciliation_sha256"
                        ),
                    }
                    if reconciliation is not None
                    else None
                ),
            }
        )
    return {
        **_task_candidate_reconciliation_view(candidate),
        "phase": "adjudication",
        "independentReviews": reviews,
        "modelOutputsVisible": False,
    }


def _development_task_validation_packet() -> dict[str, Any]:
    settings = get_settings()
    configured_path = settings.development_task_validation_packet_path
    expected_sha256 = settings.development_task_validation_packet_sha256
    if not configured_path or not expected_sha256:
        raise HTTPException(status_code=503, detail="development task validation is not configured")
    path = Path(configured_path)
    if path.is_symlink() or not path.is_file():
        raise HTTPException(
            status_code=503, detail="development task validation packet is unavailable"
        )
    try:
        document = json.loads(path.read_bytes())
        if not isinstance(document, dict):
            raise DevelopmentTaskValidationError("validation packet is not an object")
        verify_validation_packet(document)
    except (OSError, json.JSONDecodeError, DevelopmentTaskValidationError) as error:
        raise HTTPException(
            status_code=503,
            detail="development task validation packet failed verification",
        ) from error
    if document.get("artifact_sha256") != expected_sha256:
        raise HTTPException(
            status_code=503, detail="development task validation packet is not pinned"
        )
    return document


def _contamination_scan_bundle() -> ContaminationScanBundle:
    settings = get_settings()
    if not (settings.contamination_scan_bundle_path and settings.contamination_scan_bundle_sha256):
        raise HTTPException(status_code=503, detail="contamination replay corpus is not configured")
    try:
        return load_contamination_scan_bundle(
            settings.contamination_scan_bundle_path,
            expected_sha256=settings.contamination_scan_bundle_sha256,
        )
    except TaskEvidenceError as exc:
        raise HTTPException(
            status_code=503,
            detail="contamination replay corpus failed verification",
        ) from exc


def _validator_calibration() -> tuple[ValidatorCalibrationArtifact, dict[str, object]]:
    settings = get_settings()
    if not (
        settings.validator_calibration_artifact_path
        and settings.validator_calibration_artifact_sha256
    ):
        raise HTTPException(
            status_code=503,
            detail="validator calibration artifact is not configured",
        )
    try:
        return load_validator_calibration(
            settings.validator_calibration_artifact_path,
            expected_sha256=settings.validator_calibration_artifact_sha256,
            expected_container_image_digest=settings.build_image_digest,
        )
    except TaskEvidenceError as exc:
        raise HTTPException(
            status_code=503,
            detail="validator calibration artifact failed verification",
        ) from exc


def _contamination_calibration(
    scan_bundle: ContaminationScanBundle | None = None,
) -> tuple[ContaminationCalibrationArtifact, dict[str, object]]:
    settings = get_settings()
    if not (
        settings.contamination_calibration_artifact_path
        and settings.contamination_calibration_artifact_sha256
    ):
        raise HTTPException(
            status_code=503,
            detail="contamination calibration artifact is not configured",
        )
    scan_bundle = scan_bundle or _contamination_scan_bundle()
    try:
        return load_contamination_calibration(
            settings.contamination_calibration_artifact_path,
            expected_sha256=settings.contamination_calibration_artifact_sha256,
            scan_bundle=scan_bundle,
            expected_container_image_digest=settings.build_image_digest,
        )
    except ContaminationCalibrationError as exc:
        raise HTTPException(
            status_code=503,
            detail="contamination calibration artifact failed verification",
        ) from exc


def _task_validator_identity_commitment(
    verified_identity_handle: str,
    *,
    packet_sha256: str,
) -> str:
    normalized = " ".join(verified_identity_handle.strip().casefold().split())
    if not normalized:
        raise HTTPException(status_code=422, detail="verified identity handle is empty")
    message = f"flavourbench-task-validator-identity-v1\0{packet_sha256}\0{normalized}"
    return hmac.new(
        get_settings().task_validator_identity_hmac_secret.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()


def _development_task_record(packet: dict[str, Any], task_id: str) -> dict[str, Any]:
    task = next((item for item in packet["tasks"] if item.get("task_id") == task_id), None)
    if not isinstance(task, dict):
        raise HTTPException(status_code=404, detail="development validation task not found")
    return task


def _development_task_review_events(session: Session, task_id: str) -> list[RunEvent]:
    return list(
        session.scalars(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "development_task_validation",
                RunEvent.entity_id == task_id,
            )
            .order_by(RunEvent.created_at, RunEvent.id)
        ).all()
    )


def _development_task_reviewer(session: Session, authorization: str) -> ExpertReviewer:
    _, reviewer = _invited_expert_identity(session, authorization)
    profile = reviewer.profile_json
    if (
        not reviewer.qualification_verified
        or reviewer.cohort not in {"expert_independent", "expert_product_affiliated"}
        or profile.get("admission_pathway") != "development_task_validator"
        or profile.get("task_validation_status") != "active"
        or profile.get("evidence_verified_by_admin") is not True
        or not isinstance(profile.get("identity_commitment_sha256"), str)
        or not isinstance(profile.get("qualification_evidence_sha256"), str)
        or not isinstance(profile.get("independence_attestation_sha256"), str)
        or not isinstance(profile.get("verification_record_sha256"), str)
        or profile.get("identity_commitment_algorithm") != "HMAC-SHA256"
        or not _expert_consent_document_active(reviewer)
    ):
        raise HTTPException(
            status_code=403, detail="qualified task-validation admission is required"
        )
    return reviewer


def _verified_independent_task_validator(reviewer: ExpertReviewer) -> bool:
    profile = reviewer.profile_json
    return bool(
        reviewer.cohort == "expert_independent"
        and profile.get("affiliation_class") == "independent_external"
        and profile.get("independent_validation_claim") is True
        and profile.get("evidence_verified_by_admin") is True
        and isinstance(profile.get("identity_commitment_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", profile["identity_commitment_sha256"])
        and isinstance(profile.get("qualification_evidence_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", profile["qualification_evidence_sha256"])
        and isinstance(profile.get("independence_attestation_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", profile["independence_attestation_sha256"])
        and isinstance(profile.get("verification_record_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", profile["verification_record_sha256"])
        and profile.get("identity_commitment_algorithm") == "HMAC-SHA256"
        and profile.get("person_uniqueness_verified") is True
        and profile.get("person_uniqueness_method") == "admin-witnessed-season-hmac-v1"
        and isinstance(profile.get("person_uniqueness_commitment_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}",
            profile["person_uniqueness_commitment_sha256"],
        )
    )


def _development_task_adjudicator(session: Session, authorization: str) -> ExpertReviewer:
    _, reviewer = _invited_expert_identity(session, authorization)
    profile = reviewer.profile_json
    if not (
        reviewer.qualification_verified
        and reviewer.cohort == "expert_independent"
        and profile.get("admission_pathway") == "development_task_validator"
        and profile.get("task_validation_status") == "active"
        and profile.get("task_adjudication_authorized") is True
        and _verified_independent_task_validator(reviewer)
        and _expert_consent_document_active(reviewer)
    ):
        raise HTTPException(
            status_code=403, detail="independent task-adjudicator admission is required"
        )
    return reviewer


def _reviewer_development_task_events(
    events: list[RunEvent], reviewer_id: str
) -> dict[str, RunEvent]:
    return {
        event.event_type: event
        for event in events
        if event.payload_json.get("reviewer_id") == reviewer_id
    }


def _development_task_review_state(
    events: list[RunEvent],
    *,
    packet_sha256: str,
) -> dict[str, Any]:
    blind_by_reviewer: dict[str, RunEvent] = {}
    criteria_by_reviewer: dict[str, RunEvent] = {}
    adjudication: RunEvent | None = None
    for event in events:
        if event.payload_json.get("packet_sha256") != packet_sha256:
            continue
        reviewer_id = str(event.payload_json.get("reviewer_id", ""))
        if event.event_type == "development_task_adjudication_recorded":
            adjudication = event
        elif not event.payload_json.get("independent_review") or not reviewer_id:
            continue
        elif event.event_type == "development_task_blind_validity_recorded":
            blind_by_reviewer[reviewer_id] = event
        elif event.event_type == "development_task_criteria_recorded":
            criteria_by_reviewer[reviewer_id] = event

    complete_reviewers = {
        reviewer_id
        for reviewer_id, blind in blind_by_reviewer.items()
        if blind.payload_json.get("decision") != "valid" or reviewer_id in criteria_by_reviewer
    }
    decisions = Counter(
        str(blind_by_reviewer[reviewer_id].payload_json.get("decision"))
        for reviewer_id in complete_reviewers
    )
    if len(blind_by_reviewer) > REQUIRED_INDEPENDENT_REVIEWERS:
        status = "invalid_excess_source_reviews"
    elif adjudication is not None:
        status = f"adjudicated_{adjudication.payload_json.get('decision', 'unknown')}"
    elif len(complete_reviewers) == REQUIRED_INDEPENDENT_REVIEWERS:
        if decisions == Counter({"valid": REQUIRED_INDEPENDENT_REVIEWERS}):
            status = "validated_unanimous"
        else:
            status = "awaiting_independent_adjudication"
    elif len(blind_by_reviewer) == REQUIRED_INDEPENDENT_REVIEWERS:
        status = "awaiting_independent_criteria"
    else:
        status = "awaiting_independent_review"
    return {
        "status": status,
        "blind_by_reviewer": blind_by_reviewer,
        "criteria_by_reviewer": criteria_by_reviewer,
        "complete_reviewers": complete_reviewers,
        "decision_counts": dict(sorted(decisions.items())),
        "adjudication": adjudication,
    }


def _development_task_consensus_sha256(
    *,
    packet_sha256: str,
    task: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str | None:
    status = str(state["status"])
    if status != "validated_unanimous" and not status.startswith("adjudicated_"):
        return None
    review_sha256s = sorted(
        state["blind_by_reviewer"][reviewer_id].payload_json["review_sha256"]
        for reviewer_id in state["complete_reviewers"]
    )
    criteria_sha256s = sorted(
        event.payload_json["criteria_sha256"]
        for reviewer_id, event in state["criteria_by_reviewer"].items()
        if reviewer_id in state["complete_reviewers"]
    )
    adjudication = state["adjudication"]
    return _canonical_sha256(
        {
            "schema_version": "flavourbench-development-task-consensus-v1",
            "packet_sha256": packet_sha256,
            "task_id": task["task_id"],
            "task_sha256": task["task_sha256"],
            "prompt_sha256": task["prompt_sha256"],
            "status": status,
            "source_review_sha256s": review_sha256s,
            "source_criteria_sha256s": criteria_sha256s,
            "adjudication_sha256": (
                adjudication.payload_json.get("adjudication_sha256")
                if adjudication is not None
                else None
            ),
        }
    )


def _development_task_adjudication_view(
    task: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    reviews: list[dict[str, Any]] = []
    for ordinal, reviewer_id in enumerate(sorted(state["complete_reviewers"]), start=1):
        blind = state["blind_by_reviewer"][reviewer_id]
        criteria = state["criteria_by_reviewer"].get(reviewer_id)
        reviews.append(
            {
                "review": ordinal,
                "decision": blind.payload_json["decision"],
                "checks": {
                    key: blind.payload_json[key]
                    for key in (
                        "construct_fit",
                        "context_complete",
                        "coherent_question",
                        "general_track_scope",
                        "answer_leakage_absent",
                        "discrimination_value",
                    )
                },
                "issueTags": blind.payload_json.get("issue_tags", []),
                "note": blind.payload_json.get("note", ""),
                "reviewSha256": blind.payload_json["review_sha256"],
                "criterionPack": (
                    {
                        "referenceAdequacy": criteria.payload_json["reference_adequacy"],
                        "successCriteria": criteria.payload_json["success_criteria"],
                        "permittedVariations": criteria.payload_json["permitted_variations"],
                        "disqualifyingErrors": criteria.payload_json["disqualifying_errors"],
                        "objectiveChecks": criteria.payload_json.get("objective_checks", []),
                        "note": criteria.payload_json.get("note", ""),
                        "criteriaSha256": criteria.payload_json["criteria_sha256"],
                    }
                    if criteria is not None
                    else None
                ),
            }
        )
    reference = task["sealed_human_reference_stage"]
    return {
        "taskId": task["task_id"],
        "family": task["family"],
        "prompt": task["prompt"],
        "promptSha256": task["prompt_sha256"],
        "taskSha256": task["task_sha256"],
        "sourceUrl": reference["source_url"],
        "sourceLicense": reference["source_license"],
        "sourceAuthor": reference["source_author"],
        "humanReference": {
            "text": reference["reference_text"],
            "sha256": reference["reference_text_sha256"],
            "url": reference["reference_url"],
            "license": reference["reference_license"],
            "author": reference["reference_author"],
            "use": reference["reference_use"],
        },
        "independentReviews": reviews,
        "modelOutputsVisible": False,
    }


def _development_task_review_progress(
    session: Session,
    *,
    packet: dict[str, Any],
    reviewer: ExpertReviewer,
) -> dict[str, int]:
    qualified = [
        task for task in packet["tasks"] if task.get("family") in reviewer.qualification_json
    ]
    blind = 0
    criteria = 0
    for task in qualified:
        own = _reviewer_development_task_events(
            _development_task_review_events(session, str(task["task_id"])), reviewer.id
        )
        blind += int("development_task_blind_validity_recorded" in own)
        criteria += int("development_task_criteria_recorded" in own)
    return {"eligible": len(qualified), "blindDecisions": blind, "criterionPacks": criteria}


def _development_task_assignment_tiebreak(
    *,
    packet_sha256: str,
    reviewer_id: str,
    task_id: str,
) -> str:
    """Return a stable reviewer-specific tie break without exposing task order."""

    return hmac.new(
        packet_sha256.encode(),
        f"{reviewer_id}:{task_id}".encode(),
        hashlib.sha256,
    ).hexdigest()


def _development_task_blind_assignment_key(
    *,
    packet_sha256: str,
    reviewer_id: str,
    task_id: str,
    complete_independent_reviews: int,
) -> tuple[int, str]:
    """Prioritize least-reviewed tasks, then disperse reviewer-specific ties."""

    return (
        complete_independent_reviews,
        _development_task_assignment_tiebreak(
            packet_sha256=packet_sha256,
            reviewer_id=reviewer_id,
            task_id=task_id,
        ),
    )


def _author_evaluator_profile(reviewer: ExpertReviewer) -> bool:
    profile = reviewer.profile_json
    candidate_pack_sha256 = profile.get("author_evaluator_pool_sha256")
    return bool(
        reviewer.qualification_verified
        and reviewer.cohort == "expert_product_affiliated"
        and profile.get("admission_pathway") == "author_evaluator"
        and profile.get("author_evaluator_admission_status") == "active"
        and profile.get("independent_validation_claim") is False
        and isinstance(candidate_pack_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", candidate_pack_sha256)
    )


def _expert_consent_document_active(reviewer: ExpertReviewer) -> bool:
    consent_sha256 = reviewer.profile_json.get("consent_document_sha256")
    return resolve_expert_consent_document(consent_sha256).status == "active"


def _author_evaluator_active(reviewer: ExpertReviewer) -> bool:
    return _author_evaluator_profile(reviewer) and _expert_consent_document_active(reviewer)


def _anonymous_external_rater_active(session: Session, reviewer: ExpertReviewer) -> bool:
    """Return whether a no-PII external rater may use the isolated pilot pool."""

    return bool(
        _anonymous_external_rater_profile(reviewer)
        and _expert_consent_document_active(reviewer)
        and anonymous_pool_reconsented(session, reviewer)
    )


def _anonymous_external_rater_profile(reviewer: ExpertReviewer) -> bool:
    """Recognize the pathway even when consent has been suspended or superseded."""

    profile = reviewer.profile_json
    candidate_pack_sha256 = profile.get("anonymous_external_pool_sha256")
    return bool(
        not reviewer.qualification_verified
        and reviewer.cohort == "expert_independent"
        and profile.get("admission_pathway") == "anonymous_external_rater"
        and profile.get("anonymous_external_admission_status") == "active"
        and profile.get("identity_collection_prohibited") is True
        and profile.get("independence_basis") == "reviewer_self_attestation"
        and profile.get("qualification_basis") == "reviewer_self_attestation_unverified"
        and profile.get("independent_expert_validation_claim") is False
        and isinstance(candidate_pack_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", candidate_pack_sha256)
    )


def _isolated_pilot_reviewer_active(session: Session, reviewer: ExpertReviewer) -> bool:
    return _author_evaluator_active(reviewer) or _anonymous_external_rater_active(session, reviewer)


def _isolated_pilot_pool_sha256(session: Session, reviewer: ExpertReviewer) -> str | None:
    if _author_evaluator_active(reviewer):
        return str(reviewer.profile_json["author_evaluator_pool_sha256"])
    if _anonymous_external_rater_active(session, reviewer):
        return str(reviewer.profile_json["anonymous_external_pool_sha256"])
    return None


def _reviewer_admission_pathway(reviewer: ExpertReviewer) -> str:
    if _author_evaluator_profile(reviewer):
        return "author_evaluator"
    if _anonymous_external_rater_profile(reviewer):
        return "anonymous_external_rater"
    return "calibrated_expert"


def _reviewer_claim_boundary(reviewer: ExpertReviewer) -> str:
    if _author_evaluator_profile(reviewer):
        return (
            "Blinded judgments from the disclosed author-evaluator. They form a "
            "separately reported single-rater case study, not independent expert "
            "validation."
        )
    if _anonymous_external_rater_profile(reviewer):
        return (
            "Blinded judgments from one pseudonymous external rater whose independence "
            "and culinary competence are self-attested and whose identity is not "
            "collected. Report separately; this is not verified expert consensus."
        )
    if reviewer.cohort == "expert_independent":
        return (
            "Blinded judgments from a qualification-verified independent expert under "
            "the governed calibration protocol."
        )
    return (
        "Blinded judgments from a disclosed product-affiliated expert cohort. This "
        "cohort does not constitute independent expert validation."
    )


def _reviewer_acknowledgement_statements(
    reviewer: ExpertReviewer,
) -> dict[str, str]:
    conflict_statement = (
        "I am not an author or developer of Epicure or FlavourBench and have no "
        "undisclosed personal, financial, employment, or provider conflict."
        if _anonymous_external_rater_profile(reviewer)
        else (
            "My relationship with Epicure and FlavourBench and any relevant interests "
            "are accurately disclosed in the reviewer record."
        )
    )
    return {
        "conflict_disclosed": conflict_statement,
        "culinary_competence": (
            "I have relevant culinary knowledge or practice for the listed task "
            "families. This is a self-attestation, not a verified credential."
            if _anonymous_external_rater_profile(reviewer)
            else (
                "I have relevant culinary knowledge or practice for the listed task "
                "families and will pause if a task falls outside my competence."
            )
        ),
        "identity_blinding": (
            "I will judge the answers without seeking model, provider, or tool identity."
        ),
        "no_external_model_identification": (
            "I will not use external services to identify an answer."
        ),
        "no_active_batch_discussion": (
            "I will not discuss active assignments before the batch is closed."
        ),
        "voluntary_participation": (
            "I understand that participation is voluntary and I may pause at any time."
        ),
        "sealed_prompt_confidentiality": (
            "I will keep controlled prompts and answers confidential."
        ),
    }


def _reviewer_target_judgments(session: Session, reviewer: ExpertReviewer) -> int:
    if not _isolated_pilot_reviewer_active(session, reviewer):
        return int(protocol_payload()["workload"]["total_presentations"])
    primary_field = (
        "author_evaluator_primary_judgments"
        if _author_evaluator_active(reviewer)
        else "anonymous_external_primary_judgments"
    )
    primary = int(reviewer.profile_json.get(primary_field, 0))
    family_field = (
        "author_evaluator_primary_by_family"
        if _author_evaluator_active(reviewer)
        else "anonymous_external_primary_by_family"
    )
    primary_by_family = reviewer.profile_json.get(family_field)
    targets = (
        isolated_uplift_workload_cell_targets(primary_by_family)
        if isinstance(primary_by_family, dict)
        else author_evaluator_workload_cell_targets(primary)
    )
    return int(targets["total_presentations"])


def _reviewer_workload_cell_targets(
    session: Session,
    reviewer: ExpertReviewer,
    target_judgments: int,
) -> dict[str, Any]:
    if not _isolated_pilot_reviewer_active(session, reviewer):
        return workload_cell_targets(target_judgments)
    expected = _reviewer_target_judgments(session, reviewer)
    if target_judgments != expected:
        pathway_label = (
            "author-evaluator" if _author_evaluator_active(reviewer) else "anonymous external-rater"
        )
        raise HTTPException(
            status_code=409,
            detail=f"{pathway_label} sessions require exactly {expected} presentations",
        )
    primary_field = (
        "author_evaluator_primary_judgments"
        if _author_evaluator_active(reviewer)
        else "anonymous_external_primary_judgments"
    )
    primary = int(reviewer.profile_json[primary_field])
    family_field = (
        "author_evaluator_primary_by_family"
        if _author_evaluator_active(reviewer)
        else "anonymous_external_primary_by_family"
    )
    primary_by_family = reviewer.profile_json.get(family_field)
    if isinstance(primary_by_family, dict):
        return isolated_uplift_workload_cell_targets(primary_by_family)
    return author_evaluator_workload_cell_targets(primary)


def _expert_identity(session: Session, authorization: str) -> tuple[str, ExpertReviewer]:
    rater, reviewer = _invited_expert_identity(session, authorization)
    if not _expert_consent_document_active(reviewer):
        raise HTTPException(
            status_code=403,
            detail="review is paused until the bound consent document is active",
        )
    if getattr(reviewer, "_flavourbench_verified_credential_id", None) is not None:
        return rater, reviewer
    if _isolated_pilot_reviewer_active(session, reviewer):
        return rater, reviewer
    if _anonymous_external_rater_profile(reviewer):
        raise HTTPException(
            status_code=403,
            detail="review is paused until an active consent document and re-consent are recorded",
        )
    if not _calibrated_expert_admission_active(session, reviewer):
        raise HTTPException(
            status_code=403,
            detail="expert onboarding and calibration are incomplete",
        )
    return rater, reviewer


def _lock_expert_review_session(
    session: Session,
    *,
    review_session_id: str,
    reviewer_id: str,
) -> None:
    lock_material = f"expert-review-session:{review_session_id}:{reviewer_id}"
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        lock_key = int(hashlib.sha256(lock_material.encode()).hexdigest()[:16], 16)
        if lock_key >= 2**63:
            lock_key -= 2**64
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )
        return
    session.scalar(
        select(ExpertReviewer.id).where(ExpertReviewer.id == reviewer_id).with_for_update()
    )


def _controlled_reviewer_authorization(
    session: Session, reviewer: ExpertReviewer, controlled_run_id: str
) -> ControlledRunReviewer | None:
    return session.scalar(
        select(ControlledRunReviewer)
        .join(
            ControlledRun,
            ControlledRun.id == ControlledRunReviewer.controlled_run_id,
        )
        .where(
            ControlledRunReviewer.controlled_run_id == controlled_run_id,
            ControlledRunReviewer.reviewer_id == reviewer.id,
            ControlledRunReviewer.active.is_(True),
            ControlledRun.status.in_({"active", "collection_complete"}),
        )
    )


def _expert_session_event(
    session: Session,
    reviewer: ExpertReviewer,
    review_session_id: str,
) -> RunEvent:
    event = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "expert_review_session",
            RunEvent.entity_id == review_session_id,
            RunEvent.event_type == "expert_review_session_opened",
        )
    )
    if (
        event is None
        or event.payload_json.get("reviewer_id") != reviewer.id
        or event.payload_json.get("protocol_sha256") != EXPERT_PROTOCOL_SHA256
    ):
        raise HTTPException(status_code=404, detail="expert review session not found")
    closed = session.scalar(
        select(RunEvent.id).where(
            RunEvent.entity_type == "expert_review_session",
            RunEvent.entity_id == review_session_id,
            RunEvent.event_type.in_(
                {"expert_review_session_closed", "expert_review_session_withdrawn"}
            ),
        )
    )
    if closed is not None:
        raise HTTPException(status_code=409, detail="expert review session is closed")
    return event


def _expert_assignment_events(
    session: Session,
    review_session_id: str,
) -> tuple[dict[str, RunEvent], dict[str, RunEvent], dict[str, RunEvent]]:
    events = session.scalars(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_review_assignment",
            RunEvent.event_type.in_(
                {
                    "expert_review_assignment_opened",
                    "expert_review_task_assessed",
                    "expert_review_assignment_submitted",
                }
            ),
            RunEvent.payload_json["review_session_id"].as_string() == review_session_id,
        )
        .order_by(RunEvent.created_at, RunEvent.id)
    ).all()
    opened: dict[str, RunEvent] = {}
    assessed: dict[str, RunEvent] = {}
    submitted: dict[str, RunEvent] = {}
    for event in events:
        if event.payload_json.get("review_session_id") != review_session_id:
            continue
        if event.event_type == "expert_review_assignment_opened":
            target = opened
        elif event.event_type == "expert_review_task_assessed":
            target = assessed
        else:
            target = submitted
        target[event.entity_id] = event
    return opened, assessed, submitted


def _expert_reviewer_submitted_events(
    session: Session,
    reviewer_id: str,
) -> dict[str, RunEvent]:
    events = session.scalars(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_review_assignment",
            RunEvent.event_type == "expert_review_assignment_submitted",
            RunEvent.payload_json["reviewer_id"].as_string() == reviewer_id,
        )
        .order_by(RunEvent.created_at, RunEvent.id)
    ).all()
    return {event.entity_id: event for event in events}


def _qualified_expert_battles(
    session: Session,
    *,
    reviewer: ExpertReviewer,
    rater: str,
    controlled_run_id: str | None,
) -> list[Battle]:
    families = set(reviewer.qualification_json)
    voted = select(Vote.battle_id).where(
        Vote.rater_pseudonym == rater,
        Vote.cohort == reviewer.cohort,
    )
    query = select(Battle).where(
        Battle.status == "complete",
        Battle.category.in_(families),
        Battle.id.not_in(voted),
    )
    if controlled_run_id is None:
        if _isolated_pilot_reviewer_active(session, reviewer):
            query = query.where(
                Battle.data_stratum == "development",
                Battle.run_class == "pilot",
                Battle.rank_eligible.is_(False),
                Battle.manifest_sha256 == _isolated_pilot_pool_sha256(session, reviewer),
                Battle.controlled_run_id.is_(None),
            )
        else:
            query = query.where(
                Battle.data_stratum == "public_freeform",
                Battle.controlled_run_id.is_(None),
            )
    else:
        if _controlled_reviewer_authorization(session, reviewer, controlled_run_id) is None:
            raise HTTPException(status_code=404, detail="qualified assignment not found")
        query = query.where(
            Battle.data_stratum == "controlled",
            Battle.controlled_run_id == controlled_run_id,
        )
    if get_settings().execution_mode != "mock" and not _isolated_pilot_reviewer_active(
        session, reviewer
    ):
        query = query.where(
            Battle.run_class == "official",
            Battle.rank_eligible.is_(True),
        )
    candidates = list(session.scalars(query).all())
    reviewable_ids = _reviewable_battle_ids(session, {item.id for item in candidates})
    return [item for item in candidates if item.id in reviewable_ids]


def _reviewable_battle_ids(session: Session, battle_ids: set[str]) -> set[str]:
    """Return battles with exactly two complete, non-truncated response arms."""

    if not battle_ids:
        return set()
    rows = session.scalars(select(ResponseArm).where(ResponseArm.battle_id.in_(battle_ids))).all()
    by_battle: dict[str, list[ResponseArm]] = {}
    for arm in rows:
        by_battle.setdefault(arm.battle_id, []).append(arm)
    return {
        battle_id
        for battle_id, arms in by_battle.items()
        if {arm.side for arm in arms} == {"left", "right"}
        and len(arms) == 2
        and all(
            arm.status == "complete"
            and bool(arm.answer_markdown)
            and is_complete_finish_reason(arm.finish_reason)
            for arm in arms
        )
    }


def _battle_model_ids(
    session: Session,
    battle_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    if not battle_ids:
        return {}
    rows = session.execute(
        select(ResponseArm.battle_id, ResponseArm.model_id)
        .where(ResponseArm.battle_id.in_(battle_ids))
        .order_by(ResponseArm.battle_id, ResponseArm.side)
    ).all()
    grouped: dict[str, list[str]] = {}
    for battle_id, model_id in rows:
        models = grouped.setdefault(str(battle_id), [])
        if model_id not in models:
            models.append(model_id)
    return {battle_id: tuple(model_ids) for battle_id, model_ids in grouped.items()}


def _balanced_integer_targets(
    total: int,
    labels: set[str],
) -> dict[str, int]:
    ordered = sorted(labels)
    if not ordered:
        return {}
    quotient, remainder = divmod(total, len(ordered))
    return {
        label: quotient + (1 if index < remainder else 0) for index, label in enumerate(ordered)
    }


def _comparison_component_sizes(
    pairs: list[tuple[str, str]],
) -> list[int]:
    adjacency: dict[str, set[str]] = {}
    for left, right in pairs:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    unseen = set(adjacency)
    sizes: list[int] = []
    while unseen:
        frontier = {unseen.pop()}
        component: set[str] = set()
        while frontier:
            node = frontier.pop()
            if node in component:
                continue
            component.add(node)
            frontier.update(adjacency.get(node, set()) - component)
        unseen.difference_update(component)
        sizes.append(len(component))
    return sorted(sizes, reverse=True)


def _comparison_component_labels(
    pairs: list[tuple[str, str]],
) -> dict[str, int]:
    adjacency: dict[str, set[str]] = {}
    for left, right in pairs:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    labels: dict[str, int] = {}
    for node in sorted(adjacency):
        if node in labels:
            continue
        label = len(set(labels.values()))
        frontier = {node}
        while frontier:
            current = frontier.pop()
            if current in labels:
                continue
            labels[current] = label
            frontier.update(adjacency.get(current, set()) - set(labels))
    return labels


def _review_presentation(
    session: Session,
    battle: Battle,
    presented_side_map: dict[str, str],
) -> tuple[dict[str, Any], str]:
    arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
    by_side = {arm.side: arm for arm in arms}
    if set(by_side) != {"left", "right"} or any(
        arm.status != "complete"
        or not arm.answer_markdown
        or not is_complete_finish_reason(arm.finish_reason)
        for arm in arms
    ):
        raise HTTPException(
            status_code=409,
            detail="expert assignment contains an incomplete final response and is not voteable",
        )
    answer_hashes = {
        side: by_side[side].answer_markdown_sha256
        or hashlib.sha256((by_side[side].answer_markdown or "").encode()).hexdigest()
        for side in ("left", "right")
    }
    digest = presentation_sha256(
        battle_id=battle.id,
        prompt_sha256=battle.prompt_sha256,
        answer_sha256_by_canonical_side=answer_hashes,
        presented_side_map=presented_side_map,
    )
    payload = {
        "battleId": battle.id,
        "status": battle.status,
        "category": battle.category,
        "prompt": battle.prompt,
        "answers": [
            {
                "side": presented_side,
                "answerMarkdown": by_side[canonical_side].answer_markdown or "",
            }
            for presented_side, canonical_side in (
                ("left", presented_side_map["left"]),
                ("right", presented_side_map["right"]),
            )
        ],
        "reveal": None,
        "error": None,
    }
    return payload, digest


def _review_task_criterion_pack(session: Session, battle: Battle) -> dict[str, Any] | None:
    if battle.task_id is None:
        if battle.rank_eligible:
            raise HTTPException(
                status_code=409,
                detail="rank-eligible expert assignment is not bound to a frozen task",
            )
        return None
    task = session.get(Task, battle.task_id)
    if task is None:
        raise HTTPException(status_code=409, detail="expert assignment task is unavailable")
    provenance = task.provenance_json if isinstance(task.provenance_json, dict) else {}
    criterion_pack = provenance.get("criterion_pack")
    criterion_pack_sha256 = provenance.get("criterion_pack_sha256")
    if not isinstance(criterion_pack, dict) or not isinstance(criterion_pack_sha256, str):
        if battle.rank_eligible:
            raise HTTPException(
                status_code=409,
                detail="rank-eligible task lacks an independently adjudicated criterion pack",
            )
        return None
    if not hmac.compare_digest(_canonical_sha256(criterion_pack), criterion_pack_sha256):
        raise HTTPException(status_code=409, detail="task criterion pack failed hash verification")
    return {
        "family": criterion_pack.get("family"),
        "constructCellId": criterion_pack.get("construct_cell_id"),
        "difficultyTier": criterion_pack.get("difficulty_tier"),
        "successCriteria": criterion_pack.get("success_criteria", []),
        "permittedVariations": criterion_pack.get("permitted_variations", []),
        "disqualifyingErrors": criterion_pack.get("disqualifying_errors", []),
        "objectiveChecks": criterion_pack.get("objective_checks", []),
        "criterionPackSha256": criterion_pack_sha256,
        "source": "independent_answer_blind_adjudication",
    }


def _assignment_response(
    session: Session,
    opened: RunEvent,
    task_assessment: RunEvent | None = None,
) -> dict[str, Any]:
    battle = session.get(Battle, opened.payload_json.get("battle_id"))
    if battle is None:
        raise HTTPException(status_code=409, detail="expert assignment battle is unavailable")
    side_map = dict(opened.payload_json.get("presented_side_map", {}))
    assignment, digest = _review_presentation(session, battle, side_map)
    if not hmac.compare_digest(
        str(opened.payload_json.get("presentation_sha256", "")),
        digest,
    ):
        raise HTTPException(status_code=409, detail="expert assignment presentation drifted")
    assignment.update(
        {
            "reviewAssignmentId": opened.entity_id,
            "reviewSessionId": opened.payload_json["review_session_id"],
            "protocolVersion": EXPERT_PROTOCOL_VERSION,
            "protocolSha256": EXPERT_PROTOCOL_SHA256,
            "stage": "response_review" if task_assessment is not None else "task_assessment",
            "taskAssessment": (
                dict(task_assessment.payload_json.get("assessment", {}))
                if task_assessment is not None
                else None
            ),
            "taskCriterionPack": _review_task_criterion_pack(session, battle),
        }
    )
    if task_assessment is None:
        assignment["answers"] = []
    return {"assignment": assignment}


def _submitted_review_payloads(
    submitted: dict[str, RunEvent],
) -> list[dict[str, Any]]:
    return [
        event.payload_json
        for event in sorted(submitted.values(), key=lambda item: (item.created_at, item.id))
    ]


def _review_fatigue_status(
    submitted: dict[str, RunEvent],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    timestamps: list[datetime] = []
    for event in submitted.values():
        created_at = event.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        timestamps.append(created_at.astimezone(UTC))
    timestamps.sort()
    active_window = [
        timestamp for timestamp in timestamps if timestamp > current - timedelta(hours=24)
    ]
    rolling_count = len(active_window)
    daily_limit = int(WORKLOAD_TARGET["recommended_daily_limit"])
    daily_reset_at: datetime | None = None
    if rolling_count >= daily_limit:
        daily_reset_at = active_window[-daily_limit] + timedelta(hours=24)
    continuous_minutes = 0.0
    if timestamps and current - timestamps[-1] <= timedelta(minutes=15):
        block_start = timestamps[-1]
        for timestamp in reversed(timestamps[:-1]):
            if block_start - timestamp > timedelta(minutes=15):
                break
            block_start = timestamp
        continuous_minutes = max(
            0.0,
            (current - block_start).total_seconds() / 60,
        )
    return {
        "presentationsLast24Hours": rolling_count,
        "rollingDailyLimit": daily_limit,
        "dailyLimitResetsAt": (
            daily_reset_at.astimezone(UTC).isoformat() if daily_reset_at is not None else None
        ),
        "secondsUntilDailyReset": (
            max(0, math.ceil((daily_reset_at - current).total_seconds()))
            if daily_reset_at is not None
            else None
        ),
        "continuousBlockMinutes": round(continuous_minutes, 1),
        "breakRequired": continuous_minutes >= 60,
        "dailyLimitReached": rolling_count >= daily_limit,
        "minimumBreakMinutes": 15,
    }


@router.get("/expert/onboarding")
def expert_onboarding(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    _, reviewer = _invited_expert_identity(session, authorization)
    profile = reviewer.profile_json
    consent_document = resolve_expert_consent_document(profile.get("consent_document_sha256"))
    calibration_candidate = profile.get("calibration_candidate")
    if not isinstance(calibration_candidate, dict):
        calibration_candidate = {}
    calibration_accuracy = profile.get("calibration_accuracy")
    calibration_passed = bool(
        isinstance(calibration_accuracy, (int, float)) and calibration_accuracy >= 0.8
    )
    author_evaluator_profile = _author_evaluator_profile(reviewer)
    author_evaluator = _author_evaluator_active(reviewer)
    anonymous_external_profile = _anonymous_external_rater_profile(reviewer)
    anonymous_external = _anonymous_external_rater_active(session, reviewer)
    pathway_bypass = author_evaluator or anonymous_external
    isolated_pathway = author_evaluator_profile or anonymous_external_profile
    calibrated_expert = _calibrated_expert_admission_active(session, reviewer)
    review_enabled = bool(
        consent_document.status == "active" and (pathway_bypass or calibrated_expert)
    )
    blockers = []
    if not _expert_consent_document_active(reviewer):
        blockers.append("active expert-consent document")
    elif anonymous_external_profile and not anonymous_external:
        blockers.append("pool-specific anonymous reviewer re-consent")
    elif not reviewer.qualification_verified and not anonymous_external_profile:
        blockers.append("qualification, consent, training, and admission approval")
    if not isolated_pathway:
        if not calibration_candidate:
            blockers.append("registered real-output calibration candidate pool")
        elif profile.get("calibration_gold_adjudicator_count", 0) < 2:
            blockers.append("two independent gold ballots and disagreement adjudication")
        if not calibration_passed:
            blockers.append("expert calibration score of at least 0.80")
    return {
        "reviewer": {
            "reviewerCode": reviewer.reviewer_code,
            "cohort": reviewer.cohort,
            "qualifiedFamilies": reviewer.qualification_json,
            "qualificationVerified": reviewer.qualification_verified,
            "qualificationBasis": profile.get("qualification_basis"),
            "calibrationAccuracy": None if isolated_pathway else calibration_accuracy,
            "affiliationClass": profile.get("affiliation_class"),
            "admissionPathway": _reviewer_admission_pathway(reviewer),
        },
        "evidenceReferences": {
            "qualification": profile.get("qualification_reference"),
            "conflictDisclosure": profile.get("conflict_disclosure_reference"),
            "consentDocumentSha256": profile.get("consent_document_sha256"),
            "consentDocumentStatus": consent_document.status,
            "trainingMaterialSha256": profile.get("training_material_sha256"),
            "calibrationSetSha256": profile.get("calibration_set_sha256"),
            "compensation": profile.get("compensation_reference"),
        },
        "calibration": {
            "status": calibration_candidate.get("status", "not_registered"),
            "candidatePackSha256": calibration_candidate.get("candidate_pack_sha256"),
            "candidatePairs": calibration_candidate.get("candidate_pairs", 0),
            "candidatePairsByFamily": calibration_candidate.get(
                "candidate_pairs_by_family",
                {},
            ),
            "sourceArms": calibration_candidate.get("source_arms", 0),
            "realProviderCalls": calibration_candidate.get(
                "real_provider_calls",
                0,
            ),
            "realEpicureCalls": calibration_candidate.get(
                "real_epicure_calls",
                0,
            ),
            "successfulRealEpicureCalls": calibration_candidate.get(
                "successful_real_epicure_calls",
                0,
            ),
            "syntheticArms": calibration_candidate.get("synthetic_arms"),
            "independentGoldBallots": profile.get(
                "calibration_gold_adjudicator_count",
                0,
            ),
            "frozenItems": profile.get("calibration_item_count", 0),
            "requiredForAdmission": not isolated_pathway,
        },
        "admission": {
            "status": "active" if review_enabled else "pending",
            "reviewEnabled": review_enabled,
            "blockers": blockers,
            "pathway": _reviewer_admission_pathway(reviewer),
            "targetJudgments": (
                _reviewer_target_judgments(session, reviewer) if review_enabled else None
            ),
        },
        "reconsent": {
            "required": anonymous_external_profile,
            "recordedForCurrentPool": anonymous_external,
            "candidatePackSha256": profile.get("anonymous_external_pool_sha256"),
            "poolActivationSha256": profile.get("anonymous_external_pool_activation_sha256"),
            "consentDocumentSha256": profile.get("consent_document_sha256"),
            "consentDocumentStatus": consent_document.status,
            "consentDocumentText": consent_document.text,
            "statement": RECONSENT_STATEMENT if anonymous_external_profile else None,
            "statementSha256": (RECONSENT_STATEMENT_SHA256 if anonymous_external_profile else None),
        },
        "claimBoundary": _reviewer_claim_boundary(reviewer),
    }


@router.post("/expert/anonymous-external-reconsent")
def record_anonymous_external_reconsent(
    request: AnonymousExternalReconsentCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    """Append reviewer-authenticated consent for one exact pool activation."""

    _, invited = _invited_expert_identity(session, authorization)
    reviewer = reviewer_control_lock(session, invited.id)
    if reviewer is None or not reviewer.active:
        raise HTTPException(status_code=401, detail="expert invitation is invalid or revoked")
    if not _anonymous_external_rater_profile(reviewer):
        raise HTTPException(
            status_code=409,
            detail="anonymous external-rater admission is incomplete",
        )
    if not _expert_consent_document_active(reviewer):
        raise HTTPException(
            status_code=409,
            detail="the bound expert-consent document is not active",
        )
    profile = reviewer.profile_json
    expected = {
        "candidate_pack_sha256": profile.get("anonymous_external_pool_sha256"),
        "pool_activation_sha256": profile.get("anonymous_external_pool_activation_sha256"),
        "consent_document_sha256": profile.get("consent_document_sha256"),
        "consent_statement_sha256": RECONSENT_STATEMENT_SHA256,
    }
    observed = {
        "candidate_pack_sha256": request.candidate_pack_sha256,
        "pool_activation_sha256": request.pool_activation_sha256,
        "consent_document_sha256": request.consent_document_sha256,
        "consent_statement_sha256": request.consent_statement_sha256,
    }
    if observed != expected:
        raise HTTPException(
            status_code=409,
            detail="re-consent does not match the current pool activation and consent document",
        )
    try:
        event, appended = append_pool_reconsent(session, reviewer)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    return {
        "recorded": True,
        "idempotent": not appended,
        "eventId": event.id,
        "candidatePackSha256": request.candidate_pack_sha256,
        "poolActivationSha256": request.pool_activation_sha256,
        "consentDocumentSha256": request.consent_document_sha256,
        "consentStatementSha256": request.consent_statement_sha256,
        "reviewEnabled": True,
    }


@router.get("/expert/protocol")
def expert_protocol(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    _, reviewer = _expert_identity(session, authorization)
    payload = protocol_payload()
    author_evaluator = _author_evaluator_active(reviewer)
    anonymous_external = _anonymous_external_rater_active(session, reviewer)
    pathway_bypass = author_evaluator or anonymous_external
    cohort_use = {
        "primaryLabel": _reviewer_admission_pathway(reviewer),
        "admissibleClaim": _reviewer_claim_boundary(reviewer),
        "inadmissibleClaim": (
            "Independent expert consensus."
            if pathway_bypass
            else "A broader rater cohort than the recorded evidence supports."
        ),
        "poolingRule": (
            "Never silently pool this pathway with author, public, provider-affiliated, "
            "other anonymous, verified independent-expert, or automated judgments."
        ),
    }
    return {
        **payload,
        "cohortUse": cohort_use,
        "acknowledgementStatements": _reviewer_acknowledgement_statements(reviewer),
        "protocolSha256": EXPERT_PROTOCOL_SHA256,
        "reviewer": {
            "reviewerCode": reviewer.reviewer_code,
            "cohort": reviewer.cohort,
            "qualifiedFamilies": reviewer.qualification_json,
            "qualificationVerified": reviewer.qualification_verified,
            "calibrationAccuracy": (
                None if pathway_bypass else reviewer.profile_json.get("calibration_accuracy")
            ),
            "batchRevealOnly": reviewer.batch_reveal_only,
            "affiliationClass": reviewer.profile_json.get("affiliation_class"),
            "admissionPathway": _reviewer_admission_pathway(reviewer),
            "targetJudgments": _reviewer_target_judgments(session, reviewer),
        },
    }


@router.post("/expert/sessions", status_code=201)
def create_expert_session(
    request: ExpertSessionCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    _, reviewer = _expert_identity(session, authorization)
    if not hmac.compare_digest(request.protocol_sha256, EXPERT_PROTOCOL_SHA256):
        raise HTTPException(status_code=409, detail="expert review protocol has changed")
    _reviewer_workload_cell_targets(session, reviewer, request.target_judgments)
    if (
        request.controlled_run_id is not None
        and _controlled_reviewer_authorization(session, reviewer, request.controlled_run_id) is None
    ):
        raise HTTPException(status_code=404, detail="qualified controlled run not found")
    review_session_id = str(uuid.uuid4())
    session.add(
        RunEvent(
            entity_type="expert_review_session",
            entity_id=review_session_id,
            event_type="expert_review_session_opened",
            payload_json={
                "reviewer_id": reviewer.id,
                "reviewer_code": reviewer.reviewer_code,
                "cohort": reviewer.cohort,
                "controlled_run_id": request.controlled_run_id,
                "protocol_version": EXPERT_PROTOCOL_VERSION,
                "protocol_sha256": EXPERT_PROTOCOL_SHA256,
                "acknowledgements": request.acknowledgements,
                "acknowledgement_statements": {
                    key: _reviewer_acknowledgement_statements(reviewer)[key]
                    for key in request.acknowledgements
                },
                "target_judgments": request.target_judgments,
                "admission_pathway": (_reviewer_admission_pathway(reviewer)),
                "author_evaluator_pool_sha256": reviewer.profile_json.get(
                    "author_evaluator_pool_sha256"
                ),
                "anonymous_external_pool_sha256": reviewer.profile_json.get(
                    "anonymous_external_pool_sha256"
                ),
                "qualification_basis": reviewer.profile_json.get("qualification_basis"),
                "independence_basis": reviewer.profile_json.get("independence_basis"),
                "identity_collection_prohibited": reviewer.profile_json.get(
                    "identity_collection_prohibited"
                ),
                "calibration_set_sha256": reviewer.profile_json.get("calibration_set_sha256"),
                "calibration_accuracy": reviewer.profile_json.get("calibration_accuracy"),
                "training_material_sha256": reviewer.profile_json.get("training_material_sha256"),
                "conflict_disclosure_reference": reviewer.profile_json.get(
                    "conflict_disclosure_reference"
                ),
            },
        )
    )
    session.commit()
    return {
        "reviewSessionId": review_session_id,
        "protocolVersion": EXPERT_PROTOCOL_VERSION,
        "protocolSha256": EXPERT_PROTOCOL_SHA256,
        "cohort": reviewer.cohort,
        "targetJudgments": request.target_judgments,
        "identityReveal": "withheld_until_governed_batch_release",
    }


@router.get("/expert/sessions/{review_session_id}/status")
def expert_session_status(
    review_session_id: str,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    rater, reviewer = _expert_identity(session, authorization)
    session_event = _expert_session_event(session, reviewer, review_session_id)
    _, _, submitted = _expert_assignment_events(session, review_session_id)
    review_payloads = _submitted_review_payloads(submitted)
    fatigue = _review_fatigue_status(_expert_reviewer_submitted_events(session, reviewer.id))
    primary = [row for row in review_payloads if row.get("mode") == "primary"]
    repeats = [row for row in review_payloads if row.get("mode") == "reliability_repeat"]
    by_family = Counter(str(row.get("category")) for row in primary)
    by_track = Counter(str(row.get("track")) for row in primary)
    primary_cell_counts = Counter(
        (str(row.get("track")), str(row.get("category"))) for row in primary
    )
    repeat_cell_counts = Counter(
        (str(row.get("track")), str(row.get("category"))) for row in repeats
    )
    durations = [
        int(row["duration_ms"])
        for row in review_payloads
        if isinstance(row.get("duration_ms"), int)
    ]
    speed_flags = sum(bool(row.get("speed_flag")) for row in review_payloads)
    invalid_tasks = sum(
        row.get("normalized_rubric", {}).get("review_metadata", {}).get("task_validity")
        == "invalid"
        for row in primary
    )
    primary_votes = session.scalars(
        select(Vote).where(
            Vote.rater_pseudonym == rater,
            Vote.cohort == reviewer.cohort,
        )
    ).all()
    primary_by_battle = {
        vote.battle_id: {
            "choice": vote.choice,
            "rubric": vote.rubric_json,
        }
        for vote in primary_votes
    }
    reliability = reliability_summary(primary_by_battle, repeats)
    target = int(session_event.payload_json.get("target_judgments", 0))
    cell_targets = _reviewer_workload_cell_targets(session, reviewer, target)
    author_evaluator = _author_evaluator_active(reviewer)
    anonymous_external = _anonymous_external_rater_active(session, reviewer)
    isolated_pilot = author_evaluator or anonymous_external
    model_ids_by_battle = _battle_model_ids(
        session,
        {str(row.get("battle_id")) for row in primary if row.get("battle_id")},
    )
    track_model_ids: dict[str, set[str]] = {
        "model_arena": set(),
        "epicure_uplift": set(),
    }
    model_exposure_counts: Counter[tuple[str, str]] = Counter()
    arena_pairs: list[tuple[str, str]] = []
    for row in primary:
        track = str(row.get("track"))
        battle_models = model_ids_by_battle.get(str(row.get("battle_id")), ())
        if track not in track_model_ids:
            continue
        track_model_ids[track].update(battle_models)
        if track == "model_arena" and len(battle_models) == 2:
            arena_pairs.append((battle_models[0], battle_models[1]))
        for model_id in battle_models:
            model_exposure_counts[(track, model_id)] += 1
    arena_exposure_total = sum(cell_targets["primary"]["model_arena"].values()) * 2
    uplift_exposure_total = sum(cell_targets["primary"]["epicure_uplift"].values())
    model_exposure_targets = {
        "model_arena": _balanced_integer_targets(
            arena_exposure_total,
            track_model_ids["model_arena"],
        ),
        "epicure_uplift": _balanced_integer_targets(
            uplift_exposure_total,
            track_model_ids["epicure_uplift"],
        ),
    }
    observed_model_ids = track_model_ids["model_arena"] | track_model_ids["epicure_uplift"]
    component_sizes = _comparison_component_sizes(arena_pairs)
    comparison_graph_connected = bool(
        len(track_model_ids["model_arena"]) == 16 and component_sizes == [16]
    )
    model_exposure_reached = bool(
        len(observed_model_ids) == 16
        and all(
            model_exposure_counts[(track, model_id)] >= exposure_target
            for track, per_model_targets in model_exposure_targets.items()
            for model_id, exposure_target in per_model_targets.items()
        )
    )
    completed = len(review_payloads)
    cell_quota_reached = all(
        primary_cell_counts[(track, family)] >= family_target
        for track, family_targets in cell_targets["primary"].items()
        for family, family_target in family_targets.items()
    ) and all(
        repeat_cell_counts[(track, family)] >= family_target
        for track, family_targets in cell_targets["reliability"].items()
        for family, family_target in family_targets.items()
    )
    if isolated_pilot:
        reliability_ready = bool(
            reliability["comparableRepeats"] >= cell_targets["reliability_repeats"]
        )
        cohort_ready = bool(completed >= target and reliability_ready and cell_quota_reached)
    else:
        reliability_ready = bool(
            reliability["comparableRepeats"] >= 40
            and reliability["exactPreferenceAgreement"] is not None
            and reliability["exactPreferenceAgreement"] >= 0.7
            and reliability["preferenceAgreementInterval95"] is not None
            and reliability["preferenceAgreementInterval95"][0] >= 0.5
            and reliability["meanAbsoluteDimensionDifference"] is not None
            and reliability["meanAbsoluteDimensionDifference"] <= 0.75
        )
        cohort_ready = bool(
            completed >= target
            and reviewer.profile_json.get("calibration_accuracy", 0) >= 0.8
            and reliability_ready
            and cell_quota_reached
            and model_exposure_reached
            and comparison_graph_connected
        )
    return {
        "reviewSessionId": review_session_id,
        "cohort": reviewer.cohort,
        "protocolVersion": EXPERT_PROTOCOL_VERSION,
        "protocolSha256": EXPERT_PROTOCOL_SHA256,
        "completedPresentations": completed,
        "primaryJudgments": len(primary),
        "reliabilityPresentations": len(repeats),
        "targetPresentations": target,
        "completionRate": round(completed / target, 4) if target else 0,
        "byFamily": {
            family: by_family.get(family, 0)
            for family in ("substitution", "composition", "cookability", "evidence")
        },
        "byTrack": {track: by_track.get(track, 0) for track in ("model_arena", "epicure_uplift")},
        "cellTargets": cell_targets,
        "modelCoverage": {
            "modelCount": len(observed_model_ids),
            "requiredModelCount": 16,
            "modelExposureReached": model_exposure_reached,
            "comparisonGraphConnected": comparison_graph_connected,
            "comparisonComponentSizes": component_sizes,
            "arenaExposureMin": min(
                (
                    model_exposure_counts[("model_arena", model_id)]
                    for model_id in track_model_ids["model_arena"]
                ),
                default=0,
            ),
            "arenaExposureMax": max(
                (
                    model_exposure_counts[("model_arena", model_id)]
                    for model_id in track_model_ids["model_arena"]
                ),
                default=0,
            ),
            "upliftExposureMin": min(
                (
                    model_exposure_counts[("epicure_uplift", model_id)]
                    for model_id in track_model_ids["epicure_uplift"]
                ),
                default=0,
            ),
            "upliftExposureMax": max(
                (
                    model_exposure_counts[("epicure_uplift", model_id)]
                    for model_id in track_model_ids["epicure_uplift"]
                ),
                default=0,
            ),
        },
        "medianReviewSeconds": (round(median(durations) / 1000, 1) if durations else None),
        "speedFlags": speed_flags,
        "invalidTaskFlags": invalid_tasks,
        "fatigue": fatigue,
        "reliability": reliability,
        "qualityGate": {
            "calibrationRequired": not isolated_pilot,
            "calibrationPassed": (
                None
                if isolated_pilot
                else reviewer.profile_json.get("calibration_accuracy", 0) >= 0.8
            ),
            "targetReached": completed >= target,
            "cellQuotaReached": cell_quota_reached,
            "modelExposureReached": model_exposure_reached,
            "comparisonGraphConnected": comparison_graph_connected,
            "reliabilityReady": reliability_ready,
            "productAffiliatedCohortReady": (False if isolated_pilot else cohort_ready),
            "authorEvaluatorCaseStudyComplete": (cohort_ready if author_evaluator else False),
            "anonymousExternalRaterComplete": (cohort_ready if anonymous_external else False),
            # Response-compatible legacy field: a self-attestation is not an
            # independently verified status.
            "independentExternalRater": False,
            "selfAttestedExternalRater": anonymous_external,
            "verifiedIndependentExternalRater": False,
            "independentExpertValidation": False,
        },
        "claimBoundary": _reviewer_claim_boundary(reviewer),
    }


@router.get(
    "/admin/expert-sessions/{review_session_id}/export",
    dependencies=[Depends(require_admin_token)],
)
def admin_export_expert_session(
    review_session_id: str,
    session: Db,
) -> dict[str, Any]:
    session_event = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "expert_review_session",
            RunEvent.entity_id == review_session_id,
            RunEvent.event_type == "expert_review_session_opened",
        )
    )
    if session_event is None:
        raise HTTPException(status_code=404, detail="expert review session not found")
    reviewer = session.get(
        ExpertReviewer,
        session_event.payload_json.get("reviewer_id"),
    )
    if reviewer is None:
        raise HTTPException(status_code=409, detail="expert reviewer record is unavailable")
    opened, assessed, submitted = _expert_assignment_events(
        session,
        review_session_id,
    )
    records: list[dict[str, Any]] = []
    for assignment_id, submitted_event in sorted(
        submitted.items(),
        key=lambda item: (item[1].created_at, item[0]),
    ):
        opened_event = opened.get(assignment_id)
        assessed_event = assessed.get(assignment_id)
        if opened_event is None or assessed_event is None:
            raise HTTPException(
                status_code=409,
                detail="expert review session has an incomplete audit chain",
            )
        battle = session.get(Battle, opened_event.payload_json.get("battle_id"))
        if battle is None:
            raise HTTPException(
                status_code=409,
                detail="expert review session battle is unavailable",
            )
        arms = session.scalars(
            select(ResponseArm).where(ResponseArm.battle_id == battle.id).order_by(ResponseArm.side)
        ).all()
        records.append(
            {
                "reviewAssignmentId": assignment_id,
                "battleId": battle.id,
                "taskId": battle.task_id,
                "taskRevision": battle.task_revision,
                "track": battle.track,
                "category": battle.category,
                "promptSha256": battle.prompt_sha256,
                "mode": submitted_event.payload_json.get("mode"),
                "presentedSideMap": opened_event.payload_json.get("presented_side_map"),
                "presentationSha256": opened_event.payload_json.get("presentation_sha256"),
                "taskAssessment": assessed_event.payload_json.get("assessment"),
                "taskAssessmentSha256": assessed_event.payload_json.get("assessment_sha256"),
                "normalizedChoice": submitted_event.payload_json.get("normalized_choice"),
                "normalizedReasonTags": submitted_event.payload_json.get("normalized_reason_tags"),
                "normalizedRubric": submitted_event.payload_json.get("normalized_rubric"),
                "reviewSha256": submitted_event.payload_json.get("review_sha256"),
                "durationMs": submitted_event.payload_json.get("duration_ms"),
                "answerReviewDurationMs": submitted_event.payload_json.get(
                    "answer_review_duration_ms"
                ),
                "speedFlag": submitted_event.payload_json.get("speed_flag"),
                "voteId": submitted_event.payload_json.get("vote_id"),
                "submittedAt": _utc_iso(submitted_event.created_at),
                "arms": [
                    {
                        "side": arm.side,
                        "modelId": arm.model_id,
                        "condition": arm.condition,
                        "executionBackend": arm.execution_backend,
                        "providerSlug": arm.actual_provider_slug or arm.provider_slug,
                        "actualModelId": arm.actual_model_id,
                        "answerMarkdownSha256": arm.answer_markdown_sha256,
                        "generationId": arm.generation_id,
                        "costMicros": arm.cost_micros,
                        "latencyMs": arm.latency_ms,
                        "finishReason": arm.finish_reason,
                    }
                    for arm in arms
                ],
            }
        )
    payload: dict[str, Any] = {
        "schemaVersion": "flavourbench-expert-session-export-v1",
        "reviewSessionId": review_session_id,
        "reviewerCode": reviewer.reviewer_code,
        "cohort": reviewer.cohort,
        "protocolVersion": session_event.payload_json.get("protocol_version"),
        "protocolSha256": session_event.payload_json.get("protocol_sha256"),
        "controlledRunId": session_event.payload_json.get("controlled_run_id"),
        "targetJudgments": session_event.payload_json.get("target_judgments"),
        "sessionOpenedAt": _utc_iso(session_event.created_at),
        "records": records,
    }
    payload["exportSha256"] = canonical_sha256(payload)
    return payload


@router.post("/expert/sessions/{review_session_id}/assignments/next")
def claim_expert_assignment(
    review_session_id: str,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    rater, reviewer = _expert_identity(session, authorization)
    _lock_expert_review_session(
        session,
        review_session_id=review_session_id,
        reviewer_id=reviewer.id,
    )
    review_session = _expert_session_event(session, reviewer, review_session_id)
    opened, assessed, submitted = _expert_assignment_events(session, review_session_id)
    for assignment_id, event in opened.items():
        if assignment_id not in submitted:
            return _assignment_response(session, event, assessed.get(assignment_id))

    completed_rows = _submitted_review_payloads(submitted)
    target_judgments = int(review_session.payload_json.get("target_judgments", 0))
    if target_judgments and len(completed_rows) >= target_judgments:
        return {"assignment": None, "queueStatus": "target_reached"}
    fatigue = _review_fatigue_status(_expert_reviewer_submitted_events(session, reviewer.id))
    if fatigue["dailyLimitReached"]:
        return {"assignment": None, "queueStatus": "daily_limit"}
    if fatigue["breakRequired"]:
        return {"assignment": None, "queueStatus": "session_break"}
    cell_targets = _reviewer_workload_cell_targets(session, reviewer, target_judgments)
    isolated_pilot = _isolated_pilot_reviewer_active(session, reviewer)
    primary_rows = [row for row in completed_rows if row.get("mode") == "primary"]
    repeat_rows = [row for row in completed_rows if row.get("mode") == "reliability_repeat"]
    controlled_run_id = review_session.payload_json.get("controlled_run_id")
    battle: Battle | None = None
    mode = "primary"
    side_map = {"left": "left", "right": "right"}

    repeat_due = (
        len(primary_rows) >= RELIABILITY_REPEAT_INTERVAL
        and len(repeat_rows) < len(primary_rows) // RELIABILITY_REPEAT_INTERVAL
    )
    if repeat_due:
        repeated_battles = {str(row.get("battle_id")) for row in repeat_rows}
        repeat_cell_counts = Counter(
            (str(row.get("track")), str(row.get("category"))) for row in repeat_rows
        )
        repeat_pool = [
            original
            for original in primary_rows[:-16]
            if str(original.get("battle_id", "")) not in repeated_battles
            and repeat_cell_counts[(str(original.get("track")), str(original.get("category")))]
            < cell_targets["reliability"]
            .get(str(original.get("track")), {})
            .get(str(original.get("category")), 0)
        ]
        repeat_pool.sort(
            key=lambda row: (
                repeat_cell_counts[(str(row.get("track")), str(row.get("category")))]
                / max(
                    1,
                    cell_targets["reliability"]
                    .get(str(row.get("track")), {})
                    .get(str(row.get("category")), 0),
                ),
                canonical_sha256(
                    {
                        "reviewSessionId": review_session_id,
                        "repeatIndex": len(repeat_rows),
                        "battleId": row.get("battle_id"),
                    }
                ),
            )
        )
        for original in repeat_pool:
            battle_id = str(original.get("battle_id", ""))
            if not battle_id:
                continue
            candidate = session.get(Battle, battle_id)
            if candidate is None or candidate.status != "complete":
                continue
            if candidate.id not in _reviewable_battle_ids(session, {candidate.id}):
                continue
            if candidate.category not in set(reviewer.qualification_json):
                continue
            if controlled_run_id is None:
                if isolated_pilot:
                    if (
                        candidate.data_stratum != "development"
                        or candidate.run_class != "pilot"
                        or candidate.rank_eligible
                        or candidate.manifest_sha256
                        != _isolated_pilot_pool_sha256(session, reviewer)
                        or candidate.controlled_run_id is not None
                    ):
                        continue
                elif (
                    candidate.data_stratum != "public_freeform"
                    or candidate.controlled_run_id is not None
                ):
                    continue
            if controlled_run_id is not None and (
                candidate.data_stratum != "controlled"
                or candidate.controlled_run_id != controlled_run_id
            ):
                continue
            battle = candidate
            mode = "reliability_repeat"
            side_map = {"left": "right", "right": "left"}
            break

    if battle is None:
        if len(primary_rows) >= cell_targets["primary_judgments"]:
            return {"assignment": None, "queueStatus": "awaiting_reliability_repeat"}
        candidates = _qualified_expert_battles(
            session,
            reviewer=reviewer,
            rater=rater,
            controlled_run_id=controlled_run_id,
        )
        cell_counts = Counter(
            (str(row.get("track")), str(row.get("category"))) for row in primary_rows
        )
        model_ids_by_battle = _battle_model_ids(
            session,
            {item.id for item in candidates}
            | {str(row.get("battle_id")) for row in primary_rows if row.get("battle_id")},
        )
        track_model_ids: dict[str, set[str]] = {
            "model_arena": set(),
            "epicure_uplift": set(),
        }
        for item in candidates:
            track_model_ids.setdefault(item.track, set()).update(
                model_ids_by_battle.get(item.id, ())
            )
        model_exposure_counts: Counter[tuple[str, str]] = Counter()
        arena_pairs: list[tuple[str, str]] = []
        for row in primary_rows:
            track = str(row.get("track"))
            battle_models = model_ids_by_battle.get(
                str(row.get("battle_id")),
                (),
            )
            track_model_ids.setdefault(track, set()).update(battle_models)
            if track == "model_arena" and len(battle_models) == 2:
                arena_pairs.append((battle_models[0], battle_models[1]))
            for model_id in battle_models:
                model_exposure_counts[(track, model_id)] += 1
        arena_component_labels = _comparison_component_labels(arena_pairs)
        model_exposure_targets = {
            "model_arena": _balanced_integer_targets(
                sum(cell_targets["primary"]["model_arena"].values()) * 2,
                track_model_ids["model_arena"],
            ),
            "epicure_uplift": _balanced_integer_targets(
                sum(cell_targets["primary"]["epicure_uplift"].values()),
                track_model_ids["epicure_uplift"],
            ),
        }
        candidates = [
            item
            for item in candidates
            if cell_counts[(item.track, item.category)]
            < cell_targets["primary"].get(item.track, {}).get(item.category, 0)
            and model_ids_by_battle.get(item.id)
            and (
                isolated_pilot
                or all(
                    model_exposure_counts[(item.track, model_id)]
                    < model_exposure_targets.get(item.track, {}).get(model_id, 0)
                    for model_id in model_ids_by_battle[item.id]
                )
            )
        ]
        if not candidates:
            return {"assignment": None, "queueStatus": "exhausted"}
        candidates.sort(
            key=lambda item: (
                cell_counts[(item.track, item.category)]
                / max(
                    1,
                    cell_targets["primary"].get(item.track, {}).get(item.category, 0),
                ),
                (
                    0
                    if item.track == "model_arena"
                    and len(model_ids_by_battle[item.id]) == 2
                    and (
                        model_ids_by_battle[item.id][0] not in arena_component_labels
                        or model_ids_by_battle[item.id][1] not in arena_component_labels
                        or arena_component_labels[model_ids_by_battle[item.id][0]]
                        != arena_component_labels[model_ids_by_battle[item.id][1]]
                    )
                    else 1
                ),
                max(
                    (
                        model_exposure_counts[(item.track, model_id)]
                        / max(
                            1,
                            model_exposure_targets.get(item.track, {}).get(
                                model_id,
                                0,
                            ),
                        )
                        for model_id in model_ids_by_battle[item.id]
                    ),
                    default=1.0,
                ),
                canonical_sha256(
                    {
                        "reviewSessionId": review_session_id,
                        "battleId": item.id,
                    }
                ),
            )
        )
        battle = candidates[0]

    _, digest = _review_presentation(session, battle, side_map)
    review_assignment_id = str(uuid.uuid4())
    opened_event = RunEvent(
        entity_type="expert_review_assignment",
        entity_id=review_assignment_id,
        event_type="expert_review_assignment_opened",
        payload_json={
            "review_session_id": review_session_id,
            "reviewer_id": reviewer.id,
            "battle_id": battle.id,
            "track": battle.track,
            "category": battle.category,
            "mode": mode,
            "presented_side_map": side_map,
            "presentation_sha256": digest,
            "protocol_sha256": EXPERT_PROTOCOL_SHA256,
        },
    )
    session.add(opened_event)
    session.commit()
    session.refresh(opened_event)
    return _assignment_response(session, opened_event)


def _record_expert_task_scope_events(
    session: Session,
    *,
    battle: Battle,
    reviewer: ExpertReviewer,
    review_session_id: str,
    review_assignment_id: str,
    assessment: dict[str, Any],
    assessment_sha256: str,
    presentation_sha256: str,
) -> None:
    """Persist fail-closed ranking consequences from the sealed task assessment."""

    common = {
        "battle_id": battle.id,
        "review_session_id": review_session_id,
        "review_assignment_id": review_assignment_id,
        "reviewer_id": reviewer.id,
        "reviewer_cohort": reviewer.cohort,
        "reviewer_qualification_verified": reviewer.qualification_verified,
        "scope_eligibility": assessment.get("scope_eligibility"),
        "specialist_domains": list(assessment.get("specialist_domains") or []),
        "assessment_sha256": assessment_sha256,
        "presentation_sha256": presentation_sha256,
        "scope_protocol_sha256": EXPERT_PROTOCOL_SHA256,
    }
    if assessment.get("general_track_eligible") is False:
        if battle.task_id is not None:
            task = session.get(Task, battle.task_id)
            session.add(
                RunEvent(
                    entity_type="task",
                    entity_id=battle.task_id,
                    event_type="task_general_track_scope_quarantined",
                    payload_json={
                        **common,
                        "task_public_id": task.public_id if task is not None else None,
                        "general_track_eligible": False,
                        "ranking_use": False,
                        "operational_use": False,
                        "status": "pending_governed_scope_adjudication",
                    },
                )
            )
        session.add(
            RunEvent(
                entity_type="battle",
                entity_id=battle.id,
                event_type="battle_ranking_restricted",
                payload_json={
                    **common,
                    "general_track_eligible": False,
                    "ranking_use": False,
                    "operational_use": False,
                    "reason": "expert_task_scope_not_general",
                    "status": "pending_governed_scope_adjudication",
                },
            )
        )
        return

    affiliation_class = reviewer.profile_json.get("affiliation_class")
    if (
        battle.data_stratum == "public_freeform"
        and battle.controlled_run_id is None
        and reviewer.qualification_verified
        and reviewer.cohort == "expert_independent"
        and affiliation_class == "independent_external"
    ):
        admission_event = calibrated_expert_admission_event(session, reviewer)
        if admission_event is None:
            return
        session.add(
            RunEvent(
                entity_type="battle",
                entity_id=battle.id,
                event_type="battle_general_track_scope_admitted",
                payload_json={
                    **common,
                    "general_track_eligible": True,
                    "ranking_use": True,
                    "admission_basis": "sealed_qualified_independent_task_assessment",
                    "affiliation_class": affiliation_class,
                    "reviewer_admission_event_id": admission_event.id,
                    "reviewer_admission_evidence_sha256": canonical_sha256(
                        admission_event.payload_json
                    ),
                    "scope_admission_quorum": 1,
                },
            )
        )


_EXPERT_SAFETY_FAILURE_TAGS = frozenset(
    {
        "safety_hazard",
        "unsupported_safety_claim",
        "allergen_or_dietary_risk",
    }
)


def _record_expert_safety_reports(
    session: Session,
    *,
    battle: Battle,
    reviewer: ExpertReviewer,
    review_session_id: str,
    review_assignment_id: str,
    normalized_rubric: dict[str, Any],
    review_sha256: str,
) -> None:
    """Record a side-specific safety signal without conditioning preference use on it."""

    metadata = normalized_rubric.get("review_metadata")
    if not isinstance(metadata, dict):
        return
    arm_ids = {"left": battle.left_arm_id, "right": battle.right_arm_id}
    for side, arm_id in arm_ids.items():
        reported_tags = sorted(
            _EXPERT_SAFETY_FAILURE_TAGS.intersection(
                set(metadata.get(f"{side}_failure_tags") or [])
            )
        )
        if not reported_tags:
            continue
        if arm_id is None:
            raise HTTPException(status_code=409, detail="reviewed response arm is missing")
        arm = session.get(ResponseArm, arm_id)
        if arm is None or arm.battle_id != battle.id:
            raise HTTPException(status_code=409, detail="reviewed response arm is invalid")
        session.add(
            RunEvent(
                entity_type="response_arm",
                entity_id=arm.id,
                event_type="reviewer_reported_potential_safety_hazard",
                payload_json={
                    "battle_id": battle.id,
                    "condition": arm.condition,
                    "reported_tags": reported_tags,
                    "review_session_id": review_session_id,
                    "review_assignment_id": review_assignment_id,
                    "reviewer_id": reviewer.id,
                    "reviewer_cohort": reviewer.cohort,
                    "reviewer_qualification_verified": reviewer.qualification_verified,
                    "review_sha256": review_sha256,
                    "protocol_sha256": EXPERT_PROTOCOL_SHA256,
                    "status": "pending_qualified_food_safety_adjudication",
                    "verified_safety_error": False,
                    "preference_exclusion_requested": False,
                    "preference_treatment": (
                        "retain_unless_task_scope_or_independent_ranking_control_excludes"
                    ),
                    "operational_use": True,
                },
            )
        )
        session.add(
            Incident(
                severity="critical",
                code="reviewer_reported_potential_safety_hazard",
                detail=(
                    "An expert review reported a potential response-level safety concern. "
                    "The report remains unverified pending qualified food-safety adjudication, "
                    "is reported separately, and does not by itself remove the comparison "
                    "from preference fitting."
                ),
                battle_id=battle.id,
            )
        )


@router.post("/expert/review-assignments/{review_assignment_id}/task-assessment")
def assess_expert_task(
    review_assignment_id: str,
    request: ExpertTaskAssessmentCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    _, reviewer = _expert_identity(session, authorization)
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    opened = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "expert_review_assignment",
            RunEvent.entity_id == review_assignment_id,
            RunEvent.event_type == "expert_review_assignment_opened",
        )
    )
    if opened is None or opened.payload_json.get("reviewer_id") != reviewer.id:
        raise HTTPException(status_code=404, detail="expert review assignment not found")
    review_session_id = str(opened.payload_json.get("review_session_id", ""))
    _lock_expert_review_session(
        session,
        review_session_id=review_session_id,
        reviewer_id=reviewer.id,
    )
    _expert_session_event(session, reviewer, review_session_id)
    prior = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "expert_review_assignment",
            RunEvent.entity_id == review_assignment_id,
            RunEvent.event_type == "expert_review_task_assessed",
        )
    )
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    assessment = request.model_dump(mode="json")
    assessment_sha256 = canonical_sha256(assessment)
    if prior is not None:
        if not hmac.compare_digest(
            str(prior.payload_json.get("idempotency_key_sha256", "")),
            idempotency_sha256,
        ) or not hmac.compare_digest(
            str(prior.payload_json.get("assessment_sha256", "")),
            assessment_sha256,
        ):
            raise HTTPException(status_code=409, detail="task assessment already recorded")
        return _assignment_response(session, opened, prior)

    battle = session.get(Battle, opened.payload_json.get("battle_id"))
    if battle is None or battle.category not in set(reviewer.qualification_json):
        raise HTTPException(status_code=404, detail="qualified assignment not found")
    side_map = dict(opened.payload_json.get("presented_side_map", {}))
    _, current_digest = _review_presentation(session, battle, side_map)
    if not hmac.compare_digest(
        str(opened.payload_json.get("presentation_sha256", "")),
        current_digest,
    ):
        raise HTTPException(status_code=409, detail="expert assignment presentation drifted")
    opened_at = opened.created_at
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=UTC)
    assessed_event = RunEvent(
        entity_type="expert_review_assignment",
        entity_id=review_assignment_id,
        event_type="expert_review_task_assessed",
        payload_json={
            "review_session_id": review_session_id,
            "reviewer_id": reviewer.id,
            "battle_id": battle.id,
            "assessment": assessment,
            "assessment_sha256": assessment_sha256,
            "protocol_sha256": EXPERT_PROTOCOL_SHA256,
            "presentation_sha256": current_digest,
            "duration_ms": max(
                0,
                int((datetime.now(UTC) - opened_at).total_seconds() * 1000),
            ),
            "idempotency_key_sha256": idempotency_sha256,
        },
    )
    session.add(assessed_event)
    _record_expert_task_scope_events(
        session,
        battle=battle,
        reviewer=reviewer,
        review_session_id=review_session_id,
        review_assignment_id=review_assignment_id,
        assessment=assessment,
        assessment_sha256=assessment_sha256,
        presentation_sha256=current_digest,
    )
    session.commit()
    session.refresh(assessed_event)
    return _assignment_response(session, opened, assessed_event)


@router.post("/expert/review-assignments/{review_assignment_id}")
def submit_expert_review(
    review_assignment_id: str,
    request: ExpertReviewCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    rater, reviewer = _expert_identity(session, authorization)
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    opened = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "expert_review_assignment",
            RunEvent.entity_id == review_assignment_id,
            RunEvent.event_type == "expert_review_assignment_opened",
        )
    )
    if opened is None or opened.payload_json.get("reviewer_id") != reviewer.id:
        raise HTTPException(status_code=404, detail="expert review assignment not found")
    review_session_id = str(opened.payload_json.get("review_session_id", ""))
    _lock_expert_review_session(
        session,
        review_session_id=review_session_id,
        reviewer_id=reviewer.id,
    )
    _expert_session_event(session, reviewer, review_session_id)
    task_assessment = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "expert_review_assignment",
            RunEvent.entity_id == review_assignment_id,
            RunEvent.event_type == "expert_review_task_assessed",
        )
    )
    if task_assessment is None:
        raise HTTPException(
            status_code=409,
            detail="task validity must be sealed before response review",
        )
    prior = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "expert_review_assignment",
            RunEvent.entity_id == review_assignment_id,
            RunEvent.event_type == "expert_review_assignment_submitted",
        )
    )
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    if prior is not None:
        if not hmac.compare_digest(
            str(prior.payload_json.get("idempotency_key_sha256", "")),
            idempotency_sha256,
        ):
            raise HTTPException(status_code=409, detail="expert assignment already submitted")
        return {
            "reviewAssignmentId": review_assignment_id,
            "voteId": prior.payload_json.get("vote_id"),
            "recorded": True,
            "reveal": None,
            "revealStatus": "withheld_pending_governed_batch_reveal",
        }
    battle = session.get(Battle, opened.payload_json.get("battle_id"))
    if battle is None or battle.category not in set(reviewer.qualification_json):
        raise HTTPException(status_code=404, detail="qualified assignment not found")
    controlled_run_id = battle.controlled_run_id
    if battle.data_stratum == "controlled":
        if (
            controlled_run_id is None
            or _controlled_reviewer_authorization(session, reviewer, controlled_run_id) is None
        ):
            raise HTTPException(status_code=404, detail="qualified assignment not found")
    elif _isolated_pilot_reviewer_active(session, reviewer):
        if (
            battle.data_stratum != "development"
            or battle.run_class != "pilot"
            or battle.rank_eligible
            or battle.manifest_sha256 != _isolated_pilot_pool_sha256(session, reviewer)
            or controlled_run_id is not None
        ):
            raise HTTPException(status_code=404, detail="qualified assignment not found")
    elif battle.data_stratum != "public_freeform" or controlled_run_id is not None:
        raise HTTPException(status_code=404, detail="qualified assignment not found")
    side_map = dict(opened.payload_json.get("presented_side_map", {}))
    _, current_digest = _review_presentation(session, battle, side_map)
    if not hmac.compare_digest(
        str(opened.payload_json.get("presentation_sha256", "")),
        current_digest,
    ):
        raise HTTPException(status_code=409, detail="expert assignment presentation drifted")

    request_payload = request.model_dump(mode="json")
    normalized_payload = {
        "choice": normalize_choice(str(request_payload["choice"]), side_map),
        "reasonTags": list(request_payload["reason_tags"]),
        "rubric": normalize_rubric(request_payload["rubric"], side_map),
    }
    normalized_request = ExpertReviewCreate.model_validate(normalized_payload)
    normalized_metadata = normalized_request.model_dump(mode="json")["rubric"]["review_metadata"]
    sealed_assessment = dict(task_assessment.payload_json.get("assessment", {}))
    submitted_assessment = {
        key: normalized_metadata.get(key)
        for key in (
            "task_validity",
            "task_issue_tags",
            "task_note",
            "answerability",
            "family_fit",
            "scope_eligibility",
            "specialist_domains",
            "general_track_eligible",
        )
    }
    if not hmac.compare_digest(
        canonical_sha256(sealed_assessment),
        canonical_sha256(submitted_assessment),
    ):
        raise HTTPException(
            status_code=409,
            detail="sealed task assessment cannot change after answers are revealed",
        )
    mode = str(opened.payload_json.get("mode", "primary"))
    vote: Vote | None = None
    if mode == "primary":
        reviewer_binding = None
        reviewer_admission = None
        credential_binding_id = getattr(reviewer, "_flavourbench_identity_binding_id", None)
        if credential_binding_id is not None:
            try:
                verified_provenance = resolve_verified_vote_admission(
                    session,
                    reviewer=reviewer,
                    battle=battle,
                )
            except ReviewerIdentityError as exc:
                raise HTTPException(
                    status_code=409,
                    detail="expert reviewer admission is ambiguous",
                ) from exc
            if verified_provenance is None:
                raise HTTPException(
                    status_code=403,
                    detail="expert reviewer lacks an active admission for this task family",
                )
            reviewer_binding, reviewer_admission = verified_provenance
            if reviewer_binding.id != credential_binding_id:
                raise HTTPException(
                    status_code=403,
                    detail="expert credential does not match the active reviewer admission",
                )
        vote = _record_vote(
            session,
            battle=battle,
            rater=rater,
            cohort=reviewer.cohort,
            request=normalized_request,
            idempotency_key=idempotency_key,
            reviewer=(reviewer if reviewer_binding is not None else None),
            reviewer_binding=reviewer_binding,
            reviewer_admission=reviewer_admission,
            commit=False,
        )
    elif mode != "reliability_repeat":
        raise HTTPException(status_code=409, detail="expert assignment mode is invalid")

    opened_at = opened.created_at
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=UTC)
    duration_ms = max(0, int((datetime.now(UTC) - opened_at).total_seconds() * 1000))
    assessed_at = task_assessment.created_at
    if assessed_at.tzinfo is None:
        assessed_at = assessed_at.replace(tzinfo=UTC)
    answer_review_duration_ms = max(
        0,
        int((datetime.now(UTC) - assessed_at).total_seconds() * 1000),
    )
    normalized_rubric = normalized_request.model_dump(mode="json")["rubric"]
    review_record = {
        "review_session_id": review_session_id,
        "reviewer_id": reviewer.id,
        "battle_id": battle.id,
        "track": battle.track,
        "category": battle.category,
        "mode": mode,
        "normalized_choice": normalized_request.choice.value,
        "normalized_reason_tags": normalized_request.reason_tags,
        "normalized_rubric": normalized_rubric,
        "review_sha256": canonical_sha256(
            {
                "choice": normalized_request.choice.value,
                "reasonTags": normalized_request.reason_tags,
                "rubric": normalized_rubric,
            }
        ),
        "presentation_sha256": current_digest,
        "task_assessment_sha256": task_assessment.payload_json.get("assessment_sha256"),
        "protocol_sha256": EXPERT_PROTOCOL_SHA256,
        "duration_ms": duration_ms,
        "answer_review_duration_ms": answer_review_duration_ms,
        "speed_flag": duration_ms < 30_000,
        "idempotency_key_sha256": idempotency_sha256,
        "vote_id": vote.id if vote else None,
    }
    _record_expert_safety_reports(
        session,
        battle=battle,
        reviewer=reviewer,
        review_session_id=review_session_id,
        review_assignment_id=review_assignment_id,
        normalized_rubric=normalized_rubric,
        review_sha256=str(review_record["review_sha256"]),
    )
    session.add(
        RunEvent(
            entity_type="expert_review_assignment",
            entity_id=review_assignment_id,
            event_type="expert_review_assignment_submitted",
            payload_json=review_record,
        )
    )
    session.commit()
    return {
        "reviewAssignmentId": review_assignment_id,
        "voteId": vote.id if vote else None,
        "recorded": True,
        "reveal": None,
        "revealStatus": "withheld_pending_governed_batch_reveal",
    }


@router.get("/expert/assignments/next")
def expert_assignment(
    session: Db,
    authorization: Annotated[str, Header()] = "",
    controlled_run_id: str | None = Query(default=None),
) -> dict:
    _invited_expert_identity(session, authorization)
    raise HTTPException(
        status_code=410,
        detail="legacy expert assignments are retired; use versioned review sessions",
    )


@router.post("/expert/battles/{battle_id}/votes")
def expert_vote(
    battle_id: str,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict:
    _invited_expert_identity(session, authorization)
    raise HTTPException(
        status_code=410,
        detail="legacy expert voting is retired; use versioned review sessions",
    )


def _invited_task_contributor(
    session: Session,
    authorization: str,
) -> ExpertReviewer:
    _, contributor = _invited_expert_identity(session, authorization)
    if (
        contributor.cohort != "expert_independent"
        or contributor.qualification_verified
        or contributor.profile_json.get("admission_pathway") != "task_contributor"
    ):
        raise HTTPException(status_code=403, detail="task-contributor invitation is required")
    return contributor


@router.get("/task-contributions/protocol")
def task_contributor_protocol(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    contributor = _invited_task_contributor(session, authorization)
    try:
        text_value = task_contributor_protocol_text()
    except TaskContributorProtocolError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    accepted = _task_contributor_protocol_binding_active(session, contributor)
    return {
        "protocolVersion": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
        "protocolSha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
        "protocolScope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
        "protocolText": text_value,
        "accepted": accepted,
        "acceptanceRequired": not accepted,
        "contributionEnabled": accepted,
        "claimBoundary": (
            "This agreement governs task authorship and redistribution only. It does not admit "
            "the contributor as an output rater, expert reviewer, or research participant."
        ),
    }


@router.post("/task-contributions/protocol-acceptance")
def accept_task_contributor_protocol(
    request: TaskContributorProtocolAcceptanceCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    invited = _invited_task_contributor(session, authorization)
    contributor = reviewer_control_lock(session, invited.id)
    if contributor is None or not contributor.active:
        raise HTTPException(status_code=401, detail="task-contributor invitation is invalid")
    if _task_contributor_protocol_binding_active(session, contributor):
        return {
            "recorded": True,
            "idempotent": True,
            "protocolVersion": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
            "protocolSha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
            "contributionEnabled": True,
            "eventId": contributor.profile_json.get(
                "task_contributor_protocol_acceptance_event_id"
            ),
        }

    now = datetime.now(UTC)
    previous_profile = dict(contributor.profile_json)
    event = RunEvent(
        entity_type="task_contributor",
        entity_id=contributor.id,
        event_type="task_contributor_protocol_accepted",
        payload_json={
            "schema_version": "flavourbench-task-contributor-protocol-acceptance-v1",
            "protocol_version": request.protocol_version,
            "protocol_sha256": request.protocol_sha256,
            "protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
            "voluntary_participation_accepted": request.voluntary_participation_accepted,
            "task_contribution_agreement_accepted": (request.task_contribution_agreement_accepted),
            "human_only_methods_acknowledged": request.human_only_methods_acknowledged,
            "accepted_at": now.isoformat(),
            "previous_profile_sha256": _canonical_sha256(previous_profile),
        },
    )
    session.add(event)
    session.flush()
    updated_profile = {
        **previous_profile,
        "task_contributor_status": "active",
        "task_contributor_protocol_version": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
        "task_contributor_protocol_sha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
        "task_contributor_protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
        "task_contributor_protocol_accepted": True,
        "task_contributor_protocol_accepted_at": now.isoformat(),
        "task_contributor_protocol_acceptance_event_id": event.id,
    }
    legacy_consent_sha256 = updated_profile.pop("consent_document_sha256", None)
    if isinstance(legacy_consent_sha256, str):
        updated_profile.setdefault("legacy_consent_document_sha256", legacy_consent_sha256)
    contributor.profile_json = updated_profile
    session.add(contributor)
    session.commit()
    return {
        "recorded": True,
        "idempotent": False,
        "protocolVersion": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
        "protocolSha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
        "contributionEnabled": True,
        "eventId": event.id,
    }


@router.get("/task-contributions/onboarding")
def task_contributor_onboarding(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    contributor = _task_contributor_identity(session, authorization)
    own_candidates = [
        event
        for event in _task_candidate_events(session)
        if event.payload_json.get("author_reviewer_id") == contributor.id
    ]
    return {
        "admissionPathway": "anonymous_task_contributor",
        "qualifiedFamilies": contributor.qualification_json,
        "identityCollectionProhibited": False,
        "rawIdentityRetentionProhibited": True,
        "personUniquenessVerified": True,
        "personUniquenessMethod": "admin-witnessed-season-hmac-v1",
        "humanAuthorshipRequired": True,
        "syntheticTasksAccepted": False,
        "protocolVersion": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
        "protocolSha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
        "protocolScope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
        "protocolAcceptanceEventId": contributor.profile_json.get(
            "task_contributor_protocol_acceptance_event_id"
        ),
        "constructBlueprintSha256": BLUEPRINT_SHA256,
        "constructCells": {
            family: list(CONSTRUCT_CELLS[family]) for family in contributor.qualification_json
        },
        "difficultyTiers": sorted(DIFFICULTY_TIERS),
        "submittedCandidates": len(own_candidates),
        "claimBoundary": (
            "Task authorship is self-attested. A submission cannot enter Season 1 until two "
            "distinct qualification-verified reviewers independently solve it before seeing "
            "the author pack, reconcile their criterion packs, and a third independent "
            "adjudicator freezes the task definition."
        ),
    }


@router.get("/task-contributions")
def list_task_contributions(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    contributor = _task_contributor_identity(session, authorization)
    records = []
    for event in _task_candidate_events(session):
        if event.payload_json.get("author_reviewer_id") != contributor.id:
            continue
        review_status = _task_candidate_status(
            _task_candidate_review_events(session, event.entity_id)
        )
        imported_task = _task_candidate_imported_task(session, event.entity_id)
        records.append(
            {
                "candidateId": event.entity_id,
                "family": event.payload_json.get("family"),
                "promptSha256": event.payload_json.get("prompt_sha256"),
                "recordSha256": event.payload_json.get("candidate_record_sha256"),
                "submittedAt": _utc_iso(event.created_at),
                "withdrawalEligible": (
                    review_status["status"] != "withdrawn" and imported_task is None
                ),
                **_task_candidate_status_view(review_status),
            }
        )
    return {"candidates": records}


@router.post("/task-contributions", status_code=201)
def create_task_contribution(
    request: TaskContributionCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    contributor = _task_contributor_identity(session, authorization)
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    if request.family.value not in contributor.qualification_json:
        raise HTTPException(status_code=403, detail="task family is outside the contributor record")

    events = _task_candidate_events(session)
    prompt_sha256 = hashlib.sha256(request.prompt.encode()).hexdigest()
    nonce_sha256 = hashlib.sha256(request.client_nonce.encode()).hexdigest()
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    candidate_record = {
        "schema_version": "flavourbench-human-task-candidate-v2",
        "family": request.family.value,
        "prompt": request.prompt,
        "prompt_sha256": prompt_sha256,
        "construct_blueprint_sha256": request.construct_blueprint_sha256,
        "construct_cell_id": request.construct_cell_id,
        "difficulty_tier": request.difficulty_tier,
        "subskills": request.subskills,
        "explicit_constraints": request.explicit_constraints,
        "unacceptable_outcomes": request.unacceptable_outcomes,
        "acceptable_solution_outline": request.acceptable_solution_outline,
        "objective_validator_possible": request.objective_validator_possible,
        "validator_notes": request.validator_notes,
        "rights_basis": request.rights_basis,
        "human_authorship_attestation": request.human_authorship_attestation,
        "no_personal_data_attestation": request.no_personal_data_attestation,
        "research_use_consent": request.research_use_consent,
        "task_contributor_protocol_version": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
        "task_contributor_protocol_sha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
        "task_contributor_protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
        "task_contributor_protocol_acceptance_event_id": contributor.profile_json[
            "task_contributor_protocol_acceptance_event_id"
        ],
    }
    record_sha256 = _canonical_sha256(candidate_record)
    for event in events:
        payload = event.payload_json
        same_author_nonce = bool(
            payload.get("author_reviewer_id") == contributor.id
            and payload.get("client_nonce_sha256") == nonce_sha256
        )
        if same_author_nonce:
            if (
                payload.get("idempotency_key_sha256") == idempotency_sha256
                and payload.get("candidate_record_sha256") == record_sha256
            ):
                return {
                    "candidateId": event.entity_id,
                    "recordSha256": record_sha256,
                    "status": _task_candidate_status(
                        _task_candidate_review_events(session, event.entity_id)
                    )["status"],
                    "idempotent": True,
                }
            raise HTTPException(status_code=409, detail="client nonce already identifies a task")
        if payload.get("prompt_sha256") == prompt_sha256:
            raise HTTPException(
                status_code=409,
                detail="an identical task prompt is already sealed",
            )

    now = datetime.now(UTC)
    recent_count = sum(
        event.payload_json.get("author_reviewer_id") == contributor.id
        and as_utc(event.created_at) >= now - timedelta(hours=24)
        for event in events
    )
    if recent_count >= 8:
        raise HTTPException(status_code=429, detail="daily task-contribution limit reached")

    candidate_id = str(uuid.uuid4())
    session.add(
        RunEvent(
            entity_type="task_candidate",
            entity_id=candidate_id,
            event_type="task_candidate_submitted",
            payload_json={
                **candidate_record,
                "author_reviewer_id": contributor.id,
                "candidate_record_sha256": record_sha256,
                "client_nonce_sha256": nonce_sha256,
                "idempotency_key_sha256": idempotency_sha256,
                "rank_eligible": False,
            },
        )
    )
    session.commit()
    return {
        "candidateId": candidate_id,
        "recordSha256": record_sha256,
        "status": "awaiting_independent_review",
        "idempotent": False,
    }


@router.post("/task-contributions/{candidate_id}/withdrawal")
def withdraw_task_contribution(
    candidate_id: str,
    request: TaskContributionWithdrawalCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    """Append a contributor withdrawal before the candidate enters a frozen task bank."""

    contributor = _task_contributor_identity(session, authorization)
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    candidate = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "task_candidate",
            RunEvent.entity_id == candidate_id,
            RunEvent.event_type == "task_candidate_submitted",
        )
        .with_for_update()
    )
    if candidate is None or candidate.payload_json.get("author_reviewer_id") != contributor.id:
        raise HTTPException(status_code=404, detail="task candidate not found")
    candidate_record_sha256 = str(candidate.payload_json.get("candidate_record_sha256", ""))
    if not hmac.compare_digest(
        request.candidate_record_sha256,
        candidate_record_sha256,
    ):
        raise HTTPException(status_code=409, detail="candidate receipt hash does not match")

    lifecycle_events = _task_candidate_review_events(session, candidate_id)
    state = _task_candidate_status(lifecycle_events)
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    client_nonce_sha256 = hashlib.sha256(request.client_nonce.encode()).hexdigest()
    withdrawal_record = {
        "schema_version": "flavourbench-task-candidate-withdrawal-v1",
        "candidate_id": candidate_id,
        "candidate_record_sha256": candidate_record_sha256,
        "author_reviewer_id": contributor.id,
        "reason_category": request.reason_category,
        "note": request.note,
        "withdrawal_confirmed": request.withdrawal_confirmed,
        "client_nonce_sha256": client_nonce_sha256,
        "idempotency_key_sha256": idempotency_sha256,
        "task_contributor_protocol_acceptance_event_id": contributor.profile_json[
            "task_contributor_protocol_acceptance_event_id"
        ],
    }
    withdrawal_record_sha256 = _canonical_sha256(withdrawal_record)
    prior = state["withdrawal"]
    if prior is not None:
        if (
            prior.payload_json.get("idempotency_key_sha256") == idempotency_sha256
            and prior.payload_json.get("withdrawal_record_sha256") == withdrawal_record_sha256
        ):
            return {
                "candidateId": candidate_id,
                "recordSha256": candidate_record_sha256,
                "withdrawalSha256": withdrawal_record_sha256,
                "status": "withdrawn",
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail="task candidate is already withdrawn")
    if _task_candidate_imported_task(session, candidate_id) is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "task candidate is already frozen in a task bank; use the public task "
                "challenge and correction process"
            ),
        )

    event = RunEvent(
        entity_type="task_candidate",
        entity_id=candidate_id,
        event_type="task_candidate_withdrawal_recorded",
        payload_json={
            **withdrawal_record,
            "withdrawal_record_sha256": withdrawal_record_sha256,
            "rank_eligible": False,
        },
    )
    session.add(event)
    session.commit()
    return {
        "candidateId": candidate_id,
        "recordSha256": candidate_record_sha256,
        "withdrawalSha256": withdrawal_record_sha256,
        "status": "withdrawn",
        "idempotent": False,
    }


@router.get("/expert/task-candidates/next")
def next_task_candidate_review(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    reviewer = _development_task_reviewer(session, authorization)
    if not _verified_independent_task_validator(reviewer):
        raise HTTPException(status_code=403, detail="independent task-review admission is required")
    for event in _task_candidate_events(session):
        payload = event.payload_json
        if (
            payload.get("author_reviewer_id") == reviewer.id
            or payload.get("family") not in reviewer.qualification_json
        ):
            continue
        reviews = _task_candidate_review_events(session, event.entity_id)
        state = _task_candidate_status(reviews)
        own = _reviewer_task_candidate_events(reviews, reviewer.id)
        blind = own.get("task_candidate_blind_validity_recorded")
        reconciliation = own.get("task_candidate_reconciliation_recorded")
        if reconciliation is not None or (
            blind is not None and blind.payload_json.get("decision") != "valid"
        ):
            continue
        if state["status"] in {
            "rejected",
            "approved_for_bank_assembly",
            "revision_requested",
            "awaiting_independent_adjudication",
            "invalid_excess_source_reviews",
            "invalid_duplicate_withdrawals",
            "withdrawn",
        }:
            continue
        if blind is not None:
            return {
                "candidate": _task_candidate_reconciliation_view(event),
                "protocol": "flavourbench-confirmatory-task-validation-v2",
            }
        if len(state["blindByReviewer"]) < 2:
            return {
                "candidate": _task_candidate_blind_view(event),
                "protocol": "flavourbench-confirmatory-task-validation-v2",
            }
    return {
        "candidate": None,
        "protocol": "flavourbench-confirmatory-task-validation-v2",
    }


@router.post("/expert/task-candidates/{candidate_id}/blind-validity")
def record_task_candidate_blind_validity(
    candidate_id: str,
    request: TaskCandidateBlindValidityCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    reviewer = _development_task_reviewer(session, authorization)
    if not _verified_independent_task_validator(reviewer):
        raise HTTPException(status_code=403, detail="independent task-review admission is required")
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    candidate = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "task_candidate",
            RunEvent.entity_id == candidate_id,
            RunEvent.event_type == "task_candidate_submitted",
        )
        .with_for_update()
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="task candidate not found")
    payload = candidate.payload_json
    if (
        payload.get("author_reviewer_id") == reviewer.id
        or payload.get("family") not in reviewer.qualification_json
    ):
        raise HTTPException(status_code=404, detail="qualified task candidate not found")
    if request.decision == "valid":
        assert request.family_classification is not None
        assert request.construct_cell_classification is not None
        assert request.difficulty_tier_classification is not None
        try:
            validate_task_binding(
                family=request.family_classification.value,
                construct_blueprint_sha256=BLUEPRINT_SHA256,
                construct_cell_id=request.construct_cell_classification,
                difficulty_tier=request.difficulty_tier_classification,
            )
        except ConstructBlueprintError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"independent construct classification is invalid: {exc}",
            ) from exc

    _lock_expert_review_session(
        session,
        review_session_id=f"task-candidate:{candidate_id}:blind-validity",
        reviewer_id="global",
    )
    reviews = _task_candidate_review_events(session, candidate_id)
    state = _task_candidate_status(reviews)
    own = _reviewer_task_candidate_events(reviews, reviewer.id)
    blind_payload = request.model_dump(mode="json")
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    record_payload = {
        "candidate_id": candidate_id,
        "candidate_record_sha256": payload.get("candidate_record_sha256"),
        "reviewer_id": reviewer.id,
        "blind_review": blind_payload,
    }
    blind_review_sha256 = _canonical_sha256(record_payload)
    prior = own.get("task_candidate_blind_validity_recorded")
    if prior is not None:
        if (
            prior.payload_json.get("idempotency_key_sha256") == idempotency_sha256
            and prior.payload_json.get("blind_review_sha256") == blind_review_sha256
        ):
            return {
                "candidateId": candidate_id,
                "blindReviewSha256": blind_review_sha256,
                "nextPhase": "reconciliation" if request.decision == "valid" else "complete",
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail="blind validity decision is already sealed")
    if state["withdrawal"] is not None:
        raise HTTPException(status_code=409, detail="task candidate has been withdrawn")
    if state["adjudication"] is not None:
        raise HTTPException(status_code=409, detail="task adjudication is already sealed")
    if len(state["blindByReviewer"]) >= 2:
        raise HTTPException(status_code=409, detail="independent source-review slate is full")
    if state["legacyReviewCount"]:
        raise HTTPException(
            status_code=409,
            detail="legacy candidate reviews must be quarantined before v2 validation",
        )

    event = RunEvent(
        entity_type="task_candidate",
        entity_id=candidate_id,
        event_type="task_candidate_blind_validity_recorded",
        payload_json={
            **blind_payload,
            "reviewer_id": reviewer.id,
            "candidate_record_sha256": payload.get("candidate_record_sha256"),
            "prompt_sha256": payload.get("prompt_sha256"),
            "blind_review_sha256": blind_review_sha256,
            "idempotency_key_sha256": idempotency_sha256,
            "author_pack_visible": False,
            "model_outputs_visible": False,
            "independent_review": True,
            "identity_commitment_sha256": reviewer.profile_json["identity_commitment_sha256"],
            "qualification_evidence_sha256": reviewer.profile_json["qualification_evidence_sha256"],
            "independence_attestation_sha256": reviewer.profile_json[
                "independence_attestation_sha256"
            ],
        },
    )
    session.add(event)
    session.commit()
    return {
        "candidateId": candidate_id,
        "blindReviewSha256": blind_review_sha256,
        "nextPhase": "reconciliation" if request.decision == "valid" else "complete",
        "idempotent": False,
    }


@router.post("/expert/task-candidates/{candidate_id}/reconciliation")
def record_task_candidate_reconciliation(
    candidate_id: str,
    request: TaskCandidateReconciliationCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    reviewer = _development_task_reviewer(session, authorization)
    if not _verified_independent_task_validator(reviewer):
        raise HTTPException(status_code=403, detail="independent task-review admission is required")
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    candidate = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "task_candidate",
            RunEvent.entity_id == candidate_id,
            RunEvent.event_type == "task_candidate_submitted",
        )
        .with_for_update()
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="task candidate not found")
    payload = candidate.payload_json
    if (
        payload.get("author_reviewer_id") == reviewer.id
        or payload.get("family") not in reviewer.qualification_json
    ):
        raise HTTPException(status_code=404, detail="qualified task candidate not found")

    _lock_expert_review_session(
        session,
        review_session_id=f"task-candidate:{candidate_id}:reconciliation",
        reviewer_id=reviewer.id,
    )
    reviews = _task_candidate_review_events(session, candidate_id)
    state = _task_candidate_status(reviews)
    own = _reviewer_task_candidate_events(reviews, reviewer.id)
    blind = own.get("task_candidate_blind_validity_recorded")
    if blind is None or blind.payload_json.get("decision") != "valid":
        raise HTTPException(
            status_code=409,
            detail="a sealed valid prompt-only review is required before reconciliation",
        )
    expected_construct_agreement = bool(
        blind.payload_json.get("family_classification") == payload.get("family")
        and blind.payload_json.get("construct_cell_classification")
        == payload.get("construct_cell_id")
    )
    expected_difficulty_agreement = bool(
        blind.payload_json.get("difficulty_tier_classification") == payload.get("difficulty_tier")
    )
    if (
        request.construct_label_agreement != expected_construct_agreement
        or request.difficulty_label_agreement != expected_difficulty_agreement
    ):
        raise HTTPException(
            status_code=409,
            detail="reconciliation label-agreement fields do not match the sealed blind review",
        )

    reconciliation_payload = request.model_dump(mode="json")
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    record_payload = {
        "candidate_id": candidate_id,
        "candidate_record_sha256": payload.get("candidate_record_sha256"),
        "blind_review_sha256": blind.payload_json.get("blind_review_sha256"),
        "reviewer_id": reviewer.id,
        "reconciliation": reconciliation_payload,
    }
    reconciliation_sha256 = _canonical_sha256(record_payload)
    prior = own.get("task_candidate_reconciliation_recorded")
    if prior is not None:
        if (
            prior.payload_json.get("idempotency_key_sha256") == idempotency_sha256
            and prior.payload_json.get("reconciliation_sha256") == reconciliation_sha256
        ):
            return {
                "candidateId": candidate_id,
                "reconciliationSha256": reconciliation_sha256,
                "complete": True,
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail="candidate reconciliation is already sealed")
    if state["withdrawal"] is not None:
        raise HTTPException(status_code=409, detail="task candidate has been withdrawn")
    if state["adjudication"] is not None:
        raise HTTPException(status_code=409, detail="task adjudication is already sealed")

    event = RunEvent(
        entity_type="task_candidate",
        entity_id=candidate_id,
        event_type="task_candidate_reconciliation_recorded",
        payload_json={
            **reconciliation_payload,
            "reviewer_id": reviewer.id,
            "candidate_record_sha256": payload.get("candidate_record_sha256"),
            "prompt_sha256": payload.get("prompt_sha256"),
            "blind_review_sha256": blind.payload_json.get("blind_review_sha256"),
            "reconciliation_sha256": reconciliation_sha256,
            "idempotency_key_sha256": idempotency_sha256,
            "author_pack_visible": True,
            "model_outputs_visible": False,
            "independent_review": True,
            "identity_commitment_sha256": reviewer.profile_json["identity_commitment_sha256"],
            "qualification_evidence_sha256": reviewer.profile_json["qualification_evidence_sha256"],
            "independence_attestation_sha256": reviewer.profile_json[
                "independence_attestation_sha256"
            ],
        },
    )
    session.add(event)
    session.commit()
    return {
        "candidateId": candidate_id,
        "reconciliationSha256": reconciliation_sha256,
        "complete": True,
        "idempotent": False,
    }


@router.get("/expert/task-candidates/adjudication/next")
def next_task_candidate_adjudication(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    adjudicator = _development_task_adjudicator(session, authorization)
    for candidate in _task_candidate_events(session):
        payload = candidate.payload_json
        if (
            payload.get("author_reviewer_id") == adjudicator.id
            or payload.get("family") not in adjudicator.qualification_json
        ):
            continue
        reviews = _task_candidate_review_events(session, candidate.entity_id)
        if any(event.payload_json.get("reviewer_id") == adjudicator.id for event in reviews):
            continue
        state = _task_candidate_status(reviews)
        if state["status"] == "awaiting_independent_adjudication":
            return {
                "candidate": _task_candidate_adjudication_view(candidate, state),
                "protocol": "flavourbench-confirmatory-task-validation-v2",
            }
    return {
        "candidate": None,
        "protocol": "flavourbench-confirmatory-task-validation-v2",
    }


@router.post("/expert/task-candidates/{candidate_id}/adjudication")
def record_task_candidate_adjudication(
    candidate_id: str,
    request: TaskCandidateAdjudicationCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    adjudicator = _development_task_adjudicator(session, authorization)
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    candidate = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "task_candidate",
            RunEvent.entity_id == candidate_id,
            RunEvent.event_type == "task_candidate_submitted",
        )
        .with_for_update()
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="task candidate not found")
    payload = candidate.payload_json
    if (
        payload.get("author_reviewer_id") == adjudicator.id
        or payload.get("family") not in adjudicator.qualification_json
    ):
        raise HTTPException(status_code=404, detail="qualified task candidate not found")
    _lock_expert_review_session(
        session,
        review_session_id=f"task-candidate:{candidate_id}:adjudication",
        reviewer_id="global",
    )
    reviews = _task_candidate_review_events(session, candidate_id)
    state = _task_candidate_status(reviews)
    if any(
        event.payload_json.get("reviewer_id") == adjudicator.id
        and event.event_type
        in {
            "task_candidate_blind_validity_recorded",
            "task_candidate_reconciliation_recorded",
        }
        for event in reviews
    ):
        raise HTTPException(status_code=409, detail="a source reviewer cannot adjudicate the task")

    source_blind_sha256s = sorted(
        state["blindByReviewer"][reviewer_id].payload_json["blind_review_sha256"]
        for reviewer_id in state["completeReviewers"]
    )
    source_reconciliation_sha256s = sorted(
        event.payload_json["reconciliation_sha256"]
        for reviewer_id, event in state["reconciliationByReviewer"].items()
        if reviewer_id in state["completeReviewers"]
    )
    adjudication_payload = request.model_dump(mode="json")
    criterion_pack_payload = {
        key: adjudication_payload[key]
        for key in (
            "family",
            "construct_cell_id",
            "difficulty_tier",
            "success_criteria",
            "permitted_variations",
            "disqualifying_errors",
            "objective_checks",
        )
    }
    criterion_pack_sha256 = _canonical_sha256(criterion_pack_payload)
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    record_payload = {
        "candidate_id": candidate_id,
        "candidate_record_sha256": payload.get("candidate_record_sha256"),
        "source_blind_sha256s": source_blind_sha256s,
        "source_reconciliation_sha256s": source_reconciliation_sha256s,
        "adjudicator_id": adjudicator.id,
        "adjudication": adjudication_payload,
        "criterion_pack_sha256": criterion_pack_sha256,
    }
    adjudication_sha256 = _canonical_sha256(record_payload)
    prior = state["adjudication"]
    if prior is not None:
        if (
            prior.payload_json.get("reviewer_id") == adjudicator.id
            and prior.payload_json.get("idempotency_key_sha256") == idempotency_sha256
            and prior.payload_json.get("adjudication_sha256") == adjudication_sha256
        ):
            return {
                "candidateId": candidate_id,
                "decision": request.decision,
                "adjudicationSha256": adjudication_sha256,
                "criterionPackSha256": criterion_pack_sha256,
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail="task adjudication is already sealed")
    if state["withdrawal"] is not None:
        raise HTTPException(status_code=409, detail="task candidate has been withdrawn")
    if state["status"] != "awaiting_independent_adjudication":
        raise HTTPException(
            status_code=409,
            detail="two complete independent source reviews are required before adjudication",
        )
    if request.decision == "approve":
        if (
            state["sourceDecisionCounts"] != {"approve": 2}
            or len(source_reconciliation_sha256s) != 2
        ):
            raise HTTPException(
                status_code=409,
                detail="approval requires two independent valid and reconciled source reviews",
            )
        if (
            request.family is None
            or request.family.value != payload.get("family")
            or request.construct_cell_id != payload.get("construct_cell_id")
            or request.difficulty_tier != payload.get("difficulty_tier")
        ):
            raise HTTPException(
                status_code=409,
                detail="adjudicated construct labels do not match the sealed candidate",
            )
        try:
            validate_task_binding(
                family=request.family.value,
                construct_blueprint_sha256=payload.get("construct_blueprint_sha256"),
                construct_cell_id=request.construct_cell_id,
                difficulty_tier=request.difficulty_tier,
            )
        except ConstructBlueprintError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"adjudicated construct binding failed: {exc}",
            ) from exc

    event = RunEvent(
        entity_type="task_candidate",
        entity_id=candidate_id,
        event_type="task_candidate_adjudication_recorded",
        payload_json={
            **adjudication_payload,
            "reviewer_id": adjudicator.id,
            "candidate_record_sha256": payload.get("candidate_record_sha256"),
            "prompt_sha256": payload.get("prompt_sha256"),
            "source_blind_sha256s": source_blind_sha256s,
            "source_reconciliation_sha256s": source_reconciliation_sha256s,
            "criterion_pack_sha256": criterion_pack_sha256,
            "adjudication_sha256": adjudication_sha256,
            "idempotency_key_sha256": idempotency_sha256,
            "source_reviewer": False,
            "model_outputs_visible": False,
            "identity_commitment_sha256": adjudicator.profile_json["identity_commitment_sha256"],
            "qualification_evidence_sha256": adjudicator.profile_json[
                "qualification_evidence_sha256"
            ],
            "independence_attestation_sha256": adjudicator.profile_json[
                "independence_attestation_sha256"
            ],
        },
    )
    session.add(event)
    session.commit()
    return {
        "candidateId": candidate_id,
        "decision": request.decision,
        "adjudicationSha256": adjudication_sha256,
        "criterionPackSha256": criterion_pack_sha256,
        "status": _task_candidate_status([*reviews, event])["status"],
        "idempotent": False,
    }


def _approved_task_candidate_for_evidence_review(
    session: Session,
    *,
    candidate_id: str,
    reviewer: ExpertReviewer,
) -> tuple[RunEvent, dict[str, Any], set[str]]:
    candidate = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "task_candidate",
            RunEvent.entity_id == candidate_id,
            RunEvent.event_type == "task_candidate_submitted",
        )
        .with_for_update()
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="task candidate not found")
    payload = candidate.payload_json
    reviews = _task_candidate_review_events(session, candidate_id)
    state = _task_candidate_status(reviews)
    adjudication = state["adjudication"]
    role_ids = {
        str(payload.get("author_reviewer_id", "")),
        *state["completeReviewers"],
    }
    if adjudication is not None:
        role_ids.add(str(adjudication.payload_json.get("reviewer_id", "")))
    if (
        state["status"] != "approved_for_bank_assembly"
        or reviewer.id in role_ids
        or payload.get("family") not in reviewer.qualification_json
    ):
        raise HTTPException(
            status_code=404,
            detail="approved, role-independent task evidence assignment not found",
        )
    if not _verified_independent_task_validator(reviewer):
        raise HTTPException(
            status_code=403,
            detail="independent task-evidence review admission is required",
        )
    return candidate, state, role_ids


def _task_evidence_review_event_payload(
    *,
    candidate_id: str,
    candidate_record_sha256: str,
    task_public_id: str,
    reviewer: ExpertReviewer,
    evidence_type: Literal["validator_contract", "contamination_audit"],
    artifact_sha256: str,
    verification_receipt_sha256: str,
    review_payload: dict[str, Any],
    idempotency_sha256: str,
) -> dict[str, Any]:
    review_event_sha256 = task_evidence_review_sha256(
        candidate_id=candidate_id,
        candidate_record_sha256=candidate_record_sha256,
        task_public_id=task_public_id,
        reviewer_id=reviewer.id,
        evidence_type=evidence_type,
        artifact_sha256=artifact_sha256,
        verification_receipt_sha256=verification_receipt_sha256,
        review=review_payload,
    )
    return {
        "candidate_record_sha256": candidate_record_sha256,
        "task_public_id": task_public_id,
        "reviewer_id": reviewer.id,
        "evidence_type": evidence_type,
        "artifact_sha256": artifact_sha256,
        "verification_receipt_sha256": verification_receipt_sha256,
        "review": review_payload,
        "review_event_sha256": review_event_sha256,
        "idempotency_key_sha256": idempotency_sha256,
        "artifact_visible": True,
        "model_outputs_visible": False,
        "independent_of_task_roles": True,
        "identity_commitment_sha256": reviewer.profile_json["identity_commitment_sha256"],
        "qualification_evidence_sha256": reviewer.profile_json["qualification_evidence_sha256"],
        "independence_attestation_sha256": reviewer.profile_json["independence_attestation_sha256"],
    }


def _seal_task_evidence_review_event(
    session: Session,
    *,
    candidate_id: str,
    event_type: str,
    event_payload: dict[str, Any],
) -> tuple[RunEvent, bool]:
    evidence_events = _task_candidate_evidence_review_events(session, candidate_id)
    same_type = [event for event in evidence_events if event.event_type == event_type]
    if len(same_type) > 1:
        raise HTTPException(status_code=409, detail="duplicate sealed evidence-review events")
    prior = same_type[0] if same_type else None
    if prior is not None:
        if (
            prior.payload_json.get("idempotency_key_sha256")
            == event_payload["idempotency_key_sha256"]
            and prior.payload_json.get("review_event_sha256")
            == event_payload["review_event_sha256"]
        ):
            return prior, True
        raise HTTPException(status_code=409, detail="task-evidence review is already sealed")
    if any(
        event.payload_json.get("reviewer_id") == event_payload["reviewer_id"]
        for event in evidence_events
    ):
        raise HTTPException(
            status_code=409,
            detail="validator and contamination evidence require distinct reviewers",
        )
    event = RunEvent(
        entity_type="task_candidate",
        entity_id=candidate_id,
        event_type=event_type,
        payload_json=event_payload,
    )
    session.add(event)
    session.commit()
    return event, False


@router.post("/expert/task-candidates/{candidate_id}/validator-contract-review")
def record_task_candidate_validator_contract_review(
    candidate_id: str,
    request: TaskValidatorContractReviewCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    reviewer = _development_task_reviewer(session, authorization)
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    candidate, _, role_ids = _approved_task_candidate_for_evidence_review(
        session,
        candidate_id=candidate_id,
        reviewer=reviewer,
    )
    contract = request.validator_contract
    if contract.verifier_reviewer_id != reviewer.id:
        raise HTTPException(
            status_code=409,
            detail="validator contract verifier does not match the authenticated reviewer",
        )
    payload = candidate.payload_json
    try:
        receipt = verify_validator_contract(
            contract,
            task_public_id=contract.task_public_id,
            task_family=str(payload.get("family", "")),
            task_revision=contract.task_revision,
            prompt_sha256=str(payload.get("prompt_sha256", "")),
            objective_validator_possible=payload.get("objective_validator_possible") is True,
            expected_container_image_digest=get_settings().build_image_digest,
        )
    except TaskEvidenceError as exc:
        raise HTTPException(status_code=409, detail=f"validator review failed: {exc}") from exc
    if reviewer.id in role_ids:
        raise HTTPException(status_code=409, detail="validator reviewer is not role-independent")
    _lock_expert_review_session(
        session,
        review_session_id=f"task-candidate:{candidate_id}:validator-contract-review",
        reviewer_id="global",
    )
    review_payload = request.model_dump(
        mode="json",
        exclude={"validator_contract"},
    )
    event_payload = _task_evidence_review_event_payload(
        candidate_id=candidate_id,
        candidate_record_sha256=str(payload.get("candidate_record_sha256", "")),
        task_public_id=contract.task_public_id,
        reviewer=reviewer,
        evidence_type="validator_contract",
        artifact_sha256=contract.artifact_sha256,
        verification_receipt_sha256=str(receipt["receipt_sha256"]),
        review_payload=review_payload,
        idempotency_sha256=hashlib.sha256(idempotency_key.encode()).hexdigest(),
    )
    event, idempotent = _seal_task_evidence_review_event(
        session,
        candidate_id=candidate_id,
        event_type="task_candidate_validator_contract_verified",
        event_payload=event_payload,
    )
    return {
        "candidateId": candidate_id,
        "evidenceType": "validator_contract",
        "artifactSha256": contract.artifact_sha256,
        "verificationReceiptSha256": receipt["receipt_sha256"],
        "reviewEventSha256": event.payload_json["review_event_sha256"],
        "idempotent": idempotent,
    }


@router.post("/expert/task-candidates/{candidate_id}/contamination-audit-review")
def record_task_candidate_contamination_audit_review(
    candidate_id: str,
    request: TaskContaminationAuditReviewCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    reviewer = _development_task_reviewer(session, authorization)
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    candidate, _, role_ids = _approved_task_candidate_for_evidence_review(
        session,
        candidate_id=candidate_id,
        reviewer=reviewer,
    )
    audit = request.contamination_audit
    if audit.auditor_reviewer_id != reviewer.id:
        raise HTTPException(
            status_code=409,
            detail="contamination auditor does not match the authenticated reviewer",
        )
    payload = candidate.payload_json
    try:
        receipt = verify_contamination_audit(
            audit,
            scan_bundle=_contamination_scan_bundle(),
            prompt=str(payload.get("prompt", "")),
            task_public_id=audit.task_public_id,
            task_family=str(payload.get("family", "")),
            task_revision=audit.task_revision,
            prompt_sha256=str(payload.get("prompt_sha256", "")),
            expected_container_image_digest=get_settings().build_image_digest,
            forbidden_reviewer_ids=role_ids,
        )
    except TaskEvidenceError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"contamination review failed: {exc}",
        ) from exc
    _lock_expert_review_session(
        session,
        review_session_id=f"task-candidate:{candidate_id}:contamination-audit-review",
        reviewer_id="global",
    )
    review_payload = request.model_dump(
        mode="json",
        exclude={"contamination_audit"},
    )
    event_payload = _task_evidence_review_event_payload(
        candidate_id=candidate_id,
        candidate_record_sha256=str(payload.get("candidate_record_sha256", "")),
        task_public_id=audit.task_public_id,
        reviewer=reviewer,
        evidence_type="contamination_audit",
        artifact_sha256=audit.artifact_sha256,
        verification_receipt_sha256=str(receipt["receipt_sha256"]),
        review_payload=review_payload,
        idempotency_sha256=hashlib.sha256(idempotency_key.encode()).hexdigest(),
    )
    event, idempotent = _seal_task_evidence_review_event(
        session,
        candidate_id=candidate_id,
        event_type="task_candidate_contamination_audit_verified",
        event_payload=event_payload,
    )
    return {
        "candidateId": candidate_id,
        "evidenceType": "contamination_audit",
        "artifactSha256": audit.artifact_sha256,
        "verificationReceiptSha256": receipt["receipt_sha256"],
        "reviewEventSha256": event.payload_json["review_event_sha256"],
        "idempotent": idempotent,
    }


@router.post("/expert/task-candidates/{candidate_id}/reviews")
def review_task_candidate(
    candidate_id: str,
    request: TaskCandidateReviewCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    del candidate_id, request, session, authorization, idempotency_key
    raise HTTPException(
        status_code=410,
        detail=(
            "one-stage candidate review is retired; use blind-validity, reconciliation, "
            "and independent adjudication"
        ),
    )


@router.get("/expert/development-tasks/next")
def next_development_task_validation(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    reviewer = _development_task_reviewer(session, authorization)
    packet = _development_task_validation_packet()
    progress = _development_task_review_progress(
        session,
        packet=packet,
        reviewer=reviewer,
    )
    independent_claim = _verified_independent_task_validator(reviewer)
    assignments: list[
        tuple[tuple[int, int, str], dict[str, Any], dict[str, Any], dict[str, RunEvent]]
    ] = []
    for task in packet["tasks"]:
        if task.get("family") not in reviewer.qualification_json:
            continue
        task_id = str(task["task_id"])
        events = _development_task_review_events(session, task_id)
        state = _development_task_review_state(
            events,
            packet_sha256=packet["artifact_sha256"],
        )
        if state["adjudication"] is not None:
            continue
        own = _reviewer_development_task_events(events, reviewer.id)
        blind = own.get("development_task_blind_validity_recorded")
        criteria = own.get("development_task_criteria_recorded")
        if blind is not None and blind.payload_json.get("decision") == "valid" and criteria is None:
            assignments.append(((0, 0, task_id), task, state, own))
            continue
        if blind is not None:
            continue
        if independent_claim and len(state["blind_by_reviewer"]) >= REQUIRED_INDEPENDENT_REVIEWERS:
            continue
        assignments.append(
            (
                (
                    1,
                    *_development_task_blind_assignment_key(
                        packet_sha256=packet["artifact_sha256"],
                        reviewer_id=reviewer.id,
                        task_id=task_id,
                        complete_independent_reviews=len(state["blind_by_reviewer"]),
                    ),
                ),
                task,
                state,
                own,
            )
        )

    if assignments:
        _, task, _, own = min(assignments, key=lambda row: row[0])
        task_id = str(task["task_id"])
        blind = own.get("development_task_blind_validity_recorded")
        if blind is None:
            task_payload = {
                "phase": "blind_validity",
                "taskId": task_id,
                "family": task["family"],
                "prompt": task["prompt"],
                "promptSha256": task["prompt_sha256"],
                "taskSha256": task["task_sha256"],
                "sourceUrl": None,
                "humanReference": None,
            }
        else:
            reference = task["sealed_human_reference_stage"]
            task_payload = {
                "phase": "criteria",
                "taskId": task_id,
                "family": task["family"],
                "prompt": task["prompt"],
                "promptSha256": task["prompt_sha256"],
                "taskSha256": task["task_sha256"],
                "sourceUrl": reference["source_url"],
                "sourceLicense": reference["source_license"],
                "sourceAuthor": reference["source_author"],
                "humanReference": {
                    "text": reference["reference_text"],
                    "sha256": reference["reference_text_sha256"],
                    "url": reference["reference_url"],
                    "license": reference["reference_license"],
                    "author": reference["reference_author"],
                    "use": reference["reference_use"],
                },
            }
        return {
            "task": task_payload,
            "progress": progress,
            "reviewerCohort": reviewer.cohort,
            "independentClaim": independent_claim,
            "packetSha256": packet["artifact_sha256"],
        }
    return {
        "task": None,
        "progress": progress,
        "reviewerCohort": reviewer.cohort,
        "independentClaim": independent_claim,
        "packetSha256": packet["artifact_sha256"],
    }


@router.post("/expert/development-tasks/{task_id}/blind-validity")
def record_development_task_blind_validity(
    task_id: str,
    request: DevelopmentTaskBlindValidityCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    reviewer = _development_task_reviewer(session, authorization)
    packet = _development_task_validation_packet()
    task = _development_task_record(packet, task_id)
    if task["family"] not in reviewer.qualification_json:
        raise HTTPException(status_code=404, detail="qualified development task not found")
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    _lock_expert_review_session(
        session,
        review_session_id=f"development-task:{task_id}:blind-validity",
        reviewer_id="global",
    )
    events = _development_task_review_events(session, task_id)
    state = _development_task_review_state(
        events,
        packet_sha256=packet["artifact_sha256"],
    )
    own = _reviewer_development_task_events(events, reviewer.id)
    if "development_task_criteria_recorded" in own:
        raise HTTPException(status_code=409, detail="criterion pack is already sealed")

    review_payload = request.model_dump(mode="json")
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    record_payload = {
        "packet_sha256": packet["artifact_sha256"],
        "task_id": task_id,
        "task_sha256": task["task_sha256"],
        "prompt_sha256": task["prompt_sha256"],
        "reviewer_id": reviewer.id,
        "review": review_payload,
    }
    review_sha256 = _canonical_sha256(record_payload)
    prior = own.get("development_task_blind_validity_recorded")
    if prior is not None:
        if (
            prior.payload_json.get("idempotency_key_sha256") == idempotency_sha256
            and prior.payload_json.get("review_sha256") == review_sha256
        ):
            return {
                "taskId": task_id,
                "reviewSha256": review_sha256,
                "nextPhase": "criteria" if request.decision == "valid" else "complete",
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail="blind validity decision is already sealed")
    if state["adjudication"] is not None:
        raise HTTPException(status_code=409, detail="task adjudication is already sealed")
    if (
        _verified_independent_task_validator(reviewer)
        and len(state["blind_by_reviewer"]) >= REQUIRED_INDEPENDENT_REVIEWERS
    ):
        raise HTTPException(status_code=409, detail="independent source-review slate is full")

    event = RunEvent(
        entity_type="development_task_validation",
        entity_id=task_id,
        event_type="development_task_blind_validity_recorded",
        payload_json={
            **review_payload,
            "packet_sha256": packet["artifact_sha256"],
            "task_sha256": task["task_sha256"],
            "prompt_sha256": task["prompt_sha256"],
            "reviewer_id": reviewer.id,
            "reviewer_cohort": reviewer.cohort,
            "independent_review": _verified_independent_task_validator(reviewer),
            "author_affiliated": reviewer.cohort == "expert_product_affiliated",
            "model_outputs_visible": False,
            "human_reference_visible": False,
            "identity_commitment_sha256": reviewer.profile_json["identity_commitment_sha256"],
            "qualification_evidence_sha256": reviewer.profile_json["qualification_evidence_sha256"],
            "independence_attestation_sha256": reviewer.profile_json[
                "independence_attestation_sha256"
            ],
            "review_sha256": review_sha256,
            "idempotency_key_sha256": idempotency_sha256,
        },
    )
    session.add(event)
    session.commit()
    return {
        "taskId": task_id,
        "reviewSha256": review_sha256,
        "nextPhase": "criteria" if request.decision == "valid" else "complete",
        "idempotent": False,
    }


@router.post("/expert/development-tasks/{task_id}/criteria")
def record_development_task_criteria(
    task_id: str,
    request: DevelopmentTaskCriteriaCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    reviewer = _development_task_reviewer(session, authorization)
    packet = _development_task_validation_packet()
    task = _development_task_record(packet, task_id)
    if task["family"] not in reviewer.qualification_json:
        raise HTTPException(status_code=404, detail="qualified development task not found")
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    _lock_expert_review_session(
        session,
        review_session_id=f"development-task:{task_id}:criteria",
        reviewer_id=reviewer.id,
    )
    events = _development_task_review_events(session, task_id)
    state = _development_task_review_state(
        events,
        packet_sha256=packet["artifact_sha256"],
    )
    own = _reviewer_development_task_events(events, reviewer.id)
    blind = own.get("development_task_blind_validity_recorded")
    if blind is None or blind.payload_json.get("decision") != "valid":
        raise HTTPException(
            status_code=409,
            detail="a sealed valid blind decision is required before criterion authoring",
        )

    criteria_payload = request.model_dump(mode="json")
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    record_payload = {
        "packet_sha256": packet["artifact_sha256"],
        "task_id": task_id,
        "task_sha256": task["task_sha256"],
        "prompt_sha256": task["prompt_sha256"],
        "reference_sha256": task["sealed_human_reference_stage"]["reference_text_sha256"],
        "reviewer_id": reviewer.id,
        "criteria": criteria_payload,
    }
    criteria_sha256 = _canonical_sha256(record_payload)
    prior = own.get("development_task_criteria_recorded")
    if prior is not None:
        if (
            prior.payload_json.get("idempotency_key_sha256") == idempotency_sha256
            and prior.payload_json.get("criteria_sha256") == criteria_sha256
        ):
            return {
                "taskId": task_id,
                "criteriaSha256": criteria_sha256,
                "complete": True,
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail="criterion pack is already sealed")
    if state["adjudication"] is not None:
        raise HTTPException(status_code=409, detail="task adjudication is already sealed")

    event = RunEvent(
        entity_type="development_task_validation",
        entity_id=task_id,
        event_type="development_task_criteria_recorded",
        payload_json={
            **criteria_payload,
            "packet_sha256": packet["artifact_sha256"],
            "task_sha256": task["task_sha256"],
            "prompt_sha256": task["prompt_sha256"],
            "reference_sha256": task["sealed_human_reference_stage"]["reference_text_sha256"],
            "blind_validity_review_sha256": blind.payload_json["review_sha256"],
            "reviewer_id": reviewer.id,
            "reviewer_cohort": reviewer.cohort,
            "independent_review": _verified_independent_task_validator(reviewer),
            "author_affiliated": reviewer.cohort == "expert_product_affiliated",
            "model_outputs_visible": False,
            "identity_commitment_sha256": reviewer.profile_json["identity_commitment_sha256"],
            "qualification_evidence_sha256": reviewer.profile_json["qualification_evidence_sha256"],
            "independence_attestation_sha256": reviewer.profile_json[
                "independence_attestation_sha256"
            ],
            "criteria_sha256": criteria_sha256,
            "idempotency_key_sha256": idempotency_sha256,
        },
    )
    session.add(event)
    session.commit()
    return {
        "taskId": task_id,
        "criteriaSha256": criteria_sha256,
        "complete": True,
        "idempotent": False,
    }


@router.get("/expert/development-tasks/adjudication/next")
def next_development_task_adjudication(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    adjudicator = _development_task_adjudicator(session, authorization)
    packet = _development_task_validation_packet()
    for task in packet["tasks"]:
        if task.get("family") not in adjudicator.qualification_json:
            continue
        events = _development_task_review_events(session, str(task["task_id"]))
        if any(event.payload_json.get("reviewer_id") == adjudicator.id for event in events):
            continue
        state = _development_task_review_state(
            events,
            packet_sha256=packet["artifact_sha256"],
        )
        if state["status"] == "awaiting_independent_adjudication":
            return {
                "task": _development_task_adjudication_view(task, state),
                "packetSha256": packet["artifact_sha256"],
                "independentClaim": True,
            }
    return {
        "task": None,
        "packetSha256": packet["artifact_sha256"],
        "independentClaim": True,
    }


@router.post("/expert/development-tasks/{task_id}/adjudication")
def record_development_task_adjudication(
    task_id: str,
    request: DevelopmentTaskAdjudicationCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    adjudicator = _development_task_adjudicator(session, authorization)
    packet = _development_task_validation_packet()
    task = _development_task_record(packet, task_id)
    if task["family"] not in adjudicator.qualification_json:
        raise HTTPException(status_code=404, detail="qualified development task not found")
    if not idempotency_key or len(idempotency_key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    _lock_expert_review_session(
        session,
        review_session_id=f"development-task:{task_id}:adjudication",
        reviewer_id="global",
    )
    events = _development_task_review_events(session, task_id)
    if any(
        event.payload_json.get("reviewer_id") == adjudicator.id
        and event.event_type
        in {"development_task_blind_validity_recorded", "development_task_criteria_recorded"}
        for event in events
    ):
        raise HTTPException(
            status_code=409,
            detail="a source reviewer cannot adjudicate the same task",
        )
    state = _development_task_review_state(
        events,
        packet_sha256=packet["artifact_sha256"],
    )
    source_review_sha256s = sorted(
        state["blind_by_reviewer"][reviewer_id].payload_json["review_sha256"]
        for reviewer_id in state["complete_reviewers"]
    )
    source_criteria_sha256s = sorted(
        criteria.payload_json["criteria_sha256"]
        for reviewer_id, criteria in state["criteria_by_reviewer"].items()
        if reviewer_id in state["complete_reviewers"]
    )
    adjudication_payload = request.model_dump(mode="json")
    idempotency_sha256 = hashlib.sha256(idempotency_key.encode()).hexdigest()
    record_payload = {
        "packet_sha256": packet["artifact_sha256"],
        "task_id": task_id,
        "task_sha256": task["task_sha256"],
        "prompt_sha256": task["prompt_sha256"],
        "reference_sha256": task["sealed_human_reference_stage"]["reference_text_sha256"],
        "source_review_sha256s": source_review_sha256s,
        "source_criteria_sha256s": source_criteria_sha256s,
        "adjudicator_id": adjudicator.id,
        "adjudication": adjudication_payload,
    }
    adjudication_sha256 = _canonical_sha256(record_payload)
    prior = state["adjudication"]
    if prior is not None:
        if (
            prior.payload_json.get("reviewer_id") == adjudicator.id
            and prior.payload_json.get("idempotency_key_sha256") == idempotency_sha256
            and prior.payload_json.get("adjudication_sha256") == adjudication_sha256
        ):
            return {
                "taskId": task_id,
                "decision": request.decision,
                "adjudicationSha256": adjudication_sha256,
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail="task adjudication is already sealed")
    if state["status"] != "awaiting_independent_adjudication":
        raise HTTPException(
            status_code=409,
            detail=(
                "three complete independent source reviews with a non-unanimous "
                "decision are required before adjudication"
            ),
        )

    event = RunEvent(
        entity_type="development_task_validation",
        entity_id=task_id,
        event_type="development_task_adjudication_recorded",
        payload_json={
            **adjudication_payload,
            "packet_sha256": packet["artifact_sha256"],
            "task_sha256": task["task_sha256"],
            "prompt_sha256": task["prompt_sha256"],
            "reference_sha256": task["sealed_human_reference_stage"]["reference_text_sha256"],
            "source_review_sha256s": source_review_sha256s,
            "source_criteria_sha256s": source_criteria_sha256s,
            "reviewer_id": adjudicator.id,
            "reviewer_cohort": adjudicator.cohort,
            "source_reviewer": False,
            "model_outputs_visible": False,
            "adjudication_sha256": adjudication_sha256,
            "idempotency_key_sha256": idempotency_sha256,
        },
    )
    session.add(event)
    session.commit()
    return {
        "taskId": task_id,
        "decision": request.decision,
        "adjudicationSha256": adjudication_sha256,
        "idempotent": False,
    }


@router.get(
    "/admin/development-tasks/status",
    dependencies=[Depends(require_admin_token)],
)
def admin_development_task_validation_status(session: Db) -> dict[str, Any]:
    packet = _development_task_validation_packet()
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    source_reviews = 0
    criterion_packs = 0
    current_events: list[RunEvent] = []
    completed_blind_records: list[dict[str, Any]] = []
    completed_criterion_records: list[dict[str, Any]] = []
    for task in packet["tasks"]:
        task_events = _development_task_review_events(session, str(task["task_id"]))
        current_events.extend(
            event
            for event in task_events
            if event.payload_json.get("packet_sha256") == packet["artifact_sha256"]
        )
        state = _development_task_review_state(
            task_events,
            packet_sha256=packet["artifact_sha256"],
        )
        status_counts[state["status"]] += 1
        source_reviews += len(state["complete_reviewers"])
        criterion_packs += len(state["criteria_by_reviewer"])
        completed_blind_records.extend(
            state["blind_by_reviewer"][reviewer_id].payload_json
            for reviewer_id in state["complete_reviewers"]
        )
        completed_criterion_records.extend(
            state["criteria_by_reviewer"][reviewer_id].payload_json
            for reviewer_id in state["complete_reviewers"]
            if reviewer_id in state["criteria_by_reviewer"]
        )
        rows.append(
            {
                "taskId": task["task_id"],
                "family": task["family"],
                "status": state["status"],
                "completeIndependentReviews": len(state["complete_reviewers"]),
                "decisionCounts": state["decision_counts"],
                "consensusSha256": _development_task_consensus_sha256(
                    packet_sha256=packet["artifact_sha256"],
                    task=task,
                    state=state,
                ),
                "adjudicationSha256": (
                    state["adjudication"].payload_json.get("adjudication_sha256")
                    if state["adjudication"] is not None
                    else None
                ),
            }
        )
    independent_reviewer_ids = {
        str(event.payload_json.get("reviewer_id"))
        for event in current_events
        if event.payload_json.get("independent_review") is True
        and event.payload_json.get("reviewer_id")
    }
    event_commitments = sorted(
        (
            {
                "event_id": event.id,
                "event_type": event.event_type,
                "payload_sha256": _canonical_sha256(event.payload_json),
            }
            for event in current_events
        ),
        key=lambda row: (row["event_id"], row["event_type"]),
    )
    try:
        statistics = summarize_task_validation(
            task_rows=rows,
            blind_records=completed_blind_records,
            criterion_records=completed_criterion_records,
            required_reviews_per_task=REQUIRED_INDEPENDENT_REVIEWERS,
        )
    except DevelopmentTaskStatisticsError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"development task-validation statistics failed closed: {exc}",
        ) from exc
    payload = {
        "schemaVersion": "flavourbench-development-task-validation-status-v1",
        "packetSha256": packet["artifact_sha256"],
        "taskCount": len(rows),
        "requiredIndependentReviewsPerTask": REQUIRED_INDEPENDENT_REVIEWERS,
        "completeIndependentReviews": source_reviews,
        "humanCriterionPacks": criterion_packs,
        "distinctIndependentReviewers": len(independent_reviewer_ids),
        "independentlyValidatedTasks": sum(
            status_counts[status] for status in ("validated_unanimous", "adjudicated_valid")
        ),
        "tasksAwaitingAdjudication": status_counts["awaiting_independent_adjudication"],
        "sourceEventCount": len(current_events),
        "sourceEventSetSha256": _canonical_sha256(event_commitments),
        "statusCounts": dict(sorted(status_counts.items())),
        "statistics": statistics,
        "claimBoundary": {
            "realHumanReviewsOnly": True,
            "modelOutputsVisibleDuringValidation": False,
            "packetItselfIsHumanValidityEvidence": False,
            "publicDevelopmentTasks": True,
            "confirmatoryEligible": False,
            "rankEligible": False,
        },
        "tasks": rows,
    }
    return {**payload, "artifactSha256": _canonical_sha256(payload)}


@router.post("/admin/task-contributors", dependencies=[Depends(require_admin_token)])
def admin_create_task_contributor(
    request: TaskContributorInviteCreate,
    session: Db,
) -> dict[str, Any]:
    existing = session.scalar(
        select(ExpertReviewer).where(ExpertReviewer.reviewer_code == request.contributor_code)
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="contributor code already exists")
    person_commitment_sha256 = _season_person_uniqueness_commitment(
        request.verified_identity_handle
    )
    duplicate_person = next(
        (
            reviewer
            for reviewer in session.scalars(select(ExpertReviewer)).all()
            if reviewer.profile_json.get("person_uniqueness_commitment_sha256")
            == person_commitment_sha256
        ),
        None,
    )
    if duplicate_person is not None:
        raise HTTPException(
            status_code=409,
            detail="this person already has a Season 1 task-contributor account",
        )
    invitation = secrets.token_urlsafe(32)
    contributor = ExpertReviewer(
        reviewer_code=request.contributor_code,
        invitation_sha256=hashlib.sha256(invitation.encode()).hexdigest(),
        qualification_json=[family.value for family in request.qualified_families],
        qualification_verified=False,
        cohort="expert_independent",
        profile_json={
            "admission_pathway": "task_contributor",
            "task_contributor_status": "pending_protocol_acceptance",
            "identity_collection_prohibited": False,
            "raw_identity_retention_prohibited": True,
            "person_uniqueness_verified": True,
            "person_uniqueness_method": "admin-witnessed-season-hmac-v1",
            "person_uniqueness_commitment_sha256": person_commitment_sha256,
            "person_uniqueness_evidence_sha256": (request.person_uniqueness_evidence_sha256),
            "qualification_basis": "self_attested_task_authorship_scope",
            "task_contributor_protocol_offer_version": request.protocol_version,
            "task_contributor_protocol_offer_sha256": request.protocol_sha256,
            "task_contributor_protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
            "task_contributor_protocol_accepted": False,
            "human_authorship_attestation_required": True,
            "independent_task_approval_claim": False,
        },
        batch_reveal_only=True,
    )
    session.add(contributor)
    session.flush()
    session.add(
        RunEvent(
            entity_type="task_contributor",
            entity_id=contributor.id,
            event_type="task_contributor_invited",
            payload_json={
                "qualified_families": contributor.qualification_json,
                "identity_collection_prohibited": False,
                "raw_identity_retention_prohibited": True,
                "person_uniqueness_verified": True,
                "person_uniqueness_method": "admin-witnessed-season-hmac-v1",
                "person_uniqueness_commitment_sha256": person_commitment_sha256,
                "person_uniqueness_evidence_sha256": (request.person_uniqueness_evidence_sha256),
                "protocol_version": request.protocol_version,
                "protocol_sha256": request.protocol_sha256,
                "protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
                "admission_status": "pending_protocol_acceptance",
            },
        )
    )
    session.commit()
    return {
        "contributorId": contributor.id,
        "invitation": invitation,
        "qualifiedFamilies": contributor.qualification_json,
        "rawIdentityRetained": False,
        "personUniquenessCommitmentSha256": person_commitment_sha256,
        "protocolVersion": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
        "protocolSha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
        "admissionStatus": "pending_protocol_acceptance",
        "notice": "The invitation is returned once and is not stored in plaintext.",
    }


@router.post("/admin/task-validators", dependencies=[Depends(require_admin_token)])
def admin_create_task_validator(
    request: TaskValidatorInviteCreate,
    session: Db,
) -> dict[str, Any]:
    existing = session.scalar(
        select(ExpertReviewer).where(ExpertReviewer.reviewer_code == request.validator_code)
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="validator code already exists")
    packet = _development_task_validation_packet()
    identity_commitment_sha256 = _task_validator_identity_commitment(
        request.verified_identity_handle,
        packet_sha256=packet["artifact_sha256"],
    )
    person_commitment_sha256 = _season_person_uniqueness_commitment(
        request.verified_identity_handle
    )
    duplicate_identity = next(
        (
            reviewer
            for reviewer in session.scalars(select(ExpertReviewer)).all()
            if reviewer.profile_json.get("identity_commitment_sha256") == identity_commitment_sha256
            or reviewer.profile_json.get("person_uniqueness_commitment_sha256")
            == person_commitment_sha256
        ),
        None,
    )
    if duplicate_identity is not None:
        raise HTTPException(
            status_code=409,
            detail="a task-validator invitation already exists for this verified identity",
        )
    consent = resolve_expert_consent_document(request.consent_document_sha256)
    if consent.status != "active":
        raise HTTPException(status_code=409, detail="task-validator consent document is not active")
    invitation = secrets.token_urlsafe(32)
    reviewer = ExpertReviewer(
        reviewer_code=request.validator_code,
        invitation_sha256=hashlib.sha256(invitation.encode()).hexdigest(),
        qualification_json=[family.value for family in request.qualified_families],
        qualification_verified=True,
        cohort=(
            "expert_independent"
            if request.affiliation_class == "independent_external"
            else "expert_product_affiliated"
        ),
        profile_json={
            "admission_pathway": "development_task_validator",
            "task_validation_status": "active",
            "model_response_review_authorized": False,
            "task_adjudication_authorized": request.adjudication_authorized,
            "identity_commitment_sha256": identity_commitment_sha256,
            "qualification_evidence_sha256": request.qualification_evidence_sha256,
            "independence_attestation_sha256": request.independence_attestation_sha256,
            "verification_record_sha256": request.verification_record_sha256,
            "identity_commitment_scope_sha256": packet["artifact_sha256"],
            "identity_commitment_algorithm": "HMAC-SHA256",
            "person_uniqueness_verified": True,
            "person_uniqueness_method": "admin-witnessed-season-hmac-v1",
            "person_uniqueness_commitment_sha256": person_commitment_sha256,
            "person_uniqueness_evidence_sha256": request.verification_record_sha256,
            "raw_identity_retention_prohibited": True,
            "qualification_reference_sha256": hashlib.sha256(
                request.qualification_reference.encode()
            ).hexdigest(),
            "affiliation_class": request.affiliation_class,
            "conflict_disclosure_reference_sha256": hashlib.sha256(
                request.conflict_disclosure_reference.encode()
            ).hexdigest(),
            "consent_document_sha256": request.consent_document_sha256,
            "evidence_verified_by_admin": request.evidence_verified_by_admin,
            "private_identity_verification_recorded": True,
            "public_identity_pseudonymous": True,
            "independent_validation_claim": (request.affiliation_class == "independent_external"),
        },
        batch_reveal_only=True,
    )
    session.add(reviewer)
    session.flush()
    session.add(
        RunEvent(
            entity_type="development_task_validator",
            entity_id=reviewer.id,
            event_type="development_task_validator_admitted",
            payload_json={
                "qualified_families": reviewer.qualification_json,
                "identity_commitment_sha256": identity_commitment_sha256,
                "qualification_evidence_sha256": request.qualification_evidence_sha256,
                "independence_attestation_sha256": request.independence_attestation_sha256,
                "verification_record_sha256": request.verification_record_sha256,
                "identity_commitment_scope_sha256": packet["artifact_sha256"],
                "identity_commitment_algorithm": "HMAC-SHA256",
                "person_uniqueness_verified": True,
                "person_uniqueness_method": "admin-witnessed-season-hmac-v1",
                "person_uniqueness_commitment_sha256": person_commitment_sha256,
                "person_uniqueness_evidence_sha256": request.verification_record_sha256,
                "raw_identity_retention_prohibited": True,
                "qualification_reference_sha256": hashlib.sha256(
                    request.qualification_reference.encode()
                ).hexdigest(),
                "affiliation_class": request.affiliation_class,
                "conflict_disclosure_reference_sha256": hashlib.sha256(
                    request.conflict_disclosure_reference.encode()
                ).hexdigest(),
                "consent_document_sha256": request.consent_document_sha256,
                "model_response_review_authorized": False,
                "task_adjudication_authorized": request.adjudication_authorized,
                "evidence_verified_by_admin": request.evidence_verified_by_admin,
                "public_identity_pseudonymous": True,
            },
        )
    )
    session.commit()
    return {
        "reviewerId": reviewer.id,
        "invitation": invitation,
        "qualifiedFamilies": reviewer.qualification_json,
        "cohort": reviewer.cohort,
        "qualificationVerified": True,
        "modelResponseReviewAuthorized": False,
        "taskAdjudicationAuthorized": request.adjudication_authorized,
        "publicIdentityPseudonymous": True,
        "notice": "The invitation is returned once and is not stored in plaintext.",
    }


@router.post(
    "/admin/seasons/{season_slug}/participant-enrollment-offers",
    dependencies=[Depends(require_admin_token)],
)
def admin_issue_participant_enrollment_offer(
    season_slug: str,
    request: ParticipantEnrollmentOfferCreate,
    session: Db,
) -> dict[str, Any]:
    """Return one identity-free enrollment credential exactly once."""

    season = session.scalar(select(Season).where(Season.slug == season_slug))
    if season is None:
        raise HTTPException(status_code=404, detail="season not found")
    try:
        token, offer = issue_enrollment_offer(
            session,
            season=season,
            consent_document_sha256=request.consent_document_sha256,
            ttl_seconds=request.ttl_seconds,
        )
        session.commit()
    except (ParticipantLifecycleError, IntegrityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="participant enrollment is not currently authorized",
        ) from exc
    return {
        "enrollmentCredential": token,
        "consentDocumentSha256": offer.consent_document_sha256,
        "activationManifestSha256": offer.activation_manifest_sha256,
        "expiresAt": _utc_iso(offer.expires_at),
        "maximumUses": 1,
        "identityCollected": False,
        "notice": "The plaintext credential is returned once and persisted only as an HMAC.",
    }


@router.get("/participant/enrollment/consent")
def participant_enrollment_consent(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    try:
        return enrollment_consent_view(
            session,
            enrollment_token=_participant_bearer(authorization),
        )
    except ParticipantLifecycleError as exc:
        raise HTTPException(
            status_code=401,
            detail="participant enrollment credential is invalid or unavailable",
        ) from exc


@router.post("/participant/enrollment/consent-acceptance")
def participant_accept_consent(
    request: ParticipantConsentAcceptanceCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    try:
        result = accept_participant_consent(
            session,
            enrollment_token=_participant_bearer(authorization),
            consent_document_sha256=request.consent_document_sha256,
            activation_manifest_sha256=request.activation_manifest_sha256,
            confirmations=list(request.confirmations),
            idempotency_key=request.idempotency_key,
        )
        session.commit()
    except (ParticipantLifecycleError, IntegrityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="participant consent acceptance was not authorized",
        ) from exc
    return {
        "consentReceiptCredential": result.receipt_credential,
        "consentReceiptSha256": result.acceptance.receipt_sha256,
        "consentDocumentSha256": result.acceptance.consent_document_sha256,
        "activationManifestSha256": result.acceptance.activation_manifest_sha256,
        "acceptedAt": _utc_iso(result.acceptance.accepted_at),
        "identityCollected": False,
        "idempotent": result.idempotent,
        "notice": "The receipt credential is returned once and persisted only as an HMAC.",
    }


@router.post("/participant/enrollment/identity")
def participant_enroll_identity(
    request: ParticipantIdentityEnrollmentCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    try:
        result = enroll_participant_identity(
            session,
            receipt_credential=_participant_bearer(authorization),
            identity_issuer=request.identity_issuer,
            issuer_subject=request.issuer_subject.get_secret_value(),
            identity_evidence_sha256=request.identity_evidence_sha256,
            roles=list(request.roles),
            qualified_families=[family.value for family in request.qualified_families],
            affiliation_class=request.affiliation_class,
        )
        session.commit()
    except (ParticipantLifecycleError, IntegrityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="participant identity enrollment was not authorized",
        ) from exc
    return {
        "participationStatus": "active",
        "auditMarkerSha256": result.lifecycle.audit_marker_sha256,
        "reviewerCredential": result.reviewer_credential,
        "reviewerCredentialExpiresAt": _utc_iso(
            result.reviewer_credential_expires_at
        ),
        "rawIdentityPersisted": False,
        "contactDataPersisted": False,
        "notice": "The issuer subject was transformed in memory and was not persisted.",
    }


@router.get("/participant/status")
def participant_status(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    try:
        return privacy_safe_participant_status(
            session,
            receipt_credential=_participant_bearer(authorization),
        )
    except ParticipantLifecycleError as exc:
        raise HTTPException(status_code=401, detail="participant credential is invalid") from exc


@router.post("/participant/withdrawal")
def participant_withdraw(
    request: ParticipantWithdrawalCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    try:
        receipt = withdraw_participant(
            session,
            receipt_credential=_participant_bearer(authorization),
            idempotency_key=request.idempotency_key,
            reason_code=request.reason_code,
        )
        session.commit()
    except (ParticipantLifecycleError, IntegrityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="participant withdrawal could not be completed",
        ) from exc
    return {
        "participationStatus": "withdrawn",
        "withdrawalReceiptSha256": receipt.receipt_sha256,
        "effectiveAt": _utc_iso(receipt.effective_at),
        "credentialsRevokedCount": receipt.credentials_revoked_count,
        "assignmentsStoppedCount": receipt.assignments_stopped_count,
        "priorJudgmentsPreserved": receipt.prior_judgments_preserved,
    }


@router.post("/participant/private-payload-deletion")
def participant_delete_private_payload(
    request: ReviewerPrivatePayloadDeletionCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    try:
        receipt = execute_participant_private_payload_deletion(
            session,
            receipt_credential=_participant_bearer(authorization),
            idempotency_key=request.idempotency_key,
        )
        session.commit()
    except (ParticipantLifecycleError, IntegrityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="participant private-payload deletion could not be completed",
        ) from exc
    return {
        "participationStatus": "redacted",
        "deletionReceiptSha256": receipt.receipt_sha256,
        "executedAt": _utc_iso(receipt.executed_at),
        "redactedFields": receipt.redacted_fields_json,
        "auditMarkerSha256": receipt.audit_marker_sha256,
        "pseudonymousAuditRetainUntil": _utc_iso(
            receipt.pseudonymous_audit_retain_until
        ),
        "priorJudgmentsPreserved": receipt.prior_judgments_preserved,
    }


@router.post(
    "/admin/reviewers/{reviewer_id}/retention-schedule",
    dependencies=[Depends(require_admin_token)],
)
def admin_create_reviewer_retention_schedule(
    reviewer_id: str,
    request: ReviewerRetentionScheduleCreate,
    session: Db,
) -> dict[str, Any]:
    try:
        schedule = create_retention_schedule(
            session,
            reviewer_id=reviewer_id,
            analysis_freeze_at=request.analysis_freeze_at,
            first_public_release_at=request.first_public_release_at,
        )
        session.commit()
    except (ParticipantLifecycleError, IntegrityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="reviewer retention schedule could not be created",
        ) from exc
    return {
        "scheduleSha256": schedule.schedule_sha256,
        "directPayloadDeleteDueAt": _utc_iso(schedule.direct_payload_delete_due_at),
        "pseudonymousAuditRetainUntil": _utc_iso(
            schedule.pseudonymous_audit_retain_until
        ),
    }


@router.post(
    "/admin/reviewers/{reviewer_id}/scheduled-private-payload-deletion",
    dependencies=[Depends(require_admin_token)],
)
def admin_execute_scheduled_reviewer_deletion(
    reviewer_id: str,
    request: ReviewerPrivatePayloadDeletionCreate,
    session: Db,
) -> dict[str, Any]:
    try:
        receipt = execute_private_payload_deletion(
            session,
            reviewer_id=reviewer_id,
            idempotency_key=request.idempotency_key,
            execution_basis="scheduled_retention",
        )
        session.commit()
    except (ParticipantLifecycleError, IntegrityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="scheduled reviewer private-payload deletion could not be completed",
        ) from exc
    return {
        "deletionReceiptSha256": receipt.receipt_sha256,
        "executedAt": _utc_iso(receipt.executed_at),
        "redactedFields": receipt.redacted_fields_json,
        "auditMarkerSha256": receipt.audit_marker_sha256,
        "priorJudgmentsPreserved": receipt.prior_judgments_preserved,
    }


@router.post("/admin/experts", dependencies=[Depends(require_admin_token)])
def admin_create_expert(request: ExpertInviteCreate, session: Db) -> dict:
    if get_settings().environment == "production":
        raise HTTPException(
            status_code=409,
            detail="production reviewer enrollment requires participant-owned consent",
        )
    existing = session.scalar(
        select(ExpertReviewer).where(ExpertReviewer.reviewer_code == request.reviewer_code)
    )
    if existing:
        raise HTTPException(status_code=409, detail="reviewer code already exists")
    if resolve_expert_consent_document(request.consent_document_sha256).status != "active":
        raise HTTPException(
            status_code=409,
            detail="expert invitation requires an operationally governed active consent document",
        )
    invitation = secrets.token_urlsafe(32)
    reviewer = ExpertReviewer(
        reviewer_code=request.reviewer_code,
        invitation_sha256=hashlib.sha256(invitation.encode()).hexdigest(),
        qualification_json=[family.value for family in request.qualified_families],
        qualification_verified=False,
        cohort={
            "independent_external": "expert_independent",
            "product_affiliated": "expert_product_affiliated",
            "provider_affiliated": "expert_provider_affiliated",
        }[request.affiliation_class],
        profile_json={
            "qualification_reference": request.qualification_reference,
            "affiliation_class": request.affiliation_class,
            "conflict_disclosure_reference": request.conflict_disclosure_reference,
            "consent_document_sha256": request.consent_document_sha256,
            "training_material_sha256": request.training_material_sha256,
            "calibration_set_sha256": request.calibration_set_sha256,
            "calibration_accuracy": None,
            "claimed_calibration_accuracy": request.calibration_accuracy,
            "requested_qualification_verified": request.qualification_verified,
            "compensation_reference": request.compensation_reference,
        },
        batch_reveal_only=True,
    )
    session.add(reviewer)
    session.commit()
    return {
        "reviewerId": reviewer.id,
        "invitation": invitation,
        "qualifiedFamilies": reviewer.qualification_json,
        "cohort": reviewer.cohort,
        "qualificationVerified": False,
        "notice": (
            "The invitation is returned once and is not stored in plaintext. "
            "Review remains disabled until evidence-bound admission is complete."
        ),
    }


@router.post(
    "/admin/seasons/{season_slug}/reviewers/{reviewer_id}/identity-binding",
    dependencies=[Depends(require_admin_token)],
)
def admin_bind_reviewer_identity(
    season_slug: str,
    reviewer_id: str,
    request: ReviewerIdentityBindingCreate,
    session: Db,
) -> dict[str, Any]:
    """Bind one person to one season without retaining the issuer subject."""

    if get_settings().environment == "production":
        raise HTTPException(
            status_code=409,
            detail="production identity binding requires participant-owned consent",
        )

    season = session.scalar(select(Season).where(Season.slug == season_slug))
    reviewer = session.get(ExpertReviewer, reviewer_id)
    if season is None or reviewer is None or not reviewer.active:
        raise HTTPException(status_code=404, detail="active season reviewer not found")
    if not _expert_consent_document_active(reviewer):
        raise HTTPException(
            status_code=409,
            detail=(
                "reviewer identity binding requires an operationally governed active "
                "consent document"
            ),
        )
    try:
        binding = bind_reviewer_identity(
            session,
            season=season,
            reviewer=reviewer,
            identity_issuer=request.identity_issuer,
            issuer_subject=request.issuer_subject.get_secret_value(),
            identity_evidence_sha256=request.identity_evidence_sha256,
            roles=request.roles,
        )
        enrollment_token, credential = issue_reviewer_credential(
            session,
            binding=binding,
            credential_kind="enrollment_once",
            scopes=["exchange_review_session"],
        )
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="this reviewer or person already has a season identity binding",
        ) from exc
    except (ReviewerIdentityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="reviewer identity binding is invalid") from exc
    return {
        "season": season.slug,
        "reviewerId": reviewer.id,
        "identityBindingId": binding.id,
        "roles": binding.roles_json,
        "enrollmentCredential": enrollment_token,
        "enrollmentCredentialExpiresAt": credential.expires_at.isoformat(),
        "enrollmentCredentialMaximumUses": 1,
        "notice": (
            "The enrollment credential is returned once. The issuer subject and plaintext "
            "credential were not persisted."
        ),
    }


@router.post("/expert/credentials/exchange")
def exchange_reviewer_enrollment_credential(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="enrollment credential is required")
    try:
        session_token, credential = exchange_enrollment_credential(
            session,
            enrollment_token=token,
            session_scopes=["expert_review"],
        )
        session.commit()
    except ReviewerIdentityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=401,
            detail="enrollment credential is invalid, expired, or already consumed",
        ) from exc
    return {
        "reviewerCredential": session_token,
        "expiresAt": credential.expires_at.isoformat(),
        "maximumUses": credential.maximum_uses,
        "notice": "The bounded review credential is returned once and stored only as an HMAC.",
    }


@router.post(
    "/admin/seasons/{season_slug}/reviewers/{reviewer_id}/qualification-evidence",
    dependencies=[Depends(require_admin_token)],
)
def admin_record_reviewer_qualification_evidence(
    season_slug: str,
    reviewer_id: str,
    request: ReviewerQualificationEvidenceCreate,
    session: Db,
) -> dict[str, Any]:
    season = session.scalar(select(Season).where(Season.slug == season_slug))
    reviewer = session.get(ExpertReviewer, reviewer_id)
    if season is None or reviewer is None or not reviewer.active:
        raise HTTPException(status_code=404, detail="active season reviewer not found")
    binding = session.scalar(
        select(ReviewerIdentityBinding).where(
            ReviewerIdentityBinding.season_id == season.id,
            ReviewerIdentityBinding.reviewer_id == reviewer.id,
            ReviewerIdentityBinding.assurance_level == "server_verified",
        )
    )
    expected_cohort = {
        "independent_external": "expert_independent",
        "product_affiliated": "expert_product_affiliated",
        "provider_affiliated": "expert_provider_affiliated",
    }[request.affiliation_class]
    if (
        binding is None
        or request.family.value not in reviewer.qualification_json
        or reviewer.cohort != expected_cohort
        or resolve_expert_consent_document(request.consent_document_sha256).status != "active"
    ):
        raise HTTPException(
            status_code=409,
            detail="reviewer qualification evidence is outside the admitted season scope",
        )
    try:
        evidence = record_qualification_evidence(
            session,
            binding=binding,
            family=request.family.value,
            affiliation_class=request.affiliation_class,
            independence_verified=request.independence_verified,
            conflict_cleared=request.conflict_cleared,
            qualification_evidence_sha256=request.qualification_evidence_sha256,
            independence_evidence_sha256=request.independence_evidence_sha256,
            conflict_disclosure_sha256=request.conflict_disclosure_sha256,
            consent_document_sha256=request.consent_document_sha256,
            training_material_sha256=request.training_material_sha256,
            verifier_principal_sha256=request.verifier_principal_sha256,
            verified_at=request.verified_at,
            valid_until=request.valid_until,
        )
        session.commit()
    except (IntegrityError, ReviewerIdentityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="reviewer qualification evidence is invalid or already registered",
        ) from exc
    return {
        "qualificationEvidenceId": evidence.id,
        "identityBindingId": binding.id,
        "family": evidence.family,
        "affiliationClass": evidence.affiliation_class,
        "verified": True,
    }


@router.post(
    "/admin/seasons/{season_slug}/reviewer-calibration-sets",
    dependencies=[Depends(require_admin_token)],
)
def admin_freeze_reviewer_calibration_set(
    season_slug: str,
    request: ReviewerCalibrationSetCreate,
    session: Db,
) -> dict[str, Any]:
    season = session.scalar(select(Season).where(Season.slug == season_slug))
    if season is None:
        raise HTTPException(status_code=404, detail="season not found")
    try:
        calibration_set = freeze_calibration_set(
            session,
            season=season,
            family=request.family.value,
            calibration_set_sha256=request.calibration_set_sha256,
            source_artifact_sha256=request.source_artifact_sha256,
            scoring_key_sha256=request.scoring_key_sha256,
            item_count=request.item_count,
            real_source_arms=request.real_source_arms,
            frozen_at=request.frozen_at,
        )
        session.commit()
    except (IntegrityError, ReviewerIdentityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="reviewer calibration set is invalid or already frozen",
        ) from exc
    return {
        "calibrationSetId": calibration_set.id,
        "family": calibration_set.family,
        "itemCount": calibration_set.item_count,
        "realSourceArms": calibration_set.real_source_arms,
        "syntheticArms": 0,
    }


@router.post(
    "/admin/seasons/{season_slug}/reviewers/{reviewer_id}/calibration-ballots",
    dependencies=[Depends(require_admin_token)],
)
def admin_record_reviewer_calibration_ballot(
    season_slug: str,
    reviewer_id: str,
    request: ReviewerCalibrationBallotCreate,
    session: Db,
) -> dict[str, Any]:
    season = session.scalar(select(Season).where(Season.slug == season_slug))
    reviewer = session.get(ExpertReviewer, reviewer_id)
    if season is None or reviewer is None or not reviewer.active:
        raise HTTPException(status_code=404, detail="active season reviewer not found")
    binding = session.scalar(
        select(ReviewerIdentityBinding).where(
            ReviewerIdentityBinding.season_id == season.id,
            ReviewerIdentityBinding.reviewer_id == reviewer.id,
        )
    )
    calibration_set = session.get(ReviewerCalibrationSet, request.calibration_set_id)
    if (
        binding is None
        or calibration_set is None
        or calibration_set.season_id != season.id
        or request.correct_count > calibration_set.item_count
    ):
        raise HTTPException(status_code=409, detail="calibration ballot scope is invalid")
    try:
        ballot = record_calibration_ballot(
            session,
            binding=binding,
            calibration_set=calibration_set,
            ballot_sha256=request.ballot_sha256,
            scoring_result_sha256=request.scoring_result_sha256,
            correct_count=request.correct_count,
            minimum_accuracy_milli=request.minimum_accuracy_milli,
            completed_at=request.completed_at,
        )
        session.commit()
    except (IntegrityError, ReviewerIdentityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="calibration ballot is invalid or already recorded",
        ) from exc
    return {
        "calibrationBallotId": ballot.id,
        "accuracyMilli": ballot.accuracy_milli,
        "passed": ballot.passed,
    }


@router.post(
    "/admin/seasons/{season_slug}/reviewers/{reviewer_id}/family-admissions",
    dependencies=[Depends(require_admin_token)],
)
def admin_derive_reviewer_family_admission(
    season_slug: str,
    reviewer_id: str,
    request: ReviewerFamilyAdmissionCreate,
    session: Db,
) -> dict[str, Any]:
    season = session.scalar(select(Season).where(Season.slug == season_slug))
    reviewer = session.get(ExpertReviewer, reviewer_id)
    if season is None or reviewer is None or not reviewer.active:
        raise HTTPException(status_code=404, detail="active season reviewer not found")
    binding = session.scalar(
        select(ReviewerIdentityBinding).where(
            ReviewerIdentityBinding.season_id == season.id,
            ReviewerIdentityBinding.reviewer_id == reviewer.id,
        )
    )
    qualification = session.get(
        ReviewerQualificationEvidence,
        request.qualification_evidence_id,
    )
    calibration_ballot = (
        session.get(ReviewerCalibrationBallot, request.calibration_ballot_id)
        if request.calibration_ballot_id is not None
        else None
    )
    if (
        binding is None
        or qualification is None
        or qualification.identity_binding_id != binding.id
        or qualification.family != request.family.value
        or request.review_role not in binding.roles_json
        or (
            request.calibration_ballot_id is not None
            and (calibration_ballot is None or calibration_ballot.identity_binding_id != binding.id)
        )
    ):
        raise HTTPException(status_code=409, detail="reviewer family admission scope is invalid")
    policy = {
        "schema_version": "flavourbench-reviewer-admission-policy-v1",
        "requires_calibration": request.requires_calibration,
        "minimum_accuracy_milli": request.minimum_accuracy_milli,
    }
    try:
        admission = derive_family_admission(
            session,
            binding=binding,
            qualification=qualification,
            calibration_ballot=calibration_ballot,
            family=request.family.value,
            review_role=request.review_role,
            cohort=reviewer.cohort,
            admission_policy=policy,
            decision_reference_sha256=request.decision_reference_sha256,
            valid_from=request.valid_from,
            valid_until=request.valid_until,
        )
        session.commit()
    except (IntegrityError, ReviewerIdentityError, ValueError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="reviewer family admission is invalid or already recorded",
        ) from exc
    return {
        "familyAdmissionId": admission.id,
        "identityBindingId": binding.id,
        "family": admission.family,
        "reviewRole": admission.review_role,
        "cohort": admission.cohort,
        "validFrom": admission.valid_from.isoformat(),
        "validUntil": admission.valid_until.isoformat(),
    }


@router.put(
    "/admin/experts/{reviewer_id}/calibration-candidate",
    dependencies=[Depends(require_admin_token)],
)
def admin_register_expert_calibration_candidate(
    reviewer_id: str,
    request: ExpertCalibrationCandidateRegister,
    session: Db,
) -> dict[str, Any]:
    reviewer = reviewer_control_lock(session, reviewer_id)
    if reviewer is None or not reviewer.active:
        raise HTTPException(status_code=404, detail="active reviewer not found")
    if reviewer.qualification_verified:
        raise HTTPException(
            status_code=409,
            detail="an admitted reviewer cannot replace calibration candidate evidence",
        )
    candidate = request.model_dump(mode="json")
    existing = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_reviewer",
            RunEvent.entity_id == reviewer.id,
            RunEvent.event_type == "expert_calibration_candidate_registered",
        )
        .order_by(RunEvent.created_at.desc())
    )
    if existing is not None:
        existing_candidate = existing.payload_json.get("candidate")
        if existing_candidate != candidate:
            raise HTTPException(
                status_code=409,
                detail=(
                    "calibration candidate evidence is immutable; register a governed correction"
                ),
            )
        return {
            "reviewerId": reviewer.id,
            "reviewerCode": reviewer.reviewer_code,
            "candidatePackSha256": request.candidate_pack_sha256,
            "candidatePairs": request.candidate_pairs,
            "status": request.status,
            "idempotent": True,
        }

    reviewer.profile_json = {
        **reviewer.profile_json,
        "calibration_candidate": candidate,
    }
    event = RunEvent(
        entity_type="expert_reviewer",
        entity_id=reviewer.id,
        event_type="expert_calibration_candidate_registered",
        payload_json={
            "reviewer_code": reviewer.reviewer_code,
            "cohort": reviewer.cohort,
            "candidate": candidate,
            "candidate_record_sha256": canonical_sha256(candidate),
        },
    )
    session.add_all([reviewer, event])
    session.commit()
    return {
        "reviewerId": reviewer.id,
        "reviewerCode": reviewer.reviewer_code,
        "candidatePackSha256": request.candidate_pack_sha256,
        "candidatePairs": request.candidate_pairs,
        "status": request.status,
        "idempotent": False,
        "eventId": event.id,
    }


@router.put(
    "/admin/experts/{reviewer_id}/admission",
    dependencies=[Depends(require_admin_token)],
)
def admin_admit_expert(
    reviewer_id: str,
    request: ExpertAdmissionCreate,
    session: Db,
) -> dict:
    reviewer = session.get(ExpertReviewer, reviewer_id)
    if reviewer is None or not reviewer.active:
        raise HTTPException(status_code=404, detail="active reviewer not found")
    if resolve_expert_consent_document(request.consent_document_sha256).status != "active":
        raise HTTPException(
            status_code=409,
            detail="expert admission requires an operationally governed active consent document",
        )
    calibration_candidate = reviewer.profile_json.get("calibration_candidate")
    if not isinstance(calibration_candidate, dict):
        raise HTTPException(
            status_code=409,
            detail="real-output calibration candidate evidence is not registered",
        )
    if (
        calibration_candidate.get("synthetic_arms") != 0
        or calibration_candidate.get("rank_eligible") is not False
        or calibration_candidate.get("candidate_pairs", 0) < request.calibration_item_count
    ):
        raise HTTPException(
            status_code=409,
            detail="registered calibration candidate evidence is inadmissible",
        )

    admitted_profile = {
        **reviewer.profile_json,
        "qualification_reference": request.qualification_reference,
        "conflict_disclosure_reference": request.conflict_disclosure_reference,
        "consent_document_sha256": request.consent_document_sha256,
        "training_material_sha256": request.training_material_sha256,
        "calibration_set_sha256": request.calibration_set_sha256,
        "calibration_item_count": request.calibration_item_count,
        "calibration_gold_adjudicator_count": (request.calibration_gold_adjudicator_count),
        "calibration_accuracy": request.calibration_accuracy,
        "admission_decision_reference": request.admission_decision_reference,
        "admission_decision_sha256": hashlib.sha256(
            request.admission_decision_reference.encode()
        ).hexdigest(),
    }
    evidence_fields = (
        "qualification_reference",
        "conflict_disclosure_reference",
        "consent_document_sha256",
        "training_material_sha256",
        "calibration_set_sha256",
        "calibration_item_count",
        "calibration_gold_adjudicator_count",
        "calibration_accuracy",
        "admission_decision_reference",
        "admission_decision_sha256",
    )
    if reviewer.qualification_verified:
        unchanged = all(
            reviewer.profile_json.get(field) == admitted_profile[field] for field in evidence_fields
        )
        if not unchanged:
            raise HTTPException(
                status_code=409,
                detail="reviewer is already admitted under different evidence",
            )
        return {
            "reviewerId": reviewer.id,
            "reviewerCode": reviewer.reviewer_code,
            "qualificationVerified": True,
            "calibrationAccuracy": request.calibration_accuracy,
            "admissionStatus": "active",
            "idempotent": True,
        }

    reviewer.profile_json = admitted_profile
    reviewer.qualification_verified = True
    event = RunEvent(
        entity_type="expert_reviewer",
        entity_id=reviewer.id,
        event_type="expert_reviewer_admitted",
        payload_json={
            "reviewer_code": reviewer.reviewer_code,
            "cohort": reviewer.cohort,
            "qualified_families": reviewer.qualification_json,
            "affiliation_class": admitted_profile.get("affiliation_class"),
            "admission_protocol_version": "expert-admission-v2",
            "consent_active_at_admission": True,
            "calibration_candidate": calibration_candidate,
            "calibration_candidate_record_sha256": canonical_sha256(calibration_candidate),
            "evidence": {field: admitted_profile[field] for field in evidence_fields},
        },
    )
    session.add_all([reviewer, event])
    session.commit()
    return {
        "reviewerId": reviewer.id,
        "reviewerCode": reviewer.reviewer_code,
        "qualificationVerified": True,
        "calibrationAccuracy": request.calibration_accuracy,
        "admissionStatus": "active",
        "idempotent": False,
        "eventId": event.id,
    }


@router.put(
    "/admin/experts/{reviewer_id}/author-evaluator-admission",
    dependencies=[Depends(require_admin_token)],
)
def admin_admit_author_evaluator(
    reviewer_id: str,
    request: AuthorEvaluatorAdmissionCreate,
    session: Db,
) -> dict[str, Any]:
    reviewer = session.get(ExpertReviewer, reviewer_id)
    if reviewer is None or not reviewer.active:
        raise HTTPException(status_code=404, detail="active reviewer not found")
    if reviewer.cohort != "expert_product_affiliated":
        raise HTTPException(
            status_code=409,
            detail="author-evaluator admission requires the disclosed affiliated cohort",
        )
    calibration_candidate = reviewer.profile_json.get("calibration_candidate")
    if not isinstance(calibration_candidate, dict):
        raise HTTPException(
            status_code=409,
            detail="real-output candidate evidence is not registered",
        )
    requested_family_counts = (
        {str(family): int(count) for family, count in request.primary_by_family.items()}
        if request.primary_by_family is not None
        else None
    )
    if (
        calibration_candidate.get("candidate_pack_sha256") != request.candidate_pack_sha256
        or calibration_candidate.get("synthetic_arms") != 0
        or calibration_candidate.get("rank_eligible") is not False
        or calibration_candidate.get("candidate_pairs") != request.primary_judgments
        or calibration_candidate.get("source_arms") != request.primary_judgments * 2
        or (
            requested_family_counts is not None
            and calibration_candidate.get("candidate_pairs_by_family") != requested_family_counts
        )
    ):
        raise HTTPException(
            status_code=409,
            detail="registered real-output evidence does not match the author review pool",
        )
    targets = (
        isolated_uplift_workload_cell_targets(requested_family_counts)
        if requested_family_counts is not None
        else author_evaluator_workload_cell_targets(request.primary_judgments)
    )
    admitted_profile = {
        **reviewer.profile_json,
        "qualification_reference": request.qualification_reference,
        "conflict_disclosure_reference": request.conflict_disclosure_reference,
        "admission_pathway": "author_evaluator",
        "author_evaluator_admission_status": "active",
        "author_evaluator_pool_sha256": request.candidate_pack_sha256,
        "author_evaluator_primary_judgments": request.primary_judgments,
        "author_evaluator_reliability_repeats": targets["reliability_repeats"],
        "author_evaluator_target_judgments": targets["total_presentations"],
        "independent_validation_claim": False,
        "external_calibration_required": False,
        "calibration_candidate_use": "optional_future_external_validation",
        "reliability_basis": "concealed_side_swapped_repeats",
        "admission_decision_reference": request.admission_decision_reference,
        "admission_decision_sha256": hashlib.sha256(
            request.admission_decision_reference.encode()
        ).hexdigest(),
    }
    if requested_family_counts is not None:
        admitted_profile["author_evaluator_primary_by_family"] = requested_family_counts
    evidence_fields = [
        "qualification_reference",
        "conflict_disclosure_reference",
        "admission_pathway",
        "author_evaluator_admission_status",
        "author_evaluator_pool_sha256",
        "author_evaluator_primary_judgments",
        "author_evaluator_reliability_repeats",
        "author_evaluator_target_judgments",
        "independent_validation_claim",
        "external_calibration_required",
        "calibration_candidate_use",
        "reliability_basis",
        "admission_decision_reference",
        "admission_decision_sha256",
    ]
    if requested_family_counts is not None:
        evidence_fields.append("author_evaluator_primary_by_family")
    existing = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_reviewer",
            RunEvent.entity_id == reviewer.id,
            RunEvent.event_type == "expert_author_evaluator_admitted",
        )
        .order_by(RunEvent.created_at.desc())
    )
    if existing is not None or _author_evaluator_active(reviewer):
        unchanged = all(
            reviewer.profile_json.get(field) == admitted_profile[field] for field in evidence_fields
        )
        if not unchanged:
            raise HTTPException(
                status_code=409,
                detail="author evaluator is already admitted under different evidence",
            )
        return {
            "reviewerId": reviewer.id,
            "reviewerCode": reviewer.reviewer_code,
            "qualificationVerified": True,
            "admissionPathway": "author_evaluator",
            "admissionStatus": "active",
            "primaryJudgments": targets["primary_judgments"],
            "reliabilityRepeats": targets["reliability_repeats"],
            "targetJudgments": targets["total_presentations"],
            "independentValidationClaim": False,
            "idempotent": True,
        }

    reviewer.profile_json = admitted_profile
    reviewer.qualification_verified = True
    event = RunEvent(
        entity_type="expert_reviewer",
        entity_id=reviewer.id,
        event_type="expert_author_evaluator_admitted",
        payload_json={
            "reviewer_code": reviewer.reviewer_code,
            "cohort": reviewer.cohort,
            "qualified_families": reviewer.qualification_json,
            "evidence": {field: admitted_profile[field] for field in evidence_fields},
            "workload": targets,
            "claim_boundary": (
                "Blinded single-rater author-evaluator case study; "
                "not independent expert validation and never silently pooled."
            ),
        },
    )
    session.add_all([reviewer, event])
    session.commit()
    return {
        "reviewerId": reviewer.id,
        "reviewerCode": reviewer.reviewer_code,
        "qualificationVerified": True,
        "admissionPathway": "author_evaluator",
        "admissionStatus": "active",
        "primaryJudgments": targets["primary_judgments"],
        "reliabilityRepeats": targets["reliability_repeats"],
        "targetJudgments": targets["total_presentations"],
        "independentValidationClaim": False,
        "idempotent": False,
        "eventId": event.id,
    }


@router.put(
    "/admin/experts/{reviewer_id}/anonymous-external-admission",
    dependencies=[Depends(require_admin_token)],
)
def admin_admit_anonymous_external_rater(
    reviewer_id: str,
    request: AnonymousExternalAdmissionCreate,
    session: Db,
) -> dict[str, Any]:
    """Admit an identity-minimized rater without presenting self-attestation as verification."""

    reviewer = reviewer_control_lock(session, reviewer_id)
    if reviewer is None or not reviewer.active:
        raise HTTPException(status_code=404, detail="active reviewer not found")
    if reviewer.cohort != "expert_independent":
        raise HTTPException(
            status_code=409,
            detail="anonymous external admission requires the independent cohort",
        )
    if reviewer.qualification_verified:
        raise HTTPException(
            status_code=409,
            detail=(
                "anonymous self-attested admission cannot reuse a "
                "qualification-verified reviewer record"
            ),
        )
    if not _expert_consent_document_active(reviewer):
        raise HTTPException(
            status_code=409,
            detail=(
                "anonymous external admission requires a consent document in the "
                "active expert-consent registry"
            ),
        )
    if not reviewer.reviewer_code.startswith("fbr-anon-"):
        raise HTTPException(
            status_code=409,
            detail="anonymous reviewer codes must use the fbr-anon- prefix",
        )
    calibration_candidate = reviewer.profile_json.get("calibration_candidate")
    if not isinstance(calibration_candidate, dict):
        raise HTTPException(
            status_code=409,
            detail="real-output candidate evidence is not registered",
        )
    requested_family_counts = (
        {str(family): int(count) for family, count in request.primary_by_family.items()}
        if request.primary_by_family is not None
        else None
    )
    if (
        calibration_candidate.get("candidate_pack_sha256") != request.candidate_pack_sha256
        or calibration_candidate.get("synthetic_arms") != 0
        or calibration_candidate.get("rank_eligible") is not False
        or calibration_candidate.get("candidate_pairs") != request.primary_judgments
        or calibration_candidate.get("source_arms") != request.primary_judgments * 2
        or (
            requested_family_counts is not None
            and calibration_candidate.get("candidate_pairs_by_family") != requested_family_counts
        )
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "registered real-output evidence does not match the anonymous external review pool"
            ),
        )

    targets = (
        isolated_uplift_workload_cell_targets(requested_family_counts)
        if requested_family_counts is not None
        else author_evaluator_workload_cell_targets(request.primary_judgments)
    )
    admission_decision_sha256 = hashlib.sha256(
        request.admission_decision_reference.encode()
    ).hexdigest()
    pool_activation_sha256 = admission_activation_sha256(
        reviewer_id=reviewer.id,
        pool_sha256=request.candidate_pack_sha256,
        admission_decision_sha256=admission_decision_sha256,
    )
    admitted_profile = {
        **reviewer.profile_json,
        "qualification_reference": (
            "No identity or credential collected; culinary competence is "
            "self-attested when the review session opens."
        ),
        "qualification_basis": "reviewer_self_attestation_unverified",
        "conflict_disclosure_reference": (
            "Independence and absence of undisclosed conflicts are self-attested "
            "when the review session opens."
        ),
        "independence_basis": "reviewer_self_attestation",
        "admission_pathway": "anonymous_external_rater",
        "anonymous_external_admission_status": "active",
        "anonymous_external_pool_sha256": request.candidate_pack_sha256,
        "anonymous_external_pool_activation_sha256": pool_activation_sha256,
        "anonymous_external_primary_judgments": request.primary_judgments,
        "anonymous_external_reliability_repeats": targets["reliability_repeats"],
        "anonymous_external_target_judgments": targets["total_presentations"],
        "identity_collection_prohibited": request.identity_collection_prohibited,
        "independence_self_attestation_required": (request.independence_self_attestation_required),
        "qualification_self_attestation_required": (
            request.qualification_self_attestation_required
        ),
        "independent_expert_validation_claim": False,
        "external_calibration_required": False,
        "reliability_basis": "concealed_side_swapped_repeats",
        "admission_decision_reference": request.admission_decision_reference,
        "admission_decision_sha256": admission_decision_sha256,
    }
    if requested_family_counts is not None:
        admitted_profile["anonymous_external_primary_by_family"] = requested_family_counts
    evidence_fields = [
        "qualification_reference",
        "qualification_basis",
        "conflict_disclosure_reference",
        "independence_basis",
        "admission_pathway",
        "anonymous_external_admission_status",
        "anonymous_external_pool_sha256",
        "anonymous_external_pool_activation_sha256",
        "anonymous_external_primary_judgments",
        "anonymous_external_reliability_repeats",
        "anonymous_external_target_judgments",
        "identity_collection_prohibited",
        "independence_self_attestation_required",
        "qualification_self_attestation_required",
        "independent_expert_validation_claim",
        "external_calibration_required",
        "reliability_basis",
        "admission_decision_reference",
        "admission_decision_sha256",
    ]
    if requested_family_counts is not None:
        evidence_fields.append("anonymous_external_primary_by_family")
    existing = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_reviewer",
            RunEvent.entity_id == reviewer.id,
            RunEvent.event_type == "expert_anonymous_external_rater_admitted",
        )
        .order_by(RunEvent.created_at.desc())
    )
    if existing is not None or _anonymous_external_rater_profile(reviewer):
        unchanged = all(
            reviewer.profile_json.get(field) == admitted_profile[field] for field in evidence_fields
        )
        if not unchanged:
            raise HTTPException(
                status_code=409,
                detail=("anonymous external rater is already admitted under different evidence"),
            )
        return {
            "reviewerId": reviewer.id,
            "reviewerCode": reviewer.reviewer_code,
            "qualificationVerified": False,
            "qualificationBasis": "reviewer_self_attestation_unverified",
            "admissionPathway": "anonymous_external_rater",
            "admissionStatus": "pending_reconsent",
            "reviewEnabled": anonymous_pool_reconsented(session, reviewer),
            "poolActivationSha256": pool_activation_sha256,
            "primaryJudgments": targets["primary_judgments"],
            "reliabilityRepeats": targets["reliability_repeats"],
            "targetJudgments": targets["total_presentations"],
            "independentExpertValidationClaim": False,
            "idempotent": True,
        }

    reviewer.profile_json = admitted_profile
    event = RunEvent(
        entity_type="expert_reviewer",
        entity_id=reviewer.id,
        event_type="expert_anonymous_external_rater_admitted",
        payload_json={
            "reviewer_code": reviewer.reviewer_code,
            "cohort": reviewer.cohort,
            "qualified_families": reviewer.qualification_json,
            "evidence": {field: admitted_profile[field] for field in evidence_fields},
            "workload": targets,
            "claim_boundary": _reviewer_claim_boundary(reviewer),
        },
    )
    session.add_all([reviewer, event])
    session.commit()
    return {
        "reviewerId": reviewer.id,
        "reviewerCode": reviewer.reviewer_code,
        "qualificationVerified": False,
        "qualificationBasis": "reviewer_self_attestation_unverified",
        "admissionPathway": "anonymous_external_rater",
        "admissionStatus": "pending_reconsent",
        "reviewEnabled": False,
        "poolActivationSha256": pool_activation_sha256,
        "primaryJudgments": targets["primary_judgments"],
        "reliabilityRepeats": targets["reliability_repeats"],
        "targetJudgments": targets["total_presentations"],
        "independentExpertValidationClaim": False,
        "idempotent": False,
        "eventId": event.id,
    }


@router.put(
    "/admin/controlled-runs/{run_id}/reviewers/{reviewer_id}",
    dependencies=[Depends(require_admin_token)],
)
def admin_authorize_controlled_reviewer(
    run_id: str,
    reviewer_id: str,
    request: ControlledReviewerAuthorizationCreate,
    session: Db,
) -> dict:
    _, run = _locked_controlled_run(session, run_id)
    reviewer = session.get(ExpertReviewer, reviewer_id)
    if reviewer is None or not reviewer.qualification_verified:
        raise HTTPException(status_code=404, detail="controlled run or reviewer not found")
    if run.status not in {"active", "collection_complete"}:
        raise HTTPException(
            status_code=409,
            detail="controlled run is not accepting reviewer authorization",
        )
    if request.active:
        try:
            lifecycle = require_active_participant_authority(
                session,
                reviewer_id=reviewer.id,
                season_id=run.season_id,
            )
        except ParticipantLifecycleError as exc:
            raise HTTPException(
                status_code=409,
                detail="reviewer assignment lacks current participant authority",
            ) from exc
        if lifecycle is None and get_settings().environment == "production":
            raise HTTPException(
                status_code=409,
                detail="production reviewer assignment requires participant-owned consent",
            )
    authorization = session.scalar(
        select(ControlledRunReviewer).where(
            ControlledRunReviewer.controlled_run_id == run.id,
            ControlledRunReviewer.reviewer_id == reviewer.id,
        )
    )
    reference_sha256 = hashlib.sha256(request.authorization_reference.encode()).hexdigest()
    if authorization is None:
        authorization = ControlledRunReviewer(
            controlled_run_id=run.id,
            reviewer_id=reviewer.id,
            authorization_reference_sha256=reference_sha256,
            active=request.active,
        )
        session.add(authorization)
    else:
        authorization.authorization_reference_sha256 = reference_sha256
        authorization.active = request.active
    session.add(
        RunEvent(
            entity_type="controlled_run_reviewer",
            entity_id=f"{run.id}:{reviewer.id}",
            event_type=(
                "controlled_run_reviewer_authorized"
                if request.active
                else "controlled_run_reviewer_revoked"
            ),
            payload_json={
                "authorization_reference_sha256": reference_sha256,
                "rater_plan_sha256": run.rater_plan_sha256,
            },
        )
    )
    session.commit()
    return {
        "runId": run.id,
        "reviewerId": reviewer.id,
        "active": authorization.active,
        "raterPlanSha256": run.rater_plan_sha256,
    }


@router.post(
    "/admin/epicure-releases",
    dependencies=[Depends(require_admin_token)],
    status_code=201,
)
def admin_register_epicure_release(
    request: EpicureReleaseRegisterCreate,
    session: Db,
) -> dict:
    if session.get(EpicureRelease, request.release_id) is not None:
        raise HTTPException(status_code=409, detail="Epicure release already exists")
    lineage_manifest = {
        "schema_version": "flavourbench-epicure-release-lineage-v1",
        "release_id": request.release_id,
        "bundle_sha256": request.bundle_sha256,
        "application_sha256": request.application_sha256,
        "public_release_uri": request.public_release_uri,
        "release_artifact_sha256": request.release_artifact_sha256,
        "rights_clearance_sha256": request.rights_clearance_sha256,
        "verification_report_sha256": request.verification_report_sha256,
        "public_release_match": request.public_release_match,
        "redistribution_rights_cleared": request.redistribution_rights_cleared,
        "reproducibility_verified": request.reproducibility_verified,
    }
    if _canonical_sha256(lineage_manifest) != request.lineage_manifest_sha256:
        raise HTTPException(
            status_code=409,
            detail="Epicure lineage manifest hash does not match its content",
        )
    official_eligible = bool(
        request.public_release_match
        and request.redistribution_rights_cleared
        and request.reproducibility_verified
        and _lineage_release_is_named_for_official_use(request.release_id)
    )
    release = EpicureRelease(
        release_id=request.release_id,
        bundle_sha256=request.bundle_sha256,
        application_sha256=request.application_sha256,
        public_release_uri=request.public_release_uri,
        release_artifact_sha256=request.release_artifact_sha256,
        rights_clearance_sha256=request.rights_clearance_sha256,
        verification_report_sha256=request.verification_report_sha256,
        lineage_manifest_json=lineage_manifest,
        lineage_manifest_sha256=request.lineage_manifest_sha256,
        public_release_match=request.public_release_match,
        redistribution_rights_cleared=request.redistribution_rights_cleared,
        reproducibility_verified=request.reproducibility_verified,
        official_eligible=official_eligible,
    )
    session.add(release)
    session.add(
        RunEvent(
            entity_type="epicure_release",
            entity_id=request.release_id,
            event_type="epicure_release_registered",
            payload_json={
                "lineage_manifest_sha256": request.lineage_manifest_sha256,
                "official_eligible": official_eligible,
            },
        )
    )
    session.commit()
    return {
        "releaseId": release.release_id,
        "lineageManifestSha256": release.lineage_manifest_sha256,
        "officialEligible": release.official_eligible,
    }


@router.post(
    "/admin/seasons",
    dependencies=[Depends(require_admin_token)],
    status_code=201,
)
def admin_provision_season(request: SeasonProvisionCreate, session: Db) -> dict:
    if session.scalar(select(Season.id).where(Season.slug == request.slug)) is not None:
        raise HTTPException(status_code=409, detail="season slug already exists")
    season = Season(
        slug=request.slug,
        name=request.name,
        status="draft",
        official=False,
        epicure_release_id="unresolved",
        epicure_bundle_sha256="unresolved",
        epicure_application_sha256="unresolved",
    )
    session.add(season)
    session.flush()
    session.add(
        RunEvent(
            entity_type="season",
            entity_id=season.id,
            event_type="season_provisioned",
            payload_json={"slug": season.slug, "name": season.name},
        )
    )
    session.commit()
    return {"season": season.slug, "status": season.status}


@router.post(
    "/admin/seasons/{season_slug}/tasks/confirmatory",
    dependencies=[Depends(require_admin_token)],
    status_code=201,
)
def admin_import_confirmatory_tasks(
    season_slug: str,
    request: ConfirmatoryTaskBankCreate,
    session: Db,
) -> dict:
    season = session.scalar(select(Season).where(Season.slug == season_slug).with_for_update())
    if season is None or season.status != "draft" or season.frozen_at is not None:
        raise HTTPException(
            status_code=409,
            detail="confirmatory tasks can be imported only into an empty draft season",
        )
    existing = session.scalars(select(Task).where(Task.season_id == season.id)).all()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                "official and development task banks occupy separate seasons; "
                "the target season is not empty"
            ),
        )
    prompt_hashes = [
        hashlib.sha256(item.prompt.encode("utf-8")).hexdigest() for item in request.tasks
    ]
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise HTTPException(status_code=409, detail="confirmatory prompts must be unique")
    validator_calibration, validator_calibration_receipt = _validator_calibration()
    if request.validator_calibration_artifact_sha256 != validator_calibration.artifact_sha256:
        raise HTTPException(
            status_code=409,
            detail="bank validator-calibration digest does not match the configured artifact",
        )
    contamination_scan_bundle = _contamination_scan_bundle()
    contamination_calibration, contamination_calibration_receipt = _contamination_calibration(
        contamination_scan_bundle
    )
    if (
        request.contamination_calibration_artifact_sha256
        != contamination_calibration.artifact_sha256
    ):
        raise HTTPException(
            status_code=409,
            detail=("bank contamination-calibration digest does not match the configured artifact"),
        )
    expected_bank_manifest_sha256 = _canonical_sha256(
        {
            "construct_blueprint_sha256": BLUEPRINT_SHA256,
            "validator_calibration_artifact_sha256": (validator_calibration.artifact_sha256),
            "contamination_calibration_artifact_sha256": (
                contamination_calibration.artifact_sha256
            ),
            "tasks": sorted(
                [
                    {
                        "public_id": item.public_id,
                        "task_record_sha256": item.task_record_sha256,
                        "task_evidence_root_sha256": item.task_evidence_root_sha256,
                    }
                    for item in request.tasks
                ],
                key=lambda row: row["public_id"],
            ),
        }
    )
    if expected_bank_manifest_sha256 != request.bank_manifest_sha256:
        raise HTTPException(
            status_code=409,
            detail="bankManifestSha256 does not match the task-record registry",
        )
    approving_reviewer_ids = {
        review.reviewer_id for item in request.tasks for review in item.independent_reviews
    }
    adjudicator_reviewer_ids = {item.adjudication.adjudicator_reviewer_id for item in request.tasks}
    evidence_reviewer_ids = {
        reviewer_id
        for item in request.tasks
        for reviewer_id in (
            item.validator_contract_review.reviewer_id,
            item.contamination_audit_review.reviewer_id,
        )
    }
    reviewer_ids = approving_reviewer_ids | adjudicator_reviewer_ids | evidence_reviewer_ids
    reviewers = {
        reviewer.id: reviewer
        for reviewer in session.scalars(
            select(ExpertReviewer).where(ExpertReviewer.id.in_(reviewer_ids))
        ).all()
    }
    if set(reviewers) != reviewer_ids:
        raise HTTPException(
            status_code=409,
            detail="confirmatory review evidence names an unknown reviewer",
        )
    author_ids = {item.human_author_id for item in request.tasks}
    contributors = {
        contributor.id: contributor
        for contributor in session.scalars(
            select(ExpertReviewer).where(ExpertReviewer.id.in_(author_ids))
        ).all()
    }
    if set(contributors) != author_ids:
        raise HTTPException(
            status_code=409,
            detail="confirmatory task evidence names an unknown human contributor",
        )
    if any(
        not contributor.active
        or contributor.qualification_verified
        or contributor.cohort != "expert_independent"
        or contributor.profile_json.get("admission_pathway") != "task_contributor"
        or contributor.profile_json.get("task_contributor_status") != "active"
        or contributor.profile_json.get("person_uniqueness_verified") is not True
        or contributor.profile_json.get("person_uniqueness_method")
        != "admin-witnessed-season-hmac-v1"
        or not isinstance(
            contributor.profile_json.get("person_uniqueness_commitment_sha256"),
            str,
        )
        or not _task_contributor_protocol_binding_active(session, contributor)
        for contributor in contributors.values()
    ):
        raise HTTPException(
            status_code=409,
            detail="confirmatory task author lacks privacy-preserving person verification",
        )
    author_person_commitments = {
        str(contributor.profile_json["person_uniqueness_commitment_sha256"])
        for contributor in contributors.values()
    }
    if len(author_person_commitments) != len(contributors):
        raise HTTPException(
            status_code=409,
            detail="confirmatory task author accounts do not represent unique people",
        )
    candidate_ids = {item.source_candidate_id for item in request.tasks}
    candidate_events = {
        event.entity_id: event
        for event in session.scalars(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "task_candidate",
                RunEvent.event_type == "task_candidate_submitted",
                RunEvent.entity_id.in_(candidate_ids),
            )
            .with_for_update()
        ).all()
    }
    if set(candidate_events) != candidate_ids:
        raise HTTPException(
            status_code=409,
            detail="confirmatory task bank names an unsealed source candidate",
        )

    imported: list[Task] = []
    verified_evidence: dict[str, dict[str, Any]] = {}
    container_image_digest = get_settings().build_image_digest
    for item, prompt_sha256 in zip(request.tasks, prompt_hashes, strict=True):
        contributor = contributors[item.human_author_id]
        task_role_ids = {
            item.human_author_id,
            item.adjudication.adjudicator_reviewer_id,
            *(review.reviewer_id for review in item.independent_reviews),
            item.validator_contract_review.reviewer_id,
            item.contamination_audit_review.reviewer_id,
        }
        task_role_commitments = {
            str(
                (
                    contributor if reviewer_id == item.human_author_id else reviewers[reviewer_id]
                ).profile_json.get("person_uniqueness_commitment_sha256", "")
            )
            for reviewer_id in task_role_ids
        }
        if (
            item.family.value not in contributor.qualification_json
            or "" in task_role_commitments
            or len(task_role_commitments) != len(task_role_ids)
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{item.public_id} task roles do not represent distinct, "
                    "family-qualified people"
                ),
            )
        if any(
            not reviewers[review.reviewer_id].active
            or not reviewers[review.reviewer_id].qualification_verified
            or reviewers[review.reviewer_id].cohort != "expert_independent"
            or not _verified_independent_task_validator(reviewers[review.reviewer_id])
            or item.family.value not in reviewers[review.reviewer_id].qualification_json
            for review in item.independent_reviews
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} is not approved by two family-qualified reviewers",
            )
        adjudicator = reviewers[item.adjudication.adjudicator_reviewer_id]
        if (
            not adjudicator.active
            or not adjudicator.qualification_verified
            or adjudicator.cohort != "expert_independent"
            or item.family.value not in adjudicator.qualification_json
            or not _verified_independent_task_validator(adjudicator)
            or not adjudicator.profile_json.get("task_adjudication_authorized")
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} lacks a qualified independent adjudicator",
            )
        evidence_reviewers = (
            item.validator_contract_review.reviewer_id,
            item.contamination_audit_review.reviewer_id,
        )
        if any(
            not reviewers[reviewer_id].active
            or not reviewers[reviewer_id].qualification_verified
            or reviewers[reviewer_id].cohort != "expert_independent"
            or not _verified_independent_task_validator(reviewers[reviewer_id])
            or item.family.value not in reviewers[reviewer_id].qualification_json
            for reviewer_id in evidence_reviewers
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} evidence was not verified by qualified reviewers",
            )
        candidate_event = candidate_events[item.source_candidate_id]
        candidate_payload = candidate_event.payload_json
        try:
            validate_candidate_binding(
                candidate_payload,
                family=item.family.value,
                construct_cell_id=item.construct_cell_id,
                difficulty_tier=item.difficulty_tier,
            )
        except ConstructBlueprintError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} candidate construct binding failed: {exc}",
            ) from exc
        candidate_reviews = _task_candidate_review_events(session, item.source_candidate_id)
        candidate_state = _task_candidate_status(candidate_reviews)
        if (
            candidate_payload.get("candidate_record_sha256") != item.candidate_record_sha256
            or candidate_payload.get("author_reviewer_id") != item.human_author_id
            or candidate_payload.get("family") != item.family.value
            or candidate_payload.get("prompt_sha256") != prompt_sha256
            or candidate_payload.get("task_contributor_protocol_version")
            != TASK_CONTRIBUTOR_PROTOCOL_VERSION
            or candidate_payload.get("task_contributor_protocol_sha256")
            != TASK_CONTRIBUTOR_PROTOCOL_SHA256
            or candidate_payload.get("task_contributor_protocol_scope")
            != TASK_CONTRIBUTOR_PROTOCOL_SCOPE
            or candidate_payload.get("task_contributor_protocol_acceptance_event_id")
            != contributor.profile_json.get("task_contributor_protocol_acceptance_event_id")
            or candidate_state["status"] != "approved_for_bank_assembly"
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} does not match an approved human task candidate",
            )
        approving_ids = {review.reviewer_id for review in item.independent_reviews}
        if (
            candidate_state["completeReviewers"] != approving_ids
            or set(candidate_state["blindByReviewer"]) != approving_ids
            or set(candidate_state["reconciliationByReviewer"]) != approving_ids
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{item.public_id} source-review slate does not match the append-only ledger"
                ),
            )
        for review in item.independent_reviews:
            reviewer = reviewers[review.reviewer_id]
            blind_event = candidate_state["blindByReviewer"][review.reviewer_id]
            reconciliation_event = candidate_state["reconciliationByReviewer"][review.reviewer_id]
            try:
                blind_request = TaskCandidateBlindValidityCreate.model_validate(
                    blind_event.payload_json
                )
                reconciliation_request = TaskCandidateReconciliationCreate.model_validate(
                    reconciliation_event.payload_json
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=f"{item.public_id} source-review payload is not schema-valid",
                ) from exc
            expected_blind_sha256 = _canonical_sha256(
                {
                    "candidate_id": item.source_candidate_id,
                    "candidate_record_sha256": item.candidate_record_sha256,
                    "reviewer_id": review.reviewer_id,
                    "blind_review": blind_request.model_dump(mode="json"),
                }
            )
            expected_reconciliation_sha256 = _canonical_sha256(
                {
                    "candidate_id": item.source_candidate_id,
                    "candidate_record_sha256": item.candidate_record_sha256,
                    "blind_review_sha256": expected_blind_sha256,
                    "reviewer_id": review.reviewer_id,
                    "reconciliation": reconciliation_request.model_dump(mode="json"),
                }
            )
            if (
                blind_event.payload_json.get("decision") != "valid"
                or blind_event.payload_json.get("blind_review_sha256")
                != review.blind_review_event_sha256
                or review.blind_review_event_sha256 != expected_blind_sha256
                or blind_event.payload_json.get("author_pack_visible") is not False
                or blind_event.payload_json.get("model_outputs_visible") is not False
                or blind_event.payload_json.get("independent_review") is not True
                or blind_event.payload_json.get("candidate_record_sha256")
                != item.candidate_record_sha256
                or reconciliation_event.payload_json.get("decision") != "approve"
                or reconciliation_event.payload_json.get("reconciliation_sha256")
                != review.reconciliation_event_sha256
                or review.reconciliation_event_sha256 != expected_reconciliation_sha256
                or reconciliation_event.payload_json.get("blind_review_sha256")
                != review.blind_review_event_sha256
                or reconciliation_event.payload_json.get("author_pack_visible") is not True
                or reconciliation_event.payload_json.get("model_outputs_visible") is not False
                or reconciliation_event.payload_json.get("independent_review") is not True
                or reconciliation_event.payload_json.get("candidate_record_sha256")
                != item.candidate_record_sha256
                or blind_event.payload_json.get("identity_commitment_sha256")
                != reviewer.profile_json.get("identity_commitment_sha256")
                or reconciliation_event.payload_json.get("identity_commitment_sha256")
                != reviewer.profile_json.get("identity_commitment_sha256")
                or blind_event.payload_json.get("qualification_evidence_sha256")
                != reviewer.profile_json.get("qualification_evidence_sha256")
                or reconciliation_event.payload_json.get("qualification_evidence_sha256")
                != reviewer.profile_json.get("qualification_evidence_sha256")
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{item.public_id} two-stage review records do not match the "
                        "append-only ledger"
                    ),
                )
        adjudication_event = candidate_state["adjudication"]
        try:
            adjudication_request = TaskCandidateAdjudicationCreate.model_validate(
                adjudication_event.payload_json if adjudication_event is not None else {}
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} adjudication payload is not schema-valid",
            ) from exc
        criterion_pack_payload = {
            key: adjudication_request.model_dump(mode="json")[key]
            for key in (
                "family",
                "construct_cell_id",
                "difficulty_tier",
                "success_criteria",
                "permitted_variations",
                "disqualifying_errors",
                "objective_checks",
            )
        }
        expected_criterion_pack_sha256 = _canonical_sha256(criterion_pack_payload)
        expected_adjudication_sha256 = _canonical_sha256(
            {
                "candidate_id": item.source_candidate_id,
                "candidate_record_sha256": item.candidate_record_sha256,
                "source_blind_sha256s": sorted(
                    review.blind_review_event_sha256 for review in item.independent_reviews
                ),
                "source_reconciliation_sha256s": sorted(
                    review.reconciliation_event_sha256 for review in item.independent_reviews
                ),
                "adjudicator_id": item.adjudication.adjudicator_reviewer_id,
                "adjudication": adjudication_request.model_dump(mode="json"),
                "criterion_pack_sha256": expected_criterion_pack_sha256,
            }
        )
        if (
            adjudication_event is None
            or adjudication_event.payload_json.get("reviewer_id")
            != item.adjudication.adjudicator_reviewer_id
            or adjudication_event.payload_json.get("decision") != "approve"
            or adjudication_event.payload_json.get("adjudication_sha256")
            != item.adjudication.adjudication_event_sha256
            or item.adjudication.adjudication_event_sha256 != expected_adjudication_sha256
            or adjudication_event.payload_json.get("criterion_pack_sha256")
            != item.adjudication.criterion_pack_sha256
            or item.adjudication.criterion_pack_sha256 != expected_criterion_pack_sha256
            or adjudication_event.payload_json.get("source_blind_sha256s")
            != sorted(review.blind_review_event_sha256 for review in item.independent_reviews)
            or adjudication_event.payload_json.get("source_reconciliation_sha256s")
            != sorted(review.reconciliation_event_sha256 for review in item.independent_reviews)
            or adjudication_event.payload_json.get("model_outputs_visible") is not False
            or adjudication_event.payload_json.get("source_reviewer") is not False
            or adjudication_event.payload_json.get("identity_commitment_sha256")
            != adjudicator.profile_json.get("identity_commitment_sha256")
            or adjudication_event.payload_json.get("qualification_evidence_sha256")
            != adjudicator.profile_json.get("qualification_evidence_sha256")
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} adjudication does not match the append-only ledger",
            )
        expected_review_history_sha256 = _canonical_sha256(
            {
                "candidate_id": item.source_candidate_id,
                "candidate_record_sha256": item.candidate_record_sha256,
                "blind_review_sha256s": sorted(
                    review.blind_review_event_sha256 for review in item.independent_reviews
                ),
                "reconciliation_sha256s": sorted(
                    review.reconciliation_event_sha256 for review in item.independent_reviews
                ),
                "adjudication_sha256": item.adjudication.adjudication_event_sha256,
                "criterion_pack_sha256": item.adjudication.criterion_pack_sha256,
            }
        )
        if item.review_history_sha256 != expected_review_history_sha256:
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} reviewHistorySha256 does not match the ledger",
            )
        reviews = [review.model_dump(mode="json") for review in item.independent_reviews]
        adjudication = item.adjudication.model_dump(mode="json")
        forbidden_evidence_reviewers = {
            item.human_author_id,
            item.adjudication.adjudicator_reviewer_id,
            *approving_ids,
        }
        if item.validator_contract.verifier_reviewer_id in forbidden_evidence_reviewers:
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} validator verifier is not role-independent",
            )
        try:
            validator_receipt = verify_validator_contract(
                item.validator_contract,
                task_public_id=item.public_id,
                task_family=item.family.value,
                task_revision=item.revision,
                prompt_sha256=prompt_sha256,
                objective_validator_possible=bool(
                    candidate_payload.get("objective_validator_possible")
                ),
                expected_container_image_digest=container_image_digest,
            )
            contamination_receipt = verify_contamination_audit(
                item.contamination_audit,
                scan_bundle=contamination_scan_bundle,
                prompt=item.prompt,
                task_public_id=item.public_id,
                task_family=item.family.value,
                task_revision=item.revision,
                prompt_sha256=prompt_sha256,
                expected_container_image_digest=container_image_digest,
                forbidden_reviewer_ids=forbidden_evidence_reviewers,
            )
        except TaskEvidenceError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} task evidence failed verification: {exc}",
            ) from exc
        evidence_events = _task_candidate_evidence_review_events(
            session,
            item.source_candidate_id,
        )
        if len(evidence_events) != 2:
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} requires two sealed task-evidence reviews",
            )
        evidence_event_by_type = {
            str(event.payload_json.get("evidence_type")): event for event in evidence_events
        }
        if set(evidence_event_by_type) != {
            "validator_contract",
            "contamination_audit",
        }:
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} task-evidence review slate is invalid",
            )
        evidence_bindings = (
            (
                item.validator_contract_review,
                item.validator_contract,
                validator_receipt,
                TaskValidatorContractReviewCreate,
                "validator_contract",
                "task_candidate_validator_contract_verified",
                "validator_contract",
            ),
            (
                item.contamination_audit_review,
                item.contamination_audit,
                contamination_receipt,
                TaskContaminationAuditReviewCreate,
                "contamination_audit",
                "task_candidate_contamination_audit_verified",
                "contamination_audit",
            ),
        )
        for (
            binding,
            artifact,
            receipt,
            review_schema,
            evidence_type,
            expected_event_type,
            artifact_field,
        ) in evidence_bindings:
            event = evidence_event_by_type[evidence_type]
            event_payload = event.payload_json
            reviewer = reviewers[binding.reviewer_id]
            try:
                parsed_review = review_schema.model_validate(
                    {
                        **dict(event_payload.get("review") or {}),
                        artifact_field: artifact.model_dump(mode="json", by_alias=True),
                    }
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=f"{item.public_id} {evidence_type} review is not schema-valid",
                ) from exc
            review_payload = parsed_review.model_dump(
                mode="json",
                exclude={artifact_field},
            )
            expected_review_sha256 = task_evidence_review_sha256(
                candidate_id=item.source_candidate_id,
                candidate_record_sha256=item.candidate_record_sha256,
                task_public_id=item.public_id,
                reviewer_id=binding.reviewer_id,
                evidence_type=evidence_type,
                artifact_sha256=artifact.artifact_sha256,
                verification_receipt_sha256=str(receipt["receipt_sha256"]),
                review=review_payload,
            )
            if (
                event.event_type != expected_event_type
                or event_payload.get("candidate_record_sha256") != item.candidate_record_sha256
                or event_payload.get("task_public_id") != item.public_id
                or event_payload.get("reviewer_id") != binding.reviewer_id
                or event_payload.get("artifact_sha256") != artifact.artifact_sha256
                or binding.artifact_sha256 != artifact.artifact_sha256
                or event_payload.get("verification_receipt_sha256") != receipt["receipt_sha256"]
                or binding.verification_receipt_sha256 != receipt["receipt_sha256"]
                or event_payload.get("review") != review_payload
                or event_payload.get("review_event_sha256") != expected_review_sha256
                or binding.review_event_sha256 != expected_review_sha256
                or event_payload.get("model_outputs_visible") is not False
                or event_payload.get("artifact_visible") is not True
                or event_payload.get("independent_of_task_roles") is not True
                or event_payload.get("identity_commitment_sha256")
                != reviewer.profile_json.get("identity_commitment_sha256")
                or event_payload.get("qualification_evidence_sha256")
                != reviewer.profile_json.get("qualification_evidence_sha256")
                or event_payload.get("independence_attestation_sha256")
                != reviewer.profile_json.get("independence_attestation_sha256")
                or adjudication_event is None
                or as_utc(event.created_at) < as_utc(adjudication_event.created_at)
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{item.public_id} {evidence_type} review does not match the "
                        "append-only ledger"
                    ),
                )
        record_payload = {
            "public_id": item.public_id,
            "family": item.family.value,
            "split": item.split.value,
            "prompt_sha256": prompt_sha256,
            "revision": item.revision,
            "construct_blueprint_sha256": item.construct_blueprint_sha256,
            "construct_cell_id": item.construct_cell_id,
            "difficulty_tier": item.difficulty_tier,
            "human_author_id": item.human_author_id,
            "source_candidate_id": item.source_candidate_id,
            "candidate_record_sha256": item.candidate_record_sha256,
            "independent_reviews": reviews,
            "adjudication": adjudication,
            "validator_contract_sha256": item.validator_contract.artifact_sha256,
            "validator_contract_review": item.validator_contract_review.model_dump(mode="json"),
            "review_history_sha256": item.review_history_sha256,
            "contamination_audit_sha256": item.contamination_audit.artifact_sha256,
            "contamination_audit_review": item.contamination_audit_review.model_dump(mode="json"),
        }
        if _canonical_sha256(record_payload) != item.task_record_sha256:
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} taskRecordSha256 does not match its content",
            )
        evidence_root_sha256 = task_evidence_root_sha256(
            task_record_sha256=item.task_record_sha256,
            candidate_record_sha256=item.candidate_record_sha256,
            review_history_sha256=item.review_history_sha256,
            validator_contract_sha256=item.validator_contract.artifact_sha256,
            contamination_audit_sha256=item.contamination_audit.artifact_sha256,
            validator_receipt_sha256=str(validator_receipt["receipt_sha256"]),
            contamination_receipt_sha256=str(contamination_receipt["receipt_sha256"]),
            validator_review_event_sha256=(item.validator_contract_review.review_event_sha256),
            contamination_review_event_sha256=(item.contamination_audit_review.review_event_sha256),
        )
        if not hmac.compare_digest(item.task_evidence_root_sha256, evidence_root_sha256):
            raise HTTPException(
                status_code=409,
                detail=f"{item.public_id} taskEvidenceRootSha256 does not match evidence",
            )
        verified_evidence[item.public_id] = {
            "validator_receipt": validator_receipt,
            "contamination_receipt": contamination_receipt,
            "task_evidence_root_sha256": evidence_root_sha256,
        }
        authored_at = as_utc(candidate_event.created_at)
        sealed_at = datetime.now(UTC)
        lifecycle_seal_sha256 = task_lifecycle_seal_sha256(
            task_public_id=item.public_id,
            task_revision=item.revision,
            candidate_record_sha256=item.candidate_record_sha256,
            task_record_sha256=item.task_record_sha256,
            task_evidence_root_sha256=evidence_root_sha256,
            authored_at=authored_at,
            sealed_at=sealed_at,
        )
        task = Task(
            public_id=item.public_id,
            season_id=season.id,
            family=item.family.value,
            prompt=item.prompt,
            prompt_sha256=prompt_sha256,
            revision=item.revision,
            split=item.split.value,
            review_status="reviewed",
            provenance_json={
                "origin_type": "human_authored",
                "human_author_id": item.human_author_id,
                "human_author_person_commitment_sha256": contributor.profile_json[
                    "person_uniqueness_commitment_sha256"
                ],
                "task_contributor_protocol_version": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
                "task_contributor_protocol_sha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
                "task_contributor_protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
                "task_contributor_protocol_acceptance_event_id": contributor.profile_json[
                    "task_contributor_protocol_acceptance_event_id"
                ],
                "source_candidate_id": item.source_candidate_id,
                "candidate_record_sha256": item.candidate_record_sha256,
                "construct_blueprint_sha256": item.construct_blueprint_sha256,
                "construct_cell_id": item.construct_cell_id,
                "difficulty_tier": item.difficulty_tier,
                "human_reviewed": True,
                "confirmatory_eligible": True,
                "season1_eligible": True,
                "objective_validator_possible": bool(
                    candidate_payload.get("objective_validator_possible")
                ),
                "independent_reviews": reviews,
                "adjudication": adjudication,
                "criterion_pack": criterion_pack_payload,
                "criterion_pack_sha256": item.adjudication.criterion_pack_sha256,
                "contamination_audit_status": "pass",
                "contamination_audit_sha256": item.contamination_audit.artifact_sha256,
                "contamination_audit_review": (
                    item.contamination_audit_review.model_dump(mode="json")
                ),
                "contamination_scan_bundle_sha256": contamination_scan_bundle.artifact_sha256,
                "contamination_calibration_artifact_sha256": (
                    contamination_calibration.artifact_sha256
                ),
                "contamination_calibration_receipt_sha256": (
                    contamination_calibration_receipt["receipt_sha256"]
                ),
                "validator_contract_sha256": item.validator_contract.artifact_sha256,
                "validator_contract_review": (
                    item.validator_contract_review.model_dump(mode="json")
                ),
                "validator_calibration_artifact_sha256": (validator_calibration.artifact_sha256),
                "validator_calibration_receipt_sha256": (
                    validator_calibration_receipt["receipt_sha256"]
                ),
                "review_history_sha256": item.review_history_sha256,
                "author_reviewer_independence_audit": "pass",
                "task_record_sha256": item.task_record_sha256,
                "validator_receipt_sha256": validator_receipt["receipt_sha256"],
                "contamination_receipt_sha256": contamination_receipt["receipt_sha256"],
                "task_evidence_root_sha256": evidence_root_sha256,
                "evidence_registry_status": "verified",
                "authored_at": authored_at.isoformat(),
                "sealed_at": sealed_at.isoformat(),
                "task_lifecycle_seal_sha256": lifecycle_seal_sha256,
            },
            created_at=sealed_at,
        )
        session.add(task)
        imported.append(task)
    session.flush()
    for item, task in zip(request.tasks, imported, strict=True):
        evidence = verified_evidence[item.public_id]
        evidence_rows = (
            (
                "validator_contract",
                item.validator_contract.schema_version,
                item.validator_contract.model_dump(mode="json", by_alias=True),
                item.validator_contract.artifact_sha256,
                evidence["validator_receipt"],
            ),
            (
                "contamination_audit",
                item.contamination_audit.schema_version,
                item.contamination_audit.model_dump(mode="json", by_alias=True),
                item.contamination_audit.artifact_sha256,
                evidence["contamination_receipt"],
            ),
        )
        for (
            evidence_type,
            schema_version,
            artifact_json,
            artifact_sha256,
            receipt,
        ) in evidence_rows:
            receipt_sha256 = str(receipt["receipt_sha256"])
            task_binding_sha256 = task_evidence_sha256(
                {
                    "artifact_sha256": artifact_sha256,
                    "evidence_type": evidence_type,
                    "revision_ordinal": 1,
                    "supersedes_artifact_id": None,
                    "task_id": task.id,
                    "verification_receipt_sha256": receipt_sha256,
                }
            )
            session.add(
                TaskEvidenceArtifact(
                    task_id=task.id,
                    evidence_type=evidence_type,
                    schema_version=str(schema_version),
                    revision_ordinal=1,
                    artifact_json=artifact_json,
                    artifact_sha256=artifact_sha256,
                    task_binding_sha256=task_binding_sha256,
                    verification_receipt_json=receipt,
                    verification_receipt_sha256=receipt_sha256,
                )
            )
        session.add(
            RunEvent(
                entity_type="task",
                entity_id=task.id,
                event_type="confirmatory_task_authorship_recorded",
                payload_json={
                    "public_id": task.public_id,
                    "family": task.family,
                    "split": task.split,
                    "revision": task.revision,
                    "prompt_sha256": task.prompt_sha256,
                    "construct_blueprint_sha256": item.construct_blueprint_sha256,
                    "construct_cell_id": item.construct_cell_id,
                    "difficulty_tier": item.difficulty_tier,
                    "task_record_sha256": item.task_record_sha256,
                    "source_candidate_id": item.source_candidate_id,
                    "candidate_record_sha256": item.candidate_record_sha256,
                    "task_evidence_root_sha256": item.task_evidence_root_sha256,
                    "authored_at": task.provenance_json["authored_at"],
                    "sealed_at": task.provenance_json["sealed_at"],
                    "lifecycle_seal_sha256": task.provenance_json["task_lifecycle_seal_sha256"],
                    "human_author_reference_sha256": hashlib.sha256(
                        item.human_author_id.encode()
                    ).hexdigest(),
                    "human_author_person_commitment_sha256": task.provenance_json[
                        "human_author_person_commitment_sha256"
                    ],
                },
                created_at=task.created_at,
            )
        )
        session.add(
            RunEvent(
                entity_type="task",
                entity_id=task.id,
                event_type="confirmatory_task_sealed",
                payload_json={
                    "public_id": task.public_id,
                    "family": task.family,
                    "split": task.split,
                    "revision": task.revision,
                    "prompt_sha256": task.prompt_sha256,
                    "task_record_sha256": item.task_record_sha256,
                    "task_evidence_root_sha256": item.task_evidence_root_sha256,
                    "authored_at": task.provenance_json["authored_at"],
                    "sealed_at": task.provenance_json["sealed_at"],
                    "lifecycle_seal_sha256": task.provenance_json["task_lifecycle_seal_sha256"],
                },
                created_at=task.created_at,
            )
        )
        for evidence_review in (
            item.validator_contract_review,
            item.contamination_audit_review,
        ):
            source_event = next(
                event
                for event in _task_candidate_evidence_review_events(
                    session,
                    item.source_candidate_id,
                )
                if event.payload_json.get("evidence_type") == evidence_review.evidence_type
            )
            session.add(
                RunEvent(
                    entity_type="task",
                    entity_id=task.id,
                    event_type="confirmatory_task_evidence_review_recorded",
                    payload_json={
                        **evidence_review.model_dump(mode="json"),
                        "public_id": task.public_id,
                        "family": task.family,
                        "split": task.split,
                        "source_candidate_id": item.source_candidate_id,
                        "candidate_record_sha256": item.candidate_record_sha256,
                        "source_event_id": source_event.id,
                        "task_record_sha256": item.task_record_sha256,
                        "task_evidence_root_sha256": item.task_evidence_root_sha256,
                    },
                )
            )
        for review in item.independent_reviews:
            session.add(
                RunEvent(
                    entity_type="task",
                    entity_id=task.id,
                    event_type="confirmatory_task_review_recorded",
                    payload_json={
                        "public_id": task.public_id,
                        "family": task.family,
                        "split": task.split,
                        "construct_blueprint_sha256": item.construct_blueprint_sha256,
                        "construct_cell_id": item.construct_cell_id,
                        "difficulty_tier": item.difficulty_tier,
                        "reviewer_id": review.reviewer_id,
                        "identity_commitment_sha256": reviewers[
                            review.reviewer_id
                        ].profile_json.get("identity_commitment_sha256"),
                        "qualification_evidence_sha256": reviewers[
                            review.reviewer_id
                        ].profile_json.get("qualification_evidence_sha256"),
                        "blind_review_event_sha256": review.blind_review_event_sha256,
                        "reconciliation_event_sha256": review.reconciliation_event_sha256,
                        "decision": review.decision,
                        "independent_of_author": review.independent_of_author,
                        "source_candidate_id": item.source_candidate_id,
                        "candidate_record_sha256": item.candidate_record_sha256,
                        "task_record_sha256": item.task_record_sha256,
                        "task_evidence_root_sha256": item.task_evidence_root_sha256,
                    },
                )
            )
        session.add(
            RunEvent(
                entity_type="task",
                entity_id=task.id,
                event_type="confirmatory_task_adjudication_recorded",
                payload_json={
                    "public_id": task.public_id,
                    "family": task.family,
                    "split": task.split,
                    "adjudicator_reviewer_id": item.adjudication.adjudicator_reviewer_id,
                    "adjudication_event_sha256": item.adjudication.adjudication_event_sha256,
                    "criterion_pack_sha256": item.adjudication.criterion_pack_sha256,
                    "decision": item.adjudication.decision,
                    "independent_of_author_and_reviewers": (
                        item.adjudication.independent_of_author_and_reviewers
                    ),
                    "source_candidate_id": item.source_candidate_id,
                    "candidate_record_sha256": item.candidate_record_sha256,
                    "task_record_sha256": item.task_record_sha256,
                    "task_evidence_root_sha256": item.task_evidence_root_sha256,
                },
            )
        )
    season.prompt_registry_sha256 = _task_registry_sha256(imported)
    task_evidence_registry_sha256 = _canonical_sha256(
        {
            "tasks": sorted(
                [
                    {
                        "public_id": item.public_id,
                        "task_record_sha256": item.task_record_sha256,
                        "task_evidence_root_sha256": item.task_evidence_root_sha256,
                    }
                    for item in request.tasks
                ],
                key=lambda row: row["public_id"],
            )
        }
    )
    import_reference_sha256 = hashlib.sha256(request.import_reference.encode()).hexdigest()
    session.add(
        RunEvent(
            entity_type="season",
            entity_id=season.id,
            event_type="confirmatory_task_bank_imported",
            payload_json={
                "task_count": len(imported),
                "prompt_registry_sha256": season.prompt_registry_sha256,
                "bank_manifest_sha256": request.bank_manifest_sha256,
                "construct_blueprint_sha256": BLUEPRINT_SHA256,
                "validator_calibration_artifact_sha256": (validator_calibration.artifact_sha256),
                "validator_calibration_receipt_sha256": (
                    validator_calibration_receipt["receipt_sha256"]
                ),
                "validator_calibration_case_count": validator_calibration_receipt["case_count"],
                "validator_calibration_status": validator_calibration_receipt["status"],
                "contamination_calibration_artifact_sha256": (
                    contamination_calibration.artifact_sha256
                ),
                "contamination_calibration_receipt_sha256": (
                    contamination_calibration_receipt["receipt_sha256"]
                ),
                "contamination_calibration_case_count": (
                    contamination_calibration_receipt["case_count"]
                ),
                "contamination_calibration_precision_milli": (
                    contamination_calibration_receipt["precision_milli"]
                ),
                "contamination_calibration_recall_milli": (
                    contamination_calibration_receipt["recall_milli"]
                ),
                "contamination_calibration_paraphrase_recall_milli": (
                    contamination_calibration_receipt["paraphrase_recall_milli"]
                ),
                "contamination_calibration_status": contamination_calibration_receipt["status"],
                "task_evidence_registry_sha256": task_evidence_registry_sha256,
                "import_reference_sha256": import_reference_sha256,
            },
        )
    )
    session.commit()
    return {
        "season": season.slug,
        "tasksImported": len(imported),
        "constructBlueprintSha256": BLUEPRINT_SHA256,
        "validatorCalibrationArtifactSha256": validator_calibration.artifact_sha256,
        "validatorCalibrationReceiptSha256": validator_calibration_receipt["receipt_sha256"],
        "contaminationCalibrationArtifactSha256": (contamination_calibration.artifact_sha256),
        "contaminationCalibrationReceiptSha256": contamination_calibration_receipt[
            "receipt_sha256"
        ],
        "promptRegistrySha256": season.prompt_registry_sha256,
        "taskEvidenceRegistrySha256": task_evidence_registry_sha256,
    }


@router.post("/admin/catalog/sync", dependencies=[Depends(require_admin_token)])
async def admin_catalog_sync(session: Db) -> dict:
    items = await fetch_openrouter_catalog()
    counts = sync_catalog(session, items)
    session.commit()
    return {"catalog": counts}


@router.post(
    "/admin/provider-account-authorizations",
    dependencies=[Depends(require_admin_token)],
)
def admin_provider_account_authorization(
    request: ProviderAccountAuthorizationCreate,
    session: Db,
) -> dict:
    """Create or rotate an authorization epoch without replacing its spend ledger."""

    backend = request.execution_backend.value
    if backend not in {"bedrock", "kimi_direct", "openrouter", "qwencloud_direct"}:
        raise HTTPException(
            status_code=409,
            detail="only paid execution backends have account authorizations",
        )
    required_cap = provider_account_hard_cap_micros(backend)
    if request.budget_cap_micros != required_cap:
        raise HTTPException(
            status_code=409,
            detail=f"{backend} account authorization must use the reviewed hard cap",
        )
    now = datetime.now(UTC)
    valid_until = request.valid_until
    if valid_until.tzinfo is None or valid_until <= now:
        raise HTTPException(
            status_code=409,
            detail="provider account authorization expiry must be future and timezone-aware",
        )
    binding = request.credential_binding.model_dump(mode="json")
    observed_at = request.credential_binding.observed_at
    expected_binding_kind = {
        "bedrock": "bedrock_control_plane_v1",
        "kimi_direct": "kimi_catalog_endpoint_v1",
        "openrouter": "openrouter_account_endpoint_v1",
        "qwencloud_direct": "qwencloud_catalog_endpoint_v1",
    }[backend]
    if (
        request.credential_binding.binding_kind != expected_binding_kind
        or request.credential_binding.credential_scope_sha256
        != provider_account_scope_sha256(backend)
        or (backend == "bedrock" and not request.credential_binding.target_arn_sha256s)
        or observed_at.tzinfo is None
        or observed_at > now + timedelta(minutes=5)
        or observed_at < now - timedelta(hours=24)
    ):
        raise HTTPException(
            status_code=409,
            detail="provider credential binding is stale, future-dated, or for another backend",
        )
    sources = [source.model_dump(mode="json") for source in request.opening_balance_sources]
    artifact_hashes = [source["artifact_sha256"] for source in sources]
    if len(set(artifact_hashes)) != len(artifact_hashes):
        raise HTTPException(
            status_code=409,
            detail="opening-balance source artifacts must be unique",
        )
    opening_used = sum(source["governed_used_micros"] for source in sources)
    opening_reserved = sum(source["governed_reserved_micros"] for source in sources)
    if opening_used + opening_reserved > required_cap:
        raise HTTPException(
            status_code=409,
            detail="opening provider exposure exceeds the reviewed account cap",
        )
    account_scope_sha256 = provider_account_scope_sha256(backend)
    ledger = session.scalar(
        select(ProviderAccountBudget)
        .where(
            ProviderAccountBudget.execution_backend == backend,
            ProviderAccountBudget.account_scope_sha256 == account_scope_sha256,
        )
        .with_for_update()
    )
    opening_balance = {
        "schema_version": "flavourbench-provider-opening-balance-v1",
        "execution_backend": backend,
        "sources": sorted(sources, key=lambda item: item["artifact_sha256"]),
        "governed_used_micros": opening_used,
        "governed_reserved_micros": opening_reserved,
    }
    opening_balance_sha256 = _canonical_sha256(opening_balance)
    credential_binding_sha256 = _canonical_sha256(binding)
    active_authorization = (
        account_authorization(session, ledger, for_update=True) if ledger is not None else None
    )
    latest_authorization = active_authorization
    if ledger is not None and latest_authorization is None:
        latest_authorization = session.scalar(
            select(ProviderAccountAuthorization)
            .where(ProviderAccountAuthorization.provider_account_budget_id == ledger.id)
            .order_by(ProviderAccountAuthorization.created_at.desc())
            .limit(1)
            .with_for_update()
        )
    supersedes_sha256 = request.supersedes_authorization_envelope_sha256
    if ledger is not None:
        if (
            ledger.currency != request.currency
            or ledger.budget_cap_micros != required_cap
            or ledger.account_scope_sha256 != account_scope_sha256
            or ledger.status == "revoked"
            or opening_used != ledger.budget_used_micros
            or opening_reserved != ledger.budget_reserved_micros
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "authorization rotation must preserve the permanent account "
                    "ledger and its cumulative exposure"
                ),
            )
        if latest_authorization is None:
            if supersedes_sha256 is not None:
                raise HTTPException(
                    status_code=409,
                    detail="legacy ledger activation cannot supersede a missing epoch",
                )
        elif supersedes_sha256 != latest_authorization.authorization_envelope_sha256:
            raise HTTPException(
                status_code=409,
                detail="authorization rotation does not supersede the latest epoch",
            )
    elif supersedes_sha256 is not None:
        raise HTTPException(
            status_code=409,
            detail="an initial authorization cannot name a superseded epoch",
        )

    ledger_id = ledger.id if ledger is not None else str(uuid.uuid4())
    settings = get_settings()
    envelope = {
        "schema_version": "flavourbench-provider-account-authorization-v4",
        "signing_key_id": settings.budget_authorization_signing_key_id,
        "provider_account_budget_id": ledger_id,
        "execution_backend": backend,
        "currency": request.currency,
        "budget_cap_micros": required_cap,
        "account_scope_sha256": account_scope_sha256,
        "authorization_reference_sha256": request.authorization_reference_sha256,
        "ledger_opening_balance_sha256": (
            ledger.opening_balance_sha256 if ledger is not None else opening_balance_sha256
        ),
        "exposure_attestation_sha256": opening_balance_sha256,
        "cumulative_used_micros": opening_used,
        "cumulative_reserved_micros": opening_reserved,
        "credential_binding_sha256": credential_binding_sha256,
        "supersedes_authorization_envelope_sha256": supersedes_sha256,
        "valid_until": valid_until.isoformat(),
    }
    envelope_sha256 = _canonical_sha256(envelope)
    authorization_hmac = hmac.new(
        settings.budget_authorization_signing_secret.encode(),
        envelope_sha256.encode(),
        hashlib.sha256,
    ).hexdigest()
    if ledger is None:
        ledger = ProviderAccountBudget(
            id=ledger_id,
            execution_backend=backend,
            currency=request.currency,
            status="active",
            budget_cap_micros=required_cap,
            budget_used_micros=opening_used,
            budget_reserved_micros=opening_reserved,
            opening_used_micros=opening_used,
            opening_reserved_micros=opening_reserved,
            account_scope_sha256=account_scope_sha256,
            authorization_reference_sha256=request.authorization_reference_sha256,
            opening_balance_json=opening_balance,
            opening_balance_sha256=opening_balance_sha256,
            credential_binding_json=binding,
            credential_binding_sha256=credential_binding_sha256,
            authorization_envelope_json=envelope,
            authorization_envelope_sha256=envelope_sha256,
            authorization_hmac_sha256=authorization_hmac,
            valid_until=valid_until,
        )
        session.add(ledger)
        session.flush()
    elif ledger.status == "pending_verification":
        ledger.status = "active"
    if active_authorization is not None:
        active_authorization.status = "revoked"
        active_authorization.revoked_at = now
        session.flush()
    authorization = ProviderAccountAuthorization(
        provider_account_budget_id=ledger.id,
        execution_backend=backend,
        account_scope_sha256=account_scope_sha256,
        status="active",
        supersedes_authorization_id=(
            latest_authorization.id if latest_authorization is not None else None
        ),
        authorization_reference_sha256=request.authorization_reference_sha256,
        exposure_attestation_json=opening_balance,
        exposure_attestation_sha256=opening_balance_sha256,
        authorized_used_micros=opening_used,
        authorized_reserved_micros=opening_reserved,
        credential_binding_json=binding,
        credential_binding_sha256=credential_binding_sha256,
        authorization_envelope_json=envelope,
        authorization_envelope_sha256=envelope_sha256,
        authorization_hmac_sha256=authorization_hmac,
        valid_until=valid_until,
    )
    session.add(authorization)
    session.flush()
    session.add(
        RunEvent(
            entity_type="provider_account_authorization",
            entity_id=authorization.id,
            event_type=(
                "provider_account_authorization_rotated"
                if latest_authorization is not None
                else "provider_account_authorized"
            ),
            payload_json={
                "provider_account_budget_id": ledger.id,
                "execution_backend": backend,
                "account_scope_sha256": account_scope_sha256,
                "authorization_envelope_sha256": envelope_sha256,
                "opening_balance_sha256": ledger.opening_balance_sha256,
                "exposure_attestation_sha256": opening_balance_sha256,
                "credential_binding_sha256": credential_binding_sha256,
                "budget_cap_micros": required_cap,
                "opening_used_micros": opening_used,
                "opening_reserved_micros": opening_reserved,
                "supersedes_authorization_id": (
                    latest_authorization.id if latest_authorization is not None else None
                ),
            },
        )
    )
    session.commit()
    return {
        "executionBackend": backend,
        "accountScopeSha256": account_scope_sha256,
        "authorizationEnvelopeSha256": envelope_sha256,
        "openingBalanceSha256": ledger.opening_balance_sha256,
        "credentialBindingSha256": credential_binding_sha256,
        "budgetCapMicros": required_cap,
        "usedMicros": ledger.budget_used_micros,
        "reservedMicros": ledger.budget_reserved_micros,
        "validUntil": valid_until.isoformat(),
    }


@router.post(
    "/admin/provider-account-authorizations/{execution_backend}/revoke",
    dependencies=[Depends(require_admin_token)],
)
def admin_revoke_provider_account_authorization(
    execution_backend: str,
    request: ProviderAccountAuthorizationRevokeCreate,
    session: Db,
) -> dict:
    if execution_backend not in {
        "bedrock",
        "kimi_direct",
        "openrouter",
        "qwencloud_direct",
    }:
        raise HTTPException(status_code=404, detail="paid provider backend not found")
    ledger = session.scalar(
        select(ProviderAccountBudget)
        .where(
            ProviderAccountBudget.execution_backend == execution_backend,
            ProviderAccountBudget.account_scope_sha256
            == provider_account_scope_sha256(execution_backend),
        )
        .with_for_update()
    )
    authorization = (
        account_authorization(
            session,
            ledger,
            envelope_sha256=request.authorization_envelope_sha256,
            for_update=True,
        )
        if ledger is not None
        else None
    )
    if ledger is None or authorization is None:
        raise HTTPException(
            status_code=409,
            detail="active provider account authorization was not found",
        )
    authorization.status = "revoked"
    authorization.revoked_at = datetime.now(UTC)
    session.add(
        RunEvent(
            entity_type="provider_account_authorization",
            entity_id=authorization.id,
            event_type="provider_account_authorization_revoked",
            payload_json={
                "execution_backend": execution_backend,
                "account_scope_sha256": ledger.account_scope_sha256,
                "authorization_envelope_sha256": (authorization.authorization_envelope_sha256),
                "revocation_reference_sha256": (request.revocation_reference_sha256),
            },
        )
    )
    session.commit()
    return {
        "executionBackend": execution_backend,
        "authorizationEnvelopeSha256": (authorization.authorization_envelope_sha256),
        "status": authorization.status,
        "revokedAt": authorization.revoked_at.isoformat(),
    }


@router.post(
    "/admin/seasons/{season_slug}/bedrock-billing-crosschecks",
    dependencies=[Depends(require_admin_token)],
)
def admin_bedrock_billing_crosscheck(
    season_slug: str,
    request: BedrockBillingCrosscheckCreate,
    session: Db,
) -> dict:
    """Attach immutable AWS aggregate billing evidence to an exact arm set."""

    postgresql_budget_authority = _uses_postgresql_budget_authority(session)
    season_statement = select(Season).where(Season.slug == season_slug)
    if not postgresql_budget_authority:
        season_statement = season_statement.with_for_update()
    season = session.scalar(season_statement)
    if season is None or season.status not in {"pilot", "active", "cost_halted"}:
        raise HTTPException(status_code=409, detail="season cannot accept billing evidence")
    _require_budget_integrity(
        session,
        season.id,
        lock_aggregates=not postgresql_budget_authority,
    )
    arm_ids = sorted(request.arm_ids)
    arms_statement = (
        select(ResponseArm)
        .join(Battle, Battle.id == ResponseArm.battle_id)
        .where(
            ResponseArm.id.in_(arm_ids),
            Battle.season_id == season.id,
        )
        .order_by(ResponseArm.id)
    )
    if not postgresql_budget_authority:
        arms_statement = arms_statement.with_for_update()
    arms = session.scalars(arms_statement).all()
    if len(arms) != len(arm_ids) or [arm.id for arm in arms] != arm_ids:
        raise HTTPException(
            status_code=409,
            detail="billing crosscheck does not resolve to the exact season arm set",
        )
    superseded_ids = {
        value
        for value in session.scalars(
            select(BedrockBillingCrosscheck.supersedes_crosscheck_id).where(
                BedrockBillingCrosscheck.supersedes_crosscheck_id.is_not(None)
            )
        ).all()
        if value is not None
    }
    existing_memberships = session.execute(
        select(
            BedrockBillingCrosscheckArm.crosscheck_id,
            BedrockBillingCrosscheckArm.arm_id,
        ).where(BedrockBillingCrosscheckArm.arm_id.in_(arm_ids))
    ).all()
    active_existing_ids = {
        crosscheck_id
        for crosscheck_id, _arm_id in existing_memberships
        if crosscheck_id not in superseded_ids
    }
    superseded: BedrockBillingCrosscheck | None = None
    if request.supersedes_crosscheck_id is None:
        if active_existing_ids:
            raise HTTPException(
                status_code=409,
                detail="one or more Bedrock arms already have active billing evidence",
            )
    else:
        superseded_statement = select(BedrockBillingCrosscheck).where(
            BedrockBillingCrosscheck.id == request.supersedes_crosscheck_id,
            BedrockBillingCrosscheck.season_id == season.id,
        )
        if not postgresql_budget_authority:
            superseded_statement = superseded_statement.with_for_update()
        superseded = session.scalar(superseded_statement)
        prior_arm_ids = sorted(
            session.scalars(
                select(BedrockBillingCrosscheckArm.arm_id).where(
                    BedrockBillingCrosscheckArm.crosscheck_id == request.supersedes_crosscheck_id
                )
            ).all()
        )
        if (
            superseded is None
            or superseded.id in superseded_ids
            or active_existing_ids != {superseded.id}
            or prior_arm_ids != arm_ids
        ):
            raise HTTPException(
                status_code=409,
                detail="billing correction must supersede the one active exact-arm record",
            )
    generation_map: list[dict[str, Any]] = []
    authorization_epoch_hashes: set[str] = set()
    authorization_attempt_times: list[datetime] = []
    for arm in arms:
        generation_ids = sorted(set(arm.provider_generation_ids_json or []))
        completed_at = arm.completed_at
        if completed_at is not None and completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=UTC)
        if (
            arm.execution_backend != "bedrock"
            or not arm.cost_reconciled
            or arm.billing_reconciliation_status != "pending_aws_billing_crosscheck"
            or not generation_ids
            or completed_at is None
            or completed_at < request.coverage_start
            or completed_at > request.coverage_end
        ):
            raise HTTPException(
                status_code=409,
                detail="billing arm is incomplete, outside coverage, or not pending AWS evidence",
            )
        actual_event = session.scalar(
            select(CostEvent).where(
                CostEvent.arm_id == arm.id,
                CostEvent.kind == "actual",
            )
        )
        if (
            actual_event is None
            or actual_event.amount_micros != arm.cost_micros
            or actual_event.accounting_json.get("billing_reconciliation_status")
            != "pending_aws_billing_crosscheck"
        ):
            raise HTTPException(
                status_code=409,
                detail="billing arm lacks matching immutable rate-card accounting evidence",
            )
        request_attempts = session.scalars(
            select(GenerationAttempt).where(
                GenerationAttempt.arm_id == arm.id,
                GenerationAttempt.event_type == "request_started",
            )
        ).all()
        authorization_attempt_times.extend(attempt.created_at for attempt in request_attempts)
        arm_authorization_hashes = {
            str(
                attempt.metadata_json.get(
                    "verified_provider_account_authorization_envelope_sha256",
                    "",
                )
            )
            for attempt in request_attempts
        }
        arm_authorization_hashes.discard("")
        if len(arm_authorization_hashes) != 1:
            raise HTTPException(
                status_code=409,
                detail="billing arm lacks one exact governed credential epoch",
            )
        authorization_epoch_sha256 = next(iter(arm_authorization_hashes))
        authorization_epoch_hashes.add(authorization_epoch_sha256)
        generation_map.append(
            {
                "arm_id": arm.id,
                "generation_ids": generation_ids,
                "account_authorization_envelope_sha256": (authorization_epoch_sha256),
                "generation_set_sha256": _canonical_sha256({"generation_ids": generation_ids}),
            }
        )
    generation_request_map_sha256 = _canonical_sha256({"arms": generation_map})
    if generation_request_map_sha256 != request.generation_request_map_sha256:
        raise HTTPException(
            status_code=409,
            detail="billing generation/request map differs from the service journal",
        )
    account_scope = provider_account_scope_sha256("bedrock")
    provider_budget_statement = select(SeasonProviderBudget).where(
        SeasonProviderBudget.season_id == season.id,
        SeasonProviderBudget.execution_backend == "bedrock",
    )
    account_budget_statement = select(ProviderAccountBudget).where(
        ProviderAccountBudget.execution_backend == "bedrock",
        ProviderAccountBudget.account_scope_sha256 == account_scope,
    )
    if not postgresql_budget_authority:
        provider_budget_statement = provider_budget_statement.with_for_update()
        account_budget_statement = account_budget_statement.with_for_update()
    provider_budget = session.scalar(provider_budget_statement)
    account_budget = session.scalar(account_budget_statement)
    if provider_budget is None or account_budget is None:
        raise HTTPException(
            status_code=409,
            detail="Bedrock billing evidence has no governing provider ledger",
        )
    if len(authorization_epoch_hashes) != 1:
        raise HTTPException(
            status_code=409,
            detail=(
                "one billing crosscheck may cover only one credential epoch; "
                "submit separate exact-arm records"
            ),
        )
    billing_authorization_sha256 = next(iter(authorization_epoch_hashes))
    billing_authorization = account_authorization(
        session,
        account_budget,
        envelope_sha256=billing_authorization_sha256,
        active_only=False,
        for_update=not postgresql_budget_authority,
    )
    authorization_interval_valid = billing_authorization is not None and all(
        as_utc(billing_authorization.created_at) <= as_utc(attempt_time)
        and as_utc(attempt_time) < as_utc(billing_authorization.valid_until)
        and (
            billing_authorization.revoked_at is None
            or as_utc(attempt_time) < as_utc(billing_authorization.revoked_at)
        )
        for attempt_time in authorization_attempt_times
    )
    if (
        not account_authorization_chain_valid(
            session,
            account_budget,
            billing_authorization,
            root_envelope_sha256=(provider_budget.account_authorization_envelope_sha256),
            signing_secret=get_settings().budget_authorization_signing_secret,
            verification_keys=budget_authorization_verification_keyring(get_settings()),
            require_head_active=False,
        )
        or billing_authorization is None
        or not authorization_interval_valid
        or request.authorization_reference_sha256
        != billing_authorization.authorization_reference_sha256
    ):
        raise HTTPException(
            status_code=409,
            detail="Bedrock billing evidence does not match its account authorization epoch",
        )
    rate_card_estimated_micros = sum(arm.cost_micros for arm in arms)
    tolerance_micros = max(10_000, (rate_card_estimated_micros + 49) // 50)
    billing_difference = request.billed_usage_micros - rate_card_estimated_micros
    crosscheck_status = "accepted" if abs(billing_difference) <= tolerance_micros else "discrepant"
    if crosscheck_status == "discrepant":
        if not postgresql_budget_authority:
            season.status = "cost_halted"
        session.add(
            Incident(
                severity="critical",
                code="bedrock_billing_crosscheck_discrepancy",
                detail=(
                    "AWS gross usage differs from the frozen rate-card estimate beyond "
                    "the fixed two-percent-or-one-cent tolerance."
                ),
            )
        )
    arm_set_sha256 = _canonical_sha256({"arm_ids": arm_ids})
    request_payload = request.model_dump(mode="json")
    # Pydantic emits UTC datetimes with a ``Z`` suffix in JSON mode, while
    # ``datetime.isoformat()`` emits ``+00:00``.  Persist and hash one explicit
    # representation so the application and PostgreSQL authority derive the
    # same evidence document.
    request_payload["coverage_start"] = request.coverage_start.isoformat()
    request_payload["coverage_end"] = request.coverage_end.isoformat()
    evidence = {
        "schema_version": "flavourbench-bedrock-billing-crosscheck-v1",
        "season_slug": season.slug,
        "account_scope_sha256": account_scope,
        "source_kind": request.source_kind,
        "source_artifact_uri": request.source_artifact_uri,
        "source_artifact_sha256": request.source_artifact_sha256,
        "statement_sha256": request.statement_sha256,
        "coverage_start": request.coverage_start.isoformat(),
        "coverage_end": request.coverage_end.isoformat(),
        "arm_set_sha256": arm_set_sha256,
        "generation_request_map_sha256": generation_request_map_sha256,
        "rate_card_estimated_micros": rate_card_estimated_micros,
        "billed_usage_micros": request.billed_usage_micros,
        "billing_difference_micros": billing_difference,
        "crosscheck_status": crosscheck_status,
        "supersedes_crosscheck_id": (superseded.id if superseded is not None else None),
        "tolerance_micros": tolerance_micros,
        "credits_policy": request.credits_policy,
        "authorization_reference_sha256": request.authorization_reference_sha256,
    }
    ledger_delta = billing_difference - (
        superseded.billing_difference_micros if superseded is not None else 0
    )
    # Admission uses a conservative exposure ledger. A later invoice correction
    # may reduce the reconciled bill, but it must never restore spend authority.
    governed_budget_delta = max(0, ledger_delta)
    evidence["ledger_delta_micros"] = ledger_delta
    evidence["governed_budget_delta_micros"] = governed_budget_delta
    evidence_sha256 = _canonical_sha256(evidence)
    if postgresql_budget_authority:
        request_json = json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            registration = (
                session.execute(
                    text(
                        "SELECT crosscheck_id, evidence_sha256, arm_set_sha256, "
                        "generation_request_map_sha256, rate_card_estimated_micros, "
                        "billed_usage_micros, billing_difference_micros, "
                        "ledger_delta_micros, governed_delta_micros, tolerance_micros, "
                        "crosscheck_status, cost_halted "
                        "FROM public.flavourbench_register_bedrock_billing_adjustment("
                        ":season_id, CAST(:request_json AS jsonb))"
                    ),
                    {"season_id": season.id, "request_json": request_json},
                )
                .mappings()
                .one()
            )
        except DBAPIError as exc:
            sqlstate = getattr(exc.orig, "sqlstate", None)
            if sqlstate in {"FB001", "23505"}:
                session.rollback()
                raise HTTPException(
                    status_code=409,
                    detail="billing evidence conflicts with the active record",
                ) from exc
            raise
        if (
            str(registration["evidence_sha256"]) != evidence_sha256
            or str(registration["arm_set_sha256"]) != arm_set_sha256
            or str(registration["generation_request_map_sha256"]) != generation_request_map_sha256
            or int(registration["rate_card_estimated_micros"]) != rate_card_estimated_micros
            or int(registration["billed_usage_micros"]) != request.billed_usage_micros
            or int(registration["billing_difference_micros"]) != billing_difference
            or int(registration["ledger_delta_micros"]) != ledger_delta
            or int(registration["governed_delta_micros"]) != governed_budget_delta
            or int(registration["tolerance_micros"]) != tolerance_micros
            or str(registration["crosscheck_status"]) != crosscheck_status
        ):
            raise HTTPException(
                status_code=409,
                detail="database billing registration disagrees with application semantics",
            )
        session.expire_all()
        season = session.get(Season, season.id)
        crosscheck = session.get(
            BedrockBillingCrosscheck,
            str(registration["crosscheck_id"]),
        )
        if season is None:
            raise HTTPException(status_code=409, detail="billing season disappeared")
        if crosscheck is None:
            raise HTTPException(status_code=409, detail="billing crosscheck disappeared")
        if bool(registration["cost_halted"]) and crosscheck_status != "discrepant":
            session.add(
                Incident(
                    severity="critical",
                    code="bedrock_billing_adjustment_exceeded_cap",
                    detail="AWS billing reconciliation pushed a governed ledger above its cap.",
                )
            )
    else:
        crosscheck = BedrockBillingCrosscheck(
            season_id=season.id,
            provider_account_budget_id=account_budget.id,
            status=crosscheck_status,
            supersedes_crosscheck_id=(superseded.id if superseded is not None else None),
            source_kind=request.source_kind,
            source_artifact_uri=request.source_artifact_uri,
            source_artifact_sha256=request.source_artifact_sha256,
            statement_sha256=request.statement_sha256,
            coverage_start=request.coverage_start,
            coverage_end=request.coverage_end,
            arm_set_sha256=arm_set_sha256,
            generation_request_map_sha256=generation_request_map_sha256,
            rate_card_estimated_micros=rate_card_estimated_micros,
            billed_usage_micros=request.billed_usage_micros,
            billing_difference_micros=billing_difference,
            ledger_delta_micros=ledger_delta,
            tolerance_micros=tolerance_micros,
            credits_policy=request.credits_policy,
            authorization_reference_sha256=request.authorization_reference_sha256,
            evidence_json=evidence,
            evidence_sha256=evidence_sha256,
        )
        session.add(crosscheck)
        session.flush()
        for item in generation_map:
            session.add(
                BedrockBillingCrosscheckArm(
                    crosscheck_id=crosscheck.id,
                    arm_id=item["arm_id"],
                    generation_set_sha256=item["generation_set_sha256"],
                )
            )
        season.budget_used_micros += governed_budget_delta
        provider_budget.budget_used_micros += governed_budget_delta
        account_budget.budget_used_micros += governed_budget_delta
        if (
            season.budget_used_micros > season.budget_cap_micros
            or provider_budget.budget_used_micros > provider_budget.budget_cap_micros
            or account_budget.budget_used_micros > account_budget.budget_cap_micros
        ):
            season.status = "cost_halted"
            session.add(
                Incident(
                    severity="critical",
                    code="bedrock_billing_adjustment_exceeded_cap",
                    detail="AWS billing reconciliation pushed a governed ledger above its cap.",
                )
            )
        session.add(
            CostEvent(
                season_id=season.id,
                kind="bedrock_billing_adjustment",
                amount_micros=ledger_delta,
                provider="bedrock",
                accounting_json={
                    "crosscheck_id": crosscheck.id,
                    "evidence_sha256": evidence_sha256,
                    "arm_set_sha256": arm_set_sha256,
                    "account_scope_sha256": account_scope,
                    "governed_budget_delta_micros": governed_budget_delta,
                },
            )
        )
    session.add(
        RunEvent(
            entity_type="bedrock_billing_crosscheck",
            entity_id=crosscheck.id,
            event_type="aws_billing_crosschecked",
            payload_json={
                "evidence_sha256": evidence_sha256,
                "arm_set_sha256": arm_set_sha256,
                "arm_count": len(arms),
                "ledger_delta_micros": ledger_delta,
                "governed_budget_delta_micros": governed_budget_delta,
                "supersedes_crosscheck_id": (superseded.id if superseded is not None else None),
            },
        )
    )
    withdrawal_reason = (
        "bedrock_billing_crosscheck_discrepant"
        if crosscheck_status == "discrepant"
        else "bedrock_billing_crosscheck_superseded"
        if superseded is not None
        else "bedrock_billing_evidence_added"
    )
    withdrawn_snapshot_ids = _withdraw_published_snapshots(
        session,
        season_id=season.id,
        reason_code=withdrawal_reason,
    )
    session.flush()
    _require_budget_integrity(session, season.id)
    session.commit()
    return {
        "crosscheckId": crosscheck.id,
        "evidenceSha256": evidence_sha256,
        "armSetSha256": arm_set_sha256,
        "armsCrosschecked": len(arms),
        "rateCardEstimatedMicros": rate_card_estimated_micros,
        "withdrawnSnapshotIds": withdrawn_snapshot_ids,
        "billedUsageMicros": request.billed_usage_micros,
        "billingDifferenceMicros": billing_difference,
        "ledgerDeltaMicros": ledger_delta,
        "governedBudgetDeltaMicros": governed_budget_delta,
        "toleranceMicros": tolerance_micros,
        "seasonStatus": season.status,
        "crosscheckStatus": crosscheck_status,
    }


@router.post("/admin/models/{model_id:path}/smoke", dependencies=[Depends(require_admin_token)])
def admin_model_smoke(model_id: str, request: ModelSmokeCreate, session: Db) -> dict:
    model = session.get(CatalogModel, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="catalog model not found")
    if get_settings().execution_mode == "live" and request.provider_slug in {
        "openrouter",
        "mock",
    }:
        raise HTTPException(
            status_code=409,
            detail="live smoke requires a concrete pinned provider endpoint",
        )
    if get_settings().execution_mode == "live" and request.endpoint_rate_card is None:
        raise HTTPException(
            status_code=409,
            detail="live smoke requires endpoint-specific pricing evidence",
        )
    if request.backend_contract:
        if (
            request.backend_contract_sha256 is None
            or _canonical_sha256(request.backend_contract) != request.backend_contract_sha256
        ):
            raise HTTPException(
                status_code=409,
                detail="backend contract hash does not match its content",
            )
        try:
            bedrock_contract = parse_bedrock_endpoint_contract(request.backend_contract)
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail="Bedrock backend contract is invalid",
            ) from exc
        if (
            request.execution_backend.value != "bedrock"
            or bedrock_contract.canonical_model_id != request.expected_actual_model_id
            or request.expected_actual_provider_slug != "amazon-bedrock"
            or not bedrock_contract.season_eligible
        ):
            raise HTTPException(
                status_code=409,
                detail="Bedrock backend contract does not match the asserted endpoint",
            )
    discovery_sha256 = model.endpoint_json.get("catalog_discovery_sha256")
    if get_settings().execution_mode == "live" and (
        not isinstance(discovery_sha256, str) or len(discovery_sha256) != 64
    ):
        raise HTTPException(
            status_code=409,
            detail="live smoke requires a content-addressed catalog discovery record",
        )
    artifact = request.evidence_artifact.model_dump(mode="json")
    artifact_sha256 = _canonical_sha256(artifact)
    if artifact_sha256 != request.evidence_artifact_sha256:
        raise HTTPException(
            status_code=409,
            detail="smoke evidence artifact hash does not match its content",
        )
    if (
        request.evidence_artifact.actual_model_id != request.expected_actual_model_id
        or request.evidence_artifact.actual_provider_slug != request.expected_actual_provider_slug
        or request.evidence_artifact.tools_passed != request.tools_passed
        or request.evidence_artifact.structured_output_passed != request.structured_output_passed
        or request.evidence_artifact.data_collection_denied != request.data_collection_denied
        or request.evidence_artifact.schema_sha256 != FINAL_SCHEMA_SHA256
    ):
        raise HTTPException(
            status_code=409,
            detail="smoke evidence artifact does not match the asserted endpoint result",
        )
    passed = (
        request.tools_passed
        and request.structured_output_passed
        and request.data_collection_denied
        and request.evidence_artifact.cost_reconciled
    )
    if passed and request.expected_actual_model_id != model.canonical_slug:
        raise HTTPException(
            status_code=409,
            detail="paid smoke actual model must equal the current canonical catalog slug",
        )
    model.status = "smoke_passed" if passed else "compatible"
    model.supports_tools = request.tools_passed
    model.supports_structured_outputs = request.structured_output_passed
    decoding = request.decoding.model_dump(exclude_none=True)
    smoke_contract = endpoint_contract_payload(
        model_id=model_id,
        provider_slug=request.provider_slug,
        expected_actual_model_id=request.expected_actual_model_id,
        expected_actual_provider_slug=request.expected_actual_provider_slug,
        supported_parameters=request.supported_parameters,
        decoding=decoding,
        endpoint_max_completion_tokens=request.endpoint_max_completion_tokens,
        endpoint_document_sha256=request.endpoint_document_sha256,
    )
    model.endpoint_json = {
        **model.endpoint_json,
        "execution_backend": request.execution_backend.value,
        "approved_provider": request.provider_slug,
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny" if request.data_collection_denied else "unverified",
        "zdr_compatible": request.zdr_compatible,
        "smoke_evidence_reference": request.evidence_reference,
        "smoke_evidence_artifact": artifact,
        "smoke_evidence_artifact_sha256": artifact_sha256,
        "smoke_catalog_discovery_sha256": discovery_sha256,
        "smoke_endpoint_contract": smoke_contract,
        "smoke_endpoint_contract_sha256": endpoint_contract_sha256(**smoke_contract),
        "smoke_endpoint_rate_card": (
            request.endpoint_rate_card.model_dump(mode="json")
            if request.endpoint_rate_card is not None
            else None
        ),
        "smoke_endpoint_rate_card_sha256": (
            _canonical_sha256(request.endpoint_rate_card.model_dump(mode="json"))
            if request.endpoint_rate_card is not None
            else None
        ),
        "smoke_backend_contract": dict(request.backend_contract),
        "smoke_backend_contract_sha256": request.backend_contract_sha256,
    }
    session.commit()
    return {"modelId": model.model_id, "status": model.status}


@router.post(
    "/admin/seasons/{season_slug}/tasks/freeze", dependencies=[Depends(require_admin_token)]
)
def admin_task_freeze(
    season_slug: str,
    request: TaskRegistryFreezeCreate,
    session: Db,
) -> dict:
    season = session.scalar(select(Season).where(Season.slug == season_slug))
    if season is None or season.status != "draft":
        raise HTTPException(status_code=409, detail="only a draft task registry can freeze")
    tasks = session.scalars(select(Task).where(Task.season_id == season.id)).all()
    actual = {task.public_id: task.prompt_sha256 for task in tasks}
    if len(tasks) != CONFIRMATORY_TASK_COUNT or request.task_hashes != actual:
        raise HTTPException(
            status_code=409,
            detail=f"review does not match all {CONFIRMATORY_TASK_COUNT} prompts",
        )
    splits = {task.split for task in tasks}
    season1_splits = {"scored", "development", "private_reserve"}
    if splits not in ({"development"}, season1_splits):
        raise HTTPException(
            status_code=409,
            detail="development and confirmatory task banks must occupy separate seasons",
        )
    registry_class = (
        "season1_scored_development_private_reserve"
        if splits == season1_splits
        else "developer_exposed_engineering_tasks"
    )
    for task in tasks:
        provenance = task.provenance_json if isinstance(task.provenance_json, dict) else {}
        if task.split in season1_splits and (
            task.review_status != "reviewed"
            or provenance.get("origin_type") != "human_authored"
            or provenance.get("construct_blueprint_sha256") != BLUEPRINT_SHA256
            or not provenance.get("construct_cell_id")
            or provenance.get("difficulty_tier") not in {"foundation", "integrative", "stress"}
            or provenance.get("confirmatory_eligible") is not True
            or len(provenance.get("independent_reviews") or []) < 2
            or provenance.get("contamination_audit_status") != "pass"
            or not provenance.get("contamination_audit_sha256")
            or not provenance.get("validator_contract_sha256")
            or not provenance.get("review_history_sha256")
            or provenance.get("author_reviewer_independence_audit") != "pass"
            or not provenance.get("task_record_sha256")
            or provenance.get("evidence_registry_status") != "verified"
            or not provenance.get("task_evidence_root_sha256")
            or not _verified_task_evidence_registry(session, task)
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{task.public_id} lacks confirmatory task evidence",
            )
        task.review_status = "frozen"
        task.provenance_json = {
            **provenance,
            "registry_class": registry_class,
            "review_reference_sha256": hashlib.sha256(
                request.review_reference.encode()
            ).hexdigest(),
        }
        if splits == {"development"}:
            task.provenance_json.update({"human_reviewed": False, "confirmatory_eligible": False})
    season.prompt_registry_sha256 = _task_registry_sha256(tasks)
    session.add(
        RunEvent(
            entity_type="season",
            entity_id=season.id,
            event_type=(
                "confirmatory_task_registry_frozen"
                if splits == season1_splits
                else "engineering_task_registry_frozen"
            ),
            payload_json={
                "prompt_registry_sha256": season.prompt_registry_sha256,
                "registry_class": registry_class,
                "review_reference_sha256": hashlib.sha256(
                    request.review_reference.encode()
                ).hexdigest(),
            },
        )
    )
    session.commit()
    return {"season": season.slug, "tasksFrozen": len(tasks)}


@router.post("/admin/seasons/{season_slug}/freeze", dependencies=[Depends(require_admin_token)])
def admin_season_freeze(season_slug: str, request: SeasonFreezeCreate, session: Db) -> dict:
    season = session.scalar(select(Season).where(Season.slug == season_slug).with_for_update())
    if season is None or season.status != "draft" or season.frozen_at is not None:
        raise HTTPException(status_code=409, detail="season is not an unfrozen draft")
    tasks = session.scalars(select(Task).where(Task.season_id == season.id)).all()
    if len(tasks) != CONFIRMATORY_TASK_COUNT or any(
        task.review_status != "frozen" for task in tasks
    ):
        raise HTTPException(status_code=409, detail="the reviewed task registry must freeze first")
    if _task_registry_sha256(tasks) != season.prompt_registry_sha256:
        raise HTTPException(status_code=409, detail="task registry hash is inconsistent")
    if len({entry.model_id for entry in request.models}) != SEASON_MODEL_COUNT:
        raise HTTPException(
            status_code=409,
            detail=f"manifest requires {SEASON_MODEL_COUNT} unique models",
        )
    role_counts = {
        role: sum(entry.slot_role == role for entry in request.models)
        for role in ("closed_family", "open_weight", "efficiency", "reasoning")
    }
    expected_roles = SEASON_SLOT_ROLE_COUNTS
    if role_counts != expected_roles:
        raise HTTPException(
            status_code=409,
            detail=(
                "manifest slot roles must be "
                + "/".join(str(expected_roles[role]) for role in expected_roles)
            ),
        )

    requested_backends = {entry.execution_backend.value for entry in request.models}
    authorizations = {
        authorization.execution_backend.value: authorization
        for authorization in request.provider_budget_authorizations
    }
    if len(authorizations) != len(request.provider_budget_authorizations):
        raise HTTPException(
            status_code=409,
            detail="provider budget authorizations must name unique execution backends",
        )
    if get_settings().execution_mode != "mock" and set(authorizations) != requested_backends:
        raise HTTPException(
            status_code=409,
            detail="every live execution backend requires a distinct budget authorization",
        )
    if get_settings().execution_mode == "mock" and authorizations:
        raise HTTPException(
            status_code=409,
            detail="mock seasons must not assert paid-provider budget authorization",
        )
    authorization_rows: list[SeasonProviderBudget] = []
    account_budgets_by_backend: dict[str, ProviderAccountBudget] = {}
    account_authorizations_by_backend: dict[str, ProviderAccountAuthorization] = {}
    now = datetime.now(UTC)
    for backend, authorization in sorted(authorizations.items()):
        if authorization.valid_until.tzinfo is None or authorization.valid_until <= now:
            raise HTTPException(
                status_code=409,
                detail=f"{backend} budget authorization is expired or timezone-naive",
            )
        expected_scope = provider_account_scope_sha256(backend)
        if authorization.account_scope_sha256 != expected_scope:
            raise HTTPException(
                status_code=409,
                detail=f"{backend} account scope is not the installation-wide authority",
            )
        account_budget = session.scalar(
            select(ProviderAccountBudget)
            .where(
                ProviderAccountBudget.execution_backend == backend,
                ProviderAccountBudget.account_scope_sha256 == expected_scope,
            )
            .with_for_update()
        )
        if account_budget is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{backend} account authorization must be provisioned before "
                    "a season can freeze"
                ),
            )
        account_authorization_epoch = account_authorization(
            session,
            account_budget,
            envelope_sha256=(authorization.account_authorization_envelope_sha256),
            for_update=True,
        )
        account_valid_until = (
            account_authorization_epoch.valid_until
            if account_authorization_epoch is not None
            else now
        )
        if account_valid_until.tzinfo is None:
            account_valid_until = account_valid_until.replace(tzinfo=UTC)
        if (
            account_budget.status != "active"
            or account_budget.currency != authorization.currency
            or account_budget.budget_cap_micros != provider_account_hard_cap_micros(backend)
            or account_budget.budget_used_micros + account_budget.budget_reserved_micros
            > account_budget.budget_cap_micros
            or account_budget.opening_used_micros + account_budget.opening_reserved_micros
            > account_budget.budget_cap_micros
            or _canonical_sha256(account_budget.opening_balance_json)
            != account_budget.opening_balance_sha256
            or not account_authorization_chain_valid(
                session,
                account_budget,
                account_authorization_epoch,
                root_envelope_sha256=(authorization.account_authorization_envelope_sha256),
                signing_secret=(get_settings().budget_authorization_signing_secret),
                verification_keys=budget_authorization_verification_keyring(get_settings()),
                now=now,
            )
            or authorization.budget_cap_micros > account_budget.budget_cap_micros
            or authorization.valid_until > account_valid_until
            or account_valid_until <= now
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{backend} account-wide provider authorization is invalid",
            )
        account_budgets_by_backend[backend] = account_budget
        assert account_authorization_epoch is not None
        account_authorizations_by_backend[backend] = account_authorization_epoch
        envelope = {
            "schema_version": "flavourbench-provider-budget-authorization-v2",
            "season_slug": season.slug,
            "execution_backend": backend,
            "currency": authorization.currency,
            "budget_cap_micros": authorization.budget_cap_micros,
            "account_scope_sha256": authorization.account_scope_sha256,
            "authorization_reference_sha256": (authorization.authorization_reference_sha256),
            "account_authorization_envelope_sha256": (
                authorization.account_authorization_envelope_sha256
            ),
            "valid_until": authorization.valid_until.isoformat(),
        }
        if _canonical_sha256(envelope) != authorization.authorization_envelope_sha256:
            raise HTTPException(
                status_code=409,
                detail=f"{backend} budget authorization envelope hash is invalid",
            )
        authorization_rows.append(
            SeasonProviderBudget(
                season_id=season.id,
                execution_backend=backend,
                currency=authorization.currency,
                budget_cap_micros=authorization.budget_cap_micros,
                account_scope_sha256=authorization.account_scope_sha256,
                authorization_reference_sha256=(authorization.authorization_reference_sha256),
                account_authorization_envelope_sha256=(
                    authorization.account_authorization_envelope_sha256
                ),
                authorization_envelope_json=envelope,
                authorization_envelope_sha256=(authorization.authorization_envelope_sha256),
                valid_until=authorization.valid_until,
            )
        )

    existing_slots = session.scalars(
        select(SeasonModel).where(SeasonModel.season_id == season.id)
    ).all()
    for slot in existing_slots:
        slot.eligible = False
    manifest_rows = []
    selected_slots = []
    smoke_evidence_by_model: dict[str, str] = {}
    for entry in request.models:
        model = session.get(CatalogModel, entry.model_id)
        if model is None or model.status != "smoke_passed":
            raise HTTPException(
                status_code=409, detail=f"{entry.model_id} has not passed contracts"
            )
        if get_settings().execution_mode != "mock" and entry.provider_slug in {
            "openrouter",
            "mock",
        }:
            raise HTTPException(
                status_code=409,
                detail=f"{entry.model_id} does not name a concrete provider endpoint",
            )
        if get_settings().execution_mode != "mock" and entry.execution_backend.value == "mock":
            raise HTTPException(
                status_code=409,
                detail=f"{entry.model_id} cannot use the mock execution backend in a live season",
            )
        if get_settings().execution_mode != "mock" and entry.endpoint_rate_card is None:
            raise HTTPException(
                status_code=409,
                detail=f"{entry.model_id} lacks endpoint-specific pricing evidence",
            )
        backend_contract_sha256 = entry.backend_contract_sha256 or _canonical_sha256({})
        if _canonical_sha256(entry.backend_contract) != backend_contract_sha256:
            raise HTTPException(
                status_code=409,
                detail=f"{entry.model_id} backend contract hash is invalid",
            )
        if entry.execution_backend.value == "bedrock":
            try:
                bedrock_contract = parse_bedrock_endpoint_contract(entry.backend_contract)
            except ValueError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=f"{entry.model_id} Bedrock backend contract is invalid",
                ) from exc
            rate_card_input = (
                entry.endpoint_rate_card.model_dump(mode="json")
                if entry.endpoint_rate_card is not None
                else {}
            )
            if (
                bedrock_contract.canonical_model_id != entry.expected_actual_model_id
                or entry.expected_actual_provider_slug != "amazon-bedrock"
                or not bedrock_contract.season_eligible
                or Decimal(bedrock_contract.price.input_per_million_usd) / Decimal(1_000_000)
                != Decimal(str(rate_card_input.get("prompt_price_per_token", "-1")))
                or Decimal(bedrock_contract.price.output_per_million_usd) / Decimal(1_000_000)
                != Decimal(str(rate_card_input.get("completion_price_per_token", "-1")))
                or bedrock_contract.price.source_uri != rate_card_input.get("pricing_source_uri")
                or bedrock_contract.price.observed_at != rate_card_input.get("pricing_observed_at")
                or bedrock_contract.bedrock_target_arn.original_sha256
                not in account_authorizations_by_backend["bedrock"].credential_binding_json.get(
                    "target_arn_sha256s", []
                )
            ):
                raise HTTPException(
                    status_code=409,
                    detail=f"{entry.model_id} Bedrock identity and price contracts disagree",
                )
        elif entry.execution_backend.value == "kimi_direct":
            kimi_contract = entry.backend_contract
            if (
                kimi_contract.get("schema_version")
                != "flavourbench-kimi-direct-endpoint-contract-v1"
                or kimi_contract.get("requested_model_id") != entry.expected_actual_model_id
                or kimi_contract.get("expected_actual_provider_slug") != "kimi-code-direct"
                or entry.expected_actual_provider_slug != "kimi-code-direct"
                or entry.provider_slug != "kimi-code-direct"
                or str(kimi_contract.get("base_url") or "").rstrip("/")
                != get_settings().kimi_base_url.rstrip("/")
                or not re.fullmatch(r"[0-9a-f]{64}", str(kimi_contract.get("catalog_sha256") or ""))
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(kimi_contract.get("catalog_entry_sha256") or ""),
                )
                or kimi_contract.get("allow_fallbacks") is not False
                or kimi_contract.get("season_eligible") is not True
            ):
                raise HTTPException(
                    status_code=409,
                    detail=f"{entry.model_id} direct Kimi identity contract is invalid",
                )
        elif entry.execution_backend.value == "qwencloud_direct":
            qwencloud_contract = entry.backend_contract
            if (
                qwencloud_contract.get("schema_version")
                != "flavourbench-qwencloud-direct-endpoint-contract-v1"
                or qwencloud_contract.get("requested_model_id") != entry.expected_actual_model_id
                or qwencloud_contract.get("expected_actual_provider_slug") != "qwencloud-direct"
                or entry.expected_actual_provider_slug != "qwencloud-direct"
                or entry.provider_slug != "qwencloud-direct"
                or str(qwencloud_contract.get("base_url") or "").rstrip("/")
                != get_settings().qwencloud_base_url.rstrip("/")
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(qwencloud_contract.get("catalog_sha256") or "")
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(qwencloud_contract.get("catalog_entry_sha256") or ""),
                )
                or qwencloud_contract.get("identity_kind") != "immutable_dated_release"
                or qwencloud_contract.get("allow_fallbacks") is not False
                or qwencloud_contract.get("structured_outputs_supported") is not True
                or qwencloud_contract.get("cost_reconciliation") != "provider_charge_available"
                or qwencloud_contract.get("season_eligible") is not True
                or qwencloud_contract.get("rank_eligible") is not True
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"{entry.model_id} QwenCloud route remains exploratory; immutable "
                        "identity, structured output, and charge reconciliation are required"
                    ),
                )
        decoding = entry.decoding.model_dump(exclude_none=True)
        contract = endpoint_contract_payload(
            model_id=entry.model_id,
            provider_slug=entry.provider_slug,
            expected_actual_model_id=entry.expected_actual_model_id,
            expected_actual_provider_slug=entry.expected_actual_provider_slug,
            supported_parameters=entry.supported_parameters,
            decoding=decoding,
            endpoint_max_completion_tokens=entry.endpoint_max_completion_tokens,
            endpoint_document_sha256=entry.endpoint_document_sha256,
        )
        contract_sha256 = endpoint_contract_sha256(**contract)
        rate_card, rate_card_sha256, computed_worst_case_micros = _rate_card_contract(
            model,
            max_completion_tokens=entry.decoding.max_tokens,
            endpoint_rate_card=(
                entry.endpoint_rate_card.model_dump(mode="json")
                if entry.endpoint_rate_card is not None
                else None
            ),
        )
        if entry.worst_case_cost_micros < computed_worst_case_micros or (
            get_settings().execution_mode != "mock" and entry.worst_case_cost_micros <= 0
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{entry.model_id} worst-case reservation is below its "
                    "frozen rate-card envelope"
                ),
            )
        if (
            model.endpoint_json.get("approved_provider") != entry.provider_slug
            or model.endpoint_json.get("execution_backend") != entry.execution_backend.value
            or model.endpoint_json.get("data_collection") != "deny"
            or model.endpoint_json.get("smoke_endpoint_contract") != contract
            or model.endpoint_json.get("smoke_endpoint_contract_sha256") != contract_sha256
            or model.endpoint_json.get("smoke_backend_contract_sha256")
            != entry.backend_contract_sha256
            or (
                entry.endpoint_rate_card is not None
                and model.endpoint_json.get("smoke_endpoint_rate_card_sha256")
                != _canonical_sha256(entry.endpoint_rate_card.model_dump(mode="json"))
            )
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{entry.model_id} endpoint contract does not match its smoke evidence",
            )
        smoke_artifact = model.endpoint_json.get("smoke_evidence_artifact")
        smoke_artifact_sha256 = model.endpoint_json.get("smoke_evidence_artifact_sha256")
        if get_settings().execution_mode != "mock" and (
            not isinstance(smoke_artifact, dict)
            or not isinstance(smoke_artifact_sha256, str)
            or _canonical_sha256(smoke_artifact) != smoke_artifact_sha256
            or smoke_artifact.get("actual_model_id") != entry.expected_actual_model_id
            or smoke_artifact.get("actual_provider_slug") != entry.expected_actual_provider_slug
            or smoke_artifact.get("tools_passed") is not True
            or smoke_artifact.get("structured_output_passed") is not True
            or smoke_artifact.get("data_collection_denied") is not True
            or smoke_artifact.get("cost_reconciled") is not True
            or smoke_artifact.get("schema_sha256") != FINAL_SCHEMA_SHA256
            or smoke_artifact.get("tool_schema_sha256") != request.tool_schema_sha256
            or model.endpoint_json.get("smoke_catalog_discovery_sha256")
            != model.endpoint_json.get("catalog_discovery_sha256")
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{entry.model_id} lacks intact content-addressed smoke evidence",
            )
        if isinstance(smoke_artifact_sha256, str):
            smoke_evidence_by_model[entry.model_id] = smoke_artifact_sha256
        if entry.slot_role == "closed_family" and model.open_weight:
            raise HTTPException(status_code=409, detail=f"{entry.model_id} is not closed-family")
        if entry.slot_role == "open_weight" and not model.open_weight:
            raise HTTPException(status_code=409, detail=f"{entry.model_id} is not open-weight")
        slot = session.scalar(
            select(SeasonModel).where(
                SeasonModel.season_id == season.id,
                SeasonModel.model_id == entry.model_id,
            )
        )
        if slot is None:
            slot = SeasonModel(season_id=season.id, model_id=entry.model_id)
            session.add(slot)
        slot.slot_role = entry.slot_role
        slot.execution_backend = entry.execution_backend.value
        slot.provider_slug = entry.provider_slug
        slot.expected_actual_model_id = entry.expected_actual_model_id
        slot.expected_actual_provider_slug = entry.expected_actual_provider_slug
        slot.supported_parameters_json = entry.supported_parameters
        slot.decoding_json = decoding
        slot.endpoint_max_completion_tokens = entry.endpoint_max_completion_tokens
        slot.endpoint_document_sha256 = entry.endpoint_document_sha256
        slot.endpoint_contract_sha256 = contract_sha256
        slot.backend_contract_json = dict(entry.backend_contract)
        slot.backend_contract_sha256 = backend_contract_sha256
        slot.rate_card_json = rate_card
        slot.rate_card_sha256 = rate_card_sha256
        slot.worst_case_cost_micros = entry.worst_case_cost_micros
        slot.eligible = True
        selected_slots.append(slot)
        manifest_rows.append(
            {
                **entry.model_dump(mode="json"),
                "rate_card_sha256": rate_card_sha256,
                "computed_worst_case_cost_micros": computed_worst_case_micros,
                "smoke_evidence_artifact_sha256": smoke_artifact_sha256,
            }
        )

    digest = hashlib.sha256(
        json.dumps(sorted(manifest_rows, key=lambda row: row["model_id"]), sort_keys=True).encode()
    ).hexdigest()
    for slot in selected_slots:
        slot.manifest_sha256 = digest
    season.manifest_sha256 = digest
    season.tool_registry_sha256 = request.tool_schema_sha256
    season.epicure_release_id = request.epicure_release_id
    season.epicure_bundle_sha256 = request.epicure_bundle_sha256
    season.epicure_application_sha256 = request.epicure_application_sha256
    season.analysis_plan_sha256 = request.analysis_plan_sha256
    model_smoke_registry_sha256 = _canonical_sha256(smoke_evidence_by_model)
    protocol_bundle, protocol_bundle_sha256 = build_protocol_bundle(
        tool_registry_sha256=season.tool_registry_sha256,
        epicure_release_id=season.epicure_release_id,
        epicure_bundle_sha256=season.epicure_bundle_sha256,
        epicure_application_sha256=season.epicure_application_sha256,
        analysis_plan_sha256=season.analysis_plan_sha256,
        model_smoke_registry_sha256=model_smoke_registry_sha256,
    )
    season.protocol_bundle_json = protocol_bundle
    season.protocol_bundle_sha256 = protocol_bundle_sha256
    if request.budget_cap_micros > sum(row.budget_cap_micros for row in authorization_rows):
        raise HTTPException(
            status_code=409,
            detail="season cap exceeds its provider-scoped authorizations",
        )
    season.budget_cap_micros = request.budget_cap_micros
    season.status = "pilot"
    season.frozen_at = datetime.now(UTC)
    session.add_all(authorization_rows)
    session.add(
        RunEvent(
            entity_type="season",
            entity_id=season.id,
            event_type="season_manifest_frozen",
            payload_json={
                "manifest_sha256": digest,
                "analysis_plan_sha256": season.analysis_plan_sha256,
                "protocol_bundle_sha256": season.protocol_bundle_sha256,
                "model_smoke_registry_sha256": model_smoke_registry_sha256,
                "epicure_application_sha256": request.epicure_application_sha256,
                "lineage_reference_sha256": hashlib.sha256(
                    request.lineage_reference.encode()
                ).hexdigest(),
                "budget_authorization_reference_sha256": hashlib.sha256(
                    request.budget_authorization_reference.encode()
                ).hexdigest(),
                "provider_budget_authorization_sha256": {
                    row.execution_backend: row.authorization_envelope_sha256
                    for row in authorization_rows
                },
                "provider_account_authorization_sha256": {
                    backend: row.authorization_envelope_sha256
                    for backend, row in sorted(account_authorizations_by_backend.items())
                },
            },
        )
    )
    session.commit()
    return {"season": season.slug, "status": season.status, "manifestSha256": digest}


@router.post(
    "/admin/seasons/{season_slug}/officialize", dependencies=[Depends(require_admin_token)]
)
def admin_season_officialize(
    season_slug: str,
    request: SeasonOfficializeCreate,
    session: Db,
) -> dict:
    settings = get_settings()
    if (
        settings.environment != "production"
        or session.bind is None
        or session.bind.dialect.name != "postgresql"
    ):
        raise HTTPException(
            status_code=409,
            detail="officialization requires the production PostgreSQL service",
        )
    try:
        database_readiness(session, expected_role="flavourbench_api")
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail="officialization requires the governed production database",
        ) from exc
    season = session.scalar(select(Season).where(Season.slug == season_slug).with_for_update())
    if season is None or season.status != "pilot" or season.manifest_sha256 == "unfrozen":
        raise HTTPException(status_code=409, detail="only a frozen pilot can become official")
    if (
        season.budget_cap_micros <= 0
        or season.epicure_bundle_sha256 in {"", "unresolved"}
        or season.epicure_application_sha256 in {"", "unresolved"}
    ):
        raise HTTPException(status_code=409, detail="budget and model lineage remain unresolved")
    _require_budget_integrity(session, season.id)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", settings.build_image_digest):
        raise HTTPException(
            status_code=409,
            detail="officialization requires the deployed OCI image digest",
        )
    release = _verified_official_epicure_release(session, season)
    if request.task_registry_manifest_sha256 != season.prompt_registry_sha256:
        raise HTTPException(
            status_code=409,
            detail="officialization approval does not match the frozen task registry",
        )
    if request.analysis_plan_sha256 != season.analysis_plan_sha256:
        raise HTTPException(
            status_code=409,
            detail="officialization approval does not match the frozen analysis plan",
        )
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
    if (
        season.protocol_bundle_sha256 in UNFROZEN_VALUES
        or season.protocol_bundle_json != protocol_bundle
        or season.protocol_bundle_sha256 != protocol_bundle_sha256
    ):
        raise HTTPException(
            status_code=409,
            detail="the frozen execution and analysis protocol has drifted",
        )
    slots = session.scalars(
        select(SeasonModel).where(
            SeasonModel.season_id == season.id,
            SeasonModel.eligible.is_(True),
        )
    ).all()
    if len(slots) != SEASON_MODEL_COUNT:
        raise HTTPException(
            status_code=409,
            detail=f"official manifest must contain {SEASON_MODEL_COUNT} models",
        )
    for slot in slots:
        if (
            slot.manifest_sha256 != season.manifest_sha256
            or slot.execution_backend == "mock"
            or slot.provider_slug == "mock"
            or slot.expected_actual_model_id in UNFROZEN_VALUES
            or slot.expected_actual_provider_slug in UNFROZEN_VALUES
            or slot.endpoint_document_sha256 in UNFROZEN_VALUES
            or slot.endpoint_contract_sha256 in UNFROZEN_VALUES
            or slot.backend_contract_sha256 in UNFROZEN_VALUES
            or _canonical_sha256(slot.backend_contract_json) != slot.backend_contract_sha256
            or slot.rate_card_sha256 in UNFROZEN_VALUES
            or _canonical_sha256(slot.rate_card_json) != slot.rate_card_sha256
            or slot.worst_case_cost_micros <= 0
            or endpoint_contract_sha256(
                model_id=slot.model_id,
                provider_slug=slot.provider_slug,
                expected_actual_model_id=slot.expected_actual_model_id,
                expected_actual_provider_slug=slot.expected_actual_provider_slug,
                supported_parameters=slot.supported_parameters_json,
                decoding=slot.decoding_json,
                endpoint_max_completion_tokens=slot.endpoint_max_completion_tokens,
                endpoint_document_sha256=slot.endpoint_document_sha256,
            )
            != slot.endpoint_contract_sha256
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{slot.model_id} lacks an intact frozen endpoint contract",
            )
    required_backends = {slot.execution_backend for slot in slots}
    provider_budgets = session.scalars(
        select(SeasonProviderBudget).where(SeasonProviderBudget.season_id == season.id)
    ).all()
    budgets_by_backend = {row.execution_backend: row for row in provider_budgets}
    if set(budgets_by_backend) != required_backends:
        raise HTTPException(
            status_code=409,
            detail="officialization requires a distinct authorization for every provider backend",
        )
    now = datetime.now(UTC)
    for backend, budget in budgets_by_backend.items():
        account_budget = session.scalar(
            select(ProviderAccountBudget)
            .where(
                ProviderAccountBudget.execution_backend == backend,
                ProviderAccountBudget.account_scope_sha256
                == provider_account_scope_sha256(backend),
            )
            .with_for_update()
        )
        valid_until = budget.valid_until
        if valid_until.tzinfo is None:
            valid_until = valid_until.replace(tzinfo=UTC)
        account_authorization_epoch = (
            account_authorization(session, account_budget, for_update=True)
            if account_budget is not None
            else None
        )
        account_valid_until = (
            account_authorization_epoch.valid_until
            if account_authorization_epoch is not None
            else now
        )
        if account_valid_until.tzinfo is None:
            account_valid_until = account_valid_until.replace(tzinfo=UTC)
        if (
            budget.currency != "USD"
            or budget.budget_cap_micros <= 0
            or budget.budget_used_micros + budget.budget_reserved_micros > budget.budget_cap_micros
            or valid_until <= now
            or _canonical_sha256(budget.authorization_envelope_json)
            != budget.authorization_envelope_sha256
            or budget.authorization_envelope_json.get("execution_backend") != backend
            or budget.authorization_envelope_json.get("season_slug") != season.slug
            or budget.authorization_envelope_json.get("budget_cap_micros")
            != budget.budget_cap_micros
            or account_budget is None
            or account_budget.status != "active"
            or account_budget.budget_cap_micros != provider_account_hard_cap_micros(backend)
            or account_budget.budget_used_micros + account_budget.budget_reserved_micros
            > account_budget.budget_cap_micros
            or account_valid_until <= now
            or not account_authorization_chain_valid(
                session,
                account_budget,
                account_authorization_epoch,
                root_envelope_sha256=(budget.account_authorization_envelope_sha256),
                signing_secret=(get_settings().budget_authorization_signing_secret),
                verification_keys=budget_authorization_verification_keyring(get_settings()),
                now=now,
            )
            or budget.account_scope_sha256 != account_budget.account_scope_sha256
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{backend} provider budget authorization is invalid",
            )
    reviewers = session.scalars(
        select(ExpertReviewer).where(
            ExpertReviewer.active.is_(True),
            ExpertReviewer.qualification_verified.is_(True),
            ExpertReviewer.cohort == "expert_independent",
        )
    ).all()
    required = {"substitution", "composition", "cookability", "evidence"}
    family_counts = {
        family: sum(family in reviewer.qualification_json for reviewer in reviewers)
        for family in required
    }
    if any(count < 2 for count in family_counts.values()):
        raise HTTPException(
            status_code=409,
            detail="official collection requires two verified independent experts per family",
        )
    tasks = session.scalars(select(Task).where(Task.season_id == season.id)).all()
    expected_split_counts = SEASON_TASK_SPLIT_COUNTS
    observed_split_counts = {
        split: sum(task.split == split for task in tasks) for split in expected_split_counts
    }
    if len(tasks) != CONFIRMATORY_TASK_COUNT or observed_split_counts != expected_split_counts:
        raise HTTPException(
            status_code=409,
            detail=(
                "official seasons require a separate "
                f"{CONFIRMATORY_TASK_COUNT}-task held-out registry"
            ),
        )
    family_task_counts = {
        family: sum(task.family == family for task in tasks) for family in required
    }
    if any(count != CONFIRMATORY_TASKS_PER_FAMILY for count in family_task_counts.values()):
        raise HTTPException(
            status_code=409,
            detail=(
                "official seasons require exactly "
                f"{CONFIRMATORY_TASKS_PER_FAMILY} confirmatory tasks per family"
            ),
        )
    family_split_counts = {
        (family, split): sum(task.family == family and task.split == split for task in tasks)
        for family in required
        for split in expected_split_counts
    }
    if any(
        family_split_counts[(family, split)] != per_family
        for family in required
        for split, per_family in SEASON_TASK_SPLIT_COUNTS_PER_FAMILY.items()
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "each official family requires 40 scored, 10 development, and "
                "10 private-reserve tasks"
            ),
        )
    for task in tasks:
        provenance = task.provenance_json if isinstance(task.provenance_json, dict) else {}
        reviews = provenance.get("independent_reviews")
        author_id = str(provenance.get("human_author_id", ""))
        reviewer_ids = {
            str(review.get("reviewer_id"))
            for review in reviews or []
            if (
                isinstance(review, dict)
                and review.get("decision") == "approve"
                and review.get("independent_of_author") is True
                and review.get("blind_review_event_sha256")
                and review.get("reconciliation_event_sha256")
            )
        }
        adjudication = provenance.get("adjudication")
        adjudicator_id = (
            str(adjudication.get("adjudicator_reviewer_id", ""))
            if isinstance(adjudication, dict)
            else ""
        )
        if (
            task.review_status != "frozen"
            or task.split not in expected_split_counts
            or provenance.get("origin_type") != "human_authored"
            or not author_id
            or provenance.get("confirmatory_eligible") is not True
            or len(reviewer_ids) != 2
            or author_id in reviewer_ids
            or not isinstance(adjudication, dict)
            or adjudication.get("decision") != "approve"
            or adjudication.get("independent_of_author_and_reviewers") is not True
            or not adjudication.get("adjudication_event_sha256")
            or not adjudication.get("criterion_pack_sha256")
            or not adjudicator_id
            or adjudicator_id == author_id
            or adjudicator_id in reviewer_ids
            or provenance.get("criterion_pack_sha256") != adjudication.get("criterion_pack_sha256")
            or provenance.get("contamination_audit_status") != "pass"
            or not provenance.get("contamination_audit_sha256")
            or not provenance.get("validator_contract_sha256")
            or not provenance.get("review_history_sha256")
            or provenance.get("author_reviewer_independence_audit") != "pass"
            or not provenance.get("task_record_sha256")
            or provenance.get("evidence_registry_status") != "verified"
            or not provenance.get("task_evidence_root_sha256")
            or not _verified_task_evidence_registry(session, task)
        ):
            raise HTTPException(
                status_code=409,
                detail=f"{task.public_id} lacks independent official task evidence",
            )
    season.official = True
    season.status = "active"
    session.add(
        RunEvent(
            entity_type="season",
            entity_id=season.id,
            event_type="season_officialized",
            payload_json={
                **request.model_dump(),
                "epicure_lineage_manifest_sha256": release.lineage_manifest_sha256,
                "provider_budget_authorization_sha256": {
                    backend: budget.authorization_envelope_sha256
                    for backend, budget in sorted(budgets_by_backend.items())
                },
            },
        )
    )
    session.commit()
    return {"season": season.slug, "status": season.status, "official": season.official}


@router.post("/admin/retention/run", dependencies=[Depends(require_admin_token)])
def admin_retention(session: Db) -> dict:
    count = redact_expired(session)
    session.commit()
    return {"redactedBattles": count}


@router.post(
    "/admin/battles/{battle_id}/release-review",
    dependencies=[Depends(require_admin_token)],
)
def admin_release_review(battle_id: str, request: ReleaseReviewCreate, session: Db) -> dict:
    battle = session.get(Battle, battle_id)
    if battle is None or not battle.research_consent:
        raise HTTPException(status_code=404, detail="consented battle not found")
    battle.release_review_status = request.status
    battle.release_reviewed_at = datetime.now(UTC)
    session.add(
        RunEvent(
            entity_type="battle",
            entity_id=battle.id,
            event_type="research_release_reviewed",
            payload_json={
                "status": request.status,
                "review_reference": request.review_reference,
            },
        )
    )
    session.commit()
    return {"battleId": battle.id, "releaseReviewStatus": battle.release_review_status}


@router.post(
    "/admin/controlled-runs/{run_id}/postcollection-item-audits",
    dependencies=[Depends(require_admin_token)],
)
def admin_register_postcollection_item_audit(
    run_id: str,
    request: Season1PostcollectionItemAuditCreate,
    session: Db,
) -> dict:
    season, run = _locked_controlled_run(session, run_id)
    artifact = request.artifact
    if season.slug != "season-1" or run.status != "closed":
        raise HTTPException(
            status_code=409,
            detail="post-collection audit registration requires a closed Season 1 run",
        )
    if not valid_post_collection_item_audit(
        artifact,
        study_design_sha256=STUDY_DESIGN_SHA256,
    ):
        raise HTTPException(
            status_code=409,
            detail="post-collection item audit failed the frozen evidence contract",
        )
    records = artifact.get("task_records")
    task_rows = session.scalars(select(Task).where(Task.season_id == season.id)).all()
    tasks_by_public_id = {task.public_id: task for task in task_rows}
    if not isinstance(records, list):
        raise HTTPException(status_code=409, detail="post-collection audit has no task records")
    for record in records:
        task = tasks_by_public_id.get(str(record.get("task_id")))
        provenance = task.provenance_json if task is not None else None
        if (
            task is None
            or not isinstance(provenance, dict)
            or record.get("task_content_sha256") != provenance.get("task_record_sha256")
        ):
            raise HTTPException(
                status_code=409,
                detail="post-collection audit task identity or content hash is not current",
            )

    events = session.scalars(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "controlled_run",
            RunEvent.entity_id == run.id,
            RunEvent.event_type == "season1_post_collection_item_audit_verified",
        )
        .order_by(RunEvent.created_at, RunEvent.id)
        .with_for_update()
    ).all()
    superseded_ids = {
        str(event.payload_json.get("supersedes_event_id"))
        for event in events
        if event.payload_json.get("supersedes_event_id") is not None
    }
    if not superseded_ids.issubset({event.id for event in events}):
        raise HTTPException(status_code=409, detail="post-collection audit chain is invalid")
    heads = [event for event in events if event.id not in superseded_ids]
    artifact_sha256 = str(artifact["artifact_sha256"])
    if len(heads) == 1 and heads[0].payload_json.get("artifact_sha256") == artifact_sha256:
        return {
            "eventId": heads[0].id,
            "artifactSha256": artifact_sha256,
            "idempotent": True,
            "withdrawnSnapshotIds": [],
        }
    if (
        len(heads) > 1
        or (request.supersedes_event_id is None and heads)
        or (
            request.supersedes_event_id is not None
            and (len(heads) != 1 or heads[0].id != request.supersedes_event_id)
        )
    ):
        raise HTTPException(
            status_code=409,
            detail="audit correction must supersede the one active post-collection audit",
        )
    event = RunEvent(
        entity_type="controlled_run",
        entity_id=run.id,
        event_type="season1_post_collection_item_audit_verified",
        payload_json={
            "artifact": artifact,
            "artifact_sha256": artifact_sha256,
            "verification_status": "verified",
            "verifier": "flavourbench.season1_readiness",
            "study_design_artifact_sha256": STUDY_DESIGN_SHA256,
            "supersedes_event_id": request.supersedes_event_id,
        },
    )
    session.add(event)
    session.flush()
    withdrawn_snapshot_ids = _withdraw_published_snapshots(
        session,
        season_id=season.id,
        controlled_run_id=run.id,
        reason_code="postcollection_item_audit_registered_or_superseded",
    )
    session.commit()
    return {
        "eventId": event.id,
        "artifactSha256": artifact_sha256,
        "idempotent": False,
        "withdrawnSnapshotIds": withdrawn_snapshot_ids,
    }


@router.post(
    "/admin/controlled-runs/{run_id}/arena-method-validation",
    dependencies=[Depends(require_admin_token)],
)
def admin_register_arena_method_validation(
    run_id: str,
    request: Season1ArenaMethodValidationCreate,
    session: Db,
) -> dict:
    season, run = _locked_controlled_run(session, run_id)
    artifact = request.artifact
    if season.slug != "season-1" or run.status != "closed":
        raise HTTPException(
            status_code=409,
            detail="method validation registration requires a closed Season 1 run",
        )
    if not verify_production_result(artifact):
        raise HTTPException(
            status_code=409,
            detail="arena method validation is incomplete, failing, or malformed",
        )
    existing = session.scalars(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "controlled_run",
            RunEvent.entity_id == run.id,
            RunEvent.event_type == "season1_arena_monte_carlo_validation_verified",
        )
        .order_by(RunEvent.created_at, RunEvent.id)
        .with_for_update()
    ).all()
    artifact_sha256 = str(artifact["artifact_sha256"])
    if len(existing) == 1 and existing[0].payload_json.get("artifact_sha256") == artifact_sha256:
        return {
            "eventId": existing[0].id,
            "artifactSha256": artifact_sha256,
            "idempotent": True,
            "withdrawnSnapshotIds": [],
        }
    if existing:
        raise HTTPException(
            status_code=409,
            detail="a different frozen arena method validation is already registered",
        )
    event = RunEvent(
        entity_type="controlled_run",
        entity_id=run.id,
        event_type="season1_arena_monte_carlo_validation_verified",
        payload_json={
            "artifact": artifact,
            "artifact_sha256": artifact_sha256,
            "verification_status": "verified",
            "verifier": "flavourbench.season1_arena_monte_carlo",
        },
    )
    session.add(event)
    session.flush()
    withdrawn_snapshot_ids = _withdraw_published_snapshots(
        session,
        season_id=season.id,
        controlled_run_id=run.id,
        reason_code="arena_method_validation_registered",
    )
    session.commit()
    return {
        "eventId": event.id,
        "artifactSha256": artifact_sha256,
        "idempotent": False,
        "withdrawnSnapshotIds": withdrawn_snapshot_ids,
    }


def _create_leaderboard_snapshot(
    session: Db,
    season: str,
    track: str,
    cohort: str,
    category: str,
    data_stratum: str,
    controlled_run_id: str | None,
) -> dict:
    require_snapshot_analysis_process()
    _prepare_snapshot_transaction(session)
    season_row = session.scalar(select(Season).where(Season.slug == season))
    if season_row is None:
        raise HTTPException(status_code=404, detail="season not found")
    if season_row.status != "active":
        raise HTTPException(status_code=409, detail="snapshots require an active season")
    if data_stratum == "public_freeform" and not season_row.official:
        raise HTTPException(
            status_code=409,
            detail="public leaderboard snapshots require an official season",
        )
    _verify_season_protocol(season_row)
    controlled_run = None
    if data_stratum == "controlled":
        if controlled_run_id is None:
            raise HTTPException(status_code=422, detail="controlledRunId is required")
        controlled_run = session.scalar(
            select(ControlledRun).where(ControlledRun.id == controlled_run_id)
        )
        if controlled_run is None or controlled_run.season_id != season_row.id:
            raise HTTPException(status_code=404, detail="controlled run not found")
        _verify_controlled_run_contract(session, controlled_run)
        if controlled_run.status != "closed":
            raise HTTPException(
                status_code=409,
                detail="controlled snapshots require a closed, reconciled run",
            )
        controlled_assignments = session.scalars(
            select(ControlledRunAssignment).where(
                ControlledRunAssignment.controlled_run_id == controlled_run.id
            )
        ).all()
        controlled_battles = session.scalars(
            select(Battle).where(Battle.controlled_run_id == controlled_run.id)
        ).all()
        try:
            verify_controlled_run_bijection(
                session,
                controlled_assignments,
                controlled_battles,
                require_terminal=True,
            )
        except ControlledRunIntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"controlled-run snapshot integrity failed: {exc}",
            ) from exc
    elif controlled_run_id is not None:
        raise HTTPException(
            status_code=422,
            detail="controlledRunId is not valid for the public-freeform stratum",
        )
    evidence_cutoff_at = datetime.now(UTC)
    payload = (
        model_leaderboard(
            session,
            season_row,
            cohort,
            category,
            data_stratum,
            controlled_run_id,
            evidence_cutoff_at,
        )
        if track == "model_arena"
        else uplift_leaderboard(
            session,
            season_row,
            cohort,
            category,
            data_stratum,
            controlled_run_id,
            evidence_cutoff_at,
        )
    )
    digest = snapshot_hash(payload)
    try:
        evidence = _snapshot_evidence_manifest(
            session,
            season=season_row,
            track=track,
            cohort=cohort,
            category=category,
            data_stratum=data_stratum,
            controlled_run_id=controlled_run_id,
            evidence_cutoff_at=evidence_cutoff_at,
        )
    except _SnapshotVerificationError as failure:
        raise HTTPException(status_code=409, detail=failure.detail) from failure
    try:
        _verify_snapshot_observation_alignment(payload, evidence)
    except _SnapshotVerificationError as failure:
        raise HTTPException(status_code=409, detail=failure.detail) from failure
    evidence_sha256 = _canonical_sha256(evidence)
    predecessor_statement = select(LeaderboardSnapshot).where(
        LeaderboardSnapshot.season_id == season_row.id,
        LeaderboardSnapshot.track == track,
        LeaderboardSnapshot.cohort == cohort,
        LeaderboardSnapshot.category == category,
        LeaderboardSnapshot.data_stratum == data_stratum,
        LeaderboardSnapshot.publication_status == "published",
    )
    if controlled_run is None:
        predecessor_statement = predecessor_statement.where(
            LeaderboardSnapshot.controlled_run_id.is_(None)
        )
    else:
        predecessor_statement = predecessor_statement.where(
            LeaderboardSnapshot.controlled_run_id == controlled_run.id
        )
    predecessor = session.scalar(
        predecessor_statement.order_by(
            LeaderboardSnapshot.published_at.desc(),
            LeaderboardSnapshot.created_at.desc(),
            LeaderboardSnapshot.id.desc(),
        ).limit(1)
    )
    snapshot = LeaderboardSnapshot(
        season_id=season_row.id,
        track=track,
        cohort=cohort,
        category=category,
        data_stratum=data_stratum,
        controlled_run_id=controlled_run.id if controlled_run else None,
        publication_status="draft",
        input_sha256=digest,
        input_evidence_sha256=evidence_sha256,
        input_evidence_json=evidence,
        payload_sha256=digest,
        evidence_cutoff_at=evidence_cutoff_at,
        supersedes_snapshot_id=predecessor.id if predecessor else None,
        payload_json=payload,
    )
    session.add(snapshot)
    session.commit()
    return {
        "snapshotId": snapshot.id,
        "inputSha256": digest,
        "inputEvidenceSha256": evidence_sha256,
        "payloadSha256": digest,
        "evidenceCutoffAt": _utc_iso(evidence_cutoff_at),
        "publicationStatus": snapshot.publication_status,
        "controlledRunId": snapshot.controlled_run_id,
    }


@router.post("/admin/leaderboards/snapshot", dependencies=[Depends(require_admin_token)])
def admin_snapshot(
    session: Db,
    season: str = Query(default="season-0"),
    track: str = Query(default="model_arena", pattern="^(model_arena|epicure_uplift)$"),
    cohort: str = Query(
        default="public",
        pattern=(
            "^(public|expert_independent|expert_product_affiliated|"
            "expert_provider_affiliated|combined)$"
        ),
    ),
    category: str = Query(default="all"),
    data_stratum: str = Query(default="public_freeform", pattern="^(public_freeform|controlled)$"),
    controlled_run_id: str | None = Query(default=None),
) -> dict:
    try:
        require_snapshot_analysis_process()
    except InProcessSnapshotAnalysisForbidden as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if season == "season-1":
        raise HTTPException(
            status_code=409,
            detail="Season 1 snapshot analysis must run through the asynchronous worker",
        )
    return _create_leaderboard_snapshot(
        session,
        season,
        track,
        cohort,
        category,
        data_stratum,
        controlled_run_id,
    )


@router.post(
    "/admin/leaderboards/snapshot-jobs",
    status_code=202,
    dependencies=[Depends(require_admin_token)],
)
def admin_snapshot_job(
    session: Db,
    season: str = Query(default="season-1"),
    track: str = Query(default="model_arena", pattern="^(model_arena|epicure_uplift)$"),
    cohort: str = Query(
        default="public",
        pattern=(
            "^(public|expert_independent|expert_product_affiliated|"
            "expert_provider_affiliated|combined)$"
        ),
    ),
    category: str = Query(default="all"),
    data_stratum: str = Query(default="controlled", pattern="^(public_freeform|controlled)$"),
    controlled_run_id: str | None = Query(default=None),
) -> dict:
    season_row = session.scalar(select(Season).where(Season.slug == season))
    if season_row is None:
        raise HTTPException(status_code=404, detail="season not found")
    request = {
        "season": season,
        "track": track,
        "cohort": cohort,
        "category": category,
        "data_stratum": data_stratum,
        "controlled_run_id": controlled_run_id,
    }
    request_sha256 = _canonical_sha256(request)
    prior = session.scalars(
        select(Job)
        .where(
            Job.kind == "leaderboard_snapshot",
            Job.status.in_(("queued", "running")),
        )
        .order_by(Job.created_at, Job.id)
    ).all()
    for item in prior:
        if item.payload_json.get("analysis_request_sha256") == request_sha256:
            return {
                "jobId": item.id,
                "status": item.status,
                "analysisRequestSha256": request_sha256,
            }
    job = Job(
        kind="leaderboard_snapshot",
        battle_id=None,
        payload_json={
            **request,
            "analysis_request_sha256": request_sha256,
            "requested_at": _utc_iso(datetime.now(UTC)),
        },
        status="queued",
        max_attempts=1,
        available_at=datetime.now(UTC),
    )
    session.add(job)
    session.commit()
    return {
        "jobId": job.id,
        "status": job.status,
        "analysisRequestSha256": request_sha256,
    }


@router.get(
    "/admin/leaderboards/snapshot-jobs/{job_id}",
    dependencies=[Depends(require_admin_token)],
)
def admin_snapshot_job_status(job_id: str, session: Db) -> dict:
    job = session.get(Job, job_id)
    if job is None or job.kind != "leaderboard_snapshot":
        raise HTTPException(status_code=404, detail="snapshot analysis job not found")
    return {
        "jobId": job.id,
        "status": job.status,
        "attempts": job.attempts,
        "snapshotId": job.payload_json.get("snapshot_id"),
        "analysisRequestSha256": job.payload_json.get("analysis_request_sha256"),
        "completedAt": _utc_iso(job.completed_at),
        "error": job.last_error if job.status == "failed" else None,
    }


@router.post(
    "/admin/leaderboards/snapshots/{snapshot_id}/publish",
    dependencies=[Depends(require_admin_token)],
)
def admin_publish_snapshot(
    snapshot_id: str,
    request: SnapshotPublishCreate,
    session: Db,
) -> dict:
    _prepare_snapshot_transaction(session)
    probe = session.get(LeaderboardSnapshot, snapshot_id)
    if probe is None:
        raise HTTPException(status_code=404, detail="leaderboard snapshot not found")
    season = session.scalar(select(Season).where(Season.id == probe.season_id).with_for_update())
    snapshot = session.scalar(
        select(LeaderboardSnapshot).where(LeaderboardSnapshot.id == snapshot_id).with_for_update()
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="leaderboard snapshot not found")
    if snapshot.data_stratum != "public_freeform" or snapshot.controlled_run_id is not None:
        raise HTTPException(
            status_code=409,
            detail="controlled-run snapshots cannot be published through the public leaderboard",
        )
    if season is None or not season.official or season.status != "active":
        raise HTTPException(status_code=409, detail="season is not authorized for publication")
    _verify_season_protocol(season)
    _require_budget_integrity(session, season.id)
    if not snapshot.input_evidence_sha256 or not snapshot.payload_sha256:
        raise HTTPException(
            status_code=409,
            detail="legacy snapshots without evidence manifests cannot be published",
        )
    if snapshot.publication_status != "draft":
        raise HTTPException(
            status_code=409,
            detail="only a draft snapshot can be published",
        )
    try:
        payload = _verified_current_snapshot_payload(
            session,
            season=season,
            snapshot=snapshot,
        )
    except _SnapshotVerificationError as failure:
        _withdraw_snapshot_after_verification_failure(
            session,
            snapshot=snapshot,
            failure=failure,
        )
        session.commit()
        raise HTTPException(
            status_code=409,
            detail="snapshot evidence changed; create and review a new snapshot",
        ) from failure
    accounting = payload.get("accounting")
    if (
        not isinstance(accounting, dict)
        or accounting.get("complete") is not True
        or accounting.get("billing_reconciliation_complete") is not True
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "official publication requires complete generation-cost accounting "
                "and Bedrock billing cross-checks"
            ),
        )
    _require_season1_statistical_acceptance(season, snapshot, payload)
    predecessor_ids = _withdraw_published_scope_predecessors(
        session,
        snapshot=snapshot,
    )
    snapshot.publication_status = "published"
    snapshot.publication_reference_sha256 = hashlib.sha256(
        request.publication_reference.encode()
    ).hexdigest()
    snapshot.published_at = datetime.now(UTC)
    session.add(
        RunEvent(
            entity_type="leaderboard_snapshot",
            entity_id=snapshot.id,
            event_type="public_leaderboard_snapshot_published",
            payload_json={
                "input_sha256": snapshot.input_sha256,
                "input_evidence_sha256": snapshot.input_evidence_sha256,
                "publication_reference_sha256": snapshot.publication_reference_sha256,
                "withdrawn_predecessor_ids": predecessor_ids,
            },
        )
    )
    session.commit()
    return {
        "snapshotId": snapshot.id,
        "publicationStatus": snapshot.publication_status,
        "publishedAt": snapshot.published_at.isoformat(),
    }


@router.post(
    "/admin/leaderboards/snapshots/{snapshot_id}/withdraw",
    dependencies=[Depends(require_admin_token)],
)
def admin_withdraw_public_snapshot(
    snapshot_id: str,
    request: SnapshotPublishCreate,
    session: Db,
) -> dict:
    probe = session.get(LeaderboardSnapshot, snapshot_id)
    if probe is None:
        raise HTTPException(status_code=404, detail="published public snapshot not found")
    session.scalar(select(Season).where(Season.id == probe.season_id).with_for_update())
    snapshot = session.scalar(
        select(LeaderboardSnapshot)
        .where(
            LeaderboardSnapshot.id == snapshot_id,
            LeaderboardSnapshot.data_stratum == "public_freeform",
            LeaderboardSnapshot.controlled_run_id.is_(None),
            LeaderboardSnapshot.publication_status == "published",
        )
        .with_for_update()
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="published public snapshot not found")
    snapshot.publication_status = "withdrawn"
    reference_sha256 = hashlib.sha256(request.publication_reference.encode()).hexdigest()
    session.add(
        RunEvent(
            entity_type="leaderboard_snapshot",
            entity_id=snapshot.id,
            event_type="public_leaderboard_snapshot_withdrawn",
            payload_json={"withdrawal_reference_sha256": reference_sha256},
        )
    )
    session.commit()
    return {
        "snapshotId": snapshot.id,
        "publicationStatus": snapshot.publication_status,
    }


@router.post(
    "/admin/controlled-runs/{run_id}/snapshots/{snapshot_id}/publish",
    dependencies=[Depends(require_admin_token)],
)
def admin_publish_controlled_snapshot(
    run_id: str,
    snapshot_id: str,
    request: SnapshotPublishCreate,
    session: Db,
) -> dict:
    _prepare_snapshot_transaction(session)
    season, run = _locked_controlled_run(session, run_id)
    snapshot = session.scalar(
        select(LeaderboardSnapshot)
        .where(
            LeaderboardSnapshot.id == snapshot_id,
            LeaderboardSnapshot.controlled_run_id == run.id,
            LeaderboardSnapshot.data_stratum == "controlled",
        )
        .with_for_update()
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="controlled snapshot not found")
    if (
        season.status != "active"
        or run.status != "closed"
        or not _active_controlled_release_authorization(session, run, lock=True)
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "private publication requires an active season, a closed run, "
                "and customer authorization"
            ),
        )
    _verify_season_protocol(season)
    _require_budget_integrity(session, season.id)
    if not snapshot.input_evidence_sha256 or not snapshot.payload_sha256:
        raise HTTPException(
            status_code=409,
            detail="snapshot lacks a content-addressed evidence manifest",
        )
    if snapshot.publication_status != "draft":
        raise HTTPException(
            status_code=409,
            detail="only a draft controlled snapshot can be published",
        )
    try:
        payload = _verified_current_snapshot_payload(
            session,
            season=season,
            snapshot=snapshot,
        )
    except _SnapshotVerificationError as failure:
        _withdraw_snapshot_after_verification_failure(
            session,
            snapshot=snapshot,
            failure=failure,
        )
        session.commit()
        raise HTTPException(
            status_code=409,
            detail="snapshot evidence changed; create and review a new snapshot",
        ) from failure
    accounting = payload.get("accounting")
    if (
        not isinstance(accounting, dict)
        or accounting.get("complete") is not True
        or accounting.get("billing_reconciliation_complete") is not True
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "snapshot publication requires complete generation-cost accounting "
                "and Bedrock billing cross-checks"
            ),
        )
    _require_season1_statistical_acceptance(season, snapshot, payload)
    predecessor_ids = _withdraw_published_scope_predecessors(
        session,
        snapshot=snapshot,
    )
    snapshot.publication_status = "published"
    snapshot.publication_reference_sha256 = hashlib.sha256(
        request.publication_reference.encode()
    ).hexdigest()
    snapshot.published_at = datetime.now(UTC)
    if run.evaluation_order_id is not None:
        order = session.get(EvaluationOrder, run.evaluation_order_id)
        if order is None or order.publication_status != "authorized":
            raise HTTPException(
                status_code=409,
                detail="commercial publication authorization state changed",
            )
        order.publication_status = "published"
    session.add(
        RunEvent(
            entity_type="leaderboard_snapshot",
            entity_id=snapshot.id,
            event_type="controlled_snapshot_privately_published",
            payload_json={
                "controlled_run_id": run.id,
                "input_evidence_sha256": snapshot.input_evidence_sha256,
                "payload_sha256": snapshot.payload_sha256,
                "publication_reference_sha256": (snapshot.publication_reference_sha256),
                "withdrawn_predecessor_ids": predecessor_ids,
            },
        )
    )
    session.commit()
    return {
        "snapshotId": snapshot.id,
        "runId": run.id,
        "publicationStatus": snapshot.publication_status,
        "publishedAt": snapshot.published_at.isoformat(),
        "visibility": "controlled-run credential only",
    }


@router.post(
    "/admin/controlled-runs/{run_id}/snapshots/{snapshot_id}/withdraw",
    dependencies=[Depends(require_admin_token)],
)
def admin_withdraw_controlled_snapshot(
    run_id: str,
    snapshot_id: str,
    request: SnapshotPublishCreate,
    session: Db,
) -> dict:
    _, run = _locked_controlled_run(session, run_id)
    snapshot = session.scalar(
        select(LeaderboardSnapshot)
        .where(
            LeaderboardSnapshot.id == snapshot_id,
            LeaderboardSnapshot.controlled_run_id == run.id,
            LeaderboardSnapshot.data_stratum == "controlled",
            LeaderboardSnapshot.publication_status == "published",
        )
        .with_for_update()
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="published controlled snapshot not found")
    snapshot.publication_status = "withdrawn"
    if run.evaluation_order_id is not None:
        order = session.get(EvaluationOrder, run.evaluation_order_id)
        if order is not None and order.publication_status in {"authorized", "published"}:
            order.publication_status = "withdrawn"
    reference_sha256 = hashlib.sha256(request.publication_reference.encode()).hexdigest()
    session.add(
        RunEvent(
            entity_type="leaderboard_snapshot",
            entity_id=snapshot.id,
            event_type="controlled_snapshot_withdrawn",
            payload_json={
                "controlled_run_id": run.id,
                "withdrawal_reference_sha256": reference_sha256,
            },
        )
    )
    session.commit()
    return {
        "snapshotId": snapshot.id,
        "runId": run.id,
        "publicationStatus": snapshot.publication_status,
    }


@router.get("/admin/research-export", dependencies=[Depends(require_admin_token)])
def research_export(session: Db, season: str = Query(default="season-0")) -> dict:
    season_row = session.scalar(select(Season).where(Season.slug == season))
    if season_row is None:
        raise HTTPException(status_code=404, detail="season not found")
    if season_row.slug == "season-1" or season_row.official:
        raise HTTPException(
            status_code=409,
            detail=(
                "official research data must be derived from a signed, snapshot-bound "
                "research release; the live export is an operational preview only"
            ),
        )
    battles = session.scalars(
        select(Battle).where(
            Battle.season_id == season_row.id,
            Battle.research_consent.is_(True),
            Battle.release_review_status == "approved",
        )
    ).all()
    records = []
    for battle in battles:
        arms = session.scalars(select(ResponseArm).where(ResponseArm.battle_id == battle.id)).all()
        votes = session.scalars(select(Vote).where(Vote.battle_id == battle.id)).all()
        records.append(
            {
                "battleId": battle.id,
                "track": battle.track,
                "category": battle.category,
                "prompt": sanitize_for_release(battle.prompt or ""),
                "promptSha256": battle.prompt_sha256,
                "arms": [
                    {
                        "side": arm.side,
                        "modelId": arm.model_id,
                        "condition": arm.condition,
                        "answer": sanitize_for_release(arm.answer_markdown or ""),
                        "generationId": arm.generation_id,
                    }
                    for arm in arms
                ],
                "votes": [privacy_safe_vote_release(vote) for vote in votes],
            }
        )
    payload = {"season": season, "manifestSha256": season_row.manifest_sha256, "records": records}
    payload["exportSha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


app.include_router(router, prefix="/v1")
app.include_router(task_validation_router, prefix="/v1")
app.include_router(
    commercial_router,
    prefix="/v1",
    dependencies=[Depends(require_service_token)],
)


def run() -> None:
    import uvicorn

    uvicorn.run("flavourbench.main:app", host="0.0.0.0", port=8090)


if __name__ == "__main__":
    run()
