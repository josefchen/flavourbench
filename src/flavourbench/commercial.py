from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .commercial_authority import publication_authorization_binding
from .config import (
    get_settings,
    organization_api_key_hmac_keyring,
)
from .database import get_db
from .models import (
    ApiIdempotencyKey,
    CatalogModel,
    ControlledRun,
    EvaluationOrder,
    EvidenceBundle,
    GovernanceAcceptance,
    ModelRouteRevision,
    ModelSubmission,
    Organization,
    OrganizationApiKey,
    RunEvent,
    Season,
    SeasonModel,
    Task,
)
from .schemas import (
    EvaluationOrderCreate,
    EvaluationOrderDecisionCreate,
    GovernanceAcceptanceCreate,
    GovernanceAcceptanceRevocationCreate,
    ModelSubmissionCreate,
    ModelSubmissionDecisionCreate,
    OrganizationApiKeyCreate,
    OrganizationApiKeyRevokeCreate,
    OrganizationCreate,
)
from .security import require_admin_token

router = APIRouter()
Db = Annotated[Session, Depends(get_db)]

_API_KEY = re.compile(r"^fb_live_([A-Za-z0-9_-]{8,24})\.([A-Za-z0-9_-]{32,128})$")
_IDEMPOTENCY = re.compile(r"^[\x21-\x7e]{8,200}$")
_REQUIRED_ORDER_ACCEPTANCES = frozenset(
    {
        "service_terms",
        "acceptable_use",
        "benchmark_integrity_attestation",
    }
)


@dataclass(frozen=True)
class OrganizationIdentity:
    organization: Organization
    api_key: OrganizationApiKey


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _hmac_sha256(secret: str, value: str) -> str:
    return hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _iso(value: datetime | None) -> str | None:
    return _utc(value).astimezone(UTC).isoformat() if value is not None else None


def _organization_identity(
    session: Session,
    authorization: str,
    *,
    scope: str,
) -> OrganizationIdentity:
    supplied = authorization.removeprefix("Bearer ").strip()
    match = _API_KEY.fullmatch(supplied)
    if match is None:
        raise HTTPException(status_code=401, detail="organization credential is invalid")
    key_prefix = match.group(1)
    key = session.scalar(
        select(OrganizationApiKey).where(
            OrganizationApiKey.key_prefix == key_prefix,
            OrganizationApiKey.status == "active",
        )
    )
    if key is None:
        raise HTTPException(status_code=401, detail="organization credential is invalid")
    verification_secret = organization_api_key_hmac_keyring().get(key.hmac_key_id)
    if verification_secret is None or not hmac.compare_digest(
        key.secret_hmac_sha256,
        _hmac_sha256(verification_secret, supplied),
    ):
        raise HTTPException(status_code=401, detail="organization credential is invalid")
    now = datetime.now(UTC)
    if now < _utc(key.not_before) or now >= _utc(key.expires_at):
        raise HTTPException(status_code=401, detail="organization credential is expired")
    scopes = key.scopes_json if isinstance(key.scopes_json, list) else []
    if scope not in scopes:
        raise HTTPException(status_code=403, detail="organization credential lacks scope")
    organization = session.scalar(
        select(Organization).where(
            Organization.id == key.organization_id,
            Organization.status == "active",
        )
    )
    if organization is None:
        raise HTTPException(status_code=401, detail="organization credential is invalid")
    key.last_used_at = now
    return OrganizationIdentity(organization=organization, api_key=key)


def _idempotency_contract(
    session: Session,
    identity: OrganizationIdentity,
    *,
    idempotency_key: str,
    method: str,
    route_template: str,
    request_payload: dict[str, Any],
) -> tuple[str, str, ApiIdempotencyKey | None]:
    if not _IDEMPOTENCY.fullmatch(idempotency_key):
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key must be 8-200 visible ASCII characters",
        )
    key_sha256 = _sha256_text(idempotency_key)
    request_sha256 = _sha256_json(request_payload)
    existing = session.scalar(
        select(ApiIdempotencyKey).where(
            ApiIdempotencyKey.organization_id == identity.organization.id,
            ApiIdempotencyKey.method == method,
            ApiIdempotencyKey.route_template == route_template,
            ApiIdempotencyKey.idempotency_key_sha256 == key_sha256,
        )
    )
    if existing is not None and existing.request_sha256 != request_sha256:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was already used with different content",
        )
    return key_sha256, request_sha256, existing


def _record_idempotency(
    session: Session,
    identity: OrganizationIdentity,
    *,
    method: str,
    route_template: str,
    key_sha256: str,
    request_sha256: str,
    response_status: int,
    resource_type: str,
    resource_id: str,
    response: dict[str, Any],
) -> None:
    now = datetime.now(UTC)
    session.add(
        ApiIdempotencyKey(
            organization_id=identity.organization.id,
            api_key_id=identity.api_key.id,
            method=method,
            route_template=route_template,
            idempotency_key_sha256=key_sha256,
            request_sha256=request_sha256,
            response_status=response_status,
            resource_type=resource_type,
            resource_id=resource_id,
            response_json=response,
            created_at=now,
            expires_at=now + timedelta(days=30),
        )
    )


def _commit_idempotent(
    session: Session,
    identity: OrganizationIdentity,
    *,
    method: str,
    route_template: str,
    key_sha256: str,
    request_sha256: str,
    response_status: int,
    resource_type: str,
    resource_id: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    _record_idempotency(
        session,
        identity,
        method=method,
        route_template=route_template,
        key_sha256=key_sha256,
        request_sha256=request_sha256,
        response_status=response_status,
        resource_type=resource_type,
        resource_id=resource_id,
        response=response,
    )
    try:
        session.commit()
        return response
    except IntegrityError:
        session.rollback()
        existing = session.scalar(
            select(ApiIdempotencyKey).where(
                ApiIdempotencyKey.organization_id == identity.organization.id,
                ApiIdempotencyKey.method == method,
                ApiIdempotencyKey.route_template == route_template,
                ApiIdempotencyKey.idempotency_key_sha256 == key_sha256,
            )
        )
        if existing is None:
            raise
        if existing.request_sha256 != request_sha256:
            raise HTTPException(
                status_code=409,
                detail="Idempotency-Key was already used with different content",
            ) from None
        return dict(existing.response_json)


def _submission_payload(
    submission: ModelSubmission,
    route: ModelRouteRevision,
) -> dict[str, Any]:
    return {
        "submissionId": submission.id,
        "organizationId": submission.organization_id,
        "revision": submission.revision,
        "status": submission.status,
        "displayName": submission.display_name,
        "publisher": submission.publisher,
        "requestedCanonicalModelId": submission.requested_canonical_model_id,
        "exactModelVersion": submission.exact_model_version,
        "releaseDate": submission.release_date,
        "modelCardUri": submission.model_card_uri,
        "modelCardSha256": submission.model_card_sha256,
        "licenseUri": submission.license_uri,
        "licenseDocumentSha256": submission.license_document_sha256,
        "submissionPayloadSha256": submission.submission_payload_sha256,
        "catalogModelId": submission.catalog_model_id,
        "route": {
            "routeRevisionId": route.id,
            "revision": route.revision,
            "routeKind": route.route_kind,
            "executionBackend": route.execution_backend,
            "requestedModelId": route.requested_model_id,
            "expectedActualModelId": route.expected_actual_model_id,
            "expectedActualProviderSlug": route.expected_actual_provider_slug,
            "descriptorSha256": route.descriptor_sha256,
            "status": route.status,
            "approvedSeasonId": route.approved_season_id,
            "approvedSeasonManifestSha256": route.approved_season_manifest_sha256,
            "approvedEndpointContractSha256": route.approved_endpoint_contract_sha256,
            "approvalBasis": (
                "frozen-season-route-contract"
                if route.status == "approved" and route.approved_contract_test_id is None
                else "worker-contract-test"
                if route.approved_contract_test_id is not None
                else None
            ),
        },
        "createdAt": _iso(submission.created_at),
        "submittedAt": _iso(submission.submitted_at),
        "decidedAt": _iso(submission.decided_at),
    }


def _order_payload(session: Session, order: EvaluationOrder) -> dict[str, Any]:
    controlled_run_id = session.scalar(
        select(ControlledRun.id).where(ControlledRun.evaluation_order_id == order.id)
    )
    return {
        "orderId": order.id,
        "organizationId": order.organization_id,
        "modelSubmissionId": order.model_submission_id,
        "routeRevisionId": order.route_revision_id,
        "seasonId": order.season_id,
        "evaluationProfileId": order.evaluation_profile_id,
        "status": order.status,
        "billingStatus": order.billing_status,
        "publicationStatus": order.publication_status,
        "requestedVisibility": order.requested_visibility,
        "forecastCostMicros": order.forecast_cost_micros,
        "budgetCapMicros": order.budget_cap_micros,
        "currency": order.currency,
        "orderCardSha256": order.order_card_sha256,
        "controlledRunId": controlled_run_id,
        "createdAt": _iso(order.created_at),
        "submittedAt": _iso(order.submitted_at),
        "approvedAt": _iso(order.approved_at),
        "completedAt": _iso(order.completed_at),
        "deliveredAt": _iso(order.delivered_at),
    }


def _same_org_subjects(
    session: Session,
    organization_id: str,
    request: GovernanceAcceptanceCreate,
) -> None:
    subjects = (
        (ModelSubmission, request.model_submission_id),
        (ModelRouteRevision, request.route_revision_id),
        (EvaluationOrder, request.evaluation_order_id),
    )
    for model, identifier in subjects:
        if identifier is None:
            continue
        subject = session.get(model, identifier)
        if subject is None:
            raise HTTPException(status_code=404, detail="acceptance subject not found")
        if isinstance(subject, ModelRouteRevision):
            submission = session.get(ModelSubmission, subject.model_submission_id)
            subject_org = submission.organization_id if submission is not None else None
        else:
            subject_org = subject.organization_id
        if subject_org != organization_id:
            raise HTTPException(status_code=404, detail="acceptance subject not found")


def _active_acceptances(
    session: Session,
    organization_id: str,
    *,
    evaluation_order_id: str | None = None,
) -> list[GovernanceAcceptance]:
    now = datetime.now(UTC)
    rows = session.scalars(
        select(GovernanceAcceptance).where(
            GovernanceAcceptance.organization_id == organization_id,
            GovernanceAcceptance.status == "active",
        )
    ).all()
    superseded_ids = {
        row.supersedes_acceptance_id for row in rows if row.supersedes_acceptance_id is not None
    }
    return [
        row
        for row in rows
        if row.id not in superseded_ids
        and _utc(row.accepted_at) <= now
        and (row.expires_at is None or _utc(row.expires_at) > now)
        and row.model_submission_id is None
        and row.route_revision_id is None
        and (row.evaluation_order_id is None or row.evaluation_order_id == evaluation_order_id)
        and (
            row.agreement_type not in {"spend_authorization", "publication_authorization"}
            or row.evaluation_order_id == evaluation_order_id
        )
    ]


def _active_acceptance_types(
    session: Session,
    organization_id: str,
    *,
    evaluation_order_id: str | None = None,
) -> set[str]:
    return {
        row.agreement_type
        for row in _active_acceptances(
            session,
            organization_id,
            evaluation_order_id=evaluation_order_id,
        )
    }


def _spend_authorization_for_order(
    session: Session,
    order: EvaluationOrder,
    quote_reference_sha256: str,
) -> GovernanceAcceptance | None:
    expected_binding = {
        "orderCardSha256": order.order_card_sha256,
        "budgetCapMicros": order.budget_cap_micros,
        "currency": order.currency,
        "forecastCostMicros": order.forecast_cost_micros,
        "routeRevisionId": order.route_revision_id,
        "seasonId": order.season_id,
        "quoteReferenceSha256": quote_reference_sha256,
    }
    return next(
        (
            row
            for row in _active_acceptances(
                session,
                order.organization_id,
                evaluation_order_id=order.id,
            )
            if row.agreement_type == "spend_authorization"
            and row.evaluation_order_id == order.id
            and row.binding_json == expected_binding
            and row.binding_sha256 == _sha256_json(expected_binding)
        ),
        None,
    )


@router.post(
    "/admin/organizations",
    status_code=201,
    dependencies=[Depends(require_admin_token)],
)
def admin_create_organization(request: OrganizationCreate, session: Db) -> dict[str, Any]:
    now = datetime.now(UTC)
    retention_payload = dict(request.retention_policy)
    organization = Organization(
        slug=request.slug,
        legal_name=request.legal_name,
        display_name=request.display_name,
        status="active" if request.activate else "pending_verification",
        idp_tenant_reference_sha256=_sha256_text(request.idp_tenant_reference),
        billing_reference_sha256=(
            _sha256_text(request.billing_reference)
            if request.billing_reference is not None
            else None
        ),
        data_region=request.data_region,
        retention_policy_json=retention_payload,
        retention_policy_sha256=_sha256_json(retention_payload),
        created_at=now,
        verified_at=now if request.activate else None,
    )
    session.add(organization)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="organization already exists") from exc
    session.add(
        RunEvent(
            entity_type="organization",
            entity_id=organization.id,
            event_type="organization_created",
            payload_json={
                "status": organization.status,
                "data_region": organization.data_region,
                "retention_policy_sha256": organization.retention_policy_sha256,
            },
        )
    )
    session.commit()
    return {
        "organizationId": organization.id,
        "slug": organization.slug,
        "displayName": organization.display_name,
        "status": organization.status,
        "dataRegion": organization.data_region,
        "retentionPolicySha256": organization.retention_policy_sha256,
    }


@router.post(
    "/admin/organizations/{organization_id}/api-keys",
    status_code=201,
    dependencies=[Depends(require_admin_token)],
)
def admin_create_organization_api_key(
    organization_id: str,
    request: OrganizationApiKeyCreate,
    session: Db,
) -> dict[str, Any]:
    organization = session.scalar(
        select(Organization).where(Organization.id == organization_id).with_for_update()
    )
    if organization is None:
        raise HTTPException(status_code=404, detail="organization not found")
    if organization.status != "active":
        raise HTTPException(status_code=409, detail="organization is not active")
    now = datetime.now(UTC)
    expires_at = _utc(request.expires_at)
    if expires_at <= now + timedelta(minutes=5) or expires_at > now + timedelta(days=730):
        raise HTTPException(
            status_code=422,
            detail="API key expiry must be between five minutes and two years",
        )
    prefix = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32)
    plaintext = f"fb_live_{prefix}.{secret}"
    settings = get_settings()
    key = OrganizationApiKey(
        organization_id=organization.id,
        key_prefix=prefix,
        secret_hmac_sha256=_hmac_sha256(
            settings.organization_api_key_hmac_secret,
            plaintext,
        ),
        hmac_key_id=settings.organization_api_key_hmac_key_id,
        label=request.label,
        scopes_json=sorted(request.scopes),
        status="active",
        rate_limit_profile=request.rate_limit_profile,
        network_policy_json=dict(request.network_policy),
        created_by_principal_ref_sha256=_sha256_text(request.created_by_principal_reference),
        not_before=now,
        expires_at=expires_at,
        created_at=now,
    )
    session.add(key)
    session.flush()
    session.add(
        RunEvent(
            entity_type="organization_api_key",
            entity_id=key.id,
            event_type="organization_api_key_issued",
            payload_json={
                "organization_id": organization.id,
                "key_prefix": key.key_prefix,
                "hmac_key_id": key.hmac_key_id,
                "scopes": key.scopes_json,
                "expires_at": _iso(key.expires_at),
            },
        )
    )
    session.commit()
    return {
        "keyId": key.id,
        "organizationId": organization.id,
        "apiKey": plaintext,
        "keyPrefix": key.key_prefix,
        "scopes": key.scopes_json,
        "expiresAt": _iso(key.expires_at),
        "notice": "The API key is returned once and is never stored in plaintext.",
    }


@router.post(
    "/admin/organizations/{organization_id}/api-keys/{key_id}/revoke",
    dependencies=[Depends(require_admin_token)],
)
def admin_revoke_organization_api_key(
    organization_id: str,
    key_id: str,
    request: OrganizationApiKeyRevokeCreate,
    session: Db,
) -> dict[str, Any]:
    key = session.scalar(
        select(OrganizationApiKey)
        .where(
            OrganizationApiKey.id == key_id,
            OrganizationApiKey.organization_id == organization_id,
        )
        .with_for_update()
    )
    if key is None:
        raise HTTPException(status_code=404, detail="organization API key not found")
    if key.status == "revoked":
        return {"keyId": key.id, "status": key.status, "revokedAt": _iso(key.revoked_at)}
    key.status = "revoked"
    key.revoked_at = datetime.now(UTC)
    session.add(
        RunEvent(
            entity_type="organization_api_key",
            entity_id=key.id,
            event_type="organization_api_key_revoked",
            payload_json={
                "organization_id": organization_id,
                "revocation_reference_sha256": request.revocation_reference_sha256,
            },
        )
    )
    session.commit()
    return {"keyId": key.id, "status": key.status, "revokedAt": _iso(key.revoked_at)}


@router.post(
    "/admin/organizations/{organization_id}/governance-acceptances",
    status_code=201,
    dependencies=[Depends(require_admin_token)],
)
def admin_record_governance_acceptance(
    organization_id: str,
    request: GovernanceAcceptanceCreate,
    session: Db,
) -> dict[str, Any]:
    organization = session.get(Organization, organization_id)
    if organization is None:
        raise HTTPException(status_code=404, detail="organization not found")
    _same_org_subjects(session, organization_id, request)
    now = datetime.now(UTC)
    if _utc(request.accepted_at) > now + timedelta(minutes=5):
        raise HTTPException(status_code=422, detail="acceptedAt cannot be in the future")
    binding = dict(request.binding)
    if request.agreement_type == "spend_authorization":
        if request.evaluation_order_id is None:
            raise HTTPException(
                status_code=422,
                detail="spend authorization must name one evaluation order",
            )
        order = session.get(EvaluationOrder, request.evaluation_order_id)
        if order is None or order.organization_id != organization_id:
            raise HTTPException(status_code=404, detail="acceptance subject not found")
        expected_without_quote = {
            "orderCardSha256": order.order_card_sha256,
            "budgetCapMicros": order.budget_cap_micros,
            "currency": order.currency,
            "forecastCostMicros": order.forecast_cost_micros,
            "routeRevisionId": order.route_revision_id,
            "seasonId": order.season_id,
        }
        quote_reference = binding.get("quoteReferenceSha256")
        if (
            {key: binding.get(key) for key in expected_without_quote} != expected_without_quote
            or set(binding) != {*expected_without_quote, "quoteReferenceSha256"}
            or not isinstance(quote_reference, str)
            or re.fullmatch(r"[0-9a-f]{64}", quote_reference) is None
        ):
            raise HTTPException(
                status_code=422,
                detail="spend authorization binding does not match the evaluation order",
            )
    elif request.agreement_type == "publication_authorization":
        if request.evaluation_order_id is None:
            raise HTTPException(
                status_code=422,
                detail="publication authorization must name one evaluation order",
            )
        order = session.get(EvaluationOrder, request.evaluation_order_id)
        run = session.scalar(
            select(ControlledRun).where(
                ControlledRun.evaluation_order_id == request.evaluation_order_id
            )
        )
        if (
            order is None
            or run is None
            or order.organization_id != organization_id
            or run.organization_id != organization_id
        ):
            raise HTTPException(status_code=404, detail="acceptance subject not found")
        expected_binding = publication_authorization_binding(order, run)
        if binding != expected_binding:
            raise HTTPException(
                status_code=422,
                detail="publication authorization binding does not match the controlled run",
            )
    elif binding:
        raise HTTPException(
            status_code=422,
            detail="this governance acceptance type does not accept a binding payload",
        )
    acceptance = GovernanceAcceptance(
        organization_id=organization.id,
        model_submission_id=request.model_submission_id,
        route_revision_id=request.route_revision_id,
        evaluation_order_id=request.evaluation_order_id,
        agreement_type=request.agreement_type,
        agreement_version=request.agreement_version,
        document_sha256=request.document_sha256,
        external_envelope_reference_sha256=_sha256_text(request.external_envelope_reference),
        signatory_principal_reference_sha256=_sha256_text(request.signatory_principal_reference),
        authority_basis=request.authority_basis,
        binding_json=binding,
        binding_sha256=_sha256_json(binding),
        status="active",
        accepted_at=_utc(request.accepted_at),
        expires_at=_utc(request.expires_at) if request.expires_at else None,
        created_at=now,
    )
    session.add(acceptance)
    session.flush()
    session.add(
        RunEvent(
            entity_type="governance_acceptance",
            entity_id=acceptance.id,
            event_type="governance_acceptance_recorded",
            payload_json={
                "organization_id": organization.id,
                "agreement_type": acceptance.agreement_type,
                "agreement_version": acceptance.agreement_version,
                "document_sha256": acceptance.document_sha256,
                "binding_sha256": acceptance.binding_sha256,
                "subject": {
                    "model_submission_id": acceptance.model_submission_id,
                    "route_revision_id": acceptance.route_revision_id,
                    "evaluation_order_id": acceptance.evaluation_order_id,
                },
            },
        )
    )
    session.commit()
    return {
        "acceptanceId": acceptance.id,
        "organizationId": organization.id,
        "agreementType": acceptance.agreement_type,
        "agreementVersion": acceptance.agreement_version,
        "status": acceptance.status,
        "acceptedAt": _iso(acceptance.accepted_at),
        "expiresAt": _iso(acceptance.expires_at),
    }


@router.post(
    "/admin/governance-acceptances/{acceptance_id}/revoke",
    status_code=201,
    dependencies=[Depends(require_admin_token)],
)
def admin_revoke_governance_acceptance(
    acceptance_id: str,
    request: GovernanceAcceptanceRevocationCreate,
    session: Db,
) -> dict[str, Any]:
    predecessor = session.scalar(
        select(GovernanceAcceptance)
        .where(GovernanceAcceptance.id == acceptance_id)
        .with_for_update()
    )
    if predecessor is None:
        raise HTTPException(status_code=404, detail="governance acceptance not found")
    existing_successor = session.scalar(
        select(GovernanceAcceptance.id).where(
            GovernanceAcceptance.supersedes_acceptance_id == predecessor.id
        )
    )
    if existing_successor is not None:
        raise HTTPException(status_code=409, detail="governance acceptance is already superseded")
    now = datetime.now(UTC)
    accepted_at = _utc(request.accepted_at)
    if accepted_at < _utc(predecessor.accepted_at) or accepted_at > now + timedelta(minutes=5):
        raise HTTPException(status_code=422, detail="revocation chronology is invalid")
    binding = {
        "action": "revoke",
        "predecessorAgreementType": predecessor.agreement_type,
        "predecessorBindingSha256": predecessor.binding_sha256,
        "reasonCode": request.reason_code,
        "supersededAcceptanceId": predecessor.id,
    }
    revocation = GovernanceAcceptance(
        organization_id=predecessor.organization_id,
        model_submission_id=predecessor.model_submission_id,
        route_revision_id=predecessor.route_revision_id,
        evaluation_order_id=predecessor.evaluation_order_id,
        agreement_type="acceptance_revocation",
        agreement_version="flavourbench-governance-revocation-v1",
        document_sha256=request.document_sha256,
        external_envelope_reference_sha256=_sha256_text(request.external_envelope_reference),
        signatory_principal_reference_sha256=_sha256_text(request.signatory_principal_reference),
        authority_basis=request.authority_basis,
        binding_json=binding,
        binding_sha256=_sha256_json(binding),
        status="active",
        accepted_at=accepted_at,
        supersedes_acceptance_id=predecessor.id,
        created_at=now,
    )
    session.add(revocation)
    session.flush()
    session.add(
        RunEvent(
            entity_type="governance_acceptance",
            entity_id=revocation.id,
            event_type="governance_acceptance_revoked_by_supersession",
            payload_json={
                "organization_id": predecessor.organization_id,
                "superseded_acceptance_id": predecessor.id,
                "predecessor_agreement_type": predecessor.agreement_type,
                "reason_code": request.reason_code,
            },
        )
    )
    session.commit()
    return {
        "acceptanceId": predecessor.id,
        "status": "superseded",
        "revocationAcceptanceId": revocation.id,
        "revocationBindingSha256": revocation.binding_sha256,
    }


@router.get("/org")
def get_organization(
    session: Db,
    authorization: str = Header(default=""),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="models:read")
    session.commit()
    organization = identity.organization
    return {
        "organizationId": organization.id,
        "slug": organization.slug,
        "displayName": organization.display_name,
        "status": organization.status,
        "dataRegion": organization.data_region,
        "retentionPolicySha256": organization.retention_policy_sha256,
        "credential": {
            "keyId": identity.api_key.id,
            "keyPrefix": identity.api_key.key_prefix,
            "scopes": identity.api_key.scopes_json,
            "expiresAt": _iso(identity.api_key.expires_at),
        },
    }


@router.post("/org/model-submissions", status_code=201)
def create_model_submission(
    request: ModelSubmissionCreate,
    session: Db,
    authorization: str = Header(default=""),
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="models:submit")
    request_payload = request.model_dump(mode="json", by_alias=True)
    key_sha256, request_sha256, existing = _idempotency_contract(
        session,
        identity,
        idempotency_key=idempotency_key,
        method="POST",
        route_template="/v1/org/model-submissions",
        request_payload=request_payload,
    )
    if existing is not None:
        session.commit()
        return dict(existing.response_json)

    revision = 1
    predecessor: ModelSubmission | None = None
    if request.supersedes_submission_id is not None:
        predecessor = session.scalar(
            select(ModelSubmission)
            .where(
                ModelSubmission.id == request.supersedes_submission_id,
                ModelSubmission.organization_id == identity.organization.id,
            )
            .with_for_update()
        )
        if predecessor is None:
            raise HTTPException(status_code=404, detail="superseded submission not found")
        if predecessor.requested_canonical_model_id != request.requested_canonical_model_id:
            raise HTTPException(
                status_code=409,
                detail="a superseding revision must keep the canonical model identity",
            )
        revision = predecessor.revision + 1

    now = datetime.now(UTC)
    route_payload = request.route.model_dump(mode="json", by_alias=False)
    managed_route_reference = route_payload.pop("managed_route_reference")
    managed_route_reference_sha256 = _sha256_text(managed_route_reference)
    execution_backend = (
        "bedrock" if request.route.route_kind.value == "managed_bedrock" else "openrouter"
    )
    descriptor = {
        **route_payload,
        "managed_route_reference_sha256": managed_route_reference_sha256,
        "execution_backend": execution_backend,
        "organization_id": identity.organization.id,
        "submission_revision": revision,
    }
    submission_record = {
        **request_payload,
        "organizationId": identity.organization.id,
        "revision": revision,
        "route": descriptor,
    }
    submission = ModelSubmission(
        organization_id=identity.organization.id,
        revision=revision,
        supersedes_submission_id=(predecessor.id if predecessor is not None else None),
        status="draft",
        display_name=request.display_name,
        publisher=request.publisher,
        requested_canonical_model_id=request.requested_canonical_model_id,
        exact_model_version=request.exact_model_version,
        release_date=request.release_date,
        model_card_uri=request.model_card_uri,
        model_card_sha256=request.model_card_sha256,
        license_uri=request.license_uri,
        license_document_sha256=request.license_document_sha256,
        capability_claims_json=dict(request.capability_claims),
        capability_claims_sha256=_sha256_json(request.capability_claims),
        contamination_disclosure_json=dict(request.contamination_disclosure),
        contamination_disclosure_sha256=_sha256_json(request.contamination_disclosure),
        submission_payload_json=submission_record,
        submission_payload_sha256=_sha256_json(submission_record),
        submitted_by_key_id=identity.api_key.id,
        created_at=now,
    )
    session.add(submission)
    session.flush()
    route = ModelRouteRevision(
        model_submission_id=submission.id,
        revision=1,
        route_kind=request.route.route_kind.value,
        execution_backend=execution_backend,
        managed_route_reference_sha256=managed_route_reference_sha256,
        requested_model_id=request.route.requested_model_id,
        expected_actual_model_id=request.route.expected_actual_model_id,
        expected_actual_provider_slug=request.route.expected_actual_provider_slug,
        supported_parameters_json=sorted(request.route.supported_parameters),
        supported_parameters_sha256=_sha256_json(sorted(request.route.supported_parameters)),
        decoding_bounds_json=dict(request.route.decoding_bounds),
        decoding_bounds_sha256=_sha256_json(request.route.decoding_bounds),
        endpoint_document_sha256=request.route.endpoint_document_sha256,
        data_policy_json=dict(request.route.data_policy),
        data_policy_sha256=_sha256_json(request.route.data_policy),
        rate_card_json=dict(request.route.rate_card),
        rate_card_sha256=_sha256_json(request.route.rate_card),
        descriptor_json=descriptor,
        descriptor_sha256=_sha256_json(descriptor),
        status="draft",
        created_at=now,
    )
    session.add(route)
    session.flush()
    response = _submission_payload(submission, route)
    session.add(
        RunEvent(
            entity_type="model_submission",
            entity_id=submission.id,
            event_type="model_submission_draft_created",
            payload_json={
                "organization_id": identity.organization.id,
                "submission_payload_sha256": submission.submission_payload_sha256,
                "route_revision_id": route.id,
                "route_descriptor_sha256": route.descriptor_sha256,
            },
        )
    )
    return _commit_idempotent(
        session,
        identity,
        method="POST",
        route_template="/v1/org/model-submissions",
        key_sha256=key_sha256,
        request_sha256=request_sha256,
        response_status=201,
        resource_type="model_submission",
        resource_id=submission.id,
        response=response,
    )


@router.get("/org/model-submissions")
def list_model_submissions(
    session: Db,
    authorization: str = Header(default=""),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=10_000),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="models:read")
    submissions = session.scalars(
        select(ModelSubmission)
        .where(ModelSubmission.organization_id == identity.organization.id)
        .order_by(ModelSubmission.created_at.desc(), ModelSubmission.id)
        .offset(offset)
        .limit(limit)
    ).all()
    records = []
    for submission in submissions:
        route = session.scalar(
            select(ModelRouteRevision).where(
                ModelRouteRevision.model_submission_id == submission.id
            )
        )
        if route is not None:
            records.append(_submission_payload(submission, route))
    session.commit()
    return {"items": records, "limit": limit, "offset": offset}


@router.get("/org/model-submissions/{submission_id}")
def get_model_submission(
    submission_id: str,
    session: Db,
    authorization: str = Header(default=""),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="models:read")
    submission = session.scalar(
        select(ModelSubmission).where(
            ModelSubmission.id == submission_id,
            ModelSubmission.organization_id == identity.organization.id,
        )
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="model submission not found")
    route = session.scalar(
        select(ModelRouteRevision).where(ModelRouteRevision.model_submission_id == submission.id)
    )
    if route is None:
        raise HTTPException(status_code=503, detail="model route revision is unavailable")
    session.commit()
    return _submission_payload(submission, route)


@router.post("/org/model-submissions/{submission_id}/submit")
def submit_model_submission(
    submission_id: str,
    session: Db,
    authorization: str = Header(default=""),
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="models:submit")
    request_payload = {"submissionId": submission_id}
    route_template = "/v1/org/model-submissions/{submission_id}/submit"
    key_sha256, request_sha256, existing = _idempotency_contract(
        session,
        identity,
        idempotency_key=idempotency_key,
        method="POST",
        route_template=route_template,
        request_payload=request_payload,
    )
    if existing is not None:
        session.commit()
        return dict(existing.response_json)
    submission = session.scalar(
        select(ModelSubmission)
        .where(
            ModelSubmission.id == submission_id,
            ModelSubmission.organization_id == identity.organization.id,
        )
        .with_for_update()
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="model submission not found")
    if submission.status != "draft":
        detail = (
            "changes require a superseding immutable model-submission revision"
            if submission.status == "changes_requested"
            else "model submission is not submit-ready"
        )
        raise HTTPException(status_code=409, detail=detail)
    route = session.scalar(
        select(ModelRouteRevision)
        .where(ModelRouteRevision.model_submission_id == submission.id)
        .with_for_update()
    )
    if route is None or route.status != "draft":
        raise HTTPException(status_code=409, detail="model route is not submit-ready")
    now = datetime.now(UTC)
    submission.status = "submitted"
    submission.submitted_at = now
    route.status = "submitted"
    route.submitted_at = now
    response = _submission_payload(submission, route)
    session.add(
        RunEvent(
            entity_type="model_submission",
            entity_id=submission.id,
            event_type="model_submission_submitted",
            payload_json={
                "organization_id": identity.organization.id,
                "submission_payload_sha256": submission.submission_payload_sha256,
                "route_descriptor_sha256": route.descriptor_sha256,
            },
        )
    )
    return _commit_idempotent(
        session,
        identity,
        method="POST",
        route_template=route_template,
        key_sha256=key_sha256,
        request_sha256=request_sha256,
        response_status=200,
        resource_type="model_submission",
        resource_id=submission.id,
        response=response,
    )


@router.post(
    "/admin/model-submissions/{submission_id}/decision",
    dependencies=[Depends(require_admin_token)],
)
def admin_decide_model_submission(
    submission_id: str,
    request: ModelSubmissionDecisionCreate,
    session: Db,
) -> dict[str, Any]:
    submission = session.scalar(
        select(ModelSubmission).where(ModelSubmission.id == submission_id).with_for_update()
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="model submission not found")
    if submission.status != "submitted":
        raise HTTPException(status_code=409, detail="model submission is not awaiting decision")
    route = session.scalar(
        select(ModelRouteRevision)
        .where(ModelRouteRevision.model_submission_id == submission.id)
        .with_for_update()
    )
    if route is None or route.status != "submitted":
        raise HTTPException(status_code=409, detail="model route is not awaiting decision")
    now = datetime.now(UTC)
    if request.decision == "approve":
        season = session.scalar(
            select(Season).where(
                Season.slug == request.season,
                Season.status == "active",
                Season.official.is_(True),
                Season.frozen_at.is_not(None),
            )
        )
        if season is None:
            raise HTTPException(status_code=409, detail="frozen active season is unavailable")
        season_model = session.scalar(
            select(SeasonModel).where(
                SeasonModel.season_id == season.id,
                SeasonModel.model_id == submission.requested_canonical_model_id,
                SeasonModel.eligible.is_(True),
            )
        )
        catalog_model = session.get(CatalogModel, submission.requested_canonical_model_id)
        if season_model is None or catalog_model is None:
            raise HTTPException(
                status_code=409,
                detail="managed route is not admitted to the requested frozen season",
            )
        requested_id_matches = route.requested_model_id in {
            season_model.model_id,
            catalog_model.canonical_slug,
        }
        decoding_bounds_match = route.decoding_bounds_json.get(
            "maxTokens"
        ) == season_model.endpoint_max_completion_tokens == season_model.decoding_json.get(
            "max_tokens"
        ) and route.decoding_bounds_json.get(
            "temperatureMaximum"
        ) == season_model.decoding_json.get("temperature")
        data_policy_match = (
            route.data_policy_json.get("training") == "deny"
            and route.data_policy_json.get("retention") == "deny"
            and season_model.backend_contract_json.get("data_collection") == "deny"
        )
        if not (
            requested_id_matches
            and route.execution_backend == season_model.execution_backend
            and route.expected_actual_model_id == season_model.expected_actual_model_id
            and route.expected_actual_provider_slug == season_model.expected_actual_provider_slug
            and route.endpoint_document_sha256 == season_model.endpoint_document_sha256
            and sorted(route.supported_parameters_json)
            == sorted(season_model.supported_parameters_json)
            and decoding_bounds_match
            and data_policy_match
            and route.rate_card_json == season_model.rate_card_json
        ):
            raise HTTPException(
                status_code=409,
                detail="managed route does not match the frozen endpoint contract",
            )
        submission.status = "approved"
        submission.catalog_model_id = catalog_model.model_id
        route.status = "approved"
        route.approved_season_id = season.id
        route.approved_season_manifest_sha256 = season.manifest_sha256
        route.approved_endpoint_contract_sha256 = season_model.endpoint_contract_sha256
        route.approved_at = now
    elif request.decision == "changes_requested":
        submission.status = "changes_requested"
        route.status = "changes_requested"
    else:
        submission.status = "rejected"
        route.status = "rejected"
    submission.decision_reference_sha256 = request.decision_reference_sha256
    submission.decided_at = now
    session.add(
        RunEvent(
            entity_type="model_submission",
            entity_id=submission.id,
            event_type=f"model_submission_{submission.status}",
            payload_json={
                "organization_id": submission.organization_id,
                "decision_reference_sha256": request.decision_reference_sha256,
                "route_revision_id": route.id,
                "route_descriptor_sha256": route.descriptor_sha256,
                "approval_basis": (
                    "frozen-season-route-contract" if submission.status == "approved" else None
                ),
            },
        )
    )
    session.commit()
    return _submission_payload(submission, route)


@router.post("/org/evaluation-orders", status_code=201)
def create_evaluation_order(
    request: EvaluationOrderCreate,
    session: Db,
    authorization: str = Header(default=""),
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="orders:create")
    request_payload = request.model_dump(mode="json", by_alias=True)
    key_sha256, request_sha256, existing = _idempotency_contract(
        session,
        identity,
        idempotency_key=idempotency_key,
        method="POST",
        route_template="/v1/org/evaluation-orders",
        request_payload=request_payload,
    )
    if existing is not None:
        session.commit()
        return dict(existing.response_json)
    submission = session.scalar(
        select(ModelSubmission).where(
            ModelSubmission.id == request.model_submission_id,
            ModelSubmission.organization_id == identity.organization.id,
            ModelSubmission.status == "approved",
        )
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="approved model submission not found")
    route = session.scalar(
        select(ModelRouteRevision).where(
            ModelRouteRevision.id == request.route_revision_id,
            ModelRouteRevision.model_submission_id == submission.id,
            ModelRouteRevision.status == "approved",
        )
    )
    if route is None:
        raise HTTPException(status_code=404, detail="approved model route not found")
    season = session.scalar(
        select(Season).where(
            Season.slug == request.season,
            Season.status == "active",
            Season.official.is_(True),
            Season.frozen_at.is_not(None),
        )
    )
    if season is None:
        raise HTTPException(status_code=409, detail="frozen active season is unavailable")
    if (
        route.approved_season_id != season.id
        or route.approved_season_manifest_sha256 != season.manifest_sha256
    ):
        raise HTTPException(
            status_code=409,
            detail="approved model route is not bound to the requested frozen season",
        )
    season_models = session.scalars(
        select(SeasonModel).where(
            SeasonModel.season_id == season.id,
            SeasonModel.eligible.is_(True),
        )
    ).all()
    if not any(model.model_id == submission.catalog_model_id for model in season_models):
        raise HTTPException(
            status_code=409,
            detail="approved model is not admitted to the requested season",
        )
    approved_slot = next(
        (model for model in season_models if model.model_id == submission.catalog_model_id),
        None,
    )
    if (
        approved_slot is None
        or route.approved_endpoint_contract_sha256 != approved_slot.endpoint_contract_sha256
        or route.execution_backend != approved_slot.execution_backend
        or route.expected_actual_model_id != approved_slot.expected_actual_model_id
        or route.expected_actual_provider_slug != approved_slot.expected_actual_provider_slug
        or route.endpoint_document_sha256 != approved_slot.endpoint_document_sha256
    ):
        raise HTTPException(
            status_code=409,
            detail="approved model route no longer matches the frozen endpoint contract",
        )
    candidate_tasks = session.scalars(
        select(Task).where(
            Task.season_id == season.id,
            Task.review_status == "frozen",
        )
    ).all()
    task_rows = [
        task
        for task in candidate_tasks
        if task.split == "scored"
        and isinstance(task.provenance_json, dict)
        and task.provenance_json.get("confirmatory_eligible") is True
    ]
    if not task_rows:
        raise HTTPException(
            status_code=409,
            detail="season has no frozen confirmatory task release",
        )
    ordered_tasks = sorted(task_rows, key=lambda item: (item.public_id, item.revision))
    task_release = {
        "schema_version": "flavourbench-private-task-release-v1",
        "season_id": season.id,
        "season_manifest_sha256": season.manifest_sha256,
        "tasks": [
            {
                "public_id": task.public_id,
                "revision": task.revision,
                "prompt_sha256": task.prompt_sha256,
                "family": task.family,
            }
            for task in ordered_tasks
        ],
    }
    task_release_sha256 = _sha256_json(task_release)
    candidate_model_id = submission.catalog_model_id
    if candidate_model_id is None:
        raise HTTPException(status_code=409, detail="approved model identity is unavailable")
    ordered_models = sorted(season_models, key=lambda item: item.model_id)
    comparator_model_ids = [
        model.model_id for model in ordered_models if model.model_id != candidate_model_id
    ]
    if not comparator_model_ids:
        raise HTTPException(
            status_code=409,
            detail="private comparative evaluation requires at least one frozen comparator",
        )
    schedule = []
    for task in ordered_tasks:
        for comparator_model_id in comparator_model_ids:
            schedule.append(
                {
                    "taskPublicId": task.public_id,
                    "track": "model_arena",
                    "modelIds": sorted([candidate_model_id, comparator_model_id]),
                    "repetitionIndex": 1,
                }
            )
        schedule.append(
            {
                "taskPublicId": task.public_id,
                "track": "epicure_uplift",
                "modelIds": [candidate_model_id],
                "repetitionIndex": 1,
            }
        )
    slot_by_id = {model.model_id: model for model in ordered_models}
    forecast_cost_micros = 0
    for entry in schedule:
        entry_cost = sum(
            max(0, slot_by_id[model_id].worst_case_cost_micros) for model_id in entry["modelIds"]
        )
        if entry["track"] == "epicure_uplift":
            entry_cost *= 2
        forecast_cost_micros += entry_cost
    if forecast_cost_micros <= 0:
        raise HTTPException(status_code=409, detail="season cost forecast is unavailable")
    if forecast_cost_micros * 10_000 >= request.budget_cap_micros * 8_500:
        raise HTTPException(
            status_code=422,
            detail="budgetCapMicros cannot admit the forecast below the 85% stop",
        )
    comparison_plan = {
        "schema_version": "flavourbench-private-comparative-profile-v1",
        "task_release_sha256": task_release_sha256,
        "tasks": task_release["tasks"],
        "task_count": len(ordered_tasks),
        "tracks": ["model_arena", "epicure_uplift"],
        "submitted_endpoint_model_id": candidate_model_id,
        "model_ids": sorted([candidate_model_id, *comparator_model_ids]),
        "comparator_model_ids": comparator_model_ids,
        "comparator_policy": "all_other_eligible_frozen_manifest_models",
        "task_schedule": schedule,
        "task_schedule_sha256": _sha256_json(schedule),
        "customer_selectable_tasks": False,
        "customer_selectable_comparators": False,
        "repetitions": 1,
    }
    now = datetime.now(UTC)
    order_id = str(uuid.uuid4())
    client_reference_sha256 = _sha256_text(request.client_reference)
    settings = get_settings()
    order_card = {
        "schema_version": "flavourbench-evaluation-order-card-v2",
        "signing": {
            "algorithm": "HMAC-SHA256",
            "key_id": settings.run_card_signing_key_id,
            "verification_scope": "FlavourBench service-held key",
        },
        "evaluation_order_id": order_id,
        "organization_id": identity.organization.id,
        "model_submission_id": submission.id,
        "route_revision_id": route.id,
        "season_id": season.id,
        "submitted_endpoint_model_id": candidate_model_id,
        "submission_payload_sha256": submission.submission_payload_sha256,
        "route_descriptor_sha256": route.descriptor_sha256,
        "season_manifest_sha256": season.manifest_sha256,
        "prompt_registry_sha256": season.prompt_registry_sha256,
        "tool_registry_sha256": season.tool_registry_sha256,
        "epicure_release_id": season.epicure_release_id,
        "epicure_bundle_sha256": season.epicure_bundle_sha256,
        "analysis_plan_sha256": season.analysis_plan_sha256,
        "evaluation_profile_id": request.evaluation_profile_id,
        "comparison_plan_sha256": _sha256_json(comparison_plan),
        "forecast_cost_micros": forecast_cost_micros,
        "budget_cap_micros": request.budget_cap_micros,
        "currency": "USD",
        "requested_visibility": request.requested_visibility,
        "client_reference_sha256": client_reference_sha256,
    }
    order_card_sha256 = _sha256_json(order_card)
    signature = _hmac_sha256(
        settings.run_card_signing_secret,
        f"flavourbench-evaluation-order-card-v2:{order_card_sha256}",
    )
    order = EvaluationOrder(
        id=order_id,
        organization_id=identity.organization.id,
        model_submission_id=submission.id,
        route_revision_id=route.id,
        season_id=season.id,
        evaluation_profile_id=request.evaluation_profile_id,
        status="draft",
        billing_status="unquoted",
        publication_status="private",
        requested_visibility=request.requested_visibility,
        comparison_plan_json=comparison_plan,
        comparison_plan_sha256=order_card["comparison_plan_sha256"],
        rater_plan_sha256=_sha256_json(
            {"profile": "qualified-independent-plus-automated-panel-v1"}
        ),
        analysis_plan_sha256=season.analysis_plan_sha256,
        forecast_cost_micros=forecast_cost_micros,
        budget_cap_micros=request.budget_cap_micros,
        currency="USD",
        client_reference_sha256=client_reference_sha256,
        order_card_json=order_card,
        order_card_sha256=order_card_sha256,
        order_card_signature=signature,
        submitted_by_key_id=identity.api_key.id,
        created_at=now,
    )
    session.add(order)
    session.flush()
    response = _order_payload(session, order)
    session.add(
        RunEvent(
            entity_type="evaluation_order",
            entity_id=order.id,
            event_type="evaluation_order_draft_created",
            payload_json={
                "organization_id": identity.organization.id,
                "order_card_sha256": order.order_card_sha256,
                "comparison_plan_sha256": order.comparison_plan_sha256,
            },
        )
    )
    return _commit_idempotent(
        session,
        identity,
        method="POST",
        route_template="/v1/org/evaluation-orders",
        key_sha256=key_sha256,
        request_sha256=request_sha256,
        response_status=201,
        resource_type="evaluation_order",
        resource_id=order.id,
        response=response,
    )


@router.get("/org/evaluation-orders")
def list_evaluation_orders(
    session: Db,
    authorization: str = Header(default=""),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=10_000),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="orders:read")
    orders = session.scalars(
        select(EvaluationOrder)
        .where(EvaluationOrder.organization_id == identity.organization.id)
        .order_by(EvaluationOrder.created_at.desc(), EvaluationOrder.id)
        .offset(offset)
        .limit(limit)
    ).all()
    session.commit()
    return {
        "items": [_order_payload(session, order) for order in orders],
        "limit": limit,
        "offset": offset,
    }


@router.get("/org/evaluation-orders/{order_id}")
def get_evaluation_order(
    order_id: str,
    session: Db,
    authorization: str = Header(default=""),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="orders:read")
    order = session.scalar(
        select(EvaluationOrder).where(
            EvaluationOrder.id == order_id,
            EvaluationOrder.organization_id == identity.organization.id,
        )
    )
    if order is None:
        raise HTTPException(status_code=404, detail="evaluation order not found")
    session.commit()
    return _order_payload(session, order)


@router.post("/org/evaluation-orders/{order_id}/submit")
def submit_evaluation_order(
    order_id: str,
    session: Db,
    authorization: str = Header(default=""),
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="orders:create")
    route_template = "/v1/org/evaluation-orders/{order_id}/submit"
    request_payload = {"orderId": order_id}
    key_sha256, request_sha256, existing = _idempotency_contract(
        session,
        identity,
        idempotency_key=idempotency_key,
        method="POST",
        route_template=route_template,
        request_payload=request_payload,
    )
    if existing is not None:
        session.commit()
        return dict(existing.response_json)
    order = session.scalar(
        select(EvaluationOrder)
        .where(
            EvaluationOrder.id == order_id,
            EvaluationOrder.organization_id == identity.organization.id,
        )
        .with_for_update()
    )
    if order is None:
        raise HTTPException(status_code=404, detail="evaluation order not found")
    if order.status != "draft":
        raise HTTPException(status_code=409, detail="evaluation order is not submit-ready")
    observed_acceptances = _active_acceptance_types(
        session, identity.organization.id, evaluation_order_id=order.id
    )
    missing = _REQUIRED_ORDER_ACCEPTANCES - observed_acceptances
    if missing:
        raise HTTPException(
            status_code=409,
            detail="required governance acceptances are missing: " + ", ".join(sorted(missing)),
        )
    order.status = "submitted"
    order.submitted_at = datetime.now(UTC)
    response = _order_payload(session, order)
    session.add(
        RunEvent(
            entity_type="evaluation_order",
            entity_id=order.id,
            event_type="evaluation_order_submitted",
            payload_json={
                "organization_id": identity.organization.id,
                "order_card_sha256": order.order_card_sha256,
                "acceptance_types": sorted(observed_acceptances),
            },
        )
    )
    return _commit_idempotent(
        session,
        identity,
        method="POST",
        route_template=route_template,
        key_sha256=key_sha256,
        request_sha256=request_sha256,
        response_status=200,
        resource_type="evaluation_order",
        resource_id=order.id,
        response=response,
    )


@router.post("/org/evaluation-orders/{order_id}/cancel")
def cancel_evaluation_order(
    order_id: str,
    session: Db,
    authorization: str = Header(default=""),
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="orders:cancel")
    route_template = "/v1/org/evaluation-orders/{order_id}/cancel"
    request_payload = {"orderId": order_id}
    key_sha256, request_sha256, existing = _idempotency_contract(
        session,
        identity,
        idempotency_key=idempotency_key,
        method="POST",
        route_template=route_template,
        request_payload=request_payload,
    )
    if existing is not None:
        session.commit()
        return dict(existing.response_json)
    order = session.scalar(
        select(EvaluationOrder)
        .where(
            EvaluationOrder.id == order_id,
            EvaluationOrder.organization_id == identity.organization.id,
        )
        .with_for_update()
    )
    if order is None:
        raise HTTPException(status_code=404, detail="evaluation order not found")
    if order.status not in {"draft", "submitted", "approved"}:
        raise HTTPException(
            status_code=409, detail="evaluation order can no longer be cancelled directly"
        )
    order.status = "cancelled"
    order.billing_status = "void"
    response = _order_payload(session, order)
    session.add(
        RunEvent(
            entity_type="evaluation_order",
            entity_id=order.id,
            event_type="evaluation_order_cancelled",
            payload_json={"organization_id": identity.organization.id},
        )
    )
    return _commit_idempotent(
        session,
        identity,
        method="POST",
        route_template=route_template,
        key_sha256=key_sha256,
        request_sha256=request_sha256,
        response_status=200,
        resource_type="evaluation_order",
        resource_id=order.id,
        response=response,
    )


@router.post(
    "/admin/evaluation-orders/{order_id}/decision",
    dependencies=[Depends(require_admin_token)],
)
def admin_decide_evaluation_order(
    order_id: str,
    request: EvaluationOrderDecisionCreate,
    session: Db,
) -> dict[str, Any]:
    order = session.scalar(
        select(EvaluationOrder).where(EvaluationOrder.id == order_id).with_for_update()
    )
    if order is None:
        raise HTTPException(status_code=404, detail="evaluation order not found")
    if order.status != "submitted":
        raise HTTPException(status_code=409, detail="evaluation order is not awaiting decision")
    now = datetime.now(UTC)
    if request.decision == "approve":
        if request.quote_reference_sha256 is None:
            raise HTTPException(status_code=422, detail="approval requires a quote reference")
        if (
            request.forecast_cost_micros is not None
            and request.forecast_cost_micros != order.forecast_cost_micros
        ):
            raise HTTPException(
                status_code=409,
                detail="a changed forecast requires a superseding evaluation order",
            )
        acceptances = _active_acceptance_types(
            session, order.organization_id, evaluation_order_id=order.id
        )
        missing = set(_REQUIRED_ORDER_ACCEPTANCES - acceptances)
        spend_authorization = _spend_authorization_for_order(
            session,
            order,
            request.quote_reference_sha256,
        )
        if spend_authorization is None:
            missing.add("spend_authorization")
        if missing:
            raise HTTPException(
                status_code=409,
                detail="order approval is missing governance acceptances: "
                + ", ".join(sorted(missing)),
            )
        order.status = "approved"
        order.billing_status = "authorized"
        order.quote_reference_sha256 = request.quote_reference_sha256
        order.approved_at = now
    else:
        order.status = "rejected"
        order.billing_status = "void"
    session.add(
        RunEvent(
            entity_type="evaluation_order",
            entity_id=order.id,
            event_type=f"evaluation_order_{order.status}",
            payload_json={
                "organization_id": order.organization_id,
                "decision_reference_sha256": request.decision_reference_sha256,
                "quote_reference_sha256": order.quote_reference_sha256,
                "spend_authorization_id": (
                    spend_authorization.id
                    if request.decision == "approve" and spend_authorization is not None
                    else None
                ),
            },
        )
    )
    session.commit()
    return {
        **_order_payload(session, order),
        "nextAction": ("operator_provision_controlled_run" if order.status == "approved" else None),
    }


@router.get("/org/evaluation-orders/{order_id}/evidence-bundles")
def list_evidence_bundles(
    order_id: str,
    session: Db,
    authorization: str = Header(default=""),
) -> dict[str, Any]:
    identity = _organization_identity(session, authorization, scope="bundles:read")
    order = session.scalar(
        select(EvaluationOrder).where(
            EvaluationOrder.id == order_id,
            EvaluationOrder.organization_id == identity.organization.id,
        )
    )
    if order is None:
        raise HTTPException(status_code=404, detail="evaluation order not found")
    bundles = session.scalars(
        select(EvidenceBundle)
        .where(
            EvidenceBundle.evaluation_order_id == order.id,
            EvidenceBundle.organization_id == identity.organization.id,
        )
        .order_by(EvidenceBundle.created_at.desc(), EvidenceBundle.id)
    ).all()
    session.commit()
    return {
        "items": [
            {
                "bundleId": bundle.id,
                "bundleClass": bundle.bundle_class,
                "status": bundle.status,
                "schemaVersion": bundle.schema_version,
                "manifestSha256": bundle.manifest_sha256,
                "archiveSha256": bundle.archive_sha256,
                "signatureAlgorithm": bundle.signature_algorithm,
                "signingKeyId": bundle.signing_key_id,
                "sealedAt": _iso(bundle.sealed_at),
                "availableUntil": _iso(bundle.available_until),
            }
            for bundle in bundles
        ]
    }
