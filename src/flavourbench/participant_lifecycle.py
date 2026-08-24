"""Participant-owned consent, withdrawal, and privacy-retention operations.

No operation in this module stores a raw identity-provider subject or contact
channel.  Forward participation authority is bound to one exact active consent
document and activation manifest.  Withdrawal and deletion are terminal rights
operations: they revalidate the manifest at the original acceptance instant but
never grant enrollment, assignment, voting, or other forward authority.
"""

from __future__ import annotations

import base64
import calendar
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .consent_documents import resolve_expert_consent_document
from .human_study_activation import resolve_human_study_activation
from .models import (
    ControlledRunReviewer,
    ExpertReviewer,
    ReviewerAccessCredential,
    ReviewerConsentAcceptance,
    ReviewerDeletionReceipt,
    ReviewerEnrollmentOffer,
    ReviewerIdentityBinding,
    ReviewerParticipationLifecycle,
    ReviewerRetentionSchedule,
    ReviewerWithdrawalReceipt,
    Season,
    new_id,
)
from .reviewer_identity import (
    AFFILIATION_COHORT,
    REVIEWER_ROLES,
    TASK_FAMILIES,
    ReviewerIdentityError,
    bind_reviewer_identity,
    issue_reviewer_credential,
)

OFFER_DOMAIN = b"flavourbench-participant-enrollment-offer-v1\x00"
ACCEPTANCE_REQUEST_DOMAIN = b"flavourbench-participant-consent-request-v1\x00"
RECEIPT_SECRET_DOMAIN = b"flavourbench-participant-consent-receipt-secret-v1\x00"
RECEIPT_PREFIX_DOMAIN = b"flavourbench-participant-consent-receipt-prefix-v1\x00"
RECEIPT_DIGEST_DOMAIN = b"flavourbench-participant-consent-receipt-digest-v1\x00"
IDENTITY_AUDIT_DOMAIN = b"flavourbench-participant-audit-marker-v1\x00"
WITHDRAWAL_REQUEST_DOMAIN = b"flavourbench-participant-withdrawal-request-v1\x00"
DELETION_REQUEST_DOMAIN = b"flavourbench-participant-deletion-request-v1\x00"
DISABLED_INVITATION_DOMAIN = b"flavourbench-disabled-legacy-invitation-v1\x00"

CONSENT_CONFIRMATIONS = (
    "participation_is_voluntary",
    "exact_consent_document_read",
    "withdrawal_and_post_release_limits_understood",
    "retention_and_deletion_schedule_understood",
)
CONSENT_ACCEPTANCE_STATEMENT = {
    "schema_version": "flavourbench-participant-consent-acceptance-v1",
    "actor": "participant",
    "administrator_acceptance_prohibited": True,
    "confirmations": list(CONSENT_CONFIRMATIONS),
}
REDACTED_PROFILE_FIELDS = (
    "expert_reviewers.profile_json",
    "expert_reviewers.qualification_json",
    "expert_reviewers.qualification_verified",
)


class ParticipantLifecycleError(RuntimeError):
    """A participant lifecycle operation is unauthorized or inconsistent."""


@dataclass(frozen=True)
class ActiveHumanStudyBinding:
    consent_document_sha256: str
    activation_manifest_sha256: str
    retention_policy_sha256: str
    consent_text: str


@dataclass(frozen=True)
class ConsentAcceptanceResult:
    acceptance: ReviewerConsentAcceptance
    receipt_credential: str
    idempotent: bool


@dataclass(frozen=True)
class ParticipantIdentityResult:
    reviewer: ExpertReviewer
    binding: ReviewerIdentityBinding
    lifecycle: ReviewerParticipationLifecycle
    reviewer_credential: str
    reviewer_credential_expires_at: datetime


def require_active_participant_authority(
    session: Session,
    *,
    reviewer_id: str,
    season_id: str,
    identity_binding_id: str | None = None,
    at: datetime | None = None,
    settings: Settings | None = None,
) -> ReviewerParticipationLifecycle | None:
    """Revalidate current forward authority for a participant-owned identity.

    ``None`` denotes a pre-0035 legacy reviewer so historical development
    fixtures keep their 0034 behavior.  A 0035 lifecycle, once present, is
    always fail-closed against its exact consent and current activation hashes.
    Production database guards independently require the lifecycle for new
    server-verified identities, credentials, assignments, and judgments.
    """

    query = select(ReviewerParticipationLifecycle).where(
        ReviewerParticipationLifecycle.reviewer_id == reviewer_id,
        ReviewerParticipationLifecycle.season_id == season_id,
    )
    if identity_binding_id is not None:
        query = query.where(
            ReviewerParticipationLifecycle.identity_binding_id == identity_binding_id
        )
    lifecycle = session.scalar(query)
    if lifecycle is None:
        return None
    if lifecycle.status != "active":
        raise ParticipantLifecycleError("reviewer participation is not active")
    acceptance = session.get(
        ReviewerConsentAcceptance,
        lifecycle.consent_acceptance_id,
    )
    if acceptance is None:
        raise ParticipantLifecycleError("reviewer consent acceptance is unavailable")
    active = require_active_human_study(
        consent_document_sha256=acceptance.consent_document_sha256,
        at=at,
        settings=settings,
    )
    if not (
        hmac.compare_digest(
            active.activation_manifest_sha256,
            acceptance.activation_manifest_sha256,
        )
        and hmac.compare_digest(
            active.retention_policy_sha256,
            acceptance.retention_policy_sha256,
        )
    ):
        raise ParticipantLifecycleError("reviewer forward authority binding is stale")
    return lifecycle


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def participant_record_analysis_eligible(
    session: Session,
    *,
    reviewer_id: str,
    season_id: str,
    recorded_at: datetime,
    identity_binding_id: str | None = None,
) -> bool | None:
    """Return the participant-specific eligibility of one immutable research record.

    ``None`` denotes a pre-0035 reviewer with no participant lifecycle, leaving the
    established legacy admission rules controlling.  Participant records remain in
    the append-only database after withdrawal.  A withdrawal received before the
    reviewer-specific analysis freeze excludes those records from analysis; one at
    or after the freeze preserves their historical eligibility.  Missing or
    inconsistent schedule evidence fails closed.
    """

    query = select(ReviewerParticipationLifecycle).where(
        ReviewerParticipationLifecycle.reviewer_id == reviewer_id,
        ReviewerParticipationLifecycle.season_id == season_id,
    )
    if identity_binding_id is not None:
        query = query.where(
            ReviewerParticipationLifecycle.identity_binding_id == identity_binding_id
        )
    lifecycle = session.scalar(query)
    if lifecycle is None:
        return None
    record_time = _utc(recorded_at)
    if record_time < _utc(lifecycle.created_at):
        return False
    if lifecycle.status == "active":
        return True
    if lifecycle.status not in {"withdrawn", "redacted"} or lifecycle.withdrawn_at is None:
        return False
    withdrawn_at = _utc(lifecycle.withdrawn_at)
    if record_time > withdrawn_at:
        return False
    schedule = session.scalar(
        select(ReviewerRetentionSchedule).where(
            ReviewerRetentionSchedule.lifecycle_id == lifecycle.id
        )
    )
    if schedule is None:
        return False
    return withdrawn_at >= _utc(schedule.analysis_freeze_at)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _domain_hmac(settings: Settings, domain: bytes, material: bytes) -> bytes:
    return hmac.new(
        settings.reviewer_credential_hmac_secret.encode(),
        domain + material,
        hashlib.sha256,
    ).digest()


def _domain_hmac_hex(settings: Settings, domain: bytes, material: bytes) -> str:
    return _domain_hmac(settings, domain, material).hex()


def _token(prefix_marker: str, prefix: str, secret: str) -> str:
    return f"{prefix_marker}_{prefix}.{secret}"


def _parse_token(token: str, prefix_marker: str) -> tuple[str, str]:
    marker = f"{prefix_marker}_"
    if not token.startswith(marker) or token.count(".") != 1:
        raise ParticipantLifecycleError("participant credential format is invalid")
    prefix, secret = token.removeprefix(marker).split(".", 1)
    if (
        not prefix
        or not secret
        or len(prefix) > 32
        or len(secret) > 128
        or any(character not in "0123456789abcdef" for character in prefix)
    ):
        raise ParticipantLifecycleError("participant credential format is invalid")
    return prefix, secret


def _strict_manifest(path: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ParticipantLifecycleError("activation manifest contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(Path(path).read_bytes(), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParticipantLifecycleError("activation manifest cannot be read safely") from exc
    if not isinstance(value, dict):
        raise ParticipantLifecycleError("activation manifest is not an object")
    return value


def require_active_human_study(
    *,
    consent_document_sha256: str,
    at: datetime | None = None,
    settings: Settings | None = None,
) -> ActiveHumanStudyBinding:
    """Require the exact consent and current activation manifest for forward work."""

    configured = settings or get_settings()
    current = _utc(at or datetime.now(UTC))
    consent = resolve_expert_consent_document(
        consent_document_sha256,
        settings=configured,
    )
    if consent.status != "active" or consent.text is None:
        raise ParticipantLifecycleError("human-study consent document is not active")
    activation = resolve_human_study_activation(
        configured.human_study_activation_manifest_path,
        configured.human_study_activation_manifest_sha256,
        consent_sha256=consent_document_sha256,
        at=current,
    )
    if not activation.ready or activation.manifest_sha256 != (
        configured.human_study_activation_manifest_sha256
    ):
        raise ParticipantLifecycleError("human-study activation manifest is not active")
    manifest = _strict_manifest(configured.human_study_activation_manifest_path)
    retention = manifest.get("retention_operation")
    retention_policy_sha256 = (
        retention.get("retention_schedule_sha256") if isinstance(retention, dict) else None
    )
    if not isinstance(retention_policy_sha256, str) or len(retention_policy_sha256) != 64:
        raise ParticipantLifecycleError("activation retention policy is unavailable")
    return ActiveHumanStudyBinding(
        consent_document_sha256=consent_document_sha256,
        activation_manifest_sha256=activation.manifest_sha256,
        retention_policy_sha256=retention_policy_sha256,
        consent_text=consent.text,
    )


def require_original_accepted_manifest(
    acceptance: ReviewerConsentAcceptance,
    *,
    settings: Settings | None = None,
) -> None:
    """Validate terminal rights against the manifest as of acceptance.

    This deliberately does not require the study to remain currently active.
    Governance suspension must stop forward work, not withdrawal or eligible
    deletion rights.
    """

    configured = settings or get_settings()
    if not hmac.compare_digest(
        acceptance.activation_manifest_sha256,
        configured.human_study_activation_manifest_sha256,
    ):
        raise ParticipantLifecycleError("accepted activation manifest is not configured")
    activation = resolve_human_study_activation(
        configured.human_study_activation_manifest_path,
        acceptance.activation_manifest_sha256,
        consent_sha256=acceptance.consent_document_sha256,
        at=_utc(acceptance.accepted_at),
    )
    if not activation.ready or not hmac.compare_digest(
        activation.manifest_sha256 or "",
        acceptance.activation_manifest_sha256,
    ):
        raise ParticipantLifecycleError("original accepted activation manifest is invalid")


def issue_enrollment_offer(
    session: Session,
    *,
    season: Season,
    consent_document_sha256: str,
    ttl_seconds: int = 3600,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> tuple[str, ReviewerEnrollmentOffer]:
    """Issue a generic one-time offer without creating a participant identity."""

    configured = settings or get_settings()
    current = _utc(now or datetime.now(UTC))
    if ttl_seconds < 300 or ttl_seconds > 86_400:
        raise ParticipantLifecycleError("enrollment offer lifetime is invalid")
    active = require_active_human_study(
        consent_document_sha256=consent_document_sha256,
        at=current,
        settings=configured,
    )
    prefix = secrets.token_hex(8)
    token = _token("fben1", prefix, secrets.token_urlsafe(32))
    offer = ReviewerEnrollmentOffer(
        season_id=season.id,
        credential_prefix=prefix,
        secret_hmac_sha256=_domain_hmac_hex(configured, OFFER_DOMAIN, token.encode()),
        hmac_key_id=configured.reviewer_credential_hmac_key_id,
        consent_document_sha256=active.consent_document_sha256,
        activation_manifest_sha256=active.activation_manifest_sha256,
        status="active",
        not_before=current,
        expires_at=current + timedelta(seconds=ttl_seconds),
    )
    session.add(offer)
    session.flush()
    return token, offer


def _offer_for_token(
    session: Session,
    token: str,
    *,
    settings: Settings,
    lock: bool,
) -> ReviewerEnrollmentOffer:
    prefix, _ = _parse_token(token, "fben1")
    query = select(ReviewerEnrollmentOffer).where(
        ReviewerEnrollmentOffer.credential_prefix == prefix
    )
    offer = session.scalar(query.with_for_update() if lock else query)
    observed = _domain_hmac_hex(settings, OFFER_DOMAIN, token.encode())
    if (
        offer is None
        or offer.hmac_key_id != settings.reviewer_credential_hmac_key_id
        or not hmac.compare_digest(offer.secret_hmac_sha256, observed)
    ):
        raise ParticipantLifecycleError("participant enrollment credential is invalid")
    return offer


def enrollment_consent_view(
    session: Session,
    *,
    enrollment_token: str,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    configured = settings or get_settings()
    current = _utc(now or datetime.now(UTC))
    offer = _offer_for_token(session, enrollment_token, settings=configured, lock=False)
    if (
        offer.status != "active"
        or current < _utc(offer.not_before)
        or current > _utc(offer.expires_at)
    ):
        raise ParticipantLifecycleError("participant enrollment credential is expired or used")
    active = require_active_human_study(
        consent_document_sha256=offer.consent_document_sha256,
        at=current,
        settings=configured,
    )
    if not hmac.compare_digest(active.activation_manifest_sha256, offer.activation_manifest_sha256):
        raise ParticipantLifecycleError("participant enrollment activation binding changed")
    return {
        "consentDocumentSha256": active.consent_document_sha256,
        "activationManifestSha256": active.activation_manifest_sha256,
        "consentText": active.consent_text,
        "confirmations": list(CONSENT_CONFIRMATIONS),
        "acceptanceStatementSha256": _sha256(CONSENT_ACCEPTANCE_STATEMENT),
    }


def _receipt_credential(enrollment_token: str, settings: Settings) -> str:
    secret_bytes = _domain_hmac(settings, RECEIPT_SECRET_DOMAIN, enrollment_token.encode())
    prefix = _domain_hmac_hex(settings, RECEIPT_PREFIX_DOMAIN, enrollment_token.encode())[:16]
    secret = base64.urlsafe_b64encode(secret_bytes).decode().rstrip("=")
    return _token("fbcr1", prefix, secret)


def accept_participant_consent(
    session: Session,
    *,
    enrollment_token: str,
    consent_document_sha256: str,
    activation_manifest_sha256: str,
    confirmations: list[str],
    idempotency_key: str,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> ConsentAcceptanceResult:
    """Atomically consume one offer and append the participant's consent receipt."""

    configured = settings or get_settings()
    current = _utc(now or datetime.now(UTC))
    if tuple(confirmations) != CONSENT_CONFIRMATIONS:
        raise ParticipantLifecycleError("participant consent confirmations are incomplete")
    if not idempotency_key or len(idempotency_key) > 160:
        raise ParticipantLifecycleError("participant consent idempotency key is invalid")
    offer = _offer_for_token(session, enrollment_token, settings=configured, lock=True)
    request_sha256 = _domain_hmac_hex(
        configured,
        ACCEPTANCE_REQUEST_DOMAIN,
        _canonical_bytes(
            {
                "offer_id": offer.id,
                "consent_document_sha256": consent_document_sha256,
                "activation_manifest_sha256": activation_manifest_sha256,
                "confirmations": confirmations,
                "idempotency_key": idempotency_key,
            }
        ),
    )
    receipt_credential = _receipt_credential(enrollment_token, configured)
    if offer.status == "accepted":
        acceptance = session.scalar(
            select(ReviewerConsentAcceptance).where(
                ReviewerConsentAcceptance.enrollment_offer_id == offer.id
            )
        )
        if (
            acceptance is None
            or not hmac.compare_digest(acceptance.request_sha256, request_sha256)
            or not hmac.compare_digest(
                acceptance.receipt_secret_hmac_sha256,
                _domain_hmac_hex(configured, RECEIPT_DIGEST_DOMAIN, receipt_credential.encode()),
            )
        ):
            raise ParticipantLifecycleError("enrollment offer was consumed by another request")
        return ConsentAcceptanceResult(acceptance, receipt_credential, True)
    if offer.status != "active":
        raise ParticipantLifecycleError("participant enrollment credential is revoked")
    if current < _utc(offer.not_before) or current > _utc(offer.expires_at):
        raise ParticipantLifecycleError("participant enrollment credential is expired")
    active = require_active_human_study(
        consent_document_sha256=consent_document_sha256,
        at=current,
        settings=configured,
    )
    if not (
        hmac.compare_digest(offer.consent_document_sha256, consent_document_sha256)
        and hmac.compare_digest(offer.activation_manifest_sha256, activation_manifest_sha256)
        and hmac.compare_digest(active.activation_manifest_sha256, activation_manifest_sha256)
    ):
        raise ParticipantLifecycleError("participant consent hashes do not match the offer")
    consumed = session.execute(
        update(ReviewerEnrollmentOffer)
        .where(
            ReviewerEnrollmentOffer.id == offer.id,
            ReviewerEnrollmentOffer.status == "active",
            ReviewerEnrollmentOffer.accepted_at.is_(None),
        )
        .values(
            status="accepted",
            accepted_at=current,
            accepted_request_sha256=request_sha256,
        )
        .execution_options(synchronize_session=False)
    )
    if consumed.rowcount != 1:
        raise ParticipantLifecycleError(
            "participant enrollment credential was consumed concurrently"
        )
    acceptance_id = new_id()
    confirmation_sha256 = _sha256(list(CONSENT_CONFIRMATIONS))
    receipt_prefix, _ = _parse_token(receipt_credential, "fbcr1")
    receipt_payload = {
        "schema_version": "flavourbench-reviewer-consent-receipt-v1",
        "id": acceptance_id,
        "enrollment_offer_id": offer.id,
        "season_id": offer.season_id,
        "consent_document_sha256": consent_document_sha256,
        "activation_manifest_sha256": activation_manifest_sha256,
        "retention_policy_sha256": active.retention_policy_sha256,
        "acceptance_statement_sha256": _sha256(CONSENT_ACCEPTANCE_STATEMENT),
        "confirmation_set_sha256": confirmation_sha256,
        "request_sha256": request_sha256,
        "accepted_at": current.isoformat(),
    }
    acceptance = ReviewerConsentAcceptance(
        id=acceptance_id,
        enrollment_offer_id=offer.id,
        season_id=offer.season_id,
        consent_document_sha256=consent_document_sha256,
        activation_manifest_sha256=activation_manifest_sha256,
        retention_policy_sha256=active.retention_policy_sha256,
        acceptance_statement_sha256=_sha256(CONSENT_ACCEPTANCE_STATEMENT),
        confirmation_set_sha256=confirmation_sha256,
        request_sha256=request_sha256,
        receipt_prefix=receipt_prefix,
        receipt_secret_hmac_sha256=_domain_hmac_hex(
            configured, RECEIPT_DIGEST_DOMAIN, receipt_credential.encode()
        ),
        hmac_key_id=configured.reviewer_credential_hmac_key_id,
        receipt_sha256=_sha256(receipt_payload),
        accepted_at=current,
    )
    session.add(acceptance)
    session.flush()
    return ConsentAcceptanceResult(acceptance, receipt_credential, False)


def authenticate_consent_receipt(
    session: Session,
    *,
    receipt_credential: str,
    settings: Settings | None = None,
    lock: bool = False,
) -> ReviewerConsentAcceptance:
    configured = settings or get_settings()
    prefix, _ = _parse_token(receipt_credential, "fbcr1")
    query = select(ReviewerConsentAcceptance).where(
        ReviewerConsentAcceptance.receipt_prefix == prefix
    )
    acceptance = session.scalar(query.with_for_update() if lock else query)
    observed = _domain_hmac_hex(configured, RECEIPT_DIGEST_DOMAIN, receipt_credential.encode())
    if (
        acceptance is None
        or acceptance.hmac_key_id != configured.reviewer_credential_hmac_key_id
        or not hmac.compare_digest(acceptance.receipt_secret_hmac_sha256, observed)
    ):
        raise ParticipantLifecycleError("participant consent receipt credential is invalid")
    return acceptance


def enroll_participant_identity(
    session: Session,
    *,
    receipt_credential: str,
    identity_issuer: str,
    issuer_subject: str,
    identity_evidence_sha256: str,
    roles: list[str],
    qualified_families: list[str],
    affiliation_class: str,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> ParticipantIdentityResult:
    """Create identity only after participant-owned consent has committed."""

    configured = settings or get_settings()
    current = _utc(now or datetime.now(UTC))
    acceptance = authenticate_consent_receipt(
        session,
        receipt_credential=receipt_credential,
        settings=configured,
        lock=True,
    )
    active = require_active_human_study(
        consent_document_sha256=acceptance.consent_document_sha256,
        at=current,
        settings=configured,
    )
    if not (
        hmac.compare_digest(
            acceptance.activation_manifest_sha256,
            active.activation_manifest_sha256,
        )
        and hmac.compare_digest(
            acceptance.retention_policy_sha256,
            active.retention_policy_sha256,
        )
    ):
        raise ParticipantLifecycleError("participant consent activation binding is stale")
    if (
        session.scalar(
            select(ReviewerParticipationLifecycle.id).where(
                ReviewerParticipationLifecycle.consent_acceptance_id == acceptance.id
            )
        )
        is not None
    ):
        raise ParticipantLifecycleError("participant consent receipt is already identity-bound")
    normalized_roles = sorted(set(roles))
    normalized_families = sorted(set(qualified_families))
    if (
        not normalized_roles
        or set(normalized_roles) - REVIEWER_ROLES
        or not normalized_families
        or set(normalized_families) - TASK_FAMILIES
        or affiliation_class not in AFFILIATION_COHORT
    ):
        raise ParticipantLifecycleError("participant reviewer scope is invalid")
    reviewer_code = f"participant-{secrets.token_hex(12)}"
    disabled_invitation = _domain_hmac_hex(
        configured,
        DISABLED_INVITATION_DOMAIN,
        acceptance.receipt_sha256.encode(),
    )
    reviewer = ExpertReviewer(
        reviewer_code=reviewer_code,
        invitation_sha256=disabled_invitation,
        qualification_json=normalized_families,
        qualification_verified=False,
        cohort=AFFILIATION_COHORT[affiliation_class],
        profile_json={
            "schema_version": "flavourbench-participant-profile-v1",
            "consent_acceptance_sha256": acceptance.receipt_sha256,
            "activation_manifest_sha256": acceptance.activation_manifest_sha256,
            "affiliation_class": affiliation_class,
            "raw_identity_retention_prohibited": True,
            "contact_data_persisted": False,
        },
        batch_reveal_only=True,
        active=True,
        privacy_status="retained",
    )
    session.add(reviewer)
    session.flush()
    season = session.get(Season, acceptance.season_id)
    if season is None:
        raise ParticipantLifecycleError("participant consent season is unavailable")
    try:
        binding = bind_reviewer_identity(
            session,
            season=season,
            reviewer=reviewer,
            identity_issuer=identity_issuer,
            issuer_subject=issuer_subject,
            identity_evidence_sha256=identity_evidence_sha256,
            roles=normalized_roles,
            settings=configured,
        )
    except ReviewerIdentityError as exc:
        raise ParticipantLifecycleError("participant identity binding is invalid") from exc
    audit_marker_sha256 = _domain_hmac_hex(
        configured,
        IDENTITY_AUDIT_DOMAIN,
        _canonical_bytes(
            {
                "consent_acceptance_sha256": acceptance.receipt_sha256,
                "season_id": acceptance.season_id,
                "reviewer_id": reviewer.id,
                "identity_binding_id": binding.id,
            }
        ),
    )
    lifecycle = ReviewerParticipationLifecycle(
        consent_acceptance_id=acceptance.id,
        season_id=acceptance.season_id,
        reviewer_id=reviewer.id,
        identity_binding_id=binding.id,
        audit_marker_sha256=audit_marker_sha256,
        status="active",
    )
    session.add(lifecycle)
    session.flush()
    reviewer_token, credential = issue_reviewer_credential(
        session,
        binding=binding,
        credential_kind="review_session",
        scopes=["expert_review"],
        now=current,
        settings=configured,
    )
    return ParticipantIdentityResult(
        reviewer=reviewer,
        binding=binding,
        lifecycle=lifecycle,
        reviewer_credential=reviewer_token,
        reviewer_credential_expires_at=credential.expires_at,
    )


def withdraw_participant(
    session: Session,
    *,
    receipt_credential: str,
    idempotency_key: str,
    reason_code: str,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> ReviewerWithdrawalReceipt:
    """Atomically revoke forward authority while preserving judgment history."""

    configured = settings or get_settings()
    current = _utc(now or datetime.now(UTC))
    if reason_code not in {"voluntary_withdrawal", "privacy_request", "safety_concern"}:
        raise ParticipantLifecycleError("participant withdrawal reason is invalid")
    if not idempotency_key or len(idempotency_key) > 160:
        raise ParticipantLifecycleError("participant withdrawal idempotency key is invalid")
    acceptance = authenticate_consent_receipt(
        session,
        receipt_credential=receipt_credential,
        settings=configured,
        lock=True,
    )
    require_original_accepted_manifest(acceptance, settings=configured)
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle)
        .where(ReviewerParticipationLifecycle.consent_acceptance_id == acceptance.id)
        .with_for_update()
    )
    if lifecycle is None:
        raise ParticipantLifecycleError("participant identity enrollment is unavailable")
    if lifecycle.status in {"withdrawn", "redacted"}:
        existing = session.scalar(
            select(ReviewerWithdrawalReceipt).where(
                ReviewerWithdrawalReceipt.lifecycle_id == lifecycle.id
            )
        )
        if existing is None:
            raise ParticipantLifecycleError("terminal participant state lacks its receipt")
        return existing
    if lifecycle.status != "active":
        raise ParticipantLifecycleError("participant lifecycle state is invalid")
    reviewer_changed = session.execute(
        update(ExpertReviewer)
        .where(ExpertReviewer.id == lifecycle.reviewer_id, ExpertReviewer.active.is_(True))
        .values(active=False, revoked_at=current)
        .execution_options(synchronize_session=False)
    )
    if reviewer_changed.rowcount != 1:
        raise ParticipantLifecycleError("participant withdrawal raced with another transition")
    credential_result = session.execute(
        update(ReviewerAccessCredential)
        .where(
            ReviewerAccessCredential.reviewer_id == lifecycle.reviewer_id,
            ReviewerAccessCredential.status == "active",
        )
        .values(status="revoked", revoked_at=current)
        .execution_options(synchronize_session=False)
    )
    assignment_result = session.execute(
        update(ControlledRunReviewer)
        .where(
            ControlledRunReviewer.reviewer_id == lifecycle.reviewer_id,
            ControlledRunReviewer.active.is_(True),
        )
        .values(active=False)
        .execution_options(synchronize_session=False)
    )
    request_sha256 = _domain_hmac_hex(
        configured,
        WITHDRAWAL_REQUEST_DOMAIN,
        _canonical_bytes(
            {
                "lifecycle_id": lifecycle.id,
                "idempotency_key": idempotency_key,
                "reason_code": reason_code,
            }
        ),
    )
    receipt_id = new_id()
    payload = {
        "schema_version": "flavourbench-reviewer-withdrawal-receipt-v1",
        "id": receipt_id,
        "lifecycle_id": lifecycle.id,
        "consent_acceptance_id": acceptance.id,
        "season_id": lifecycle.season_id,
        "reviewer_id": lifecycle.reviewer_id,
        "identity_binding_id": lifecycle.identity_binding_id,
        "request_sha256": request_sha256,
        "reason_code": reason_code,
        "credentials_revoked_count": credential_result.rowcount,
        "assignments_stopped_count": assignment_result.rowcount,
        "prior_judgments_preserved": True,
        "effective_at": current.isoformat(),
    }
    receipt = ReviewerWithdrawalReceipt(
        id=receipt_id,
        lifecycle_id=lifecycle.id,
        consent_acceptance_id=acceptance.id,
        season_id=lifecycle.season_id,
        reviewer_id=lifecycle.reviewer_id,
        identity_binding_id=lifecycle.identity_binding_id,
        request_sha256=request_sha256,
        reason_code=reason_code,
        credentials_revoked_count=credential_result.rowcount,
        assignments_stopped_count=assignment_result.rowcount,
        prior_judgments_preserved=True,
        receipt_sha256=_sha256(payload),
        effective_at=current,
    )
    session.add(receipt)
    session.flush()
    lifecycle.status = "withdrawn"
    lifecycle.withdrawn_at = current
    lifecycle.assignments_stopped_at = current
    lifecycle.withdrawal_receipt_sha256 = receipt.receipt_sha256
    session.flush()
    return receipt


def _add_calendar_months(value: datetime, months: int) -> datetime:
    current = _utc(value)
    month_index = current.month - 1 + months
    year = current.year + month_index // 12
    month = month_index % 12 + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return current.replace(year=year, month=month, day=day)


def create_retention_schedule(
    session: Session,
    *,
    reviewer_id: str,
    analysis_freeze_at: datetime,
    first_public_release_at: datetime,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> ReviewerRetentionSchedule:
    configured = settings or get_settings()
    current = _utc(now or datetime.now(UTC))
    freeze_at = _utc(analysis_freeze_at)
    release_at = _utc(first_public_release_at)
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle)
        .where(ReviewerParticipationLifecycle.reviewer_id == reviewer_id)
        .with_for_update()
    )
    if lifecycle is None or lifecycle.status == "redacted":
        raise ParticipantLifecycleError("reviewer retention lifecycle is unavailable")
    acceptance = session.get(ReviewerConsentAcceptance, lifecycle.consent_acceptance_id)
    if acceptance is None:
        raise ParticipantLifecycleError("reviewer consent acceptance is unavailable")
    require_original_accepted_manifest(acceptance, settings=configured)
    existing = session.scalar(
        select(ReviewerRetentionSchedule).where(
            ReviewerRetentionSchedule.lifecycle_id == lifecycle.id
        )
    )
    direct_due = _add_calendar_months(freeze_at, 12)
    audit_until = _add_calendar_months(release_at, 60)
    if existing is not None:
        if not (
            _utc(existing.analysis_freeze_at) == freeze_at
            and _utc(existing.first_public_release_at) == release_at
            and _utc(existing.direct_payload_delete_due_at) == direct_due
            and _utc(existing.pseudonymous_audit_retain_until) == audit_until
        ):
            raise ParticipantLifecycleError("reviewer retention schedule already differs")
        return existing
    schedule_id = new_id()
    payload = {
        "schema_version": "flavourbench-reviewer-retention-schedule-v1",
        "id": schedule_id,
        "lifecycle_id": lifecycle.id,
        "season_id": lifecycle.season_id,
        "reviewer_id": lifecycle.reviewer_id,
        "analysis_freeze_at": freeze_at.isoformat(),
        "first_public_release_at": release_at.isoformat(),
        "direct_payload_delete_due_at": direct_due.isoformat(),
        "pseudonymous_audit_retain_until": audit_until.isoformat(),
        "retention_policy_sha256": acceptance.retention_policy_sha256,
        "created_at": current.isoformat(),
    }
    schedule = ReviewerRetentionSchedule(
        id=schedule_id,
        lifecycle_id=lifecycle.id,
        season_id=lifecycle.season_id,
        reviewer_id=lifecycle.reviewer_id,
        analysis_freeze_at=freeze_at,
        first_public_release_at=release_at,
        direct_payload_delete_due_at=direct_due,
        pseudonymous_audit_retain_until=audit_until,
        retention_policy_sha256=acceptance.retention_policy_sha256,
        schedule_sha256=_sha256(payload),
        created_at=current,
    )
    session.add(schedule)
    session.flush()
    return schedule


def execute_private_payload_deletion(
    session: Session,
    *,
    reviewer_id: str,
    idempotency_key: str,
    execution_basis: str,
    receipt_credential: str | None = None,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> ReviewerDeletionReceipt:
    """Redact only declared private payloads; retain pseudonymous audit markers."""

    configured = settings or get_settings()
    current = _utc(now or datetime.now(UTC))
    if execution_basis not in {"scheduled_retention", "participant_request"}:
        raise ParticipantLifecycleError("reviewer deletion basis is invalid")
    if not idempotency_key or len(idempotency_key) > 160:
        raise ParticipantLifecycleError("reviewer deletion idempotency key is invalid")
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle)
        .where(ReviewerParticipationLifecycle.reviewer_id == reviewer_id)
        .with_for_update()
    )
    if lifecycle is None:
        raise ParticipantLifecycleError("reviewer deletion lifecycle is unavailable")
    acceptance = session.get(ReviewerConsentAcceptance, lifecycle.consent_acceptance_id)
    if acceptance is None:
        raise ParticipantLifecycleError("reviewer consent acceptance is unavailable")
    require_original_accepted_manifest(acceptance, settings=configured)
    if execution_basis == "participant_request":
        if receipt_credential is None:
            raise ParticipantLifecycleError("participant deletion requires receipt authentication")
        authenticated = authenticate_consent_receipt(
            session,
            receipt_credential=receipt_credential,
            settings=configured,
        )
        if authenticated.id != acceptance.id:
            raise ParticipantLifecycleError("participant deletion crossed reviewer scope")
    if lifecycle.status == "redacted":
        existing = session.scalar(
            select(ReviewerDeletionReceipt).where(
                ReviewerDeletionReceipt.lifecycle_id == lifecycle.id
            )
        )
        if existing is None:
            raise ParticipantLifecycleError("redacted participant state lacks its receipt")
        return existing
    if lifecycle.status != "withdrawn":
        raise ParticipantLifecycleError("reviewer must withdraw before private deletion")
    schedule = session.scalar(
        select(ReviewerRetentionSchedule).where(
            ReviewerRetentionSchedule.lifecycle_id == lifecycle.id
        )
    )
    if schedule is None:
        raise ParticipantLifecycleError("reviewer retention schedule is unavailable")
    if execution_basis == "scheduled_retention" and current < _utc(
        schedule.direct_payload_delete_due_at
    ):
        raise ParticipantLifecycleError("reviewer private-payload deletion deadline is not due")
    reviewer = session.get(ExpertReviewer, lifecycle.reviewer_id)
    if reviewer is None or reviewer.privacy_status != "retained":
        raise ParticipantLifecycleError("reviewer private payload state is unavailable")
    private_payload_before_sha256 = _sha256(
        {
            "profile_json": reviewer.profile_json,
            "qualification_json": reviewer.qualification_json,
            "qualification_verified": reviewer.qualification_verified,
        }
    )
    request_sha256 = _domain_hmac_hex(
        configured,
        DELETION_REQUEST_DOMAIN,
        _canonical_bytes(
            {
                "lifecycle_id": lifecycle.id,
                "idempotency_key": idempotency_key,
                "execution_basis": execution_basis,
            }
        ),
    )
    receipt_id = new_id()
    payload = {
        "schema_version": "flavourbench-reviewer-deletion-receipt-v1",
        "id": receipt_id,
        "lifecycle_id": lifecycle.id,
        "retention_schedule_id": schedule.id,
        "season_id": lifecycle.season_id,
        "reviewer_id": lifecycle.reviewer_id,
        "request_sha256": request_sha256,
        "execution_basis": execution_basis,
        "redacted_fields": list(REDACTED_PROFILE_FIELDS),
        "private_payload_before_sha256": private_payload_before_sha256,
        "audit_marker_sha256": lifecycle.audit_marker_sha256,
        "direct_payload_delete_due_at": _utc(schedule.direct_payload_delete_due_at).isoformat(),
        "pseudonymous_audit_retain_until": _utc(
            schedule.pseudonymous_audit_retain_until
        ).isoformat(),
        "prior_judgments_preserved": True,
        "executed_at": current.isoformat(),
    }
    receipt_sha256 = _sha256(payload)
    redacted_profile = {
        "schema_version": "flavourbench-reviewer-redacted-profile-v1",
        "privacy_status": "redacted",
        "audit_marker_sha256": lifecycle.audit_marker_sha256,
        "private_payload_before_sha256": private_payload_before_sha256,
    }
    receipt = ReviewerDeletionReceipt(
        id=receipt_id,
        lifecycle_id=lifecycle.id,
        retention_schedule_id=schedule.id,
        season_id=lifecycle.season_id,
        reviewer_id=lifecycle.reviewer_id,
        request_sha256=request_sha256,
        execution_basis=execution_basis,
        redacted_fields_json=list(REDACTED_PROFILE_FIELDS),
        private_payload_before_sha256=private_payload_before_sha256,
        audit_marker_sha256=lifecycle.audit_marker_sha256,
        direct_payload_delete_due_at=schedule.direct_payload_delete_due_at,
        pseudonymous_audit_retain_until=schedule.pseudonymous_audit_retain_until,
        prior_judgments_preserved=True,
        receipt_sha256=receipt_sha256,
        executed_at=current,
    )
    session.add(receipt)
    session.flush()
    claimed = session.execute(
        update(ExpertReviewer)
        .where(
            ExpertReviewer.id == lifecycle.reviewer_id,
            ExpertReviewer.privacy_status == "retained",
            ExpertReviewer.active.is_(False),
        )
        .values(
            profile_json=redacted_profile,
            qualification_json=[],
            qualification_verified=False,
            privacy_status="redacted",
            privacy_redacted_at=current,
            privacy_redaction_receipt_sha256=receipt_sha256,
        )
        .execution_options(synchronize_session=False)
    )
    if claimed.rowcount != 1:
        raise ParticipantLifecycleError("reviewer private deletion raced with another transition")
    changed = session.execute(
        update(ReviewerParticipationLifecycle)
        .where(
            ReviewerParticipationLifecycle.id == lifecycle.id,
            ReviewerParticipationLifecycle.status == "withdrawn",
            ReviewerParticipationLifecycle.deletion_receipt_sha256.is_(None),
        )
        .values(
            status="redacted",
            redacted_at=current,
            deletion_receipt_sha256=receipt_sha256,
        )
        .execution_options(synchronize_session=False)
    )
    if changed.rowcount != 1:
        raise ParticipantLifecycleError("reviewer redaction raced with another transition")
    return receipt


def execute_participant_private_payload_deletion(
    session: Session,
    *,
    receipt_credential: str,
    idempotency_key: str,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> ReviewerDeletionReceipt:
    """Resolve reviewer scope only from the authenticated participant receipt."""

    configured = settings or get_settings()
    acceptance = authenticate_consent_receipt(
        session,
        receipt_credential=receipt_credential,
        settings=configured,
        lock=True,
    )
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle).where(
            ReviewerParticipationLifecycle.consent_acceptance_id == acceptance.id
        )
    )
    if lifecycle is None:
        raise ParticipantLifecycleError("participant deletion lifecycle is unavailable")
    return execute_private_payload_deletion(
        session,
        reviewer_id=lifecycle.reviewer_id,
        idempotency_key=idempotency_key,
        execution_basis="participant_request",
        receipt_credential=receipt_credential,
        now=now,
        settings=configured,
    )


def privacy_safe_participant_status(
    session: Session,
    *,
    receipt_credential: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return an allowlisted status without identity, contact, or credential locators."""

    acceptance = authenticate_consent_receipt(
        session,
        receipt_credential=receipt_credential,
        settings=settings,
    )
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle).where(
            ReviewerParticipationLifecycle.consent_acceptance_id == acceptance.id
        )
    )
    schedule = (
        session.scalar(
            select(ReviewerRetentionSchedule).where(
                ReviewerRetentionSchedule.lifecycle_id == lifecycle.id
            )
        )
        if lifecycle is not None
        else None
    )
    return {
        "consentReceiptSha256": acceptance.receipt_sha256,
        "consentDocumentSha256": acceptance.consent_document_sha256,
        "activationManifestSha256": acceptance.activation_manifest_sha256,
        "participationStatus": lifecycle.status if lifecycle is not None else "consent_accepted",
        "auditMarkerSha256": lifecycle.audit_marker_sha256 if lifecycle is not None else None,
        "withdrawalReceiptSha256": (
            lifecycle.withdrawal_receipt_sha256 if lifecycle is not None else None
        ),
        "deletionReceiptSha256": (
            lifecycle.deletion_receipt_sha256 if lifecycle is not None else None
        ),
        "directPayloadDeleteDueAt": (
            _utc(schedule.direct_payload_delete_due_at).isoformat() if schedule else None
        ),
        "pseudonymousAuditRetainUntil": (
            _utc(schedule.pseudonymous_audit_retain_until).isoformat() if schedule else None
        ),
    }
