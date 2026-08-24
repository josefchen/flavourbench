"""Season-scoped reviewer identity, credentials, admission, and vote provenance.

The module deliberately stores no names, email addresses, or identity-provider
subjects. A season-keyed HMAC supplies uniqueness inside one scored season
without creating a durable cross-season identifier.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .models import (
    Battle,
    ExpertReviewer,
    ReviewerAccessCredential,
    ReviewerCalibrationBallot,
    ReviewerCalibrationSet,
    ReviewerConsentAcceptance,
    ReviewerFamilyAdmission,
    ReviewerIdentityBinding,
    ReviewerParticipationLifecycle,
    ReviewerQualificationEvidence,
    Season,
    Vote,
    _reviewer_admission_evidence_sha256,
    _verified_vote_provenance_sha256,
)

REVIEWER_ROLES = frozenset({"task_author", "task_validator", "task_adjudicator", "output_rater"})
TASK_FAMILIES = frozenset({"substitution", "composition", "cookability", "evidence"})
AFFILIATION_COHORT = {
    "independent_external": "expert_independent",
    "product_affiliated": "expert_product_affiliated",
    "provider_affiliated": "expert_provider_affiliated",
}
SAFE_RELEASE_REASON_TAGS = frozenset(
    {
        "ignored_constraint",
        "weak_flavour_logic",
        "unclear",
        "generic",
        "overconfident",
        "safety_hazard",
        "unsupported_safety_claim",
        "allergen_or_dietary_risk",
        "impractical",
        "evidence_trace_mismatch",
        "entity_resolution_mismatch",
        "similarity_as_functional_proof",
        "similarity_as_mechanism",
        "axis_as_measured_quantity",
        "score_as_normative_truth",
        "selective_evidence",
        "irrelevant_evidence",
        "false_precision",
    }
)


class ReviewerIdentityError(RuntimeError):
    """Reviewer evidence is missing, ambiguous, expired, or inconsistent."""


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ReviewerIdentityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def season_person_commitment(
    *,
    season_id: str,
    identity_issuer: str,
    issuer_subject: str,
    settings: Settings | None = None,
) -> tuple[str, str]:
    """Return a season-only person HMAC and a non-identifying issuer digest."""

    configured = settings or get_settings()
    normalized_issuer = identity_issuer.strip().casefold()
    normalized_subject = issuer_subject.strip()
    if not normalized_issuer or not normalized_subject:
        raise ReviewerIdentityError("identity issuer and subject are required")
    if len(normalized_issuer) > 512 or len(normalized_subject) > 1024:
        raise ReviewerIdentityError("identity issuer or subject exceeds the accepted bound")
    material = json.dumps(
        {
            "schema_version": "flavourbench-season-person-commitment-v1",
            "season_id": season_id,
            "identity_issuer": normalized_issuer,
            "issuer_subject": normalized_subject,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    commitment = hmac.new(
        configured.reviewer_identity_hmac_secret.encode(),
        material,
        hashlib.sha256,
    ).hexdigest()
    issuer_sha256 = hashlib.sha256(normalized_issuer.encode()).hexdigest()
    return commitment, issuer_sha256


def bind_reviewer_identity(
    session: Session,
    *,
    season: Season,
    reviewer: ExpertReviewer,
    identity_issuer: str,
    issuer_subject: str,
    identity_evidence_sha256: str,
    roles: Iterable[str],
    settings: Settings | None = None,
) -> ReviewerIdentityBinding:
    """Create one immutable season/person binding and flush its unique constraints."""

    configured = settings or get_settings()
    normalized_roles = sorted(set(roles))
    if not normalized_roles or not set(normalized_roles).issubset(REVIEWER_ROLES):
        raise ReviewerIdentityError("reviewer roles are empty or invalid")
    if not reviewer.active:
        raise ReviewerIdentityError("reviewer is inactive")
    commitment, issuer_sha256 = season_person_commitment(
        season_id=season.id,
        identity_issuer=identity_issuer,
        issuer_subject=issuer_subject,
        settings=configured,
    )
    binding = ReviewerIdentityBinding(
        season_id=season.id,
        reviewer_id=reviewer.id,
        person_commitment_sha256=commitment,
        identity_issuer_sha256=issuer_sha256,
        identity_evidence_sha256=_require_sha256(identity_evidence_sha256, "identity evidence"),
        hmac_key_id=configured.reviewer_identity_hmac_key_id,
        verification_method="season_hmac_issuer_subject_v1",
        assurance_level="server_verified",
        roles_json=normalized_roles,
    )
    session.add(binding)
    session.flush()
    return binding


def _credential_token(prefix: str, secret: str) -> str:
    return f"fbrv1_{prefix}.{secret}"


def _credential_digest(token: str, settings: Settings) -> str:
    return hmac.new(
        settings.reviewer_credential_hmac_secret.encode(),
        token.encode(),
        hashlib.sha256,
    ).hexdigest()


def _token_prefix(token: str) -> str:
    if not token.startswith("fbrv1_") or "." not in token:
        raise ReviewerIdentityError("reviewer credential format is invalid")
    prefix, secret = token.removeprefix("fbrv1_").split(".", 1)
    if not prefix or not secret or len(prefix) > 32 or len(secret) > 128:
        raise ReviewerIdentityError("reviewer credential format is invalid")
    return prefix


def issue_reviewer_credential(
    session: Session,
    *,
    binding: ReviewerIdentityBinding,
    credential_kind: str,
    scopes: Iterable[str],
    maximum_uses: int | None = None,
    ttl_seconds: int | None = None,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> tuple[str, ReviewerAccessCredential]:
    """Issue plaintext once while persisting only a peppered HMAC."""

    configured = settings or get_settings()
    issued_at = _utc(now or datetime.now(UTC))
    normalized_scopes = sorted(set(scopes))
    if not normalized_scopes or any(not scope or len(scope) > 80 for scope in normalized_scopes):
        raise ReviewerIdentityError("reviewer credential scopes are invalid")
    if credential_kind == "enrollment_once":
        use_limit = 1
        lifetime = min(ttl_seconds or 3600, 86_400)
    elif credential_kind == "review_session":
        use_limit = maximum_uses or configured.reviewer_session_max_uses
        lifetime = ttl_seconds or configured.reviewer_session_ttl_seconds
    else:
        raise ReviewerIdentityError("reviewer credential kind is invalid")
    if use_limit < 1 or use_limit > 256 or lifetime < 300 or lifetime > 604_800:
        raise ReviewerIdentityError("reviewer credential bounds are invalid")
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle).where(
            ReviewerParticipationLifecycle.identity_binding_id == binding.id
        )
    )
    if lifecycle is None and configured.environment == "production":
        raise ReviewerIdentityError(
            "production reviewer credential requires participant-owned consent"
        )
    if lifecycle is not None and lifecycle.status != "active":
        raise ReviewerIdentityError("withdrawn reviewer cannot receive a credential")
    if lifecycle is not None:
        try:
            from .participant_lifecycle import require_active_participant_authority

            require_active_participant_authority(
                session,
                reviewer_id=binding.reviewer_id,
                season_id=binding.season_id,
                identity_binding_id=binding.id,
                at=issued_at,
                settings=configured,
            )
        except RuntimeError as exc:
            raise ReviewerIdentityError(
                "reviewer credential requires current participant authority"
            ) from exc
    prefix = secrets.token_hex(8)
    token = _credential_token(prefix, secrets.token_urlsafe(32))
    credential = ReviewerAccessCredential(
        season_id=binding.season_id,
        reviewer_id=binding.reviewer_id,
        identity_binding_id=binding.id,
        credential_prefix=prefix,
        secret_hmac_sha256=_credential_digest(token, configured),
        hmac_key_id=configured.reviewer_credential_hmac_key_id,
        credential_kind=credential_kind,
        scopes_json=normalized_scopes,
        status="active",
        maximum_uses=use_limit,
        use_count=0,
        not_before=issued_at,
        expires_at=issued_at + timedelta(seconds=lifetime),
    )
    session.add(credential)
    session.flush()
    return token, credential


def consume_reviewer_credential(
    session: Session,
    *,
    token: str,
    required_scope: str,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> ReviewerAccessCredential:
    """Atomically consume one bounded use; exhausted credentials become terminal."""

    configured = settings or get_settings()
    prefix = _token_prefix(token)
    credential = session.scalar(
        select(ReviewerAccessCredential)
        .where(ReviewerAccessCredential.credential_prefix == prefix)
        .with_for_update()
    )
    current = _utc(now or datetime.now(UTC))
    if credential is None or not hmac.compare_digest(
        credential.secret_hmac_sha256,
        _credential_digest(token, configured),
    ):
        raise ReviewerIdentityError("reviewer credential is invalid")
    if credential.hmac_key_id != configured.reviewer_credential_hmac_key_id:
        raise ReviewerIdentityError("reviewer credential key epoch is not active")
    if credential.status != "active" or credential.use_count >= credential.maximum_uses:
        raise ReviewerIdentityError("reviewer credential is exhausted or revoked")
    if current < _utc(credential.not_before) or current > _utc(credential.expires_at):
        raise ReviewerIdentityError("reviewer credential is outside its validity window")
    if required_scope not in credential.scopes_json:
        raise ReviewerIdentityError("reviewer credential lacks the required scope")
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle).where(
            ReviewerParticipationLifecycle.identity_binding_id == credential.identity_binding_id
        )
    )
    if lifecycle is None and configured.environment == "production":
        raise ReviewerIdentityError(
            "production reviewer credential requires participant-owned consent"
        )
    if lifecycle is not None and lifecycle.status != "active":
        raise ReviewerIdentityError("reviewer participation is withdrawn or redacted")
    if lifecycle is not None:
        try:
            from .participant_lifecycle import require_active_participant_authority

            require_active_participant_authority(
                session,
                reviewer_id=credential.reviewer_id,
                season_id=credential.season_id,
                identity_binding_id=credential.identity_binding_id,
                at=current,
                settings=configured,
            )
        except RuntimeError as exc:
            raise ReviewerIdentityError(
                "reviewer credential lacks current participant authority"
            ) from exc
    prior_uses = credential.use_count
    new_uses = prior_uses + 1
    terminal = new_uses == credential.maximum_uses
    result = session.execute(
        update(ReviewerAccessCredential)
        .where(
            ReviewerAccessCredential.id == credential.id,
            ReviewerAccessCredential.status == "active",
            ReviewerAccessCredential.use_count == prior_uses,
        )
        .values(
            use_count=new_uses,
            status="consumed" if terminal else "active",
            consumed_at=current if terminal else None,
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        raise ReviewerIdentityError("reviewer credential was consumed concurrently")
    session.expire(credential)
    session.flush()
    return credential


def exchange_enrollment_credential(
    session: Session,
    *,
    enrollment_token: str,
    session_scopes: Iterable[str],
    now: datetime | None = None,
    settings: Settings | None = None,
) -> tuple[str, ReviewerAccessCredential]:
    """Redeem a one-time enrollment token for one bounded review session token."""

    configured = settings or get_settings()
    enrollment = consume_reviewer_credential(
        session,
        token=enrollment_token,
        required_scope="exchange_review_session",
        now=now,
        settings=configured,
    )
    if enrollment.credential_kind != "enrollment_once":
        raise ReviewerIdentityError("only an enrollment credential may be exchanged")
    binding = session.get(ReviewerIdentityBinding, enrollment.identity_binding_id)
    if binding is None:
        raise ReviewerIdentityError("reviewer identity binding is unavailable")
    return issue_reviewer_credential(
        session,
        binding=binding,
        credential_kind="review_session",
        scopes=session_scopes,
        now=now,
        settings=configured,
    )


def record_qualification_evidence(
    session: Session,
    *,
    binding: ReviewerIdentityBinding,
    family: str,
    affiliation_class: str,
    independence_verified: bool,
    conflict_cleared: bool,
    qualification_evidence_sha256: str,
    independence_evidence_sha256: str,
    conflict_disclosure_sha256: str,
    consent_document_sha256: str,
    training_material_sha256: str,
    verifier_principal_sha256: str,
    verified_at: datetime,
    valid_until: datetime | None,
) -> ReviewerQualificationEvidence:
    if family not in TASK_FAMILIES or affiliation_class not in AFFILIATION_COHORT:
        raise ReviewerIdentityError("qualification scope is invalid")
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle).where(
            ReviewerParticipationLifecycle.identity_binding_id == binding.id
        )
    )
    if lifecycle is None and get_settings().environment == "production":
        raise ReviewerIdentityError("production qualification requires participant-owned consent")
    if lifecycle is not None:
        acceptance = session.get(
            ReviewerConsentAcceptance,
            lifecycle.consent_acceptance_id,
        )
        if (
            lifecycle.status != "active"
            or acceptance is None
            or acceptance.consent_document_sha256 != consent_document_sha256
        ):
            raise ReviewerIdentityError(
                "qualification evidence requires active participant consent"
            )
        try:
            from .participant_lifecycle import require_active_participant_authority

            require_active_participant_authority(
                session,
                reviewer_id=binding.reviewer_id,
                season_id=binding.season_id,
                identity_binding_id=binding.id,
                at=datetime.now(UTC),
            )
        except RuntimeError as exc:
            raise ReviewerIdentityError(
                "qualification evidence requires current participant authority"
            ) from exc
    evidence = ReviewerQualificationEvidence(
        season_id=binding.season_id,
        reviewer_id=binding.reviewer_id,
        identity_binding_id=binding.id,
        family=family,
        affiliation_class=affiliation_class,
        independence_verified=independence_verified,
        conflict_cleared=conflict_cleared,
        verification_status="verified",
        qualification_evidence_sha256=_require_sha256(
            qualification_evidence_sha256, "qualification evidence"
        ),
        independence_evidence_sha256=_require_sha256(
            independence_evidence_sha256, "independence evidence"
        ),
        conflict_disclosure_sha256=_require_sha256(
            conflict_disclosure_sha256, "conflict disclosure"
        ),
        consent_document_sha256=_require_sha256(consent_document_sha256, "consent document"),
        training_material_sha256=_require_sha256(training_material_sha256, "training material"),
        verifier_principal_sha256=_require_sha256(verifier_principal_sha256, "verifier principal"),
        verified_at=_utc(verified_at),
        valid_until=_utc(valid_until) if valid_until is not None else None,
    )
    session.add(evidence)
    session.flush()
    return evidence


def freeze_calibration_set(
    session: Session,
    *,
    season: Season,
    family: str,
    calibration_set_sha256: str,
    source_artifact_sha256: str,
    scoring_key_sha256: str,
    item_count: int,
    real_source_arms: int,
    frozen_at: datetime,
) -> ReviewerCalibrationSet:
    calibration_set = ReviewerCalibrationSet(
        season_id=season.id,
        family=family,
        calibration_set_sha256=_require_sha256(calibration_set_sha256, "calibration set"),
        source_artifact_sha256=_require_sha256(
            source_artifact_sha256, "calibration source artifact"
        ),
        scoring_key_sha256=_require_sha256(scoring_key_sha256, "calibration scoring key"),
        item_count=item_count,
        real_source_arms=real_source_arms,
        synthetic_arms=0,
        frozen_at=_utc(frozen_at),
    )
    session.add(calibration_set)
    session.flush()
    return calibration_set


def record_calibration_ballot(
    session: Session,
    *,
    binding: ReviewerIdentityBinding,
    calibration_set: ReviewerCalibrationSet,
    ballot_sha256: str,
    scoring_result_sha256: str,
    correct_count: int,
    minimum_accuracy_milli: int,
    completed_at: datetime,
) -> ReviewerCalibrationBallot:
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle).where(
            ReviewerParticipationLifecycle.identity_binding_id == binding.id
        )
    )
    if lifecycle is None and get_settings().environment == "production":
        raise ReviewerIdentityError("production calibration requires participant-owned consent")
    if lifecycle is not None:
        try:
            from .participant_lifecycle import require_active_participant_authority

            require_active_participant_authority(
                session,
                reviewer_id=binding.reviewer_id,
                season_id=binding.season_id,
                identity_binding_id=binding.id,
                at=datetime.now(UTC),
            )
        except RuntimeError as exc:
            raise ReviewerIdentityError(
                "calibration requires current participant authority"
            ) from exc
    accuracy_milli = (
        1000 * correct_count + calibration_set.item_count // 2
    ) // calibration_set.item_count
    ballot = ReviewerCalibrationBallot(
        season_id=binding.season_id,
        reviewer_id=binding.reviewer_id,
        identity_binding_id=binding.id,
        calibration_set_id=calibration_set.id,
        ballot_sha256=_require_sha256(ballot_sha256, "calibration ballot"),
        scoring_result_sha256=_require_sha256(scoring_result_sha256, "calibration scoring result"),
        item_count=calibration_set.item_count,
        correct_count=correct_count,
        accuracy_milli=accuracy_milli,
        passed=accuracy_milli >= minimum_accuracy_milli,
        completed_at=_utc(completed_at),
    )
    session.add(ballot)
    session.flush()
    return ballot


def derive_family_admission(
    session: Session,
    *,
    binding: ReviewerIdentityBinding,
    qualification: ReviewerQualificationEvidence,
    family: str,
    review_role: str,
    cohort: str,
    admission_policy: dict[str, Any],
    decision_reference_sha256: str,
    valid_from: datetime,
    valid_until: datetime,
    calibration_ballot: ReviewerCalibrationBallot | None = None,
) -> ReviewerFamilyAdmission:
    """Derive an admission digest from rows already verified by the server."""

    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle).where(
            ReviewerParticipationLifecycle.identity_binding_id == binding.id
        )
    )
    if lifecycle is None and get_settings().environment == "production":
        raise ReviewerIdentityError(
            "production reviewer admission requires participant-owned consent"
        )
    if lifecycle is not None:
        try:
            from .participant_lifecycle import require_active_participant_authority

            require_active_participant_authority(
                session,
                reviewer_id=binding.reviewer_id,
                season_id=binding.season_id,
                identity_binding_id=binding.id,
                at=datetime.now(UTC),
            )
        except RuntimeError as exc:
            raise ReviewerIdentityError(
                "reviewer admission requires current participant authority"
            ) from exc

    policy_sha256 = canonical_sha256(admission_policy)
    admission = ReviewerFamilyAdmission(
        season_id=binding.season_id,
        reviewer_id=binding.reviewer_id,
        identity_binding_id=binding.id,
        family=family,
        review_role=review_role,
        cohort=cohort,
        qualification_evidence_id=qualification.id,
        calibration_ballot_id=(calibration_ballot.id if calibration_ballot else None),
        admission_policy_json=admission_policy,
        admission_policy_sha256=policy_sha256,
        evidence_bundle_sha256="0" * 64,
        decision_reference_sha256=_require_sha256(decision_reference_sha256, "admission decision"),
        valid_from=_utc(valid_from),
        valid_until=_utc(valid_until),
    )
    admission.evidence_bundle_sha256 = _reviewer_admission_evidence_sha256(
        admission,
        qualification_evidence_sha256=qualification.qualification_evidence_sha256,
        calibration_ballot_sha256=(
            calibration_ballot.ballot_sha256 if calibration_ballot else None
        ),
    )
    session.add(admission)
    session.flush()
    return admission


def reviewer_rater_pseudonym(
    binding: ReviewerIdentityBinding,
    *,
    settings: Settings | None = None,
) -> str:
    configured = settings or get_settings()
    material = (f"flavourbench-expert-rater-v1\x00{binding.season_id}\x00{binding.id}").encode()
    return hmac.new(configured.pseudonym_secret.encode(), material, hashlib.sha256).hexdigest()


def resolve_verified_vote_admission(
    session: Session,
    *,
    reviewer: ExpertReviewer,
    battle: Battle,
    at: datetime | None = None,
) -> tuple[ReviewerIdentityBinding, ReviewerFamilyAdmission] | None:
    """Return the sole active output-rater admission, or fail on ambiguity."""

    timestamp = _utc(at or datetime.now(UTC))
    binding = session.scalar(
        select(ReviewerIdentityBinding).where(
            ReviewerIdentityBinding.season_id == battle.season_id,
            ReviewerIdentityBinding.reviewer_id == reviewer.id,
            ReviewerIdentityBinding.assurance_level == "server_verified",
        )
    )
    if binding is None:
        return None
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle).where(
            ReviewerParticipationLifecycle.identity_binding_id == binding.id
        )
    )
    if lifecycle is None and get_settings().environment == "production":
        return None
    if lifecycle is not None and lifecycle.status != "active":
        return None
    if lifecycle is not None:
        try:
            from .participant_lifecycle import require_active_participant_authority

            require_active_participant_authority(
                session,
                reviewer_id=reviewer.id,
                season_id=battle.season_id,
                identity_binding_id=binding.id,
                at=timestamp,
            )
        except RuntimeError:
            return None
    admissions = session.scalars(
        select(ReviewerFamilyAdmission).where(
            ReviewerFamilyAdmission.season_id == battle.season_id,
            ReviewerFamilyAdmission.reviewer_id == reviewer.id,
            ReviewerFamilyAdmission.identity_binding_id == binding.id,
            ReviewerFamilyAdmission.family == battle.category,
            ReviewerFamilyAdmission.review_role == "output_rater",
            ReviewerFamilyAdmission.cohort == reviewer.cohort,
            ReviewerFamilyAdmission.valid_from <= timestamp,
            ReviewerFamilyAdmission.valid_until >= timestamp,
        )
    ).all()
    if len(admissions) > 1:
        raise ReviewerIdentityError("multiple active reviewer admissions make provenance ambiguous")
    return (binding, admissions[0]) if admissions else None


def apply_verified_vote_provenance(
    vote: Vote,
    *,
    reviewer: ExpertReviewer,
    binding: ReviewerIdentityBinding,
    admission: ReviewerFamilyAdmission,
) -> None:
    vote.reviewer_id = reviewer.id
    vote.reviewer_identity_binding_id = binding.id
    vote.reviewer_family_admission_id = admission.id
    vote.provenance_status = "expert_verified_v1"
    vote.provenance_sha256 = _verified_vote_provenance_sha256(vote)


def verified_vote_person_commitment(session: Session, vote: Vote) -> str | None:
    """Revalidate a stored vote and return its season-only person commitment."""

    if vote.provenance_status != "expert_verified_v1" or not vote.cohort.startswith("expert_"):
        return None
    if vote.provenance_sha256 != _verified_vote_provenance_sha256(vote):
        return None
    admission = session.get(ReviewerFamilyAdmission, vote.reviewer_family_admission_id)
    if admission is None:
        return None
    qualification = session.get(
        ReviewerQualificationEvidence,
        admission.qualification_evidence_id,
    )
    ballot = (
        session.get(ReviewerCalibrationBallot, admission.calibration_ballot_id)
        if admission.calibration_ballot_id is not None
        else None
    )
    calibration_set = (
        session.get(ReviewerCalibrationSet, ballot.calibration_set_id)
        if ballot is not None
        else None
    )
    policy = (
        admission.admission_policy_json if isinstance(admission.admission_policy_json, dict) else {}
    )
    requires_calibration = policy.get("requires_calibration")
    minimum_accuracy = policy.get("minimum_accuracy_milli")
    if (
        admission.admission_policy_sha256 != canonical_sha256(policy)
        or policy.get("schema_version") != "flavourbench-reviewer-admission-policy-v1"
        or not isinstance(requires_calibration, bool)
        or not isinstance(minimum_accuracy, int)
        or not 0 <= minimum_accuracy <= 1000
        or qualification is None
        or qualification.season_id != admission.season_id
        or qualification.reviewer_id != admission.reviewer_id
        or qualification.identity_binding_id != admission.identity_binding_id
        or qualification.family != admission.family
        or AFFILIATION_COHORT.get(qualification.affiliation_class) != admission.cohort
        or (
            admission.cohort == "expert_independent"
            and not (qualification.independence_verified and qualification.conflict_cleared)
        )
        or (requires_calibration and ballot is None)
        or (
            ballot is not None
            and (
                ballot.season_id != admission.season_id
                or ballot.reviewer_id != admission.reviewer_id
                or ballot.identity_binding_id != admission.identity_binding_id
                or not ballot.passed
                or ballot.accuracy_milli < minimum_accuracy
                or calibration_set is None
                or calibration_set.season_id != admission.season_id
                or calibration_set.family != admission.family
                or calibration_set.synthetic_arms != 0
            )
        )
    ):
        return None
    expected_admission_evidence = _reviewer_admission_evidence_sha256(
        admission,
        qualification_evidence_sha256=qualification.qualification_evidence_sha256,
        calibration_ballot_sha256=ballot.ballot_sha256 if ballot is not None else None,
    )
    if admission.evidence_bundle_sha256 != expected_admission_evidence:
        return None
    lifecycle = session.scalar(
        select(ReviewerParticipationLifecycle).where(
            ReviewerParticipationLifecycle.identity_binding_id == vote.reviewer_identity_binding_id,
            ReviewerParticipationLifecycle.reviewer_id == vote.reviewer_id,
            ReviewerParticipationLifecycle.season_id == admission.season_id,
        )
    )
    try:
        from .participant_lifecycle import participant_record_analysis_eligible

        participant_eligible = participant_record_analysis_eligible(
            session,
            reviewer_id=vote.reviewer_id,
            season_id=admission.season_id,
            identity_binding_id=vote.reviewer_identity_binding_id,
            recorded_at=vote.created_at,
        )
    except RuntimeError:
        return None
    if participant_eligible is False:
        return None
    requires_current_reviewer = lifecycle is None or lifecycle.status == "active"
    row = session.execute(
        select(
            ReviewerIdentityBinding.person_commitment_sha256,
            ReviewerIdentityBinding.season_id,
            ReviewerIdentityBinding.reviewer_id,
            ReviewerIdentityBinding.assurance_level,
            ReviewerFamilyAdmission.season_id,
            ReviewerFamilyAdmission.reviewer_id,
            ReviewerFamilyAdmission.identity_binding_id,
            ReviewerFamilyAdmission.family,
            ReviewerFamilyAdmission.review_role,
            ReviewerFamilyAdmission.cohort,
            ReviewerFamilyAdmission.valid_from,
            ReviewerFamilyAdmission.valid_until,
            ExpertReviewer.active,
            Battle.season_id,
            Battle.category,
        )
        .select_from(ReviewerIdentityBinding)
        .join(
            ReviewerFamilyAdmission,
            ReviewerFamilyAdmission.id == vote.reviewer_family_admission_id,
        )
        .join(ExpertReviewer, ExpertReviewer.id == vote.reviewer_id)
        .join(Battle, Battle.id == vote.battle_id)
        .where(ReviewerIdentityBinding.id == vote.reviewer_identity_binding_id)
    ).one_or_none()
    if row is None or (
        row[1] != row[4]
        or row[1] != row[13]
        or row[2] != vote.reviewer_id
        or row[3] != "server_verified"
        or row[5] != vote.reviewer_id
        or row[6] != vote.reviewer_identity_binding_id
        or row[7] != row[14]
        or row[8] != "output_rater"
        or row[9] != vote.cohort
        or _utc(vote.created_at) < _utc(row[10])
        or _utc(vote.created_at) > _utc(row[11])
        or (requires_current_reviewer and row[12] is not True)
    ):
        return None
    return row[0]


def filter_ranking_vote_rows(
    session: Session,
    rows: Sequence[tuple[Vote, Battle]],
    *,
    expert_quorum: int,
) -> list[tuple[Vote, Battle]]:
    """Exclude legacy experts and require distinct admitted people per comparison."""

    if expert_quorum < 2:
        raise ReviewerIdentityError("expert output comparison quorum must be at least two")
    public_rows: list[tuple[Vote, Battle]] = []
    expert_rows: list[tuple[Vote, Battle]] = []
    commitments: dict[tuple[str, str], set[str]] = defaultdict(set)
    for vote, battle in rows:
        if vote.cohort == "public":
            public_rows.append((vote, battle))
            continue
        commitment = verified_vote_person_commitment(session, vote)
        if commitment is None:
            continue
        expert_rows.append((vote, battle))
        commitments[(battle.id, vote.cohort)].add(commitment)
    admitted_expert_rows = [
        (vote, battle)
        for vote, battle in expert_rows
        if len(commitments[(battle.id, vote.cohort)]) >= expert_quorum
    ]
    return sorted(
        [*public_rows, *admitted_expert_rows],
        key=lambda row: (row[1].id, row[0].cohort, row[0].id),
    )


def distinct_person_quorum(
    session: Session,
    binding_ids: Iterable[str],
    *,
    minimum: int,
) -> bool:
    """Count stable person commitments, never reviewer codes or presentation rows."""

    ids = list(binding_ids)
    if minimum < 1 or not ids:
        return False
    commitments = set(
        session.scalars(
            select(ReviewerIdentityBinding.person_commitment_sha256).where(
                ReviewerIdentityBinding.id.in_(ids),
                ReviewerIdentityBinding.assurance_level == "server_verified",
            )
        ).all()
    )
    return len(commitments) >= minimum


def privacy_safe_vote_release(vote: Vote) -> dict[str, Any]:
    """Return an allowlisted vote view with no person, credential, or admission locator."""

    return {
        "cohort": vote.cohort,
        "choice": vote.choice,
        "reasonTags": [
            tag
            for tag in vote.reason_tags_json
            if isinstance(tag, str) and tag in SAFE_RELEASE_REASON_TAGS
        ],
        "provenanceClass": (
            "verified_expert"
            if vote.provenance_status == "expert_verified_v1"
            else "public"
            if vote.cohort == "public"
            else "legacy_unverified"
        ),
    }
