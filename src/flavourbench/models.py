from __future__ import annotations

import base64
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    inspect,
    select,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .task_validation_replay_binding import (
    TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
    TASK_VALIDATION_RIGHTS_REQUIRED_IDS,
    TASK_VALIDATION_V1_REPLAY_SHA256,
    TASK_VALIDATION_V6_CAMPAIGN_SHA256,
    rights_audit_plan,
)


def utcnow() -> datetime:
    return datetime.now(UTC)


RETENTION_BASES = frozenset(
    {
        "public_nonconsented",
        "public_consented",
        "official_research",
        "commercial_private",
        "controlled_development",
        "development_research",
        "legacy_operational",
    }
)
REDACTABLE_RETENTION_BASES = frozenset(
    {
        "public_nonconsented",
        "commercial_private",
        "controlled_development",
        "development_research",
    }
)
PERMANENT_RESEARCH_RETENTION_UNTIL = datetime(9999, 12, 31, tzinfo=UTC)
RESEARCH_ARCHIVE_SIGNATURE_CONTEXT = b"flavourbench-research-release-v1\x00"


def _as_utc(value: datetime) -> datetime:
    """Normalize SQLite's timezone-naive round trip before ordering timestamps."""

    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _same_instant(left: datetime | None, right: datetime | None) -> bool:
    if left is None or right is None:
        return left is right
    return _as_utc(left) == _as_utc(right)


def _retention_authorized_for_battle(
    connection: Connection,
    battle_id: str,
) -> bool:
    row = connection.execute(
        select(
            Battle.prompt,
            Battle.prompt_redacted,
            Battle.research_consent,
            Battle.retention_until,
            Battle.retention_basis,
        ).where(Battle.id == battle_id)
    ).one_or_none()
    return bool(
        row is not None
        and row[0] is None
        and row[1] is True
        and row[2] is False
        and _as_utc(row[3]) <= datetime.now(UTC)
        and row[4] in REDACTABLE_RETENTION_BASES
    )


def _battle_id_for_arm(connection: Connection, arm_id: str) -> str | None:
    return connection.execute(
        select(ResponseArm.battle_id).where(ResponseArm.id == arm_id)
    ).scalar_one_or_none()


def _require_nonnegative_budget(record: object) -> None:
    # SQLAlchemy's Python-side column defaults are populated after
    # ``before_insert`` listeners run, so an omitted zero-valued counter is
    # still ``None`` here. Treat only that not-yet-materialized default as
    # zero; explicit negative values remain invalid.
    used = getattr(record, "budget_used_micros", 0)
    reserved = getattr(record, "budget_reserved_micros", 0)
    if (0 if used is None else used) < 0 or (0 if reserved is None else reserved) < 0:
        raise ValueError("governed spend and reservation counters must be nonnegative")


def new_id() -> str:
    return str(uuid.uuid4())


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


TOOL_CALL_REDACTION_SENTINEL = "[REDACTED AFTER OPERATIONAL RETENTION]"
TOOL_CALL_REDACTION_JSON = {"redacted": True}


class Base(DeclarativeBase):
    pass


class Season(Base):
    __tablename__ = "seasons"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    official: Mapped[bool] = mapped_column(Boolean, default=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    prompt_registry_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    tool_registry_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    epicure_release_id: Mapped[str] = mapped_column(String(160))
    epicure_bundle_sha256: Mapped[str] = mapped_column(String(64), default="unresolved")
    epicure_application_sha256: Mapped[str] = mapped_column(String(64), default="unresolved")
    analysis_plan_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    protocol_bundle_json: Mapped[dict] = mapped_column(JSON, default=dict)
    protocol_bundle_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    budget_cap_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    budget_used_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    budget_reserved_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EpicureRelease(Base):
    """Immutable, content-addressed Epicure lineage approved for benchmark use."""

    __tablename__ = "epicure_releases"

    release_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    bundle_sha256: Mapped[str] = mapped_column(String(64), index=True)
    application_sha256: Mapped[str] = mapped_column(String(64), index=True)
    public_release_uri: Mapped[str] = mapped_column(Text)
    release_artifact_sha256: Mapped[str] = mapped_column(String(64))
    rights_clearance_sha256: Mapped[str] = mapped_column(String(64))
    verification_report_sha256: Mapped[str] = mapped_column(String(64))
    lineage_manifest_json: Mapped[dict] = mapped_column(JSON)
    lineage_manifest_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    public_release_match: Mapped[bool] = mapped_column(Boolean, default=False)
    redistribution_rights_cleared: Mapped[bool] = mapped_column(Boolean, default=False)
    reproducibility_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    official_eligible: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Organization(Base):
    """Tenant identity for private model-company evaluations.

    Personal contacts and agreement documents remain in the configured identity,
    CRM, and e-sign systems.  This table stores only operational names and
    content-addressed references.
    """

    __tablename__ = "organizations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending_verification', 'active', 'suspended', 'closed')",
            name="ck_organizations_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    legal_name: Mapped[str] = mapped_column(String(240))
    display_name: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(32), default="pending_verification", index=True)
    idp_tenant_reference_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    billing_reference_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    data_region: Mapped[str] = mapped_column(String(32))
    retention_policy_json: Mapped[dict] = mapped_column(JSON)
    retention_policy_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class OrganizationApiKey(Base):
    """One-time-issued, scoped organization credential.

    Only a peppered HMAC is persisted.  The plaintext credential must never be
    written to PostgreSQL, logs, events, or evidence bundles.
    """

    __tablename__ = "organization_api_keys"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'revoked')",
            name="ck_organization_api_keys_status",
        ),
        CheckConstraint(
            "(status = 'active' AND revoked_at IS NULL) OR "
            "(status = 'revoked' AND revoked_at IS NOT NULL)",
            name="ck_organization_api_keys_revocation",
        ),
        CheckConstraint(
            "expires_at > not_before",
            name="ck_organization_api_keys_validity_window",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    key_prefix: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    secret_hmac_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    hmac_key_id: Mapped[str] = mapped_column(String(64), default="primary")
    label: Mapped[str] = mapped_column(String(120))
    scopes_json: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    rate_limit_profile: Mapped[str] = mapped_column(String(64), default="standard")
    network_policy_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by_principal_ref_sha256: Mapped[str] = mapped_column(String(64))
    not_before: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    supersedes_key_id: Mapped[str | None] = mapped_column(
        ForeignKey("organization_api_keys.id"), nullable=True, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GovernanceAcceptance(Base):
    """Append-only reference to externally executed legal or governance evidence."""

    __tablename__ = "governance_acceptances"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'revoked', 'expired', 'superseded')",
            name="ck_governance_acceptances_status",
        ),
        CheckConstraint(
            "(CASE WHEN model_submission_id IS NULL THEN 0 ELSE 1 END + "
            "CASE WHEN route_revision_id IS NULL THEN 0 ELSE 1 END + "
            "CASE WHEN evaluation_order_id IS NULL THEN 0 ELSE 1 END) <= 1",
            name="ck_governance_acceptances_subject",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    model_submission_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_submissions.id"), nullable=True, index=True
    )
    route_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_route_revisions.id"), nullable=True, index=True
    )
    evaluation_order_id: Mapped[str | None] = mapped_column(
        ForeignKey("evaluation_orders.id"), nullable=True, index=True
    )
    agreement_type: Mapped[str] = mapped_column(String(80), index=True)
    agreement_version: Mapped[str] = mapped_column(String(80))
    document_sha256: Mapped[str] = mapped_column(String(64))
    external_envelope_reference_sha256: Mapped[str] = mapped_column(String(64))
    signatory_principal_reference_sha256: Mapped[str] = mapped_column(String(64))
    authority_basis: Mapped[str] = mapped_column(String(160))
    binding_json: Mapped[dict] = mapped_column(JSON, default=dict)
    binding_sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    supersedes_acceptance_id: Mapped[str | None] = mapped_column(
        ForeignKey("governance_acceptances.id"), nullable=True, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelSubmission(Base):
    """Immutable model-version submission owned by one organization."""

    __tablename__ = "model_submissions"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "requested_canonical_model_id",
            "revision",
            name="uq_model_submissions_org_model_revision",
        ),
        CheckConstraint("revision >= 1", name="ck_model_submissions_revision"),
        CheckConstraint(
            "status IN ('draft', 'submitted', 'changes_requested', 'approved', "
            "'rejected', 'withdrawn', 'suspended', 'retired')",
            name="ck_model_submissions_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    supersedes_submission_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_submissions.id"), nullable=True, unique=True
    )
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    display_name: Mapped[str] = mapped_column(String(240))
    publisher: Mapped[str] = mapped_column(String(240))
    requested_canonical_model_id: Mapped[str] = mapped_column(String(240), index=True)
    exact_model_version: Mapped[str] = mapped_column(String(240))
    release_date: Mapped[str] = mapped_column(String(32))
    model_card_uri: Mapped[str] = mapped_column(Text)
    model_card_sha256: Mapped[str] = mapped_column(String(64))
    license_uri: Mapped[str] = mapped_column(Text)
    license_document_sha256: Mapped[str] = mapped_column(String(64))
    capability_claims_json: Mapped[dict] = mapped_column(JSON)
    capability_claims_sha256: Mapped[str] = mapped_column(String(64))
    contamination_disclosure_json: Mapped[dict] = mapped_column(JSON)
    contamination_disclosure_sha256: Mapped[str] = mapped_column(String(64))
    submission_payload_json: Mapped[dict] = mapped_column(JSON)
    submission_payload_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    catalog_model_id: Mapped[str | None] = mapped_column(
        ForeignKey("catalog_models.model_id"), nullable=True, index=True
    )
    submitted_by_key_id: Mapped[str] = mapped_column(
        ForeignKey("organization_api_keys.id"), index=True
    )
    decision_reference_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelRouteRevision(Base):
    """Content-addressed managed-route request for one submitted model version."""

    __tablename__ = "model_route_revisions"
    __table_args__ = (
        UniqueConstraint(
            "model_submission_id",
            "revision",
            name="uq_model_route_revisions_submission_revision",
        ),
        CheckConstraint("revision >= 1", name="ck_model_route_revisions_revision"),
        CheckConstraint(
            "route_kind IN ('managed_bedrock', 'managed_openrouter')",
            name="ck_model_route_revisions_kind",
        ),
        CheckConstraint(
            "(route_kind = 'managed_bedrock' AND execution_backend = 'bedrock') OR "
            "(route_kind = 'managed_openrouter' AND execution_backend = 'openrouter')",
            name="ck_model_route_revisions_backend",
        ),
        CheckConstraint(
            "status IN ('draft', 'submitted', 'contract_testing', 'approved', "
            "'changes_requested', 'rejected', 'withdrawn', 'suspended', 'retired')",
            name="ck_model_route_revisions_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_submission_id: Mapped[str] = mapped_column(ForeignKey("model_submissions.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    route_kind: Mapped[str] = mapped_column(String(32))
    execution_backend: Mapped[str] = mapped_column(String(32), index=True)
    managed_route_reference_sha256: Mapped[str] = mapped_column(String(64))
    requested_model_id: Mapped[str] = mapped_column(String(240))
    expected_actual_model_id: Mapped[str] = mapped_column(String(240))
    expected_actual_provider_slug: Mapped[str] = mapped_column(String(160))
    supported_parameters_json: Mapped[list] = mapped_column(JSON)
    supported_parameters_sha256: Mapped[str] = mapped_column(String(64))
    decoding_bounds_json: Mapped[dict] = mapped_column(JSON)
    decoding_bounds_sha256: Mapped[str] = mapped_column(String(64))
    endpoint_document_sha256: Mapped[str] = mapped_column(String(64))
    data_policy_json: Mapped[dict] = mapped_column(JSON)
    data_policy_sha256: Mapped[str] = mapped_column(String(64))
    rate_card_json: Mapped[dict] = mapped_column(JSON)
    rate_card_sha256: Mapped[str] = mapped_column(String(64))
    descriptor_json: Mapped[dict] = mapped_column(JSON)
    descriptor_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    # Application and transition guards verify that this identifier names a
    # passed test for this exact route.  Avoiding the reciprocal inline foreign
    # key keeps portable schema creation free of an FK cycle.
    approved_contract_test_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, unique=True
    )
    approved_season_id: Mapped[str | None] = mapped_column(
        ForeignKey("seasons.id"), nullable=True, index=True
    )
    approved_season_manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    approved_endpoint_contract_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RouteContractTest(Base):
    """Worker-produced compatibility evidence for one exact route descriptor."""

    __tablename__ = "route_contract_tests"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'passed', 'failed', 'inconclusive', 'cancelled')",
            name="ck_route_contract_tests_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    route_revision_id: Mapped[str] = mapped_column(
        ForeignKey("model_route_revisions.id"), index=True
    )
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    suite_version: Mapped[str] = mapped_column(String(80))
    protocol_bundle_sha256: Mapped[str] = mapped_column(String(64))
    worker_build_digest: Mapped[str] = mapped_column(String(160))
    request_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    response_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_trace_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    structured_output_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    observed_model_id: Mapped[str | None] = mapped_column(String(240), nullable=True)
    observed_provider_slug: Mapped[str | None] = mapped_column(String(160), nullable=True)
    check_results_json: Mapped[dict] = mapped_column(JSON, default=dict)
    check_results_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    generation_id: Mapped[str | None] = mapped_column(String(240), nullable=True)
    usage_json: Mapped[dict] = mapped_column(JSON, default=dict)
    cost_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    cost_accounting_basis: Mapped[str] = mapped_column(String(80), default="unrecorded")
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    incident_id: Mapped[str | None] = mapped_column(ForeignKey("incidents.id"), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationOrder(Base):
    """Commercial request whose execution materializes as a governed controlled run."""

    __tablename__ = "evaluation_orders"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "client_reference_sha256",
            name="uq_evaluation_orders_org_client_reference",
        ),
        CheckConstraint(
            "status IN ('draft', 'submitted', 'approved', 'provisioning', 'ready', "
            "'running', 'collection_complete', 'analysis_complete', 'delivered', "
            "'rejected', 'cancelling', 'cancelled', 'failed')",
            name="ck_evaluation_orders_status",
        ),
        CheckConstraint(
            "billing_status IN ('unquoted', 'quoted', 'authorized', 'reconciled', "
            "'disputed', 'void')",
            name="ck_evaluation_orders_billing_status",
        ),
        CheckConstraint(
            "publication_status IN ('private', 'authorized', 'published', 'withdrawn')",
            name="ck_evaluation_orders_publication_status",
        ),
        CheckConstraint(
            "requested_visibility IN ('private', 'public_candidate')",
            name="ck_evaluation_orders_visibility",
        ),
        CheckConstraint(
            "forecast_cost_micros >= 0 AND budget_cap_micros > 0",
            name="ck_evaluation_orders_budget",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    model_submission_id: Mapped[str] = mapped_column(ForeignKey("model_submissions.id"), index=True)
    route_revision_id: Mapped[str] = mapped_column(
        ForeignKey("model_route_revisions.id"), index=True
    )
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    evaluation_profile_id: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    billing_status: Mapped[str] = mapped_column(String(24), default="unquoted", index=True)
    publication_status: Mapped[str] = mapped_column(String(24), default="private", index=True)
    requested_visibility: Mapped[str] = mapped_column(String(24), default="private")
    comparison_plan_json: Mapped[dict] = mapped_column(JSON)
    comparison_plan_sha256: Mapped[str] = mapped_column(String(64))
    rater_plan_sha256: Mapped[str] = mapped_column(String(64))
    analysis_plan_sha256: Mapped[str] = mapped_column(String(64))
    forecast_cost_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    budget_cap_micros: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    quote_reference_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    client_reference_sha256: Mapped[str] = mapped_column(String(64))
    order_card_json: Mapped[dict] = mapped_column(JSON)
    order_card_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    order_card_signature: Mapped[str] = mapped_column(String(64))
    submitted_by_key_id: Mapped[str] = mapped_column(
        ForeignKey("organization_api_keys.id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvidenceBundle(Base):
    """Externally verifiable customer evidence archive metadata."""

    __tablename__ = "evidence_bundles"
    __table_args__ = (
        CheckConstraint(
            "bundle_class IN ('private_customer', 'public_release')",
            name="ck_evidence_bundles_class",
        ),
        CheckConstraint(
            "status IN ('building', 'sealed', 'available', 'superseded', 'revoked', 'failed')",
            name="ck_evidence_bundles_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    evaluation_order_id: Mapped[str] = mapped_column(ForeignKey("evaluation_orders.id"), index=True)
    bundle_class: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(24), default="building", index=True)
    schema_version: Mapped[str] = mapped_column(String(80))
    manifest_json: Mapped[dict] = mapped_column(JSON)
    manifest_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    archive_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    storage_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    signature_algorithm: Mapped[str | None] = mapped_column(String(32), nullable=True)
    signing_key_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    signature_base64: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_authorization_id: Mapped[str | None] = mapped_column(
        ForeignKey("governance_acceptances.id"), nullable=True
    )
    supersedes_bundle_id: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_bundles.id"), nullable=True, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    sealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    available_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ApiIdempotencyKey(Base):
    """Tenant-scoped replay record for state-changing customer API calls."""

    __tablename__ = "api_idempotency_keys"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "method",
            "route_template",
            "idempotency_key_sha256",
            name="uq_api_idempotency_scope",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("organization_api_keys.id"), index=True)
    method: Mapped[str] = mapped_column(String(12))
    route_template: Mapped[str] = mapped_column(String(160))
    idempotency_key_sha256: Mapped[str] = mapped_column(String(64))
    request_sha256: Mapped[str] = mapped_column(String(64))
    response_status: Mapped[int] = mapped_column(Integer)
    resource_type: Mapped[str] = mapped_column(String(80))
    resource_id: Mapped[str] = mapped_column(String(80))
    response_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class CatalogModel(Base):
    __tablename__ = "catalog_models"

    model_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    canonical_slug: Mapped[str] = mapped_column(String(240), index=True)
    name: Mapped[str] = mapped_column(String(240))
    family: Mapped[str] = mapped_column(String(120), default="unknown")
    catalog_source: Mapped[str] = mapped_column(String(48), default="openrouter", index=True)
    open_weight: Mapped[bool] = mapped_column(Boolean, default=False)
    open_weight_evidence_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="discovered", index=True)
    supports_tools: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_structured_outputs: Mapped[bool] = mapped_column(Boolean, default=False)
    context_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pricing_json: Mapped[dict] = mapped_column(JSON, default=dict)
    endpoint_json: Mapped[dict] = mapped_column(JSON, default=dict)
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SeasonModel(Base):
    __tablename__ = "season_models"
    __table_args__ = (UniqueConstraint("season_id", "model_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("catalog_models.model_id"), index=True)
    slot_role: Mapped[str] = mapped_column(String(48))
    execution_backend: Mapped[str] = mapped_column(String(32), default="openrouter", index=True)
    provider_slug: Mapped[str] = mapped_column(String(120), default="mock")
    expected_actual_model_id: Mapped[str] = mapped_column(String(240), default="unfrozen")
    expected_actual_provider_slug: Mapped[str] = mapped_column(String(160), default="unfrozen")
    supported_parameters_json: Mapped[list] = mapped_column(JSON, default=list)
    decoding_json: Mapped[dict] = mapped_column(JSON, default=dict)
    endpoint_max_completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    endpoint_document_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    endpoint_contract_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    backend_contract_json: Mapped[dict] = mapped_column(JSON, default=dict)
    backend_contract_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    rate_card_json: Mapped[dict] = mapped_column(JSON, default=dict)
    rate_card_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    eligible: Mapped[bool] = mapped_column(Boolean, default=True)
    manifest_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen")
    worst_case_cost_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SeasonProviderBudget(Base):
    """Provider-scoped authorization and mutable spend ledger for one season.

    ``account_authorization_envelope_sha256`` is the immutable root epoch that
    bound the season to a permanent provider-account ledger at freeze time.
    Dispatch may use only an active, cryptographically valid successor of that
    root, and records the exact successor on every provider request.
    """

    __tablename__ = "season_provider_budgets"
    __table_args__ = (
        UniqueConstraint("season_id", "execution_backend"),
        CheckConstraint(
            "execution_backend IN ('openrouter', 'bedrock', 'kimi_direct', "
            "'qwencloud_direct', 'mock')",
            name="ck_season_provider_budgets_backend",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    execution_backend: Mapped[str] = mapped_column(String(32), index=True)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    budget_cap_micros: Mapped[int] = mapped_column(BigInteger)
    budget_used_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    budget_reserved_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    account_scope_sha256: Mapped[str] = mapped_column(String(64))
    authorization_reference_sha256: Mapped[str] = mapped_column(String(64))
    account_authorization_envelope_sha256: Mapped[str] = mapped_column(String(64))
    authorization_envelope_json: Mapped[dict] = mapped_column(JSON)
    authorization_envelope_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProviderAccountBudget(Base):
    """Permanent cumulative spend ledger for one installation-wide provider scope."""

    __tablename__ = "provider_account_budgets"
    __table_args__ = (
        UniqueConstraint("execution_backend", "account_scope_sha256"),
        CheckConstraint(
            "execution_backend IN ('openrouter', 'bedrock', 'kimi_direct', 'qwencloud_direct')",
            name="ck_provider_account_budgets_backend",
        ),
        CheckConstraint(
            "status IN ('pending_verification', 'active', 'revoked')",
            name="ck_provider_account_budgets_status",
        ),
        CheckConstraint(
            "budget_cap_micros > 0 AND budget_used_micros >= 0 AND "
            "budget_reserved_micros >= 0 AND opening_used_micros >= 0 AND "
            "opening_reserved_micros >= 0",
            name="ck_provider_account_budgets_nonnegative",
        ),
        CheckConstraint(
            "(status = 'revoked' AND revoked_at IS NOT NULL) OR "
            "(status != 'revoked' AND revoked_at IS NULL)",
            name="ck_provider_account_budgets_revocation",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    execution_backend: Mapped[str] = mapped_column(String(32), index=True)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    budget_cap_micros: Mapped[int] = mapped_column(BigInteger)
    budget_used_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    budget_reserved_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    opening_used_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    opening_reserved_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    account_scope_sha256: Mapped[str] = mapped_column(String(64), index=True)
    authorization_reference_sha256: Mapped[str] = mapped_column(String(64))
    opening_balance_json: Mapped[dict] = mapped_column(JSON)
    opening_balance_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    credential_binding_json: Mapped[dict] = mapped_column(JSON)
    credential_binding_sha256: Mapped[str] = mapped_column(String(64))
    authorization_envelope_json: Mapped[dict] = mapped_column(JSON)
    authorization_envelope_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    authorization_hmac_sha256: Mapped[str] = mapped_column(String(64))
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProviderAccountAuthorization(Base):
    """Append-only credential-authorization epoch over a permanent account ledger."""

    __tablename__ = "provider_account_authorizations"
    __table_args__ = (
        CheckConstraint(
            "execution_backend IN ('openrouter', 'bedrock', 'kimi_direct', 'qwencloud_direct')",
            name="ck_provider_account_authorizations_backend",
        ),
        CheckConstraint(
            "status IN ('active', 'revoked')",
            name="ck_provider_account_authorizations_status",
        ),
        CheckConstraint(
            "(status = 'active' AND revoked_at IS NULL) OR "
            "(status = 'revoked' AND revoked_at IS NOT NULL)",
            name="ck_provider_account_authorizations_revocation",
        ),
        CheckConstraint(
            "authorized_used_micros >= 0 AND authorized_reserved_micros >= 0",
            name="ck_provider_account_authorizations_nonnegative",
        ),
        Index(
            "uq_provider_account_authorizations_active_scope",
            "execution_backend",
            "account_scope_sha256",
            unique=True,
            postgresql_where=sql_text("status = 'active'"),
            sqlite_where=sql_text("status = 'active'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    provider_account_budget_id: Mapped[str] = mapped_column(
        ForeignKey("provider_account_budgets.id"), index=True
    )
    execution_backend: Mapped[str] = mapped_column(String(32), index=True)
    account_scope_sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    supersedes_authorization_id: Mapped[str | None] = mapped_column(
        ForeignKey("provider_account_authorizations.id"),
        nullable=True,
        unique=True,
    )
    authorization_reference_sha256: Mapped[str] = mapped_column(String(64))
    exposure_attestation_json: Mapped[dict] = mapped_column(JSON)
    exposure_attestation_sha256: Mapped[str] = mapped_column(String(64))
    authorized_used_micros: Mapped[int] = mapped_column(BigInteger)
    authorized_reserved_micros: Mapped[int] = mapped_column(BigInteger)
    credential_binding_json: Mapped[dict] = mapped_column(JSON)
    credential_binding_sha256: Mapped[str] = mapped_column(String(64))
    authorization_envelope_json: Mapped[dict] = mapped_column(JSON)
    authorization_envelope_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    authorization_hmac_sha256: Mapped[str] = mapped_column(String(64))
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        UniqueConstraint("season_id", "public_id"),
        UniqueConstraint("season_id", "prompt_sha256"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    public_id: Mapped[str] = mapped_column(String(80), index=True)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    family: Mapped[str] = mapped_column(String(64), index=True)
    prompt: Mapped[str] = mapped_column(Text)
    prompt_sha256: Mapped[str] = mapped_column(String(64), index=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    split: Mapped[str] = mapped_column(String(24), default="pilot")
    review_status: Mapped[str] = mapped_column(String(32), default="candidate")
    provenance_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TaskEvidenceArtifact(Base):
    """Append-only, task-bound evidence resolved before confirmatory admission."""

    __tablename__ = "task_evidence_artifacts"
    __table_args__ = (
        CheckConstraint(
            "evidence_type IN ('validator_contract', 'contamination_audit')",
            name="ck_task_evidence_artifacts_type",
        ),
        UniqueConstraint("artifact_sha256"),
        UniqueConstraint("task_id", "evidence_type", "revision_ordinal"),
        UniqueConstraint("supersedes_artifact_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    evidence_type: Mapped[str] = mapped_column(String(32), index=True)
    schema_version: Mapped[str] = mapped_column(String(96))
    revision_ordinal: Mapped[int] = mapped_column(Integer, default=1)
    artifact_json: Mapped[dict] = mapped_column(JSON)
    artifact_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    task_binding_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    verification_receipt_json: Mapped[dict] = mapped_column(JSON)
    verification_receipt_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    supersedes_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("task_evidence_artifacts.id"), nullable=True, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ControlledRun(Base):
    """A private, content-addressed evaluation contract for one customer run.

    Its spend counters govern admission from endpoint-level generation
    accounting. Later aggregate invoice variance is governed at the season and
    provider-account levels because it is not safely attributable to one run.
    """

    __tablename__ = "controlled_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'collection_complete', 'closed', 'revoked')",
            name="ck_controlled_runs_status",
        ),
        CheckConstraint(
            "(organization_id IS NULL AND evaluation_order_id IS NULL AND "
            "route_revision_id IS NULL AND endpoint_descriptor_sha256 IS NULL AND "
            "spend_authorization_id IS NULL AND "
            "spend_authorization_binding_sha256 IS NULL) OR "
            "(organization_id IS NOT NULL AND evaluation_order_id IS NOT NULL AND "
            "route_revision_id IS NOT NULL AND endpoint_descriptor_sha256 IS NOT NULL AND "
            "spend_authorization_id IS NOT NULL AND "
            "spend_authorization_binding_sha256 IS NOT NULL)",
            name="ck_controlled_runs_commercial_binding",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    organization_id: Mapped[str | None] = mapped_column(
        ForeignKey("organizations.id"), nullable=True, index=True
    )
    evaluation_order_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "evaluation_orders.id",
            name="fk_controlled_runs_evaluation_order_id_evaluation_orders",
        ),
        nullable=True,
        unique=True,
        index=True,
    )
    route_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_route_revisions.id"), nullable=True, index=True
    )
    endpoint_descriptor_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    spend_authorization_id: Mapped[str | None] = mapped_column(
        ForeignKey("governance_acceptances.id"), nullable=True, index=True
    )
    spend_authorization_binding_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    organization_reference_sha256: Mapped[str] = mapped_column(String(64), index=True)
    access_token_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    protocol_version: Mapped[str] = mapped_column(String(80))
    rater_plan_sha256: Mapped[str] = mapped_column(String(64))
    analysis_plan_sha256: Mapped[str] = mapped_column(String(64))
    submitted_endpoint_model_id: Mapped[str] = mapped_column(String(200), default="unbound")
    submitted_model_card_sha256: Mapped[str] = mapped_column(String(64), default=lambda: "0" * 64)
    data_policy_sha256: Mapped[str] = mapped_column(String(64), default=lambda: "0" * 64)
    model_roster_json: Mapped[list] = mapped_column(JSON, default=list)
    model_roster_sha256: Mapped[str] = mapped_column(String(64), default=lambda: "0" * 64)
    task_schedule_sha256: Mapped[str] = mapped_column(String(64), default=lambda: "0" * 64)
    token_version: Mapped[int] = mapped_column(Integer, default=1)
    budget_cap_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    budget_used_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    budget_reserved_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    run_card_json: Mapped[dict] = mapped_column(JSON)
    run_card_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    run_card_signature: Mapped[str] = mapped_column(String(64))
    release_authorized: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    release_authorization_reference_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    release_authorized_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    publication_authorization_id: Mapped[str | None] = mapped_column(
        ForeignKey("governance_acceptances.id"), nullable=True, index=True
    )
    publication_authorization_binding_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    collection_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ControlledRunAssignment(Base):
    """One immutable row in a private run's frozen execution schedule."""

    __tablename__ = "controlled_run_assignments"
    __table_args__ = (
        UniqueConstraint(
            "controlled_run_id", "ordinal", name="uq_controlled_assignments_run_ordinal"
        ),
        UniqueConstraint(
            "controlled_run_id",
            "assignment_sha256",
            name="uq_controlled_assignments_run_sha256",
        ),
        UniqueConstraint("battle_id", name="uq_controlled_assignments_battle"),
        CheckConstraint(
            "track IN ('model_arena', 'epicure_uplift')",
            name="ck_controlled_run_assignments_track",
        ),
        CheckConstraint(
            "status IN ('pending', 'queued', 'cancelled')",
            name="ck_controlled_run_assignments_status",
        ),
        CheckConstraint("ordinal >= 0", name="ck_controlled_run_assignments_ordinal"),
        CheckConstraint(
            "repetition_index >= 1",
            name="ck_controlled_run_assignments_repetition_index",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    controlled_run_id: Mapped[str] = mapped_column(ForeignKey("controlled_runs.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    task_public_id: Mapped[str] = mapped_column(String(80))
    task_revision: Mapped[int] = mapped_column(Integer)
    task_prompt_sha256: Mapped[str] = mapped_column(String(64))
    task_family: Mapped[str] = mapped_column(String(64))
    track: Mapped[str] = mapped_column(String(32), index=True)
    model_ids_json: Mapped[list] = mapped_column(JSON)
    repetition_index: Mapped[int] = mapped_column(Integer)
    assignment_sha256: Mapped[str] = mapped_column(String(64), index=True)
    assignment_seed: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    battle_id: Mapped[str | None] = mapped_column(
        ForeignKey("battles.id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Battle(Base):
    __tablename__ = "battles"
    __table_args__ = (
        Index("ix_battles_season_track_status", "season_id", "track", "status"),
        Index(
            "ix_battles_rank_scope",
            "season_id",
            "rank_eligible",
            "run_class",
            "manifest_sha256",
            "track",
        ),
        Index(
            "ix_battles_controlled_run_rank_scope",
            "controlled_run_id",
            "track",
            "status",
        ),
        UniqueConstraint("season_id", "requester_pseudonym", "client_nonce_sha256"),
        CheckConstraint(
            "run_class IN ('mock', 'smoke', 'exploratory', 'pilot', 'official')",
            name="ck_battles_run_class",
        ),
        CheckConstraint(
            "data_stratum IN ('public_freeform', 'controlled', 'development', 'legacy')",
            name="ck_battles_data_stratum",
        ),
        CheckConstraint(
            "(retention_basis = 'public_nonconsented' AND "
            "data_stratum = 'public_freeform' AND research_consent = false) OR "
            "(retention_basis = 'public_consented' AND "
            "data_stratum = 'public_freeform' AND research_consent = true) OR "
            "(retention_basis IN ("
            "'official_research', 'commercial_private', 'controlled_development'"
            ") AND "
            "data_stratum = 'controlled' AND controlled_run_id IS NOT NULL AND "
            "research_consent = false) OR "
            "(retention_basis = 'development_research' AND "
            "data_stratum = 'development' AND research_consent = false) OR "
            "(retention_basis = 'legacy_operational' AND data_stratum = 'legacy')",
            name="ck_battles_retention_basis_scope",
        ),
        CheckConstraint(
            "(task_id IS NULL AND task_revision IS NULL) OR "
            "(task_id IS NOT NULL AND task_revision IS NOT NULL)",
            name="ck_battles_task_revision_pair",
        ),
        CheckConstraint(
            "left_arm_id IS NULL OR right_arm_id IS NULL OR left_arm_id <> right_arm_id",
            name="ck_battles_distinct_arm_links",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    run_class: Mapped[str] = mapped_column(String(24), default="exploratory", index=True)
    rank_eligible: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    data_stratum: Mapped[str] = mapped_column(String(32), default="public_freeform", index=True)
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("tasks.id", name="fk_battles_task_id_tasks"), nullable=True, index=True
    )
    task_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    controlled_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("controlled_runs.id", name="fk_battles_controlled_run_id_controlled_runs"),
        nullable=True,
        index=True,
    )
    manifest_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen", index=True)
    protocol_bundle_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen", index=True)
    scheduler_version: Mapped[str] = mapped_column(String(80), default="legacy-unversioned")
    assignment_seed: Mapped[str] = mapped_column(String(64), default=lambda: "0" * 64)
    track_assignment_probability: Mapped[str] = mapped_column(String(48), default="unknown")
    model_assignment_probability: Mapped[str] = mapped_column(String(48), default="unknown")
    side_assignment_probability: Mapped[str] = mapped_column(String(48), default="unknown")
    track: Mapped[str] = mapped_column(String(32), index=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_sha256: Mapped[str] = mapped_column(String(64), index=True)
    client_nonce_sha256: Mapped[str] = mapped_column(String(64), index=True)
    prompt_redacted: Mapped[bool] = mapped_column(Boolean, default=False)
    research_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    retention_basis: Mapped[str] = mapped_column(
        String(32), default="public_nonconsented", index=True
    )
    release_review_status: Mapped[str] = mapped_column(
        String(32), default="not_requested", index=True
    )
    release_reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    requester_pseudonym: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    left_arm_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    right_arm_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    reserved_cost_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    provider_reservations_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retention_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ResponseArm(Base):
    __tablename__ = "response_arms"
    __table_args__ = (
        UniqueConstraint("battle_id", "side"),
        CheckConstraint("side IN ('left', 'right')", name="ck_response_arms_side"),
        CheckConstraint(
            "status IN ('queued', 'running', 'complete', 'failed', 'uncertain')",
            name="ck_response_arms_status",
        ),
        CheckConstraint(
            "cost_micros >= 0",
            name="ck_response_arms_nonnegative_cost",
        ),
        CheckConstraint(
            "(route_revision_id IS NULL AND endpoint_descriptor_sha256 IS NULL) OR "
            "(route_revision_id IS NOT NULL AND endpoint_descriptor_sha256 IS NOT NULL)",
            name="ck_response_arms_route_binding",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    battle_id: Mapped[str] = mapped_column(ForeignKey("battles.id"), index=True)
    side: Mapped[str] = mapped_column(String(8))
    condition: Mapped[str] = mapped_column(String(32), index=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("catalog_models.model_id"), index=True)
    execution_backend: Mapped[str] = mapped_column(String(32), default="openrouter", index=True)
    route_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_route_revisions.id"), nullable=True, index=True
    )
    endpoint_descriptor_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    provider_slug: Mapped[str] = mapped_column(String(120), default="unknown")
    actual_provider_slug: Mapped[str | None] = mapped_column(String(160), nullable=True)
    actual_model_id: Mapped[str | None] = mapped_column(String(240), nullable=True)
    generation_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    provider_generation_ids_json: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    answer_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_markdown_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    output_json: Mapped[dict] = mapped_column(JSON, default=dict)
    output_json_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_sha256: Mapped[str] = mapped_column(String(64))
    system_prompt_sha256: Mapped[str] = mapped_column(String(64), default="unresolved")
    schema_sha256: Mapped[str] = mapped_column(String(64))
    tool_schema_sha256: Mapped[str] = mapped_column(String(64))
    decoding_json: Mapped[dict] = mapped_column(JSON, default=dict)
    observed_decoding_json: Mapped[dict] = mapped_column(JSON, default=dict)
    protocol_bundle_sha256: Mapped[str] = mapped_column(String(64), default="unfrozen", index=True)
    epicure_release_id: Mapped[str] = mapped_column(String(160))
    epicure_bundle_sha256: Mapped[str] = mapped_column(String(64))
    epicure_application_sha256: Mapped[str] = mapped_column(String(64), default="unresolved")
    epicure_attestation_json: Mapped[dict] = mapped_column(JSON, default=dict)
    epicure_attestation_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    cost_reconciled: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_accounting_basis: Mapped[str] = mapped_column(String(80), default="unrecorded")
    billing_reconciliation_status: Mapped[str] = mapped_column(
        String(80), default="unrecorded", index=True
    )
    backend_response_schema_sha256: Mapped[str] = mapped_column(String(64), default="unresolved")
    backend_tool_schema_sha256: Mapped[str] = mapped_column(String(64), default="unresolved")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    finish_reason: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(120), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class GenerationAttempt(Base):
    """Append-only lifecycle evidence written before and after provider I/O."""

    __tablename__ = "generation_attempts"
    __table_args__ = (
        Index("ix_generation_attempts_arm_created", "arm_id", "created_at"),
        Index("ix_generation_attempts_attempt", "attempt_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    attempt_id: Mapped[str] = mapped_column(String(36), index=True)
    arm_id: Mapped[str] = mapped_column(ForeignKey("response_arms.id"), index=True)
    request_key_sha256: Mapped[str] = mapped_column(String(64))
    phase: Mapped[str] = mapped_column(String(80), index=True)
    attempt_index: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    generation_id: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(160), nullable=True)
    payload_sha256: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ToolCall(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (
        UniqueConstraint(
            "arm_id",
            "round_index",
            "call_index",
            name="uq_tool_calls_arm_round_call",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    arm_id: Mapped[str] = mapped_column(ForeignKey("response_arms.id"), index=True)
    round_index: Mapped[int] = mapped_column(Integer)
    call_index: Mapped[int] = mapped_column(Integer, default=0)
    tool_call_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(120))
    arguments_json: Mapped[dict] = mapped_column(JSON)
    arguments_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_text: Mapped[str] = mapped_column(Text)
    structured_content_json: Mapped[dict] = mapped_column(JSON, default=dict)
    structured_content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_sha256: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    is_error: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ValidatorResult(Base):
    __tablename__ = "validator_results"
    __table_args__ = (
        UniqueConstraint(
            "arm_id",
            "validator_name",
            "validator_version",
            name="uq_validator_results_arm_name_version",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    arm_id: Mapped[str] = mapped_column(ForeignKey("response_arms.id"), index=True)
    validator_name: Mapped[str] = mapped_column(String(120), index=True)
    validator_version: Mapped[str] = mapped_column(String(48))
    status: Mapped[str] = mapped_column(String(24), index=True)
    score_milli: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detail_json: Mapped[dict] = mapped_column(JSON, default=dict)
    detail_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_claim", "status", "available_at", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    battle_id: Mapped[str | None] = mapped_column(ForeignKey("battles.id"), nullable=True)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    claimed_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Vote(Base):
    __tablename__ = "votes"
    __table_args__ = (
        UniqueConstraint("battle_id", "rater_pseudonym", "cohort"),
        UniqueConstraint("idempotency_key"),
        CheckConstraint(
            "choice IN ('left', 'right', 'tie', 'both_bad')",
            name="ck_votes_choice",
        ),
        CheckConstraint(
            "cohort IN ('public', 'expert_independent', "
            "'expert_product_affiliated', 'expert_provider_affiliated')",
            name="ck_votes_cohort",
        ),
        CheckConstraint(
            "provenance_status IN ('legacy_unverified', 'public_pseudonymous', "
            "'expert_verified_v1')",
            name="ck_votes_provenance_status",
        ),
        CheckConstraint(
            "(provenance_status = 'expert_verified_v1' AND reviewer_id IS NOT NULL "
            "AND reviewer_identity_binding_id IS NOT NULL "
            "AND reviewer_family_admission_id IS NOT NULL "
            "AND provenance_sha256 IS NOT NULL) OR "
            "(provenance_status <> 'expert_verified_v1' AND reviewer_id IS NULL "
            "AND reviewer_identity_binding_id IS NULL "
            "AND reviewer_family_admission_id IS NULL)",
            name="ck_votes_reviewer_provenance_shape",
        ),
        Index(
            "uq_votes_verified_person_battle",
            "battle_id",
            "reviewer_identity_binding_id",
            unique=True,
            postgresql_where=sql_text(
                "provenance_status = 'expert_verified_v1' "
                "AND reviewer_identity_binding_id IS NOT NULL"
            ),
            sqlite_where=sql_text(
                "provenance_status = 'expert_verified_v1' "
                "AND reviewer_identity_binding_id IS NOT NULL"
            ),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    battle_id: Mapped[str] = mapped_column(ForeignKey("battles.id"), index=True)
    rater_pseudonym: Mapped[str] = mapped_column(String(64), index=True)
    cohort: Mapped[str] = mapped_column(String(48), default="public", index=True)
    choice: Mapped[str] = mapped_column(String(24), index=True)
    reason_tags_json: Mapped[list] = mapped_column(JSON, default=list)
    rubric_json: Mapped[dict] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(120), index=True)
    reviewer_id: Mapped[str | None] = mapped_column(
        ForeignKey("expert_reviewers.id"), nullable=True, index=True
    )
    reviewer_identity_binding_id: Mapped[str | None] = mapped_column(
        ForeignKey("reviewer_identity_bindings.id"), nullable=True, index=True
    )
    reviewer_family_admission_id: Mapped[str | None] = mapped_column(
        ForeignKey("reviewer_family_admissions.id"), nullable=True, index=True
    )
    provenance_status: Mapped[str] = mapped_column(
        String(32),
        default="legacy_unverified",
        server_default="legacy_unverified",
        index=True,
    )
    provenance_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExpertReviewer(Base):
    __tablename__ = "expert_reviewers"
    __table_args__ = (
        CheckConstraint(
            "(privacy_status = 'retained' AND privacy_redacted_at IS NULL "
            "AND privacy_redaction_receipt_sha256 IS NULL) OR "
            "(privacy_status = 'redacted' AND privacy_redacted_at IS NOT NULL "
            "AND privacy_redaction_receipt_sha256 IS NOT NULL)",
            name="ck_expert_reviewers_privacy_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    reviewer_code: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    invitation_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    qualification_json: Mapped[list] = mapped_column(JSON, default=list)
    qualification_verified: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    cohort: Mapped[str] = mapped_column(String(48), default="expert_independent", index=True)
    profile_json: Mapped[dict] = mapped_column(JSON, default=dict)
    batch_reveal_only: Mapped[bool] = mapped_column(Boolean, default=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    privacy_status: Mapped[str] = mapped_column(
        String(24), default="retained", server_default="retained", index=True
    )
    privacy_redacted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    privacy_redaction_receipt_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )


class ReviewerIdentityBinding(Base):
    """Season-scoped, identity-minimized binding for one natural person.

    The raw issuer subject is never stored. ``person_commitment_sha256`` is a
    server-side season HMAC, so the database can reject the same person under a
    second reviewer code or role without creating a cross-season identifier.
    """

    __tablename__ = "reviewer_identity_bindings"
    __table_args__ = (
        UniqueConstraint(
            "season_id",
            "reviewer_id",
            name="uq_reviewer_identity_bindings_season_reviewer",
        ),
        UniqueConstraint(
            "season_id",
            "person_commitment_sha256",
            name="uq_reviewer_identity_bindings_season_person",
        ),
        CheckConstraint(
            "assurance_level IN ('server_verified', 'legacy_unverified')",
            name="ck_reviewer_identity_bindings_assurance",
        ),
        CheckConstraint(
            "verification_method IN ('season_hmac_issuer_subject_v1', 'legacy_unverified')",
            name="ck_reviewer_identity_bindings_method",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    person_commitment_sha256: Mapped[str] = mapped_column(String(64), index=True)
    identity_issuer_sha256: Mapped[str] = mapped_column(String(64))
    identity_evidence_sha256: Mapped[str] = mapped_column(String(64))
    hmac_key_id: Mapped[str] = mapped_column(String(64))
    verification_method: Mapped[str] = mapped_column(
        String(64), default="season_hmac_issuer_subject_v1"
    )
    assurance_level: Mapped[str] = mapped_column(String(32), default="server_verified")
    roles_json: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerAccessCredential(Base):
    """Hash-only enrollment or bounded review-session credential."""

    __tablename__ = "reviewer_access_credentials"
    __table_args__ = (
        UniqueConstraint("credential_prefix", name="uq_reviewer_access_credentials_prefix"),
        UniqueConstraint("secret_hmac_sha256", name="uq_reviewer_access_credentials_secret"),
        CheckConstraint(
            "credential_kind IN ('enrollment_once', 'review_session')",
            name="ck_reviewer_access_credentials_kind",
        ),
        CheckConstraint(
            "status IN ('active', 'consumed', 'revoked')",
            name="ck_reviewer_access_credentials_status",
        ),
        CheckConstraint(
            "maximum_uses >= 1 AND maximum_uses <= 256 AND "
            "use_count >= 0 AND use_count <= maximum_uses",
            name="ck_reviewer_access_credentials_usage",
        ),
        CheckConstraint(
            "(credential_kind = 'enrollment_once' AND maximum_uses = 1) OR "
            "credential_kind = 'review_session'",
            name="ck_reviewer_access_credentials_one_time_enrollment",
        ),
        CheckConstraint(
            "expires_at > not_before",
            name="ck_reviewer_access_credentials_window",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    identity_binding_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_identity_bindings.id"), index=True
    )
    credential_prefix: Mapped[str] = mapped_column(String(32), index=True)
    secret_hmac_sha256: Mapped[str] = mapped_column(String(64))
    hmac_key_id: Mapped[str] = mapped_column(String(64))
    credential_kind: Mapped[str] = mapped_column(String(32))
    scopes_json: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    maximum_uses: Mapped[int] = mapped_column(Integer, default=1)
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    not_before: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerEnrollmentOffer(Base):
    """Identity-free, one-use credential for participant-owned consent."""

    __tablename__ = "reviewer_enrollment_offers"
    __table_args__ = (
        UniqueConstraint("credential_prefix", name="uq_reviewer_enrollment_offers_prefix"),
        UniqueConstraint("secret_hmac_sha256", name="uq_reviewer_enrollment_offers_secret_hmac"),
        CheckConstraint(
            "status IN ('active', 'accepted', 'revoked')",
            name="ck_reviewer_enrollment_offers_status",
        ),
        CheckConstraint(
            "expires_at > not_before",
            name="ck_reviewer_enrollment_offers_window",
        ),
        CheckConstraint(
            "(status = 'active' AND accepted_at IS NULL AND revoked_at IS NULL "
            "AND accepted_request_sha256 IS NULL) OR "
            "(status = 'accepted' AND accepted_at IS NOT NULL AND revoked_at IS NULL "
            "AND accepted_request_sha256 IS NOT NULL) OR "
            "(status = 'revoked' AND accepted_at IS NULL AND revoked_at IS NOT NULL "
            "AND accepted_request_sha256 IS NULL)",
            name="ck_reviewer_enrollment_offers_terminal_shape",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    credential_prefix: Mapped[str] = mapped_column(String(32), index=True)
    secret_hmac_sha256: Mapped[str] = mapped_column(String(64))
    hmac_key_id: Mapped[str] = mapped_column(String(64))
    consent_document_sha256: Mapped[str] = mapped_column(String(64), index=True)
    activation_manifest_sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    not_before: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    accepted_request_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerConsentAcceptance(Base):
    """Append-only consent receipt created by the participant, before identity."""

    __tablename__ = "reviewer_consent_acceptances"
    __table_args__ = (
        UniqueConstraint("enrollment_offer_id", name="uq_reviewer_consent_acceptances_offer"),
        UniqueConstraint("request_sha256", name="uq_reviewer_consent_acceptances_request"),
        UniqueConstraint("receipt_prefix", name="uq_reviewer_consent_acceptances_prefix"),
        UniqueConstraint(
            "receipt_secret_hmac_sha256",
            name="uq_reviewer_consent_acceptances_receipt_secret",
        ),
        UniqueConstraint("receipt_sha256", name="uq_reviewer_consent_acceptances_receipt"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    enrollment_offer_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_enrollment_offers.id"), index=True
    )
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    consent_document_sha256: Mapped[str] = mapped_column(String(64), index=True)
    activation_manifest_sha256: Mapped[str] = mapped_column(String(64), index=True)
    retention_policy_sha256: Mapped[str] = mapped_column(String(64))
    acceptance_statement_sha256: Mapped[str] = mapped_column(String(64))
    confirmation_set_sha256: Mapped[str] = mapped_column(String(64))
    request_sha256: Mapped[str] = mapped_column(String(64), index=True)
    receipt_prefix: Mapped[str] = mapped_column(String(32), index=True)
    receipt_secret_hmac_sha256: Mapped[str] = mapped_column(String(64))
    hmac_key_id: Mapped[str] = mapped_column(String(64))
    receipt_sha256: Mapped[str] = mapped_column(String(64), index=True)
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerParticipationLifecycle(Base):
    """Monotone participant state linked to one consent and one identity binding."""

    __tablename__ = "reviewer_participation_lifecycles"
    __table_args__ = (
        UniqueConstraint(
            "consent_acceptance_id", name="uq_reviewer_participation_lifecycles_acceptance"
        ),
        UniqueConstraint(
            "identity_binding_id", name="uq_reviewer_participation_lifecycles_binding"
        ),
        UniqueConstraint(
            "season_id", "reviewer_id", name="uq_reviewer_participation_lifecycles_reviewer"
        ),
        UniqueConstraint("audit_marker_sha256"),
        CheckConstraint(
            "status IN ('active', 'withdrawn', 'redacted')",
            name="ck_reviewer_participation_lifecycles_status",
        ),
        CheckConstraint(
            "(status = 'active' AND withdrawn_at IS NULL AND assignments_stopped_at IS NULL "
            "AND withdrawal_receipt_sha256 IS NULL AND redacted_at IS NULL "
            "AND deletion_receipt_sha256 IS NULL) OR "
            "(status = 'withdrawn' AND withdrawn_at IS NOT NULL "
            "AND assignments_stopped_at IS NOT NULL AND withdrawal_receipt_sha256 IS NOT NULL "
            "AND redacted_at IS NULL AND deletion_receipt_sha256 IS NULL) OR "
            "(status = 'redacted' AND withdrawn_at IS NOT NULL "
            "AND assignments_stopped_at IS NOT NULL AND withdrawal_receipt_sha256 IS NOT NULL "
            "AND redacted_at IS NOT NULL AND deletion_receipt_sha256 IS NOT NULL)",
            name="ck_reviewer_participation_lifecycles_terminal_shape",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    consent_acceptance_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_consent_acceptances.id"), index=True
    )
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    identity_binding_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_identity_bindings.id"), index=True
    )
    audit_marker_sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    assignments_stopped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    withdrawal_receipt_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    redacted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deletion_receipt_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerWithdrawalReceipt(Base):
    """Append-only proof of an authenticated, atomic withdrawal."""

    __tablename__ = "reviewer_withdrawal_receipts"
    __table_args__ = (
        UniqueConstraint("lifecycle_id", name="uq_reviewer_withdrawal_receipts_lifecycle"),
        UniqueConstraint("request_sha256", name="uq_reviewer_withdrawal_receipts_request"),
        UniqueConstraint("receipt_sha256", name="uq_reviewer_withdrawal_receipts_receipt"),
        CheckConstraint(
            "reason_code IN ('voluntary_withdrawal', 'privacy_request', 'safety_concern')",
            name="ck_reviewer_withdrawal_receipts_reason",
        ),
        CheckConstraint(
            "credentials_revoked_count >= 0 AND assignments_stopped_count >= 0",
            name="ck_reviewer_withdrawal_receipts_counts",
        ),
        CheckConstraint(
            "prior_judgments_preserved = true",
            name="ck_reviewer_withdrawal_receipts_history",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    lifecycle_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_participation_lifecycles.id"), index=True
    )
    consent_acceptance_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_consent_acceptances.id"), index=True
    )
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    identity_binding_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_identity_bindings.id"), index=True
    )
    request_sha256: Mapped[str] = mapped_column(String(64), index=True)
    reason_code: Mapped[str] = mapped_column(String(32))
    credentials_revoked_count: Mapped[int] = mapped_column(Integer)
    assignments_stopped_count: Mapped[int] = mapped_column(Integer)
    prior_judgments_preserved: Mapped[bool] = mapped_column(Boolean, default=True)
    receipt_sha256: Mapped[str] = mapped_column(String(64), index=True)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerRetentionSchedule(Base):
    """Immutable reviewer-specific deadlines derived from the approved policy."""

    __tablename__ = "reviewer_retention_schedules"
    __table_args__ = (
        UniqueConstraint("lifecycle_id", name="uq_reviewer_retention_schedules_lifecycle"),
        UniqueConstraint("schedule_sha256", name="uq_reviewer_retention_schedules_digest"),
        CheckConstraint(
            "direct_payload_delete_due_at > analysis_freeze_at",
            name="ck_reviewer_retention_schedules_direct_deadline",
        ),
        CheckConstraint(
            "pseudonymous_audit_retain_until > first_public_release_at",
            name="ck_reviewer_retention_schedules_audit_deadline",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    lifecycle_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_participation_lifecycles.id"), index=True
    )
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    analysis_freeze_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    first_public_release_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    direct_payload_delete_due_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    pseudonymous_audit_retain_until: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    retention_policy_sha256: Mapped[str] = mapped_column(String(64))
    schedule_sha256: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerDeletionReceipt(Base):
    """Append-only proof of bounded, one-way private-payload redaction."""

    __tablename__ = "reviewer_deletion_receipts"
    __table_args__ = (
        UniqueConstraint("lifecycle_id", name="uq_reviewer_deletion_receipts_lifecycle"),
        UniqueConstraint("request_sha256", name="uq_reviewer_deletion_receipts_request"),
        UniqueConstraint("receipt_sha256", name="uq_reviewer_deletion_receipts_receipt"),
        CheckConstraint(
            "execution_basis IN ('scheduled_retention', 'participant_request')",
            name="ck_reviewer_deletion_receipts_basis",
        ),
        CheckConstraint(
            "prior_judgments_preserved = true",
            name="ck_reviewer_deletion_receipts_history",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    lifecycle_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_participation_lifecycles.id"), index=True
    )
    retention_schedule_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_retention_schedules.id"), index=True
    )
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    request_sha256: Mapped[str] = mapped_column(String(64), index=True)
    execution_basis: Mapped[str] = mapped_column(String(32))
    redacted_fields_json: Mapped[list] = mapped_column(JSON)
    private_payload_before_sha256: Mapped[str] = mapped_column(String(64))
    audit_marker_sha256: Mapped[str] = mapped_column(String(64))
    direct_payload_delete_due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    pseudonymous_audit_retain_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    prior_judgments_preserved: Mapped[bool] = mapped_column(Boolean, default=True)
    receipt_sha256: Mapped[str] = mapped_column(String(64), index=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerQualificationEvidence(Base):
    """Immutable, privacy-minimized evidence for one reviewer and task family."""

    __tablename__ = "reviewer_qualification_evidence"
    __table_args__ = (
        UniqueConstraint(
            "identity_binding_id",
            "family",
            "qualification_evidence_sha256",
            name="uq_reviewer_qualification_evidence_binding_family_evidence",
        ),
        CheckConstraint(
            "family IN ('substitution', 'composition', 'cookability', 'evidence')",
            name="ck_reviewer_qualification_evidence_family",
        ),
        CheckConstraint(
            "affiliation_class IN ('independent_external', 'product_affiliated', "
            "'provider_affiliated')",
            name="ck_reviewer_qualification_evidence_affiliation",
        ),
        CheckConstraint(
            "verification_status = 'verified'",
            name="ck_reviewer_qualification_evidence_verified",
        ),
        CheckConstraint(
            "valid_until IS NULL OR valid_until > verified_at",
            name="ck_reviewer_qualification_evidence_window",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    identity_binding_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_identity_bindings.id"), index=True
    )
    family: Mapped[str] = mapped_column(String(64), index=True)
    affiliation_class: Mapped[str] = mapped_column(String(40), index=True)
    independence_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    conflict_cleared: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_status: Mapped[str] = mapped_column(String(24), default="verified")
    qualification_evidence_sha256: Mapped[str] = mapped_column(String(64))
    independence_evidence_sha256: Mapped[str] = mapped_column(String(64))
    conflict_disclosure_sha256: Mapped[str] = mapped_column(String(64))
    consent_document_sha256: Mapped[str] = mapped_column(String(64))
    training_material_sha256: Mapped[str] = mapped_column(String(64))
    verifier_principal_sha256: Mapped[str] = mapped_column(String(64))
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerCalibrationSet(Base):
    """Frozen real-output calibration set metadata; response content lives elsewhere."""

    __tablename__ = "reviewer_calibration_sets"
    __table_args__ = (
        UniqueConstraint(
            "season_id",
            "family",
            "calibration_set_sha256",
            name="uq_reviewer_calibration_sets_season_family_hash",
        ),
        CheckConstraint(
            "family IN ('substitution', 'composition', 'cookability', 'evidence')",
            name="ck_reviewer_calibration_sets_family",
        ),
        CheckConstraint(
            "item_count >= 1 AND real_source_arms >= item_count * 2 AND synthetic_arms = 0",
            name="ck_reviewer_calibration_sets_real_outputs",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    family: Mapped[str] = mapped_column(String(64), index=True)
    calibration_set_sha256: Mapped[str] = mapped_column(String(64), index=True)
    source_artifact_sha256: Mapped[str] = mapped_column(String(64))
    scoring_key_sha256: Mapped[str] = mapped_column(String(64))
    item_count: Mapped[int] = mapped_column(Integer)
    real_source_arms: Mapped[int] = mapped_column(Integer)
    synthetic_arms: Mapped[int] = mapped_column(Integer, default=0)
    frozen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerCalibrationBallot(Base):
    """Content-addressed calibration result for one bound person."""

    __tablename__ = "reviewer_calibration_ballots"
    __table_args__ = (
        UniqueConstraint(
            "calibration_set_id",
            "identity_binding_id",
            name="uq_reviewer_calibration_ballots_set_binding",
        ),
        CheckConstraint(
            "item_count >= 1 AND correct_count >= 0 AND correct_count <= item_count "
            "AND accuracy_milli >= 0 AND accuracy_milli <= 1000",
            name="ck_reviewer_calibration_ballots_score",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    identity_binding_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_identity_bindings.id"), index=True
    )
    calibration_set_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_calibration_sets.id"), index=True
    )
    ballot_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    scoring_result_sha256: Mapped[str] = mapped_column(String(64))
    item_count: Mapped[int] = mapped_column(Integer)
    correct_count: Mapped[int] = mapped_column(Integer)
    accuracy_milli: Mapped[int] = mapped_column(Integer)
    passed: Mapped[bool] = mapped_column(Boolean)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewerFamilyAdmission(Base):
    """Immutable, server-derived authorization for one family and review role."""

    __tablename__ = "reviewer_family_admissions"
    __table_args__ = (
        UniqueConstraint(
            "identity_binding_id",
            "family",
            "review_role",
            "admission_policy_sha256",
            name="uq_reviewer_family_admissions_binding_family_role_policy",
        ),
        CheckConstraint(
            "family IN ('substitution', 'composition', 'cookability', 'evidence')",
            name="ck_reviewer_family_admissions_family",
        ),
        CheckConstraint(
            "review_role IN ('task_author', 'task_validator', 'task_adjudicator', 'output_rater')",
            name="ck_reviewer_family_admissions_role",
        ),
        CheckConstraint(
            "cohort IN ('expert_independent', 'expert_product_affiliated', "
            "'expert_provider_affiliated')",
            name="ck_reviewer_family_admissions_cohort",
        ),
        CheckConstraint(
            "valid_until > valid_from",
            name="ck_reviewer_family_admissions_window",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    identity_binding_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_identity_bindings.id"), index=True
    )
    family: Mapped[str] = mapped_column(String(64), index=True)
    review_role: Mapped[str] = mapped_column(String(32), index=True)
    cohort: Mapped[str] = mapped_column(String(48), index=True)
    qualification_evidence_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_qualification_evidence.id"), index=True
    )
    calibration_ballot_id: Mapped[str | None] = mapped_column(
        ForeignKey("reviewer_calibration_ballots.id"), nullable=True, index=True
    )
    admission_policy_json: Mapped[dict] = mapped_column(JSON)
    admission_policy_sha256: Mapped[str] = mapped_column(String(64))
    evidence_bundle_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    decision_reference_sha256: Mapped[str] = mapped_column(String(64))
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TaskValidationAuditAuthorization(Base):
    """Immutable authorization for one independent campaign-level auditor."""

    __tablename__ = "task_validation_audit_authorizations"
    __table_args__ = (
        UniqueConstraint(
            "campaign_sha256",
            "audit_kind",
            name="uq_task_validation_audit_authorizations_campaign_kind",
        ),
        UniqueConstraint(
            "campaign_sha256",
            "identity_binding_id",
            name="uq_task_validation_audit_authorizations_campaign_person",
        ),
        UniqueConstraint(
            "authorization_sha256",
            name="uq_task_validation_audit_authorizations_digest",
        ),
        CheckConstraint(
            "audit_kind IN ('rights', 'contamination')",
            name="ck_task_validation_audit_authorizations_kind",
        ),
        CheckConstraint(
            "cohort = 'expert_independent'",
            name="ck_task_validation_audit_authorizations_cohort",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    campaign_sha256: Mapped[str] = mapped_column(String(64), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    identity_binding_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_identity_bindings.id"), index=True
    )
    audit_kind: Mapped[str] = mapped_column(String(32), index=True)
    cohort: Mapped[str] = mapped_column(String(48), default="expert_independent")
    qualification_evidence_sha256: Mapped[str] = mapped_column(String(64))
    conflict_evidence_sha256: Mapped[str] = mapped_column(String(64))
    automated_evidence_sha256: Mapped[str] = mapped_column(String(64))
    audit_plan_json: Mapped[dict] = mapped_column(JSON)
    audit_plan_sha256: Mapped[str] = mapped_column(String(64))
    decision_reference_sha256: Mapped[str] = mapped_column(String(64))
    authorization_sha256: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TaskValidationCampaignEvent(Base):
    """One append-only, hash-chained event in the v6 task campaign."""

    __tablename__ = "task_validation_campaign_events"
    __table_args__ = (
        Index(
            "uq_task_validation_campaign_events_audit_authorization_type",
            "campaign_sha256",
            "audit_authorization_id",
            "event_type",
            unique=True,
            postgresql_where=sql_text("audit_authorization_id IS NOT NULL"),
            sqlite_where=sql_text("audit_authorization_id IS NOT NULL"),
        ),
        UniqueConstraint(
            "campaign_sha256",
            "sequence",
            name="uq_task_validation_campaign_events_sequence",
        ),
        UniqueConstraint(
            "campaign_sha256",
            "event_id",
            name="uq_task_validation_campaign_events_event_id",
        ),
        UniqueConstraint(
            "event_sha256",
            name="uq_task_validation_campaign_events_digest",
        ),
        UniqueConstraint(
            "campaign_sha256",
            "candidate_id",
            "event_type",
            "identity_binding_id",
            name="uq_task_validation_campaign_events_candidate_person_type",
        ),
        CheckConstraint(
            "event_type IN ('blind_ballot', 'criterion_pack_confirmation', "
            "'adjudication', 'rights_batch_audit', 'contamination_batch_audit')",
            name="ck_task_validation_campaign_events_type",
        ),
        CheckConstraint(
            "sequence >= 1",
            name="ck_task_validation_campaign_events_sequence",
        ),
        CheckConstraint(
            "((event_type IN ('blind_ballot', 'criterion_pack_confirmation', "
            "'adjudication') AND candidate_id IS NOT NULL "
            "AND family_admission_id IS NOT NULL AND audit_authorization_id IS NULL) OR "
            "(event_type IN ('rights_batch_audit', 'contamination_batch_audit') "
            "AND candidate_id IS NULL AND family_admission_id IS NULL "
            "AND audit_authorization_id IS NOT NULL))",
            name="ck_task_validation_campaign_events_authorization_shape",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    campaign_sha256: Mapped[str] = mapped_column(String(64), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    event_id: Mapped[str] = mapped_column(String(160))
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    candidate_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    identity_binding_id: Mapped[str] = mapped_column(
        ForeignKey("reviewer_identity_bindings.id"), index=True
    )
    family_admission_id: Mapped[str | None] = mapped_column(
        ForeignKey("reviewer_family_admissions.id"), nullable=True, index=True
    )
    audit_authorization_id: Mapped[str | None] = mapped_column(
        ForeignKey("task_validation_audit_authorizations.id"), nullable=True, index=True
    )
    reviewer_pseudonym: Mapped[str] = mapped_column(String(120), index=True)
    person_commitment_sha256: Mapped[str] = mapped_column(String(64))
    reviewer_admission_receipt_sha256: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[dict] = mapped_column(JSON)
    previous_event_sha256: Mapped[str] = mapped_column(String(64))
    event_sha256: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ControlledRunReviewer(Base):
    """Explicit reviewer authorization for one private controlled run."""

    __tablename__ = "controlled_run_reviewers"
    __table_args__ = (UniqueConstraint("controlled_run_id", "reviewer_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    controlled_run_id: Mapped[str] = mapped_column(ForeignKey("controlled_runs.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("expert_reviewers.id"), index=True)
    authorization_reference_sha256: Mapped[str] = mapped_column(String(64))
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AdmissionEvent(Base):
    __tablename__ = "admission_events"
    __table_args__ = (Index("ix_admission_lookup", "pseudonym", "action", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    pseudonym: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    admitted: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CostEvent(Base):
    __tablename__ = "cost_events"
    __table_args__ = (
        Index(
            "uq_cost_events_battle_reconcile",
            "battle_id",
            unique=True,
            postgresql_where=sql_text("kind = 'reconcile'"),
            sqlite_where=sql_text("kind = 'reconcile'"),
        ),
        Index(
            "uq_cost_events_battle_provider_reserve",
            "battle_id",
            "provider",
            unique=True,
            postgresql_where=sql_text("kind = 'provider_reserve'"),
            sqlite_where=sql_text("kind = 'provider_reserve'"),
        ),
        Index(
            "uq_cost_events_battle_provider_account_reserve",
            "battle_id",
            "provider",
            unique=True,
            postgresql_where=sql_text("kind = 'provider_account_reserve'"),
            sqlite_where=sql_text("kind = 'provider_account_reserve'"),
        ),
        Index(
            "uq_cost_events_battle_governor_reserve",
            "battle_id",
            unique=True,
            postgresql_where=sql_text("kind = 'reserve'"),
            sqlite_where=sql_text("kind = 'reserve'"),
        ),
        Index(
            "uq_cost_events_battle_governor_release",
            "battle_id",
            unique=True,
            postgresql_where=sql_text("kind = 'release'"),
            sqlite_where=sql_text("kind = 'release'"),
        ),
        Index(
            "uq_cost_events_battle_provider_release",
            "battle_id",
            "provider",
            unique=True,
            postgresql_where=sql_text("kind = 'provider_release'"),
            sqlite_where=sql_text("kind = 'provider_release'"),
        ),
        Index(
            "uq_cost_events_battle_provider_reconcile",
            "battle_id",
            "provider",
            unique=True,
            postgresql_where=sql_text("kind = 'provider_reconcile'"),
            sqlite_where=sql_text("kind = 'provider_reconcile'"),
        ),
        Index(
            "uq_cost_events_battle_provider_account_release",
            "battle_id",
            "provider",
            unique=True,
            postgresql_where=sql_text("kind = 'provider_account_release'"),
            sqlite_where=sql_text("kind = 'provider_account_release'"),
        ),
        Index(
            "uq_cost_events_battle_provider_account_reconcile",
            "battle_id",
            "provider",
            unique=True,
            postgresql_where=sql_text("kind = 'provider_account_reconcile'"),
            sqlite_where=sql_text("kind = 'provider_account_reconcile'"),
        ),
        Index(
            "uq_cost_events_arm_actual",
            "arm_id",
            unique=True,
            postgresql_where=sql_text("kind = 'actual'"),
            sqlite_where=sql_text("kind = 'actual'"),
        ),
        Index(
            "uq_cost_events_arm_actual_settlement",
            "arm_id",
            unique=True,
            postgresql_where=sql_text("kind = 'actual_settlement'"),
            sqlite_where=sql_text("kind = 'actual_settlement'"),
        ),
        CheckConstraint(
            "kind NOT IN ('reserve', 'reconcile', 'provider_reserve', "
            "'provider_reconcile', 'provider_account_reserve', "
            "'provider_account_reconcile', 'actual', 'actual_settlement') "
            "OR amount_micros >= 0",
            name="ck_cost_events_nonnegative_charge",
        ),
        CheckConstraint(
            "kind NOT IN ('release', 'provider_release', "
            "'provider_account_release') OR amount_micros <= 0",
            name="ck_cost_events_nonpositive_release",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    battle_id: Mapped[str | None] = mapped_column(ForeignKey("battles.id"), nullable=True)
    arm_id: Mapped[str | None] = mapped_column(ForeignKey("response_arms.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))
    amount_micros: Mapped[int] = mapped_column(BigInteger)
    provider: Mapped[str] = mapped_column(String(80), default="openrouter")
    generation_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    accounting_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BedrockBillingCrosscheck(Base):
    """Immutable aggregate AWS billing evidence for an exact set of arms."""

    __tablename__ = "bedrock_billing_crosschecks"
    __table_args__ = (
        UniqueConstraint("evidence_sha256"),
        UniqueConstraint("source_artifact_sha256", "statement_sha256"),
        CheckConstraint(
            "status IN ('accepted', 'discrepant')",
            name="ck_bedrock_billing_crosschecks_status",
        ),
        CheckConstraint(
            "supersedes_crosscheck_id IS NULL OR supersedes_crosscheck_id <> id",
            name="ck_bedrock_billing_crosschecks_not_self_superseding",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    provider_account_budget_id: Mapped[str] = mapped_column(
        ForeignKey("provider_account_budgets.id"), index=True
    )
    status: Mapped[str] = mapped_column(String(24), default="accepted", index=True)
    supersedes_crosscheck_id: Mapped[str | None] = mapped_column(
        ForeignKey("bedrock_billing_crosschecks.id"), nullable=True, unique=True, index=True
    )
    source_kind: Mapped[str] = mapped_column(String(48))
    source_artifact_uri: Mapped[str] = mapped_column(Text)
    source_artifact_sha256: Mapped[str] = mapped_column(String(64), index=True)
    statement_sha256: Mapped[str] = mapped_column(String(64))
    coverage_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    coverage_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    arm_set_sha256: Mapped[str] = mapped_column(String(64), index=True)
    generation_request_map_sha256: Mapped[str] = mapped_column(String(64))
    rate_card_estimated_micros: Mapped[int] = mapped_column(BigInteger)
    billed_usage_micros: Mapped[int] = mapped_column(BigInteger)
    billing_difference_micros: Mapped[int] = mapped_column(BigInteger)
    ledger_delta_micros: Mapped[int] = mapped_column(BigInteger)
    tolerance_micros: Mapped[int] = mapped_column(BigInteger)
    credits_policy: Mapped[str] = mapped_column(String(80))
    authorization_reference_sha256: Mapped[str] = mapped_column(String(64))
    evidence_json: Mapped[dict] = mapped_column(JSON)
    evidence_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class BedrockBillingCrosscheckArm(Base):
    """Immutable membership binding one response arm to one billing crosscheck."""

    __tablename__ = "bedrock_billing_crosscheck_arms"
    __table_args__ = (UniqueConstraint("crosscheck_id", "arm_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    crosscheck_id: Mapped[str] = mapped_column(
        ForeignKey("bedrock_billing_crosschecks.id"), index=True
    )
    arm_id: Mapped[str] = mapped_column(ForeignKey("response_arms.id"), index=True)
    generation_set_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    entity_type: Mapped[str] = mapped_column(String(48), index=True)
    entity_id: Mapped[str] = mapped_column(String(80), index=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    severity: Mapped[str] = mapped_column(String(24), index=True)
    code: Mapped[str] = mapped_column(String(80), index=True)
    detail: Mapped[str] = mapped_column(Text)
    battle_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LeaderboardSnapshot(Base):
    __tablename__ = "leaderboard_snapshots"
    __table_args__ = (
        CheckConstraint(
            "publication_status IN ('draft', 'published', 'withdrawn')",
            name="ck_leaderboard_snapshots_publication_status",
        ),
        Index(
            "uq_leaderboard_snapshots_one_published_public_scope",
            "season_id",
            "track",
            "cohort",
            "category",
            "data_stratum",
            unique=True,
            postgresql_where=sql_text(
                "publication_status = 'published' AND controlled_run_id IS NULL"
            ),
            sqlite_where=sql_text("publication_status = 'published' AND controlled_run_id IS NULL"),
        ),
        Index(
            "uq_leaderboard_snapshots_one_published_controlled_scope",
            "controlled_run_id",
            "track",
            "cohort",
            "category",
            "data_stratum",
            unique=True,
            postgresql_where=sql_text(
                "publication_status = 'published' AND controlled_run_id IS NOT NULL"
            ),
            sqlite_where=sql_text(
                "publication_status = 'published' AND controlled_run_id IS NOT NULL"
            ),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    track: Mapped[str] = mapped_column(String(32), index=True)
    cohort: Mapped[str] = mapped_column(String(48), index=True)
    category: Mapped[str] = mapped_column(String(64), default="all")
    data_stratum: Mapped[str] = mapped_column(String(32), default="legacy_mixed", index=True)
    controlled_run_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "controlled_runs.id",
            name="fk_leaderboard_snapshots_controlled_run_id_controlled_runs",
        ),
        nullable=True,
        index=True,
    )
    publication_status: Mapped[str] = mapped_column(String(24), default="draft", index=True)
    publication_reference_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    input_sha256: Mapped[str] = mapped_column(String(64))
    input_evidence_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_evidence_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    payload_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    supersedes_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("leaderboard_snapshots.id"), nullable=True, index=True
    )
    payload_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ResearchReleaseArchive(Base):
    """Immutable metadata for one reproducible, externally verifiable release."""

    __tablename__ = "research_release_archives"
    __table_args__ = (
        CheckConstraint(
            "archive_class IN ('internal_official', 'sanitized_public')",
            name="ck_research_release_archives_class",
        ),
        CheckConstraint(
            "signature_algorithm = 'Ed25519'",
            name="ck_research_release_archives_signature_algorithm",
        ),
        UniqueConstraint("snapshot_set_sha256"),
        UniqueConstraint("manifest_sha256"),
        UniqueConstraint("archive_sha256"),
        UniqueConstraint("supersedes_archive_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    season_id: Mapped[str] = mapped_column(ForeignKey("seasons.id"), index=True)
    archive_class: Mapped[str] = mapped_column(String(32), index=True)
    schema_version: Mapped[str] = mapped_column(String(96))
    snapshot_ids_json: Mapped[list] = mapped_column(JSON)
    snapshot_set_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    manifest_json: Mapped[dict] = mapped_column(JSON)
    manifest_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    archive_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    storage_object_key: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    member_count: Mapped[int] = mapped_column(Integer)
    source_date_epoch: Mapped[int] = mapped_column(BigInteger, default=0)
    requirements_lock_sha256: Mapped[str] = mapped_column(String(64))
    build_image_digest: Mapped[str] = mapped_column(String(71))
    signature_algorithm: Mapped[str] = mapped_column(String(32), default="Ed25519")
    signing_key_id: Mapped[str] = mapped_column(String(160))
    public_key_pem: Mapped[str] = mapped_column(Text)
    public_key_sha256: Mapped[str] = mapped_column(String(64))
    signature_base64: Mapped[str] = mapped_column(Text)
    privacy_review_artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    supersedes_archive_id: Mapped[str | None] = mapped_column(
        ForeignKey("research_release_archives.id"), nullable=True, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


_IMMUTABLE_BATTLE_PROVENANCE_FIELDS = (
    "id",
    "season_id",
    "run_class",
    "rank_eligible",
    "data_stratum",
    "task_id",
    "task_revision",
    "controlled_run_id",
    "manifest_sha256",
    "protocol_bundle_sha256",
    "scheduler_version",
    "assignment_seed",
    "track_assignment_probability",
    "model_assignment_probability",
    "side_assignment_probability",
    "provider_reservations_json",
    "track",
    "category",
    "prompt_sha256",
    "client_nonce_sha256",
    "research_consent",
    "retention_basis",
    "requester_pseudonym",
    "created_at",
)

_IMMUTABLE_RESPONSE_ARM_CONTRACT_FIELDS = (
    "id",
    "battle_id",
    "side",
    "condition",
    "model_id",
    "execution_backend",
    "route_revision_id",
    "endpoint_descriptor_sha256",
    "provider_slug",
    "prompt_sha256",
    "system_prompt_sha256",
    "schema_sha256",
    "tool_schema_sha256",
    "decoding_json",
    "protocol_bundle_sha256",
    "epicure_release_id",
    "epicure_bundle_sha256",
    "epicure_application_sha256",
)

_TERMINAL_RESPONSE_ARM_STATUSES = frozenset({"complete", "failed", "uncertain"})
_NORMAL_RESPONSE_FINISH_REASONS = frozenset({"completed", "end_turn", "stop", "stop_sequence"})
_TERMINAL_RESPONSE_ARM_EVIDENCE_FIELDS = (
    "actual_provider_slug",
    "actual_model_id",
    "generation_id",
    "provider_generation_ids_json",
    "observed_decoding_json",
    "epicure_attestation_json",
    "epicure_attestation_sha256",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "backend_response_schema_sha256",
    "backend_tool_schema_sha256",
    "latency_ms",
    "retries",
    "finish_reason",
    "created_at",
)
_TERMINAL_RESPONSE_ARM_SETTLEMENT_FIELDS = (
    "status",
    "cost_micros",
    "cost_reconciled",
    "cost_accounting_basis",
    "billing_reconciliation_status",
    "error_code",
    "error_detail",
    "completed_at",
)

_IMMUTABLE_SEASON_MODEL_CONTRACT_FIELDS = (
    "model_id",
    "slot_role",
    "execution_backend",
    "provider_slug",
    "expected_actual_model_id",
    "expected_actual_provider_slug",
    "supported_parameters_json",
    "decoding_json",
    "endpoint_max_completion_tokens",
    "endpoint_document_sha256",
    "endpoint_contract_sha256",
    "backend_contract_json",
    "backend_contract_sha256",
    "rate_card_json",
    "rate_card_sha256",
    "worst_case_cost_micros",
    "manifest_sha256",
    "eligible",
)

_IMMUTABLE_FROZEN_TASK_FIELDS = (
    "public_id",
    "season_id",
    "family",
    "prompt",
    "prompt_sha256",
    "revision",
    "split",
    "review_status",
    "provenance_json",
)

_IMMUTABLE_FROZEN_SEASON_FIELDS = (
    "manifest_sha256",
    "prompt_registry_sha256",
    "tool_registry_sha256",
    "epicure_release_id",
    "epicure_bundle_sha256",
    "epicure_application_sha256",
    "analysis_plan_sha256",
    "protocol_bundle_json",
    "protocol_bundle_sha256",
    "frozen_at",
)

_IMMUTABLE_LEADERBOARD_SNAPSHOT_FIELDS = (
    "season_id",
    "track",
    "cohort",
    "category",
    "data_stratum",
    "controlled_run_id",
    "input_sha256",
    "input_evidence_sha256",
    "input_evidence_json",
    "payload_sha256",
    "evidence_cutoff_at",
    "supersedes_snapshot_id",
    "payload_json",
    "created_at",
)

_IMMUTABLE_CONTROLLED_RUN_CONTRACT_FIELDS = (
    "id",
    "season_id",
    "organization_id",
    "evaluation_order_id",
    "route_revision_id",
    "endpoint_descriptor_sha256",
    "spend_authorization_id",
    "spend_authorization_binding_sha256",
    "organization_reference_sha256",
    "protocol_version",
    "rater_plan_sha256",
    "analysis_plan_sha256",
    "submitted_endpoint_model_id",
    "submitted_model_card_sha256",
    "data_policy_sha256",
    "model_roster_json",
    "model_roster_sha256",
    "task_schedule_sha256",
    "budget_cap_micros",
    "run_card_json",
    "run_card_sha256",
    "run_card_signature",
    "created_at",
)

_IMMUTABLE_CONTROLLED_ASSIGNMENT_FIELDS = (
    "id",
    "controlled_run_id",
    "ordinal",
    "task_id",
    "task_public_id",
    "task_revision",
    "task_prompt_sha256",
    "task_family",
    "track",
    "model_ids_json",
    "repetition_index",
    "assignment_sha256",
    "assignment_seed",
    "created_at",
)

_IMMUTABLE_PROVIDER_BUDGET_CONTRACT_FIELDS = (
    "season_id",
    "execution_backend",
    "currency",
    "budget_cap_micros",
    "account_scope_sha256",
    "authorization_reference_sha256",
    "account_authorization_envelope_sha256",
    "authorization_envelope_json",
    "authorization_envelope_sha256",
    "valid_until",
)

_IMMUTABLE_ACCOUNT_BUDGET_CONTRACT_FIELDS = (
    "execution_backend",
    "currency",
    "budget_cap_micros",
    "opening_used_micros",
    "opening_reserved_micros",
    "account_scope_sha256",
    "authorization_reference_sha256",
    "opening_balance_json",
    "opening_balance_sha256",
    "credential_binding_json",
    "credential_binding_sha256",
    "authorization_envelope_json",
    "authorization_envelope_sha256",
    "authorization_hmac_sha256",
    "valid_until",
)

_IMMUTABLE_ACCOUNT_AUTHORIZATION_FIELDS = (
    "provider_account_budget_id",
    "execution_backend",
    "account_scope_sha256",
    "supersedes_authorization_id",
    "authorization_reference_sha256",
    "exposure_attestation_json",
    "exposure_attestation_sha256",
    "authorized_used_micros",
    "authorized_reserved_micros",
    "credential_binding_json",
    "credential_binding_sha256",
    "authorization_envelope_json",
    "authorization_envelope_sha256",
    "authorization_hmac_sha256",
    "valid_until",
)

_IMMUTABLE_ORGANIZATION_FIELDS = (
    "id",
    "slug",
    "legal_name",
    "display_name",
    "idp_tenant_reference_sha256",
    "billing_reference_sha256",
    "data_region",
    "retention_policy_json",
    "retention_policy_sha256",
    "created_at",
)

_IMMUTABLE_ORGANIZATION_API_KEY_FIELDS = (
    "id",
    "organization_id",
    "key_prefix",
    "secret_hmac_sha256",
    "hmac_key_id",
    "label",
    "scopes_json",
    "rate_limit_profile",
    "network_policy_json",
    "created_by_principal_ref_sha256",
    "not_before",
    "expires_at",
    "supersedes_key_id",
    "created_at",
)

_IMMUTABLE_MODEL_SUBMISSION_FIELDS = (
    "id",
    "organization_id",
    "revision",
    "supersedes_submission_id",
    "display_name",
    "publisher",
    "requested_canonical_model_id",
    "exact_model_version",
    "release_date",
    "model_card_uri",
    "model_card_sha256",
    "license_uri",
    "license_document_sha256",
    "capability_claims_json",
    "capability_claims_sha256",
    "contamination_disclosure_json",
    "contamination_disclosure_sha256",
    "submission_payload_json",
    "submission_payload_sha256",
    "submitted_by_key_id",
    "created_at",
)

_IMMUTABLE_MODEL_ROUTE_FIELDS = (
    "id",
    "model_submission_id",
    "revision",
    "route_kind",
    "execution_backend",
    "managed_route_reference_sha256",
    "requested_model_id",
    "expected_actual_model_id",
    "expected_actual_provider_slug",
    "supported_parameters_json",
    "supported_parameters_sha256",
    "decoding_bounds_json",
    "decoding_bounds_sha256",
    "endpoint_document_sha256",
    "data_policy_json",
    "data_policy_sha256",
    "rate_card_json",
    "rate_card_sha256",
    "descriptor_json",
    "descriptor_sha256",
    "created_at",
)

_IMMUTABLE_EVALUATION_ORDER_FIELDS = (
    "id",
    "organization_id",
    "model_submission_id",
    "route_revision_id",
    "season_id",
    "evaluation_profile_id",
    "requested_visibility",
    "comparison_plan_json",
    "comparison_plan_sha256",
    "rater_plan_sha256",
    "analysis_plan_sha256",
    "forecast_cost_micros",
    "budget_cap_micros",
    "currency",
    "client_reference_sha256",
    "order_card_json",
    "order_card_sha256",
    "order_card_signature",
    "submitted_by_key_id",
    "created_at",
)

_IMMUTABLE_EVIDENCE_BUNDLE_FIELDS = (
    "id",
    "organization_id",
    "evaluation_order_id",
    "bundle_class",
    "schema_version",
    "manifest_json",
    "manifest_sha256",
    "publication_authorization_id",
    "supersedes_bundle_id",
    "created_at",
)

_REVIEWER_ROLES = frozenset({"task_author", "task_validator", "task_adjudicator", "output_rater"})
_REVIEWER_FAMILIES = frozenset({"substitution", "composition", "cookability", "evidence"})
_REVIEWER_COHORT_BY_AFFILIATION = {
    "independent_external": "expert_independent",
    "product_affiliated": "expert_product_affiliated",
    "provider_affiliated": "expert_provider_affiliated",
}

_IMMUTABLE_REVIEWER_IDENTITY_BINDING_FIELDS = (
    "id",
    "season_id",
    "reviewer_id",
    "person_commitment_sha256",
    "identity_issuer_sha256",
    "identity_evidence_sha256",
    "hmac_key_id",
    "verification_method",
    "assurance_level",
    "roles_json",
    "created_at",
)

_IMMUTABLE_REVIEWER_ACCESS_CREDENTIAL_FIELDS = (
    "id",
    "season_id",
    "reviewer_id",
    "identity_binding_id",
    "credential_prefix",
    "secret_hmac_sha256",
    "hmac_key_id",
    "credential_kind",
    "scopes_json",
    "maximum_uses",
    "not_before",
    "expires_at",
    "created_at",
)

_IMMUTABLE_REVIEWER_ENROLLMENT_OFFER_FIELDS = (
    "id",
    "season_id",
    "credential_prefix",
    "secret_hmac_sha256",
    "hmac_key_id",
    "consent_document_sha256",
    "activation_manifest_sha256",
    "not_before",
    "expires_at",
    "created_at",
)

_IMMUTABLE_REVIEWER_CONSENT_ACCEPTANCE_FIELDS = (
    "id",
    "enrollment_offer_id",
    "season_id",
    "consent_document_sha256",
    "activation_manifest_sha256",
    "retention_policy_sha256",
    "acceptance_statement_sha256",
    "confirmation_set_sha256",
    "request_sha256",
    "receipt_prefix",
    "receipt_secret_hmac_sha256",
    "hmac_key_id",
    "receipt_sha256",
    "accepted_at",
    "created_at",
)

_IMMUTABLE_REVIEWER_PARTICIPATION_FIELDS = (
    "id",
    "consent_acceptance_id",
    "season_id",
    "reviewer_id",
    "identity_binding_id",
    "audit_marker_sha256",
    "created_at",
)

_IMMUTABLE_REVIEWER_WITHDRAWAL_RECEIPT_FIELDS = (
    "id",
    "lifecycle_id",
    "consent_acceptance_id",
    "season_id",
    "reviewer_id",
    "identity_binding_id",
    "request_sha256",
    "reason_code",
    "credentials_revoked_count",
    "assignments_stopped_count",
    "prior_judgments_preserved",
    "receipt_sha256",
    "effective_at",
    "created_at",
)

_IMMUTABLE_REVIEWER_RETENTION_SCHEDULE_FIELDS = (
    "id",
    "lifecycle_id",
    "season_id",
    "reviewer_id",
    "analysis_freeze_at",
    "first_public_release_at",
    "direct_payload_delete_due_at",
    "pseudonymous_audit_retain_until",
    "retention_policy_sha256",
    "schedule_sha256",
    "created_at",
)

_IMMUTABLE_REVIEWER_DELETION_RECEIPT_FIELDS = (
    "id",
    "lifecycle_id",
    "retention_schedule_id",
    "season_id",
    "reviewer_id",
    "request_sha256",
    "execution_basis",
    "redacted_fields_json",
    "private_payload_before_sha256",
    "audit_marker_sha256",
    "direct_payload_delete_due_at",
    "pseudonymous_audit_retain_until",
    "prior_judgments_preserved",
    "receipt_sha256",
    "executed_at",
    "created_at",
)

_IMMUTABLE_REVIEWER_QUALIFICATION_FIELDS = (
    "id",
    "season_id",
    "reviewer_id",
    "identity_binding_id",
    "family",
    "affiliation_class",
    "independence_verified",
    "conflict_cleared",
    "verification_status",
    "qualification_evidence_sha256",
    "independence_evidence_sha256",
    "conflict_disclosure_sha256",
    "consent_document_sha256",
    "training_material_sha256",
    "verifier_principal_sha256",
    "verified_at",
    "valid_until",
    "created_at",
)

_IMMUTABLE_REVIEWER_CALIBRATION_SET_FIELDS = (
    "id",
    "season_id",
    "family",
    "calibration_set_sha256",
    "source_artifact_sha256",
    "scoring_key_sha256",
    "item_count",
    "real_source_arms",
    "synthetic_arms",
    "frozen_at",
    "created_at",
)

_IMMUTABLE_REVIEWER_CALIBRATION_BALLOT_FIELDS = (
    "id",
    "season_id",
    "reviewer_id",
    "identity_binding_id",
    "calibration_set_id",
    "ballot_sha256",
    "scoring_result_sha256",
    "item_count",
    "correct_count",
    "accuracy_milli",
    "passed",
    "completed_at",
    "created_at",
)

_IMMUTABLE_REVIEWER_FAMILY_ADMISSION_FIELDS = (
    "id",
    "season_id",
    "reviewer_id",
    "identity_binding_id",
    "family",
    "review_role",
    "cohort",
    "qualification_evidence_id",
    "calibration_ballot_id",
    "admission_policy_json",
    "admission_policy_sha256",
    "evidence_bundle_sha256",
    "decision_reference_sha256",
    "valid_from",
    "valid_until",
    "created_at",
)

_IMMUTABLE_TASK_VALIDATION_AUDIT_AUTHORIZATION_FIELDS = (
    "id",
    "season_id",
    "campaign_sha256",
    "reviewer_id",
    "identity_binding_id",
    "audit_kind",
    "cohort",
    "qualification_evidence_sha256",
    "conflict_evidence_sha256",
    "automated_evidence_sha256",
    "audit_plan_json",
    "audit_plan_sha256",
    "decision_reference_sha256",
    "authorization_sha256",
    "created_at",
)

_IMMUTABLE_TASK_VALIDATION_CAMPAIGN_EVENT_FIELDS = (
    "id",
    "season_id",
    "campaign_sha256",
    "sequence",
    "event_id",
    "event_type",
    "candidate_id",
    "reviewer_id",
    "identity_binding_id",
    "family_admission_id",
    "audit_authorization_id",
    "reviewer_pseudonym",
    "person_commitment_sha256",
    "reviewer_admission_receipt_sha256",
    "payload_json",
    "previous_event_sha256",
    "event_sha256",
    "created_at",
)


def _verified_vote_provenance_sha256(vote: Vote) -> str:
    return _json_sha256(
        {
            "schema_version": "flavourbench-expert-vote-provenance-v1",
            "battle_id": vote.battle_id,
            "reviewer_id": vote.reviewer_id,
            "reviewer_identity_binding_id": vote.reviewer_identity_binding_id,
            "reviewer_family_admission_id": vote.reviewer_family_admission_id,
            "cohort": vote.cohort,
            "choice": vote.choice,
            "rater_pseudonym_sha256": _text_sha256(vote.rater_pseudonym),
            "idempotency_key_sha256": _text_sha256(vote.idempotency_key),
        }
    )


def _reviewer_admission_evidence_sha256(
    admission: ReviewerFamilyAdmission,
    *,
    qualification_evidence_sha256: str,
    calibration_ballot_sha256: str | None,
) -> str:
    return _json_sha256(
        {
            "schema_version": "flavourbench-reviewer-family-admission-evidence-v1",
            "season_id": admission.season_id,
            "reviewer_id": admission.reviewer_id,
            "identity_binding_id": admission.identity_binding_id,
            "family": admission.family,
            "review_role": admission.review_role,
            "cohort": admission.cohort,
            "qualification_evidence_id": admission.qualification_evidence_id,
            "qualification_evidence_sha256": qualification_evidence_sha256,
            "calibration_ballot_id": admission.calibration_ballot_id,
            "calibration_ballot_sha256": calibration_ballot_sha256,
            "admission_policy_sha256": admission.admission_policy_sha256,
            "decision_reference_sha256": admission.decision_reference_sha256,
            "valid_from": _as_utc(admission.valid_from).isoformat(),
            "valid_until": _as_utc(admission.valid_until).isoformat(),
        }
    )


def _task_validation_audit_authorization_sha256(
    authorization: TaskValidationAuditAuthorization,
) -> str:
    return _json_sha256(
        {
            "schema_version": "flavourbench-task-validation-audit-authorization-v1",
            "season_id": authorization.season_id,
            "campaign_sha256": authorization.campaign_sha256,
            "reviewer_id": authorization.reviewer_id,
            "identity_binding_id": authorization.identity_binding_id,
            "audit_kind": authorization.audit_kind,
            "cohort": authorization.cohort,
            "qualification_evidence_sha256": (authorization.qualification_evidence_sha256),
            "conflict_evidence_sha256": authorization.conflict_evidence_sha256,
            "automated_evidence_sha256": authorization.automated_evidence_sha256,
            "audit_plan_sha256": authorization.audit_plan_sha256,
            "decision_reference_sha256": authorization.decision_reference_sha256,
        }
    )


def _task_validation_campaign_event_sha256(
    event_record: TaskValidationCampaignEvent,
) -> str:
    return _json_sha256(
        {
            "schema_version": "flavourbench-task-validation-ledger-event-v1",
            "campaign_sha256": event_record.campaign_sha256,
            "sequence": event_record.sequence,
            "event_id": event_record.event_id,
            "event_type": event_record.event_type,
            "reviewer_pseudonym": event_record.reviewer_pseudonym,
            "person_commitment_sha256": event_record.person_commitment_sha256,
            "reviewer_admission_receipt_sha256": (event_record.reviewer_admission_receipt_sha256),
            "payload": event_record.payload_json,
            "previous_event_sha256": event_record.previous_event_sha256,
        }
    )


def _changed_fields(record: object, fields: tuple[str, ...]) -> list[str]:
    state = inspect(record)
    return [field for field in fields if state.attrs[field].history.has_changes()]


def _prior_value(record: object, field: str) -> object:
    history = inspect(record).attrs[field].history
    return history.deleted[0] if history.deleted else getattr(record, field)


@event.listens_for(Organization, "before_insert")
def validate_organization_insert(
    _mapper: object, _connection: object, organization: Organization
) -> None:
    status = organization.status or "pending_verification"
    if status not in {"pending_verification", "active"}:
        raise ValueError("organizations must be inserted pending verification or active")
    if status == "active" and organization.verified_at is None:
        raise ValueError("active organizations require a verification timestamp")
    if status == "pending_verification" and organization.verified_at is not None:
        raise ValueError("pending organizations cannot carry a verification timestamp")
    if organization.suspended_at is not None or organization.closed_at is not None:
        raise ValueError("new organizations cannot carry terminal lifecycle timestamps")
    if organization.retention_policy_sha256 != _json_sha256(organization.retention_policy_json):
        raise ValueError("organization retention policy digest does not match its payload")
    if not _is_sha256(organization.idp_tenant_reference_sha256) or (
        organization.billing_reference_sha256 is not None
        and not _is_sha256(organization.billing_reference_sha256)
    ):
        raise ValueError("organization identity references must be SHA-256 digests")


@event.listens_for(Organization, "before_update")
def prevent_organization_contract_mutation(
    _mapper: object, _connection: object, organization: Organization
) -> None:
    changed = _changed_fields(organization, _IMMUTABLE_ORGANIZATION_FIELDS)
    if changed:
        raise ValueError("organization contract is immutable: " + ", ".join(changed))
    state = inspect(organization)
    status_changed = state.attrs.status.history.has_changes()
    lifecycle_fields = ("verified_at", "suspended_at", "closed_at")
    lifecycle_changed = any(state.attrs[field].history.has_changes() for field in lifecycle_fields)
    if not status_changed:
        if lifecycle_changed:
            raise ValueError("organization lifecycle timestamps require a status transition")
        return
    prior_status = _prior_value(organization, "status")
    allowed = {
        ("pending_verification", "active"),
        ("pending_verification", "closed"),
        ("active", "suspended"),
        ("active", "closed"),
        ("suspended", "closed"),
    }
    if (prior_status, organization.status) not in allowed:
        raise ValueError("organization lifecycle transition is invalid")
    changed_lifecycle = {
        field for field in lifecycle_fields if state.attrs[field].history.has_changes()
    }
    permitted_lifecycle = {
        ("pending_verification", "active"): {"verified_at"},
        ("pending_verification", "closed"): {"closed_at"},
        ("active", "suspended"): {"suspended_at"},
        ("active", "closed"): {"closed_at"},
        ("suspended", "closed"): {"closed_at"},
    }[(prior_status, organization.status)]
    if not changed_lifecycle <= permitted_lifecycle:
        raise ValueError("organization lifecycle evidence is write-once")
    if organization.status == "active" and organization.verified_at is None:
        raise ValueError("organization activation requires a verification timestamp")
    if organization.status == "suspended" and organization.suspended_at is None:
        raise ValueError("organization suspension requires a timestamp")
    if organization.status == "closed" and organization.closed_at is None:
        raise ValueError("organization closure requires a timestamp")


@event.listens_for(Organization, "before_delete")
def prevent_organization_delete(
    _mapper: object, _connection: object, _organization: Organization
) -> None:
    raise ValueError("organizations are append-only")


@event.listens_for(OrganizationApiKey, "before_insert")
def validate_organization_api_key_insert(
    _mapper: object, _connection: object, api_key: OrganizationApiKey
) -> None:
    if (api_key.status or "active") != "active" or api_key.revoked_at is not None:
        raise ValueError("organization API keys must be inserted active")
    if _as_utc(api_key.expires_at) <= _as_utc(api_key.not_before):
        raise ValueError("organization API-key expiry must follow not-before")
    if not _is_sha256(api_key.secret_hmac_sha256) or not _is_sha256(
        api_key.created_by_principal_ref_sha256
    ):
        raise ValueError("organization API-key references must be SHA-256 digests")


@event.listens_for(OrganizationApiKey, "before_update")
def prevent_organization_api_key_mutation(
    _mapper: object, _connection: object, api_key: OrganizationApiKey
) -> None:
    changed = _changed_fields(api_key, _IMMUTABLE_ORGANIZATION_API_KEY_FIELDS)
    if changed:
        raise ValueError("organization API-key contract is immutable: " + ", ".join(changed))
    state = inspect(api_key)
    last_used_history = state.attrs.last_used_at.history
    if last_used_history.has_changes() and last_used_history.deleted:
        prior_last_used = last_used_history.deleted[0]
        if prior_last_used is not None and (
            api_key.last_used_at is None or _as_utc(api_key.last_used_at) < _as_utc(prior_last_used)
        ):
            raise ValueError("organization API-key last-used time cannot move backward")
    status_changed = state.attrs.status.history.has_changes()
    revoked_changed = state.attrs.revoked_at.history.has_changes()
    if status_changed:
        if (
            _prior_value(api_key, "status") != "active"
            or api_key.status != "revoked"
            or api_key.revoked_at is None
        ):
            raise ValueError("organization API keys may only transition active to revoked")
    elif revoked_changed:
        raise ValueError("organization API-key revocation time requires revocation")


@event.listens_for(OrganizationApiKey, "before_delete")
def prevent_organization_api_key_delete(
    _mapper: object, _connection: object, _api_key: OrganizationApiKey
) -> None:
    raise ValueError("organization API keys are append-only")


@event.listens_for(GovernanceAcceptance, "before_insert")
def validate_governance_acceptance_insert(
    _mapper: object, _connection: object, acceptance: GovernanceAcceptance
) -> None:
    if (acceptance.status or "active") != "active" or acceptance.revoked_at is not None:
        raise ValueError("governance acceptances must be inserted active")
    if acceptance.expires_at is not None and _as_utc(acceptance.expires_at) <= _as_utc(
        acceptance.accepted_at
    ):
        raise ValueError("governance acceptance expiry must follow acceptance")
    if _as_utc(acceptance.accepted_at) > _as_utc(utcnow()) + timedelta(minutes=5):
        raise ValueError("governance acceptance time cannot be in the future")
    if acceptance.binding_sha256 != _json_sha256(acceptance.binding_json):
        raise ValueError("governance acceptance binding digest mismatch")
    for field in (
        "document_sha256",
        "external_envelope_reference_sha256",
        "signatory_principal_reference_sha256",
        "binding_sha256",
    ):
        if not _is_sha256(getattr(acceptance, field)):
            raise ValueError(f"governance acceptance {field} must be a SHA-256 digest")


@event.listens_for(GovernanceAcceptance, "before_update")
@event.listens_for(GovernanceAcceptance, "before_delete")
def prevent_governance_acceptance_mutation(
    _mapper: object, _connection: object, _acceptance: GovernanceAcceptance
) -> None:
    raise ValueError("governance acceptances are append-only; add a superseding record")


@event.listens_for(ModelSubmission, "before_insert")
def validate_model_submission_insert(
    _mapper: object, _connection: object, submission: ModelSubmission
) -> None:
    if (submission.status or "draft") != "draft":
        raise ValueError("model submissions must be inserted as drafts")
    if any(
        value is not None
        for value in (
            submission.catalog_model_id,
            submission.decision_reference_sha256,
            submission.submitted_at,
            submission.decided_at,
            submission.suspended_at,
        )
    ):
        raise ValueError("draft model submissions cannot carry decision metadata")
    digest_pairs = (
        (submission.capability_claims_json, submission.capability_claims_sha256),
        (
            submission.contamination_disclosure_json,
            submission.contamination_disclosure_sha256,
        ),
        (submission.submission_payload_json, submission.submission_payload_sha256),
    )
    if any(_json_sha256(payload) != digest for payload, digest in digest_pairs):
        raise ValueError("model submission payload digest mismatch")


@event.listens_for(ModelSubmission, "before_update")
def prevent_model_submission_contract_mutation(
    _mapper: object, _connection: object, submission: ModelSubmission
) -> None:
    changed = _changed_fields(submission, _IMMUTABLE_MODEL_SUBMISSION_FIELDS)
    if changed:
        raise ValueError("model submission is immutable: " + ", ".join(changed))
    state = inspect(submission)
    status_changed = state.attrs.status.history.has_changes()
    metadata_fields = (
        "catalog_model_id",
        "decision_reference_sha256",
        "submitted_at",
        "decided_at",
        "suspended_at",
    )
    if not status_changed:
        if any(state.attrs[field].history.has_changes() for field in metadata_fields):
            raise ValueError("model submission metadata requires a lifecycle transition")
        return
    prior_status = _prior_value(submission, "status")
    allowed = {
        ("draft", "submitted"),
        ("draft", "withdrawn"),
        ("submitted", "approved"),
        ("submitted", "changes_requested"),
        ("submitted", "rejected"),
        ("submitted", "withdrawn"),
        ("approved", "suspended"),
        ("approved", "retired"),
        ("changes_requested", "withdrawn"),
        ("suspended", "retired"),
    }
    if (prior_status, submission.status) not in allowed:
        raise ValueError("model submission lifecycle transition is invalid")
    changed_metadata = {
        field for field in metadata_fields if state.attrs[field].history.has_changes()
    }
    permitted_metadata = {
        ("draft", "submitted"): {"submitted_at"},
        ("draft", "withdrawn"): set(),
        ("submitted", "approved"): {
            "catalog_model_id",
            "decision_reference_sha256",
            "decided_at",
        },
        ("submitted", "changes_requested"): {
            "decision_reference_sha256",
            "decided_at",
        },
        ("submitted", "rejected"): {
            "decision_reference_sha256",
            "decided_at",
        },
        ("submitted", "withdrawn"): set(),
        ("approved", "suspended"): {"suspended_at"},
        ("approved", "retired"): set(),
        ("changes_requested", "withdrawn"): set(),
        ("suspended", "retired"): set(),
    }[(prior_status, submission.status)]
    if not changed_metadata <= permitted_metadata:
        raise ValueError("model submission lifecycle evidence is write-once")
    if submission.status == "submitted" and submission.submitted_at is None:
        raise ValueError("submitted model submissions require a submission timestamp")
    if submission.status in {"approved", "changes_requested", "rejected"}:
        if submission.decided_at is None or not _is_sha256(submission.decision_reference_sha256):
            raise ValueError("model submission decisions require timestamped evidence")
    if submission.status == "approved" and submission.catalog_model_id is None:
        raise ValueError("approved model submissions require a catalog identity")
    if submission.status == "suspended" and submission.suspended_at is None:
        raise ValueError("model submission suspension requires a timestamp")


@event.listens_for(ModelSubmission, "before_delete")
def prevent_model_submission_delete(
    _mapper: object, _connection: object, _submission: ModelSubmission
) -> None:
    raise ValueError("model submissions are append-only")


@event.listens_for(ModelRouteRevision, "before_insert")
def validate_model_route_insert(
    _mapper: object, _connection: object, route: ModelRouteRevision
) -> None:
    if (route.status or "draft") != "draft":
        raise ValueError("model routes must be inserted as drafts")
    if any(
        value is not None
        for value in (
            route.approved_contract_test_id,
            route.approved_season_id,
            route.approved_season_manifest_sha256,
            route.approved_endpoint_contract_sha256,
            route.valid_until,
            route.submitted_at,
            route.approved_at,
            route.suspended_at,
            route.retired_at,
        )
    ):
        raise ValueError("draft model routes cannot carry approval metadata")
    digest_pairs = (
        (route.supported_parameters_json, route.supported_parameters_sha256),
        (route.decoding_bounds_json, route.decoding_bounds_sha256),
        (route.data_policy_json, route.data_policy_sha256),
        (route.rate_card_json, route.rate_card_sha256),
        (route.descriptor_json, route.descriptor_sha256),
    )
    if any(_json_sha256(payload) != digest for payload, digest in digest_pairs):
        raise ValueError("model route descriptor digest mismatch")


@event.listens_for(ModelRouteRevision, "before_update")
def prevent_model_route_contract_mutation(
    _mapper: object, _connection: object, route: ModelRouteRevision
) -> None:
    changed = _changed_fields(route, _IMMUTABLE_MODEL_ROUTE_FIELDS)
    if changed:
        raise ValueError("model route is immutable: " + ", ".join(changed))
    state = inspect(route)
    status_changed = state.attrs.status.history.has_changes()
    metadata_fields = (
        "approved_contract_test_id",
        "approved_season_id",
        "approved_season_manifest_sha256",
        "approved_endpoint_contract_sha256",
        "valid_until",
        "submitted_at",
        "approved_at",
        "suspended_at",
        "retired_at",
    )
    if not status_changed:
        if any(state.attrs[field].history.has_changes() for field in metadata_fields):
            raise ValueError("model route metadata requires a lifecycle transition")
        return
    prior_status = _prior_value(route, "status")
    allowed = {
        ("draft", "submitted"),
        ("draft", "withdrawn"),
        ("submitted", "contract_testing"),
        ("submitted", "approved"),
        ("submitted", "changes_requested"),
        ("submitted", "rejected"),
        ("submitted", "withdrawn"),
        ("contract_testing", "approved"),
        ("contract_testing", "changes_requested"),
        ("contract_testing", "rejected"),
        ("approved", "suspended"),
        ("approved", "retired"),
        ("changes_requested", "withdrawn"),
        ("suspended", "retired"),
    }
    if (prior_status, route.status) not in allowed:
        raise ValueError("model route lifecycle transition is invalid")
    changed_metadata = {
        field for field in metadata_fields if state.attrs[field].history.has_changes()
    }
    approval_metadata = {
        "approved_contract_test_id",
        "approved_season_id",
        "approved_season_manifest_sha256",
        "approved_endpoint_contract_sha256",
        "valid_until",
        "approved_at",
    }
    permitted_metadata = {
        ("draft", "submitted"): {"submitted_at"},
        ("draft", "withdrawn"): set(),
        ("submitted", "contract_testing"): set(),
        ("submitted", "approved"): approval_metadata,
        ("submitted", "changes_requested"): set(),
        ("submitted", "rejected"): set(),
        ("submitted", "withdrawn"): set(),
        ("contract_testing", "approved"): approval_metadata,
        ("contract_testing", "changes_requested"): set(),
        ("contract_testing", "rejected"): set(),
        ("approved", "suspended"): {"suspended_at"},
        ("approved", "retired"): {"retired_at"},
        ("changes_requested", "withdrawn"): set(),
        ("suspended", "retired"): {"retired_at"},
    }[(prior_status, route.status)]
    if not changed_metadata <= permitted_metadata:
        raise ValueError("model route approval evidence is write-once")
    if route.status == "submitted" and route.submitted_at is None:
        raise ValueError("submitted model routes require a submission timestamp")
    if route.status == "approved" and route.approved_at is None:
        raise ValueError("approved model routes require an approval timestamp")
    if route.status == "approved" and (
        route.approved_season_id is None
        or not _is_sha256(route.approved_season_manifest_sha256)
        or not _is_sha256(route.approved_endpoint_contract_sha256)
    ):
        raise ValueError("approved model routes require a frozen season endpoint binding")
    if route.status == "suspended" and route.suspended_at is None:
        raise ValueError("model route suspension requires a timestamp")
    if route.status == "retired" and route.retired_at is None:
        raise ValueError("model route retirement requires a timestamp")


@event.listens_for(ModelRouteRevision, "before_delete")
def prevent_model_route_delete(
    _mapper: object, _connection: object, _route: ModelRouteRevision
) -> None:
    raise ValueError("model routes are append-only")


@event.listens_for(RouteContractTest, "before_insert")
def validate_route_contract_test_insert(
    _mapper: object, _connection: object, contract_test: RouteContractTest
) -> None:
    if (contract_test.status or "queued") != "queued":
        raise ValueError("route contract tests must be inserted queued")
    if any(
        value is not None
        for value in (
            contract_test.request_sha256,
            contract_test.response_sha256,
            contract_test.tool_trace_sha256,
            contract_test.structured_output_sha256,
            contract_test.observed_model_id,
            contract_test.observed_provider_slug,
            contract_test.check_results_sha256,
            contract_test.generation_id,
            contract_test.latency_ms,
            contract_test.failure_code,
            contract_test.incident_id,
            contract_test.started_at,
            contract_test.completed_at,
            contract_test.valid_until,
        )
    ):
        raise ValueError("queued route contract tests cannot carry execution evidence")
    if (contract_test.cost_micros or 0) != 0:
        raise ValueError("queued route contract tests cannot carry cost")
    if not _is_sha256(contract_test.protocol_bundle_sha256):
        raise ValueError("route contract protocol reference must be a SHA-256 digest")


@event.listens_for(RouteContractTest, "before_update")
def prevent_route_contract_test_mutation(
    _mapper: object, _connection: object, contract_test: RouteContractTest
) -> None:
    state = inspect(contract_test)
    immutable = (
        "id",
        "route_revision_id",
        "suite_version",
        "protocol_bundle_sha256",
        "worker_build_digest",
        "created_at",
    )
    changed = _changed_fields(contract_test, immutable)
    if changed:
        raise ValueError("route contract-test contract is immutable: " + ", ".join(changed))
    status_history = state.attrs.status.history
    if not status_history.has_changes():
        if any(
            state.attrs[field].history.has_changes()
            for field in (
                "request_sha256",
                "response_sha256",
                "tool_trace_sha256",
                "structured_output_sha256",
                "observed_model_id",
                "observed_provider_slug",
                "check_results_json",
                "check_results_sha256",
                "generation_id",
                "usage_json",
                "cost_micros",
                "cost_accounting_basis",
                "latency_ms",
                "failure_code",
                "incident_id",
                "started_at",
                "completed_at",
                "valid_until",
            )
        ):
            raise ValueError("route contract-test evidence requires a lifecycle transition")
        return
    prior_status = _prior_value(contract_test, "status")
    allowed = {
        ("queued", "running"),
        ("queued", "cancelled"),
        ("running", "passed"),
        ("running", "failed"),
        ("running", "inconclusive"),
        ("running", "cancelled"),
    }
    if (prior_status, contract_test.status) not in allowed:
        raise ValueError("route contract-test lifecycle transition is invalid")
    evidence_fields = (
        "request_sha256",
        "response_sha256",
        "tool_trace_sha256",
        "structured_output_sha256",
        "observed_model_id",
        "observed_provider_slug",
        "check_results_json",
        "check_results_sha256",
        "generation_id",
        "usage_json",
        "cost_micros",
        "cost_accounting_basis",
        "latency_ms",
        "failure_code",
        "incident_id",
        "started_at",
        "completed_at",
        "valid_until",
    )
    changed_evidence = {
        field for field in evidence_fields if state.attrs[field].history.has_changes()
    }
    permitted_evidence = (
        {"request_sha256", "started_at"}
        if contract_test.status == "running"
        else set(evidence_fields) - {"started_at", "request_sha256"}
    )
    if not changed_evidence <= permitted_evidence:
        raise ValueError("route contract-test evidence is write-once")
    if contract_test.status == "running":
        if contract_test.started_at is None or contract_test.completed_at is not None:
            raise ValueError("running route contract tests require only a start timestamp")
        return
    if contract_test.completed_at is None:
        raise ValueError("terminal route contract tests require a completion timestamp")
    if contract_test.check_results_sha256 is not None and (
        contract_test.check_results_sha256 != _json_sha256(contract_test.check_results_json)
    ):
        raise ValueError("route contract-test result digest mismatch")
    if contract_test.status == "passed":
        required = (
            contract_test.request_sha256,
            contract_test.response_sha256,
            contract_test.tool_trace_sha256,
            contract_test.structured_output_sha256,
            contract_test.check_results_sha256,
            contract_test.observed_model_id,
            contract_test.observed_provider_slug,
            contract_test.generation_id,
            contract_test.valid_until,
        )
        if any(value is None for value in required):
            raise ValueError("passed route contract tests require complete execution evidence")


@event.listens_for(RouteContractTest, "before_delete")
def prevent_route_contract_test_delete(
    _mapper: object, _connection: object, _contract_test: RouteContractTest
) -> None:
    raise ValueError("route contract tests are append-only")


@event.listens_for(EvaluationOrder, "before_insert")
def validate_evaluation_order_insert(
    _mapper: object, _connection: object, order: EvaluationOrder
) -> None:
    if (order.status or "draft") != "draft":
        raise ValueError("evaluation orders must be inserted as drafts")
    if (order.billing_status or "unquoted") != "unquoted":
        raise ValueError("evaluation orders must be inserted unquoted")
    if (order.publication_status or "private") != "private":
        raise ValueError("evaluation orders must be inserted private")
    if any(
        value is not None
        for value in (
            order.quote_reference_sha256,
            order.submitted_at,
            order.approved_at,
            order.started_at,
            order.completed_at,
            order.delivered_at,
        )
    ):
        raise ValueError("draft evaluation orders cannot carry execution metadata")
    if order.comparison_plan_sha256 != _json_sha256(order.comparison_plan_json):
        raise ValueError("evaluation-order comparison-plan digest mismatch")
    if order.order_card_sha256 != _json_sha256(order.order_card_json):
        raise ValueError("evaluation-order card digest mismatch")
    for field in (
        "comparison_plan_sha256",
        "rater_plan_sha256",
        "analysis_plan_sha256",
        "client_reference_sha256",
        "order_card_sha256",
        "order_card_signature",
    ):
        if not _is_sha256(getattr(order, field)):
            raise ValueError(f"evaluation-order {field} must be a SHA-256 digest")


@event.listens_for(EvaluationOrder, "before_update")
def prevent_evaluation_order_contract_mutation(
    _mapper: object, _connection: object, order: EvaluationOrder
) -> None:
    changed = _changed_fields(order, _IMMUTABLE_EVALUATION_ORDER_FIELDS)
    if changed:
        raise ValueError("evaluation-order contract is immutable: " + ", ".join(changed))
    state = inspect(order)
    status_history = state.attrs.status.history
    mutable_metadata = (
        "billing_status",
        "publication_status",
        "quote_reference_sha256",
        "submitted_at",
        "approved_at",
        "started_at",
        "completed_at",
        "delivered_at",
    )
    if not status_history.has_changes():
        changed_metadata = {
            field for field in mutable_metadata if state.attrs[field].history.has_changes()
        }
        if changed_metadata - {"publication_status"}:
            raise ValueError("evaluation-order metadata requires a lifecycle transition")
        publication_history = state.attrs.publication_status.history
        if not publication_history.has_changes():
            return
        prior_publication = publication_history.deleted[0]
        publication_allowed = {
            ("private", "authorized"),
            ("authorized", "published"),
            ("authorized", "withdrawn"),
            ("published", "withdrawn"),
        }
        if (prior_publication, order.publication_status) not in publication_allowed:
            raise ValueError("evaluation-order publication transition is invalid")
        return
    prior_status = _prior_value(order, "status")
    allowed = {
        ("draft", "submitted"),
        ("draft", "cancelled"),
        ("submitted", "approved"),
        ("submitted", "rejected"),
        ("submitted", "cancelled"),
        ("approved", "provisioning"),
        ("approved", "cancelled"),
        ("provisioning", "ready"),
        ("provisioning", "failed"),
        ("provisioning", "cancelling"),
        ("ready", "running"),
        ("ready", "cancelled"),
        ("running", "collection_complete"),
        ("running", "cancelling"),
        ("running", "failed"),
        ("collection_complete", "analysis_complete"),
        ("collection_complete", "failed"),
        ("analysis_complete", "delivered"),
        ("analysis_complete", "failed"),
        ("cancelling", "cancelled"),
        ("cancelling", "failed"),
    }
    if (prior_status, order.status) not in allowed:
        raise ValueError("evaluation-order lifecycle transition is invalid")
    changed_metadata = {
        field for field in mutable_metadata if state.attrs[field].history.has_changes()
    }
    permitted_metadata = {
        ("draft", "submitted"): {"submitted_at"},
        ("draft", "cancelled"): {"billing_status"},
        ("submitted", "approved"): {
            "billing_status",
            "quote_reference_sha256",
            "approved_at",
        },
        ("submitted", "rejected"): {"billing_status"},
        ("submitted", "cancelled"): {"billing_status"},
        ("approved", "provisioning"): set(),
        ("approved", "cancelled"): {"billing_status"},
        ("provisioning", "ready"): set(),
        ("provisioning", "failed"): set(),
        ("provisioning", "cancelling"): set(),
        ("ready", "running"): {"started_at"},
        ("ready", "cancelled"): {"billing_status"},
        ("running", "collection_complete"): set(),
        ("running", "cancelling"): set(),
        ("running", "failed"): set(),
        ("collection_complete", "analysis_complete"): {"completed_at"},
        ("collection_complete", "failed"): set(),
        ("analysis_complete", "delivered"): {"delivered_at"},
        ("analysis_complete", "failed"): set(),
        ("cancelling", "cancelled"): {"billing_status"},
        ("cancelling", "failed"): set(),
    }[(prior_status, order.status)]
    if not changed_metadata <= permitted_metadata:
        raise ValueError("evaluation-order lifecycle evidence is write-once")
    if order.status == "submitted" and order.submitted_at is None:
        raise ValueError("submitted evaluation orders require a submission timestamp")
    if order.status == "approved":
        if (
            order.approved_at is None
            or order.billing_status != "authorized"
            or not _is_sha256(order.quote_reference_sha256)
        ):
            raise ValueError("approved evaluation orders require authorized billing evidence")
    if order.status == "provisioning":
        controlled_run_id = _connection.scalar(
            select(ControlledRun.id).where(ControlledRun.evaluation_order_id == order.id)
        )
        if controlled_run_id is None:
            raise ValueError("provisioning evaluation orders require a controlled run")
    if order.status == "running" and order.started_at is None:
        raise ValueError("running evaluation orders require a start timestamp")
    if order.status in {"analysis_complete", "delivered"} and order.completed_at is None:
        raise ValueError("completed evaluation orders require a completion timestamp")
    if order.status == "delivered" and order.delivered_at is None:
        raise ValueError("delivered evaluation orders require a delivery timestamp")
    if order.status in {"rejected", "cancelled"} and order.billing_status != "void":
        raise ValueError("rejected or cancelled evaluation orders must void billing")
    if state.attrs.publication_status.history.has_changes():
        raise ValueError("publication authorization is independent of execution lifecycle")


@event.listens_for(EvaluationOrder, "before_delete")
def prevent_evaluation_order_delete(
    _mapper: object, _connection: object, _order: EvaluationOrder
) -> None:
    raise ValueError("evaluation orders are append-only")


@event.listens_for(EvidenceBundle, "before_insert")
def validate_evidence_bundle_insert(
    _mapper: object, _connection: object, bundle: EvidenceBundle
) -> None:
    if (bundle.status or "building") != "building":
        raise ValueError("evidence bundles must be inserted building")
    if bundle.manifest_sha256 != _json_sha256(bundle.manifest_json):
        raise ValueError("evidence-bundle manifest digest mismatch")
    if any(
        value is not None
        for value in (
            bundle.archive_sha256,
            bundle.storage_object_key,
            bundle.size_bytes,
            bundle.signature_algorithm,
            bundle.signing_key_id,
            bundle.signature_base64,
            bundle.sealed_at,
            bundle.available_until,
            bundle.revoked_at,
        )
    ):
        raise ValueError("building evidence bundles cannot carry sealed archive metadata")


@event.listens_for(EvidenceBundle, "before_update")
def prevent_evidence_bundle_contract_mutation(
    _mapper: object, _connection: object, bundle: EvidenceBundle
) -> None:
    changed = _changed_fields(bundle, _IMMUTABLE_EVIDENCE_BUNDLE_FIELDS)
    if changed:
        raise ValueError("evidence-bundle manifest is immutable: " + ", ".join(changed))
    state = inspect(bundle)
    status_history = state.attrs.status.history
    archive_fields = (
        "archive_sha256",
        "storage_object_key",
        "size_bytes",
        "signature_algorithm",
        "signing_key_id",
        "signature_base64",
        "sealed_at",
        "available_until",
        "revoked_at",
    )
    if not status_history.has_changes():
        if any(state.attrs[field].history.has_changes() for field in archive_fields):
            raise ValueError("evidence-bundle archive metadata requires a status transition")
        return
    prior_status = _prior_value(bundle, "status")
    allowed = {
        ("building", "sealed"),
        ("building", "failed"),
        ("sealed", "available"),
        ("sealed", "superseded"),
        ("sealed", "revoked"),
        ("available", "superseded"),
        ("available", "revoked"),
    }
    if (prior_status, bundle.status) not in allowed:
        raise ValueError("evidence-bundle lifecycle transition is invalid")
    changed_archive = {
        field for field in archive_fields if state.attrs[field].history.has_changes()
    }
    permitted_archive = {
        ("building", "sealed"): set(archive_fields) - {"revoked_at"},
        ("building", "failed"): set(),
        ("sealed", "available"): set(),
        ("sealed", "superseded"): set(),
        ("sealed", "revoked"): {"revoked_at"},
        ("available", "superseded"): set(),
        ("available", "revoked"): {"revoked_at"},
    }[(prior_status, bundle.status)]
    if not changed_archive <= permitted_archive:
        raise ValueError("evidence-bundle archive evidence is write-once")
    if bundle.status in {"sealed", "available", "superseded", "revoked"}:
        required = (
            bundle.archive_sha256,
            bundle.storage_object_key,
            bundle.size_bytes,
            bundle.signature_algorithm,
            bundle.signing_key_id,
            bundle.signature_base64,
            bundle.sealed_at,
        )
        if any(value is None for value in required):
            raise ValueError("sealed evidence bundles require complete archive evidence")
    if bundle.status == "revoked" and bundle.revoked_at is None:
        raise ValueError("revoked evidence bundles require a revocation timestamp")


@event.listens_for(EvidenceBundle, "before_delete")
def prevent_evidence_bundle_delete(
    _mapper: object, _connection: object, _bundle: EvidenceBundle
) -> None:
    raise ValueError("evidence bundles are append-only")


@event.listens_for(ApiIdempotencyKey, "before_update")
@event.listens_for(ApiIdempotencyKey, "before_delete")
def prevent_idempotency_record_mutation(
    _mapper: object, _connection: object, _record: ApiIdempotencyKey
) -> None:
    raise ValueError("API idempotency records are immutable")


@event.listens_for(Season, "before_insert")
@event.listens_for(SeasonProviderBudget, "before_insert")
@event.listens_for(ProviderAccountBudget, "before_insert")
def require_nonnegative_budget_insert(
    _mapper: object,
    _connection: object,
    record: Season | SeasonProviderBudget | ProviderAccountBudget,
) -> None:
    _require_nonnegative_budget(record)


@event.listens_for(EpicureRelease, "before_update")
@event.listens_for(EpicureRelease, "before_delete")
def prevent_epicure_release_mutation(
    _mapper: object, _connection: object, _release: EpicureRelease
) -> None:
    raise ValueError("Epicure release records are append-only")


@event.listens_for(SeasonProviderBudget, "before_update")
def prevent_provider_budget_contract_mutation(
    _mapper: object, _connection: object, budget: SeasonProviderBudget
) -> None:
    _require_nonnegative_budget(budget)
    state = inspect(budget)
    used_history = state.attrs.budget_used_micros.history
    if (
        used_history.has_changes()
        and used_history.deleted
        and budget.budget_used_micros < used_history.deleted[0]
    ):
        raise ValueError("governed provider spend cannot move backward")
    changed = [
        field
        for field in _IMMUTABLE_PROVIDER_BUDGET_CONTRACT_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError(
            "provider budget authorization is immutable; create a new season: " + ", ".join(changed)
        )


@event.listens_for(SeasonProviderBudget, "before_delete")
def prevent_provider_budget_delete(
    _mapper: object, _connection: object, _budget: SeasonProviderBudget
) -> None:
    raise ValueError("provider budget authorizations cannot be deleted")


@event.listens_for(ProviderAccountBudget, "before_update")
def prevent_account_budget_contract_mutation(
    _mapper: object, _connection: object, budget: ProviderAccountBudget
) -> None:
    _require_nonnegative_budget(budget)
    state = inspect(budget)
    used_history = state.attrs.budget_used_micros.history
    if (
        used_history.has_changes()
        and used_history.deleted
        and budget.budget_used_micros < used_history.deleted[0]
    ):
        raise ValueError("governed provider-account spend cannot move backward")
    changed = [
        field
        for field in _IMMUTABLE_ACCOUNT_BUDGET_CONTRACT_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError("provider account authorization is immutable: " + ", ".join(changed))
    status_history = state.attrs.status.history
    revoked_history = state.attrs.revoked_at.history
    if status_history.has_changes() or revoked_history.has_changes():
        old_status = status_history.deleted[0] if status_history.deleted else budget.status
        new_status = status_history.added[0] if status_history.added else budget.status
        old_revoked = revoked_history.deleted[0] if revoked_history.deleted else None
        new_revoked = revoked_history.added[0] if revoked_history.added else budget.revoked_at
        allowed = (
            old_status == "pending_verification"
            and new_status == "active"
            and old_revoked is None
            and new_revoked is None
        ) or (
            old_status in {"pending_verification", "active"}
            and new_status == "revoked"
            and old_revoked is None
            and new_revoked is not None
        )
        if not allowed:
            raise ValueError("provider account ledger status transition is invalid")


@event.listens_for(ProviderAccountBudget, "before_delete")
def prevent_account_budget_delete(
    _mapper: object, _connection: object, _budget: ProviderAccountBudget
) -> None:
    raise ValueError("provider account ledgers cannot be deleted")


@event.listens_for(ProviderAccountAuthorization, "before_update")
def prevent_account_authorization_mutation(
    _mapper: object,
    _connection: object,
    authorization: ProviderAccountAuthorization,
) -> None:
    state = inspect(authorization)
    changed = [
        field
        for field in _IMMUTABLE_ACCOUNT_AUTHORIZATION_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError("provider account authorization epoch is immutable: " + ", ".join(changed))
    status_history = state.attrs.status.history
    revoked_history = state.attrs.revoked_at.history
    if status_history.has_changes() or revoked_history.has_changes():
        old_status = status_history.deleted[0] if status_history.deleted else authorization.status
        new_status = status_history.added[0] if status_history.added else authorization.status
        old_revoked = revoked_history.deleted[0] if revoked_history.deleted else None
        new_revoked = (
            revoked_history.added[0] if revoked_history.added else authorization.revoked_at
        )
        if not (
            old_status == "active"
            and new_status == "revoked"
            and old_revoked is None
            and new_revoked is not None
        ):
            raise ValueError("provider account authorization revocation is invalid")


@event.listens_for(ProviderAccountAuthorization, "before_delete")
def prevent_account_authorization_delete(
    _mapper: object,
    _connection: object,
    _authorization: ProviderAccountAuthorization,
) -> None:
    raise ValueError("provider account authorization epochs are append-only")


@event.listens_for(Battle, "before_insert")
def require_unlinked_battle_insert(_mapper: object, _connection: object, battle: Battle) -> None:
    """Create the battle first, then bind its two persisted arm rows exactly once."""

    if battle.prompt_redacted is None:
        battle.prompt_redacted = False
    data_stratum = battle.data_stratum or "public_freeform"
    research_consent = bool(battle.research_consent)
    expected_basis = {
        "public_freeform": ("public_consented" if research_consent else "public_nonconsented"),
        "controlled": "controlled_development",
        "development": "development_research",
        "legacy": "legacy_operational",
    }.get(data_stratum)
    if battle.retention_basis is None:
        battle.retention_basis = expected_basis
    if data_stratum == "controlled" and battle.retention_basis in {
        "official_research",
        "commercial_private",
    }:
        expected_basis = battle.retention_basis
    if battle.retention_basis != expected_basis:
        raise ValueError("battle retention basis contradicts its collection scope")
    if (
        battle.status not in {None, "queued"}
        or battle.left_arm_id is not None
        or battle.right_arm_id is not None
        or battle.completed_at is not None
        or battle.prompt is None
        or battle.prompt_redacted is not False
    ):
        raise ValueError("battles must be inserted queued, unredacted, and with null arm links")


@event.listens_for(Battle, "before_update")
def prevent_battle_provenance_mutation(
    _mapper: object, connection: Connection, battle: Battle
) -> None:
    """Keep scientific assignment provenance write-once after insertion.

    Operational lifecycle and redaction state remain mutable. Arm links permit
    only the initial null-to-ID assignment and must point to the matching side of
    this battle. Changing an assignment or its eligibility requires a new battle.
    """

    state = inspect(battle)
    changed = [
        field
        for field in _IMMUTABLE_BATTLE_PROVENANCE_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError(
            "battle scientific provenance is immutable; create a superseding battle: "
            + ", ".join(changed)
        )
    for field, side in (("left_arm_id", "left"), ("right_arm_id", "right")):
        history = state.attrs[field].history
        if not history.has_changes():
            continue
        previous = history.deleted[0] if history.deleted else None
        current = getattr(battle, field)
        if previous is not None:
            raise ValueError(f"battle {side} arm link is write-once")
        if current is None:
            raise ValueError(f"battle {side} arm link cannot be cleared")
        row = connection.execute(
            sql_text("SELECT battle_id, side FROM response_arms WHERE id = :arm_id"),
            {"arm_id": current},
        ).one_or_none()
        if row is None or row[0] != battle.id or row[1] != side:
            raise ValueError(f"battle {side} arm link does not match arm ownership and side")
    if (
        battle.left_arm_id is not None
        and battle.right_arm_id is not None
        and battle.left_arm_id == battle.right_arm_id
    ):
        raise ValueError("battle arm links must be distinct")
    status_history = state.attrs.status.history
    completed_history = state.attrs.completed_at.history
    (
        persisted_status,
        persisted_completed,
        persisted_prompt,
        persisted_prompt_redacted,
        persisted_research_consent,
        persisted_retention_until,
        persisted_retention_basis,
    ) = connection.execute(
        select(
            Battle.status,
            Battle.completed_at,
            Battle.prompt,
            Battle.prompt_redacted,
            Battle.research_consent,
            Battle.retention_until,
            Battle.retention_basis,
        ).where(Battle.id == battle.id)
    ).one()
    if not _same_instant(persisted_retention_until, battle.retention_until):
        raise ValueError("battle retention deadline is immutable")
    prompt_changed = persisted_prompt != battle.prompt
    redaction_changed = persisted_prompt_redacted != battle.prompt_redacted
    authorized_redaction = (
        persisted_research_consent is False
        and persisted_retention_basis in REDACTABLE_RETENTION_BASES
        and _as_utc(persisted_retention_until) <= datetime.now(UTC)
    )
    if (prompt_changed or redaction_changed) and not (
        authorized_redaction
        and prompt_changed
        and redaction_changed
        and persisted_prompt is not None
        and battle.prompt is None
        and persisted_prompt_redacted is False
        and battle.prompt_redacted is True
    ):
        raise ValueError("battle prompt permits only one-way retention redaction")
    if status_history.has_changes():
        previous_status = persisted_status
        if (previous_status, battle.status) not in {
            ("queued", "running"),
            ("queued", "complete"),
            ("queued", "failed"),
            ("running", "complete"),
            ("running", "failed"),
        }:
            raise ValueError("battle lifecycle transition is invalid")
    previous_status = persisted_status
    previous_completed = persisted_completed
    if completed_history.has_changes() and not (
        previous_status in {"queued", "running"}
        and battle.status in {"complete", "failed"}
        and previous_completed is None
        and battle.completed_at is not None
    ):
        raise ValueError("battle completion timestamp is write-once")
    if battle.status in {"queued", "running"} and battle.completed_at is not None:
        raise ValueError("nonterminal battle cannot carry a completion timestamp")
    if battle.status in {"running", "complete", "failed"}:
        linked = []
        for arm_id, side in (
            (battle.left_arm_id, "left"),
            (battle.right_arm_id, "right"),
        ):
            if arm_id is None:
                raise ValueError("running or terminal battles require two owned arm links")
            row = connection.execute(
                sql_text("SELECT battle_id, side FROM response_arms WHERE id = :arm_id"),
                {"arm_id": arm_id},
            ).one_or_none()
            linked.append(row is not None and row[0] == battle.id and row[1] == side)
        if not all(linked):
            raise ValueError("running or terminal battles require two owned arm links")
    if battle.status in {"complete", "failed"}:
        if battle.completed_at is None or _as_utc(battle.completed_at) < _as_utc(battle.created_at):
            raise ValueError("terminal battle record is incomplete")
        allowed_statuses = (
            ("complete",) if battle.status == "complete" else ("complete", "failed", "uncertain")
        )
        allowed_status_sql = ", ".join(f"'{status}'" for status in allowed_statuses)
        finish_reason_clause = (
            "AND LOWER(TRIM(finish_reason)) IN ('completed', 'end_turn', 'stop', 'stop_sequence')"
            if battle.status == "complete"
            else ""
        )
        for arm_id, side in (
            (battle.left_arm_id, "left"),
            (battle.right_arm_id, "right"),
        ):
            terminal_arm = connection.execute(
                sql_text(
                    f"""
                    SELECT 1 FROM response_arms
                    WHERE id = :arm_id
                      AND battle_id = :battle_id
                      AND side = :side
                      AND status IN ({allowed_status_sql})
                      {finish_reason_clause}
                      AND completed_at IS NOT NULL
                      AND completed_at <= :battle_completed_at
                    """
                ),
                {
                    "arm_id": arm_id,
                    "battle_id": battle.id,
                    "side": side,
                    "battle_completed_at": battle.completed_at,
                },
            ).one_or_none()
            if terminal_arm is None:
                raise ValueError("terminal battle record is incomplete")


@event.listens_for(ResponseArm, "before_insert")
def require_empty_queued_response_arm_insert(
    _mapper: object, connection: Connection, arm: ResponseArm
) -> None:
    if (
        arm.status not in {None, "queued"}
        or arm.completed_at is not None
        or arm.actual_provider_slug is not None
        or arm.actual_model_id is not None
        or arm.generation_id is not None
        or arm.answer_markdown is not None
        or arm.answer_markdown_sha256 is not None
        or arm.output_json_sha256 is not None
        or bool(arm.cost_micros)
        or arm.cost_reconciled is True
    ):
        raise ValueError("response arms must be inserted as empty queued records")
    binding = (arm.route_revision_id, arm.endpoint_descriptor_sha256)
    if any(value is None for value in binding) and any(value is not None for value in binding):
        raise ValueError("response-arm route binding must be complete")
    expected = connection.execute(
        select(
            ControlledRun.evaluation_order_id,
            ControlledRun.submitted_endpoint_model_id,
            ControlledRun.route_revision_id,
            ControlledRun.endpoint_descriptor_sha256,
        )
        .join(Battle, Battle.controlled_run_id == ControlledRun.id)
        .where(Battle.id == arm.battle_id)
    ).one_or_none()
    should_be_bound = bool(
        expected is not None
        and expected.evaluation_order_id is not None
        and arm.model_id == expected.submitted_endpoint_model_id
    )
    if should_be_bound:
        if (
            arm.route_revision_id != expected.route_revision_id
            or arm.endpoint_descriptor_sha256 != expected.endpoint_descriptor_sha256
        ):
            raise ValueError("submitted commercial response arm has the wrong route binding")
    elif any(value is not None for value in binding):
        raise ValueError("non-submitted response arm cannot claim a commercial route")


@event.listens_for(ResponseArm, "before_update")
def prevent_response_arm_contract_mutation(
    _mapper: object, connection: Connection, arm: ResponseArm
) -> None:
    """Keep the assigned prompt, condition, endpoint, and intervention write-once."""

    state = inspect(arm)
    changed = [
        field
        for field in _IMMUTABLE_RESPONSE_ARM_CONTRACT_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError(
            "response-arm execution contract is immutable; create a superseding arm: "
            + ", ".join(changed)
        )
    for field in ("answer_markdown_sha256", "output_json_sha256"):
        history = state.attrs[field].history
        if history.has_changes() and history.deleted and history.deleted[0] not in {None, ""}:
            raise ValueError(f"response-arm normalized output digest is immutable: {field}")

    status_history = state.attrs.status.history
    previous_status = (
        status_history.deleted[0]
        if status_history.has_changes() and status_history.deleted
        else arm.status
    )
    if arm.answer_markdown is not None and not arm.answer_markdown_sha256:
        arm.answer_markdown_sha256 = _text_sha256(arm.answer_markdown)
    if arm.output_json_sha256 is None and (
        arm.status in _TERMINAL_RESPONSE_ARM_STATUSES or bool(arm.output_json)
    ):
        arm.output_json_sha256 = _json_sha256(arm.output_json or {})
    if arm.status in {"queued", "running"} and arm.completed_at is not None:
        raise ValueError("nonterminal response arm cannot carry a completion timestamp")
    if (
        status_history.has_changes()
        and arm.status in _TERMINAL_RESPONSE_ARM_STATUSES
        and (
            arm.completed_at is None
            or _as_utc(arm.completed_at) < _as_utc(arm.created_at)
            or arm.output_json_sha256 is None
            or (
                arm.status == "complete"
                and (
                    not isinstance(arm.finish_reason, str)
                    or arm.finish_reason.strip().lower() not in _NORMAL_RESPONSE_FINISH_REASONS
                    or arm.answer_markdown is None
                    or arm.answer_markdown_sha256 is None
                    or arm.actual_provider_slug is None
                    or arm.actual_model_id is None
                    or arm.generation_id is None
                    or not arm.provider_generation_ids_json
                    or arm.cost_reconciled is not True
                    or arm.cost_accounting_basis == "unrecorded"
                    or arm.billing_reconciliation_status == "unrecorded"
                )
            )
            or (
                arm.status == "failed"
                and (
                    arm.cost_reconciled is not True
                    or arm.cost_accounting_basis == "unrecorded"
                    or arm.billing_reconciliation_status == "unrecorded"
                )
            )
            or (arm.status == "uncertain" and arm.cost_reconciled is True)
        )
    ):
        raise ValueError("terminal response-arm record is incomplete")
    if previous_status not in _TERMINAL_RESPONSE_ARM_STATUSES:
        if status_history.has_changes() and (
            previous_status,
            arm.status,
        ) not in {
            ("queued", "running"),
            ("queued", "complete"),
            ("queued", "failed"),
            ("queued", "uncertain"),
            ("running", "complete"),
            ("running", "failed"),
            ("running", "uncertain"),
        }:
            raise ValueError("response-arm lifecycle transition is invalid")
        return

    if state.attrs.completed_at.history.has_changes():
        raise ValueError("terminal response-arm completion timestamp is immutable")

    changed_evidence = [
        field
        for field in _TERMINAL_RESPONSE_ARM_EVIDENCE_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed_evidence:
        raise ValueError(
            "terminal response-arm evidence is immutable: " + ", ".join(changed_evidence)
        )

    settlement = (
        previous_status == "uncertain"
        and arm.status == "failed"
        and arm.cost_reconciled is True
        and arm.cost_micros >= 0
        and arm.cost_accounting_basis == "manual_authorized_settlement"
        and arm.billing_reconciliation_status == "manual_authorized_settlement"
        and arm.error_code == "CostExposureSettled"
        and arm.error_detail == "Provider cost exposure was settled by an authorized record."
        and arm.completed_at is not None
    )
    changed_settlement = {
        field
        for field in _TERMINAL_RESPONSE_ARM_SETTLEMENT_FIELDS
        if state.attrs[field].history.has_changes()
    }
    if settlement:
        if not {"status", "cost_reconciled", "cost_accounting_basis"}.issubset(changed_settlement):
            raise ValueError("terminal response-arm settlement record is incomplete")
    else:
        disallowed = changed_settlement - {"error_detail"}
        if disallowed:
            raise ValueError(
                "terminal response-arm result is immutable: " + ", ".join(sorted(disallowed))
            )

    raw_redactions = {
        field
        for field in ("answer_markdown", "output_json", "error_detail")
        if state.attrs[field].history.has_changes()
    }
    privacy_redactions = raw_redactions - ({"error_detail"} if settlement else set())
    if privacy_redactions:
        parent_is_redacted = connection.execute(
            sql_text(
                "SELECT 1 FROM battles "
                "WHERE id = :battle_id AND prompt_redacted = true AND prompt IS NULL "
                "AND research_consent = false "
                "AND retention_basis IN ("
                "'public_nonconsented', 'commercial_private', "
                "'controlled_development', 'development_research'"
                ") "
                "AND retention_until <= CURRENT_TIMESTAMP"
            ),
            {"battle_id": arm.battle_id},
        ).one_or_none()
        if parent_is_redacted is None:
            raise ValueError(
                "terminal response content may be redacted only with its battle prompt"
            )
    if "answer_markdown" in raw_redactions and arm.answer_markdown is not None:
        raise ValueError("terminal response content permits only one-way privacy redaction")
    if "output_json" in raw_redactions and arm.output_json != TOOL_CALL_REDACTION_JSON:
        raise ValueError("terminal structured output permits only one-way privacy redaction")
    if "error_detail" in raw_redactions and not (settlement or arm.error_detail is None):
        raise ValueError("terminal error detail permits only settlement or privacy redaction")


@event.listens_for(ResponseArm, "before_insert")
@event.listens_for(ResponseArm, "before_update")
def populate_response_arm_output_digests(
    _mapper: object, _connection: object, arm: ResponseArm
) -> None:
    """Persist normalized output digests once so retention can remove raw content safely."""

    if arm.answer_markdown is not None and not arm.answer_markdown_sha256:
        arm.answer_markdown_sha256 = _text_sha256(arm.answer_markdown)
    if arm.output_json_sha256 is None and (
        arm.status in {"complete", "failed", "uncertain"} or bool(arm.output_json)
    ):
        arm.output_json_sha256 = _json_sha256(arm.output_json or {})


@event.listens_for(ResponseArm, "before_delete")
def prevent_response_arm_delete(_mapper: object, _connection: object, _arm: ResponseArm) -> None:
    raise ValueError("response arms are append-only and cannot be deleted")


@event.listens_for(CostEvent, "before_update")
@event.listens_for(CostEvent, "before_delete")
def prevent_cost_event_mutation(_mapper: object, _connection: object, _event: CostEvent) -> None:
    raise ValueError("cost events are append-only; record a linked adjustment instead")


@event.listens_for(GenerationAttempt, "before_update")
@event.listens_for(GenerationAttempt, "before_delete")
@event.listens_for(AdmissionEvent, "before_update")
@event.listens_for(AdmissionEvent, "before_delete")
@event.listens_for(BedrockBillingCrosscheck, "before_update")
@event.listens_for(BedrockBillingCrosscheck, "before_delete")
@event.listens_for(BedrockBillingCrosscheckArm, "before_update")
@event.listens_for(BedrockBillingCrosscheckArm, "before_delete")
def prevent_append_only_evidence_mutation(
    _mapper: object,
    _connection: object,
    _record: object,
) -> None:
    raise ValueError("append-only evidence must be superseded by a new linked record")


@event.listens_for(RunEvent, "before_update")
def permit_only_run_event_redaction(
    _mapper: object,
    connection: Connection,
    record: RunEvent,
) -> None:
    previous = connection.execute(
        select(
            RunEvent.entity_type,
            RunEvent.entity_id,
            RunEvent.event_type,
            RunEvent.payload_json,
            RunEvent.created_at,
        ).where(RunEvent.id == record.id)
    ).one()
    if (
        (record.entity_type, record.entity_id, record.event_type)
        != (previous[0], previous[1], previous[2])
        or not _same_instant(record.created_at, previous[4])
        or previous[3] == TOOL_CALL_REDACTION_JSON
        or record.payload_json != TOOL_CALL_REDACTION_JSON
    ):
        raise ValueError("run-event evidence permits only one-way payload redaction")
    battle_id = (
        record.entity_id
        if record.entity_type == "battle"
        else _battle_id_for_arm(connection, record.entity_id)
        if record.entity_type == "response_arm"
        else None
    )
    if battle_id is None or not _retention_authorized_for_battle(connection, battle_id):
        raise ValueError("run-event redaction lacks an expired non-consent basis")


@event.listens_for(RunEvent, "before_delete")
def prevent_run_event_delete(
    _mapper: object,
    _connection: object,
    _record: RunEvent,
) -> None:
    raise ValueError("run events are append-only")


@event.listens_for(Incident, "before_update")
def permit_only_incident_redaction(
    _mapper: object,
    connection: Connection,
    record: Incident,
) -> None:
    previous = connection.execute(
        select(
            Incident.severity,
            Incident.code,
            Incident.detail,
            Incident.battle_id,
            Incident.created_at,
        ).where(Incident.id == record.id)
    ).one()
    if (
        (record.severity, record.code, record.battle_id) != (previous[0], previous[1], previous[3])
        or not _same_instant(record.created_at, previous[4])
        or previous[2] == TOOL_CALL_REDACTION_SENTINEL
        or record.detail != TOOL_CALL_REDACTION_SENTINEL
    ):
        raise ValueError("incident evidence permits only one-way detail redaction")
    if record.battle_id is None or not _retention_authorized_for_battle(
        connection, record.battle_id
    ):
        raise ValueError("incident redaction lacks an expired non-consent basis")


@event.listens_for(Incident, "before_delete")
def prevent_incident_delete(
    _mapper: object,
    _connection: object,
    _record: Incident,
) -> None:
    raise ValueError("incidents are append-only")


@event.listens_for(Job, "before_insert")
def require_queued_job_insert(_mapper: object, _connection: object, job: Job) -> None:
    if job.status not in {None, "queued"} or job.completed_at is not None:
        raise ValueError("jobs must be inserted queued without a completion timestamp")


@event.listens_for(Job, "before_update")
def prevent_job_terminal_evidence_mutation(
    _mapper: object, connection: Connection, job: Job
) -> None:
    state = inspect(job)
    last_error_history = state.attrs.last_error.history
    if (
        last_error_history.has_changes()
        and job.last_error == TOOL_CALL_REDACTION_SENTINEL
        and not _retention_authorized_for_battle(connection, job.battle_id)
    ):
        raise ValueError("job-error redaction lacks an expired non-consent basis")
    status_changed = state.attrs.status.history.has_changes()
    completed_changed = state.attrs.completed_at.history.has_changes()
    if not status_changed and not completed_changed:
        return
    previous_status, previous_completed = connection.execute(
        select(Job.status, Job.completed_at).where(Job.id == job.id)
    ).one()
    allowed = {
        ("queued", "running"),
        ("queued", "failed"),
        ("running", "queued"),
        ("running", "complete"),
        ("running", "failed"),
        ("running", "uncertain"),
        ("uncertain", "failed"),
    }
    if status_changed and (previous_status, job.status) not in allowed:
        raise ValueError("job lifecycle transition is invalid")
    if job.status in {"queued", "running"} and job.completed_at is not None:
        raise ValueError("nonterminal job cannot carry a completion timestamp")
    if job.status in {"complete", "failed", "uncertain"} and (
        job.completed_at is None or _as_utc(job.completed_at) < _as_utc(job.created_at)
    ):
        raise ValueError("terminal job evidence is incomplete")
    if previous_completed is not None and not _same_instant(job.completed_at, previous_completed):
        raise ValueError("job completion timestamp is write-once")


@event.listens_for(ToolCall, "before_insert")
def populate_tool_call_content_digests(
    _mapper: object, _connection: object, call: ToolCall
) -> None:
    """Bind every model-visible MCP payload before the trace row is inserted."""

    arguments_sha256 = _json_sha256({"arguments": call.arguments_json})
    result_sha256 = _text_sha256(call.result_text)
    structured_sha256 = _json_sha256({"structured": call.structured_content_json})
    if call.arguments_sha256 not in {None, arguments_sha256}:
        raise ValueError("tool-call arguments digest does not match its content")
    if call.result_sha256 != result_sha256:
        raise ValueError("tool-call result digest does not match its content")
    if call.structured_content_sha256 not in {None, structured_sha256}:
        raise ValueError("tool-call structured-content digest does not match its content")
    call.arguments_sha256 = arguments_sha256
    call.structured_content_sha256 = structured_sha256


@event.listens_for(ToolCall, "before_update")
def prevent_tool_call_mutation(_mapper: object, connection: Connection, call: ToolCall) -> None:
    """Permit only the privacy policy's one-way content erasure."""

    state = inspect(call)
    changed = {field for field in state.attrs.keys() if state.attrs[field].history.has_changes()}
    redaction_fields = {"arguments_json", "result_text", "structured_content_json"}
    is_redaction = (
        bool(changed)
        and changed <= redaction_fields
        and call.arguments_json == TOOL_CALL_REDACTION_JSON
        and call.result_text == TOOL_CALL_REDACTION_SENTINEL
        and call.structured_content_json == TOOL_CALL_REDACTION_JSON
    )
    if not is_redaction:
        raise ValueError(
            "tool-call traces are immutable except for one-way operational-retention redaction"
        )
    battle_id = _battle_id_for_arm(connection, call.arm_id)
    if battle_id is None or not _retention_authorized_for_battle(connection, battle_id):
        raise ValueError("tool-call redaction lacks an expired non-consent basis")


@event.listens_for(ToolCall, "before_delete")
def prevent_tool_call_delete(_mapper: object, _connection: object, _call: ToolCall) -> None:
    raise ValueError("tool-call trace records cannot be deleted")


@event.listens_for(ValidatorResult, "before_insert")
def populate_validator_detail_digest(
    _mapper: object, _connection: object, result: ValidatorResult
) -> None:
    detail_sha256 = _json_sha256({"detail": result.detail_json})
    if result.detail_sha256 not in {None, detail_sha256}:
        raise ValueError("validator detail digest does not match its content")
    result.detail_sha256 = detail_sha256


@event.listens_for(ValidatorResult, "before_update")
def prevent_validator_result_mutation(
    _mapper: object, connection: Connection, result: ValidatorResult
) -> None:
    state = inspect(result)
    changed = {field for field in state.attrs.keys() if state.attrs[field].history.has_changes()}
    if changed != {"detail_json"} or result.detail_json != TOOL_CALL_REDACTION_JSON:
        raise ValueError(
            "validator results are immutable except for one-way operational-retention redaction"
        )
    battle_id = _battle_id_for_arm(connection, result.arm_id)
    if battle_id is None or not _retention_authorized_for_battle(connection, battle_id):
        raise ValueError("validator-result redaction lacks an expired non-consent basis")


@event.listens_for(ValidatorResult, "before_delete")
def prevent_validator_result_delete(
    _mapper: object, _connection: object, _result: ValidatorResult
) -> None:
    raise ValueError("validator result records cannot be deleted")


@event.listens_for(TaskEvidenceArtifact, "before_insert")
def validate_task_evidence_artifact_insert(
    _mapper: object,
    _connection: object,
    artifact: TaskEvidenceArtifact,
) -> None:
    if artifact.revision_ordinal < 1:
        raise ValueError("task-evidence revision ordinal must be positive")
    if not isinstance(artifact.artifact_json, dict) or not isinstance(
        artifact.verification_receipt_json, dict
    ):
        raise ValueError("task-evidence content and verification receipt must be objects")
    embedded_artifact_sha256 = artifact.artifact_json.get("artifactSha256")
    artifact_payload = {
        key: value for key, value in artifact.artifact_json.items() if key != "artifactSha256"
    }
    if (
        embedded_artifact_sha256 != artifact.artifact_sha256
        or _evidence_json_sha256(artifact_payload) != artifact.artifact_sha256
    ):
        raise ValueError("task-evidence artifact digest mismatch")
    if artifact.artifact_json.get("schemaVersion") != artifact.schema_version:
        raise ValueError("task-evidence schema version mismatch")
    embedded_receipt_sha256 = artifact.verification_receipt_json.get("receipt_sha256")
    receipt_payload = {
        key: value
        for key, value in artifact.verification_receipt_json.items()
        if key != "receipt_sha256"
    }
    if (
        embedded_receipt_sha256 != artifact.verification_receipt_sha256
        or _evidence_json_sha256(receipt_payload) != artifact.verification_receipt_sha256
    ):
        raise ValueError("task-evidence verification receipt digest mismatch")
    expected_binding_sha256 = _evidence_json_sha256(
        {
            "artifact_sha256": artifact.artifact_sha256,
            "evidence_type": artifact.evidence_type,
            "revision_ordinal": artifact.revision_ordinal,
            "supersedes_artifact_id": artifact.supersedes_artifact_id,
            "task_id": artifact.task_id,
            "verification_receipt_sha256": artifact.verification_receipt_sha256,
        }
    )
    if artifact.task_binding_sha256 != expected_binding_sha256:
        raise ValueError("task-evidence database binding digest mismatch")


@event.listens_for(TaskEvidenceArtifact, "before_update")
@event.listens_for(TaskEvidenceArtifact, "before_delete")
def prevent_task_evidence_artifact_mutation(
    _mapper: object,
    _connection: object,
    _artifact: TaskEvidenceArtifact,
) -> None:
    raise ValueError("task-evidence artifacts are append-only")


@event.listens_for(ResearchReleaseArchive, "before_insert")
def validate_research_release_archive_insert(
    _mapper: object,
    _connection: object,
    archive: ResearchReleaseArchive,
) -> None:
    snapshot_ids = archive.snapshot_ids_json
    if (
        not isinstance(snapshot_ids, list)
        or not snapshot_ids
        or snapshot_ids != sorted(set(snapshot_ids))
        or any(not isinstance(item, str) or not item for item in snapshot_ids)
    ):
        raise ValueError("research archive snapshot IDs must be sorted and unique")
    if archive.snapshot_set_sha256 != _json_sha256({"snapshot_ids": snapshot_ids}):
        raise ValueError("research archive snapshot-set digest mismatch")
    if not isinstance(archive.manifest_json, dict) or archive.manifest_sha256 != _json_sha256(
        archive.manifest_json
    ):
        raise ValueError("research archive manifest digest mismatch")
    if (
        archive.public_key_sha256
        != hashlib.sha256(archive.public_key_pem.encode("utf-8")).hexdigest()
    ):
        raise ValueError("research archive public-key digest mismatch")
    for label, digest in (
        ("snapshot set", archive.snapshot_set_sha256),
        ("manifest", archive.manifest_sha256),
        ("archive", archive.archive_sha256),
        ("requirements lock", archive.requirements_lock_sha256),
    ):
        if not _is_sha256(digest):
            raise ValueError(f"research archive {label} digest is invalid")
    if not (
        isinstance(archive.build_image_digest, str)
        and archive.build_image_digest.startswith("sha256:")
        and _is_sha256(archive.build_image_digest.removeprefix("sha256:"))
    ):
        raise ValueError("research archive build-image digest is invalid")
    try:
        signature = base64.b64decode(archive.signature_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("research archive signature is not canonical base64") from exc
    if len(signature) != 64:
        raise ValueError("research archive Ed25519 signature must be 64 bytes")
    try:
        public_key = load_pem_public_key(archive.public_key_pem.encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("research archive public key is not valid PEM") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("research archive public key must be Ed25519")
    try:
        public_key.verify(
            signature,
            RESEARCH_ARCHIVE_SIGNATURE_CONTEXT + bytes.fromhex(archive.archive_sha256),
        )
    except InvalidSignature as exc:
        raise ValueError("research archive signature verification failed") from exc
    if archive.size_bytes <= 0 or archive.member_count <= 0:
        raise ValueError("research archive must contain non-empty sealed evidence")
    if archive.source_date_epoch != 0:
        raise ValueError("research archive requires the frozen zero source epoch")
    if archive.archive_class == "sanitized_public" and not archive.privacy_review_artifact_sha256:
        raise ValueError("public research archives require exact privacy-review evidence")


@event.listens_for(ResearchReleaseArchive, "before_update")
@event.listens_for(ResearchReleaseArchive, "before_delete")
def prevent_research_release_archive_mutation(
    _mapper: object,
    _connection: object,
    _archive: ResearchReleaseArchive,
) -> None:
    raise ValueError("research release archives are append-only")


@event.listens_for(SeasonModel, "before_update")
def prevent_frozen_endpoint_contract_mutation(
    _mapper: object, _connection: object, season_model: SeasonModel
) -> None:
    """Allow the initial freeze, then make its endpoint contract write-once."""

    state = inspect(season_model)
    manifest_history = state.attrs.manifest_sha256.history
    prior_manifest = (
        manifest_history.deleted[0] if manifest_history.deleted else season_model.manifest_sha256
    )
    if prior_manifest in {"", "unfrozen", "unresolved"}:
        return
    changed = [
        field
        for field in _IMMUTABLE_SEASON_MODEL_CONTRACT_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError(
            "frozen season endpoint contract is immutable; create a new season manifest: "
            + ", ".join(changed)
        )


@event.listens_for(Task, "before_update")
def prevent_frozen_task_mutation(_mapper: object, _connection: object, task: Task) -> None:
    """A frozen prompt registry can change only through a new task revision."""

    state = inspect(task)
    history = state.attrs.review_status.history
    prior_status = history.deleted[0] if history.deleted else task.review_status
    if prior_status != "frozen":
        return
    changed = [
        field for field in _IMMUTABLE_FROZEN_TASK_FIELDS if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError(
            "frozen task content is immutable; create a new task revision: " + ", ".join(changed)
        )


@event.listens_for(Season, "before_update")
def prevent_frozen_season_contract_mutation(
    _mapper: object, _connection: object, season: Season
) -> None:
    """Preserve frozen scientific hashes while allowing lifecycle and cost updates."""

    _require_nonnegative_budget(season)
    state = inspect(season)
    used_history = state.attrs.budget_used_micros.history
    if (
        used_history.has_changes()
        and used_history.deleted
        and season.budget_used_micros < used_history.deleted[0]
    ):
        raise ValueError("governed season spend cannot move backward")
    frozen_history = state.attrs.frozen_at.history
    prior_frozen_at = frozen_history.deleted[0] if frozen_history.deleted else season.frozen_at
    changed = []
    if prior_frozen_at is not None:
        changed.extend(
            field
            for field in _IMMUTABLE_FROZEN_SEASON_FIELDS
            if state.attrs[field].history.has_changes()
        )
    official_history = state.attrs.official.history
    prior_official = official_history.deleted[0] if official_history.deleted else season.official
    if prior_official and state.attrs.official.history.has_changes():
        changed.append("official")
    if changed:
        raise ValueError(
            "frozen season contract is immutable; create a new season: " + ", ".join(changed)
        )


@event.listens_for(LeaderboardSnapshot, "before_update")
def prevent_leaderboard_snapshot_content_mutation(
    _mapper: object, _connection: object, snapshot: LeaderboardSnapshot
) -> None:
    """Keep snapshot scope and content write-once; publication state may advance."""

    state = inspect(snapshot)
    changed = [
        field
        for field in _IMMUTABLE_LEADERBOARD_SNAPSHOT_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError(
            "leaderboard snapshot content is immutable; create a superseding snapshot: "
            + ", ".join(changed)
        )
    status_history = state.attrs.publication_status.history
    if status_history.has_changes() and status_history.deleted:
        prior_status = status_history.deleted[0]
        allowed = {
            ("draft", "published"),
            ("draft", "withdrawn"),
            ("published", "withdrawn"),
        }
        if (prior_status, snapshot.publication_status) not in allowed:
            raise ValueError(
                "leaderboard snapshot publication status cannot move backward: "
                f"{prior_status} -> {snapshot.publication_status}"
            )
        publication_metadata_changed = any(
            state.attrs[field].history.has_changes()
            for field in ("publication_reference_sha256", "published_at")
        )
        if snapshot.publication_status == "published":
            if not snapshot.publication_reference_sha256 or snapshot.published_at is None:
                raise ValueError(
                    "publishing a leaderboard snapshot requires reference and timestamp"
                )
        elif publication_metadata_changed:
            raise ValueError(
                "leaderboard publication metadata can change only during draft to published"
            )
    elif any(
        state.attrs[field].history.has_changes()
        for field in ("publication_reference_sha256", "published_at")
    ):
        raise ValueError(
            "leaderboard publication metadata can change only during draft to published"
        )


@event.listens_for(LeaderboardSnapshot, "before_insert")
def require_draft_leaderboard_snapshot_insert(
    _mapper: object, _connection: object, snapshot: LeaderboardSnapshot
) -> None:
    if snapshot.publication_status != "draft":
        raise ValueError("leaderboard snapshots must be inserted as drafts")
    if snapshot.publication_reference_sha256 is not None or snapshot.published_at is not None:
        raise ValueError("draft leaderboard snapshots cannot carry publication metadata")
    missing_or_invalid = [
        field
        for field in ("input_sha256", "input_evidence_sha256", "payload_sha256")
        if not _is_sha256(getattr(snapshot, field))
    ]
    if missing_or_invalid or snapshot.input_evidence_json is None:
        raise ValueError(
            "leaderboard snapshots must be inserted with sealed payload and evidence digests"
        )


@event.listens_for(LeaderboardSnapshot, "before_delete")
def prevent_leaderboard_snapshot_delete(
    _mapper: object, _connection: object, _snapshot: LeaderboardSnapshot
) -> None:
    raise ValueError("leaderboard snapshots are append-only and cannot be deleted")


@event.listens_for(ControlledRun, "before_insert")
def require_active_controlled_run_insert(
    _mapper: object, connection: Connection, run: ControlledRun
) -> None:
    _require_nonnegative_budget(run)
    if (
        run.status not in {None, "active"}
        or run.collection_completed_at is not None
        or run.closed_at is not None
        or run.revoked_at is not None
    ):
        raise ValueError("controlled runs must be inserted active")
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
        raise ValueError("commercial controlled-run binding must be complete")
    contract = connection.execute(
        select(
            EvaluationOrder.organization_id,
            EvaluationOrder.season_id,
            EvaluationOrder.route_revision_id,
            EvaluationOrder.status,
            EvaluationOrder.billing_status,
            EvaluationOrder.quote_reference_sha256,
            EvaluationOrder.rater_plan_sha256,
            EvaluationOrder.analysis_plan_sha256,
            EvaluationOrder.budget_cap_micros,
            EvaluationOrder.order_card_sha256,
            ModelRouteRevision.status,
            ModelRouteRevision.descriptor_sha256,
            ModelRouteRevision.data_policy_sha256,
            ModelRouteRevision.approved_season_id,
            ModelRouteRevision.approved_season_manifest_sha256,
            ModelSubmission.status,
            ModelSubmission.catalog_model_id,
            ModelSubmission.model_card_sha256,
            Season.manifest_sha256,
        )
        .join(
            ModelRouteRevision,
            ModelRouteRevision.id == EvaluationOrder.route_revision_id,
        )
        .join(ModelSubmission, ModelSubmission.id == EvaluationOrder.model_submission_id)
        .join(Season, Season.id == EvaluationOrder.season_id)
        .where(EvaluationOrder.id == run.evaluation_order_id)
    ).one_or_none()
    if contract is None:
        raise ValueError("commercial controlled-run order is unavailable")
    (
        organization_id,
        season_id,
        route_revision_id,
        order_status,
        billing_status,
        quote_reference_sha256,
        rater_plan_sha256,
        analysis_plan_sha256,
        budget_cap_micros,
        order_card_sha256,
        route_status,
        descriptor_sha256,
        data_policy_sha256,
        approved_season_id,
        approved_season_manifest_sha256,
        submission_status,
        catalog_model_id,
        model_card_sha256,
        season_manifest_sha256,
    ) = contract
    spend_authorization = connection.execute(
        select(
            GovernanceAcceptance.organization_id,
            GovernanceAcceptance.evaluation_order_id,
            GovernanceAcceptance.agreement_type,
            GovernanceAcceptance.binding_sha256,
            GovernanceAcceptance.status,
            GovernanceAcceptance.accepted_at,
            GovernanceAcceptance.expires_at,
        ).where(GovernanceAcceptance.id == run.spend_authorization_id)
    ).one_or_none()
    superseding_acceptance_id = connection.scalar(
        select(GovernanceAcceptance.id).where(
            GovernanceAcceptance.supersedes_acceptance_id == run.spend_authorization_id,
            GovernanceAcceptance.status == "active",
        )
    )
    spend_binding = {
        "orderCardSha256": order_card_sha256,
        "budgetCapMicros": budget_cap_micros,
        "currency": connection.scalar(
            select(EvaluationOrder.currency).where(EvaluationOrder.id == run.evaluation_order_id)
        ),
        "forecastCostMicros": connection.scalar(
            select(EvaluationOrder.forecast_cost_micros).where(
                EvaluationOrder.id == run.evaluation_order_id
            )
        ),
        "routeRevisionId": route_revision_id,
        "seasonId": season_id,
        "quoteReferenceSha256": quote_reference_sha256,
    }
    now = _as_utc(utcnow())
    if not (
        organization_id == run.organization_id
        and season_id == run.season_id
        and route_revision_id == run.route_revision_id
        and order_status == "approved"
        and billing_status == "authorized"
        and _is_sha256(quote_reference_sha256)
        and rater_plan_sha256 == run.rater_plan_sha256
        and analysis_plan_sha256 == run.analysis_plan_sha256
        and budget_cap_micros == run.budget_cap_micros
        and route_status == "approved"
        and descriptor_sha256 == run.endpoint_descriptor_sha256
        and data_policy_sha256 == run.data_policy_sha256
        and approved_season_id == run.season_id
        and approved_season_manifest_sha256 == season_manifest_sha256
        and submission_status == "approved"
        and catalog_model_id == run.submitted_endpoint_model_id
        and model_card_sha256 == run.submitted_model_card_sha256
        and run.organization_reference_sha256 == _text_sha256(run.organization_id or "")
        and spend_authorization is not None
        and spend_authorization.organization_id == run.organization_id
        and spend_authorization.evaluation_order_id == run.evaluation_order_id
        and spend_authorization.agreement_type == "spend_authorization"
        and spend_authorization.status == "active"
        and _as_utc(spend_authorization.accepted_at) <= now
        and (
            spend_authorization.expires_at is None or _as_utc(spend_authorization.expires_at) > now
        )
        and superseding_acceptance_id is None
        and spend_authorization.binding_sha256 == _json_sha256(spend_binding)
        and run.spend_authorization_binding_sha256 == spend_authorization.binding_sha256
    ):
        raise ValueError("commercial controlled-run contract does not match its approved order")


@event.listens_for(ControlledRun, "before_update")
def prevent_controlled_run_contract_mutation(
    _mapper: object, connection: Connection, run: ControlledRun
) -> None:
    """Preserve the signed protocol contract while allowing lifecycle changes."""

    _require_nonnegative_budget(run)
    state = inspect(run)
    used_history = state.attrs.budget_used_micros.history
    if (
        used_history.has_changes()
        and used_history.deleted
        and run.budget_used_micros < used_history.deleted[0]
    ):
        raise ValueError("governed controlled-run spend cannot move backward")
    changed = [
        field
        for field in _IMMUTABLE_CONTROLLED_RUN_CONTRACT_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError(
            "controlled-run contract is immutable; create a new controlled run: "
            + ", ".join(changed)
        )
    status_history = state.attrs.status.history
    collection_history = state.attrs.collection_completed_at.history
    closed_history = state.attrs.closed_at.history
    revoked_history = state.attrs.revoked_at.history
    if not status_history.has_changes():
        if any(
            history.has_changes()
            for history in (collection_history, closed_history, revoked_history)
        ):
            raise ValueError("controlled-run lifecycle timestamps are immutable")
        return
    previous_status, previous_collection, previous_closed, previous_revoked = connection.execute(
        select(
            ControlledRun.status,
            ControlledRun.collection_completed_at,
            ControlledRun.closed_at,
            ControlledRun.revoked_at,
        ).where(ControlledRun.id == run.id)
    ).one()
    completing = (
        previous_status == "active"
        and run.status == "collection_complete"
        and previous_collection is None
        and run.collection_completed_at is not None
        and previous_closed is None
        and run.closed_at is None
        and previous_revoked is None
        and run.revoked_at is None
    )
    closing = (
        previous_status == "collection_complete"
        and run.status == "closed"
        and previous_collection is not None
        and _same_instant(run.collection_completed_at, previous_collection)
        and previous_closed is None
        and run.closed_at is not None
        and previous_revoked is None
        and run.revoked_at is None
    )
    revoking = (
        previous_status != "revoked"
        and run.status == "revoked"
        and previous_revoked is None
        and run.revoked_at is not None
        and _same_instant(run.collection_completed_at, previous_collection)
        and _same_instant(run.closed_at, previous_closed)
    )
    if not (completing or closing or revoking):
        raise ValueError("controlled-run lifecycle transition is invalid")


@event.listens_for(ControlledRun, "before_delete")
def prevent_controlled_run_delete(
    _mapper: object, _connection: object, _run: ControlledRun
) -> None:
    raise ValueError("controlled runs are append-only")


@event.listens_for(ControlledRunAssignment, "before_insert")
def require_pending_controlled_assignment_insert(
    _mapper: object, _connection: object, assignment: ControlledRunAssignment
) -> None:
    if assignment.status not in {None, "pending"} or assignment.battle_id is not None:
        raise ValueError("controlled-run assignments must be inserted pending and unbound")


@event.listens_for(ControlledRunAssignment, "before_update")
def prevent_controlled_assignment_mutation(
    _mapper: object, connection: Connection, assignment: ControlledRunAssignment
) -> None:
    """Permit only pending-to-queued binding or pending-to-cancelled transition."""

    state = inspect(assignment)
    changed = [
        field
        for field in _IMMUTABLE_CONTROLLED_ASSIGNMENT_FIELDS
        if state.attrs[field].history.has_changes()
    ]
    if changed:
        raise ValueError(
            "controlled-run assignment is immutable; create a new controlled run: "
            + ", ".join(changed)
        )
    status_history = state.attrs.status.history
    battle_history = state.attrs.battle_id.history
    if not status_history.has_changes() and not battle_history.has_changes():
        return
    previous_status = status_history.deleted[0] if status_history.deleted else "pending"
    previous_battle_id = battle_history.deleted[0] if battle_history.deleted else None
    binding = (
        previous_status == "pending"
        and assignment.status == "queued"
        and previous_battle_id is None
        and assignment.battle_id is not None
    )
    cancellation = (
        previous_status == "pending"
        and assignment.status == "cancelled"
        and previous_battle_id is None
        and assignment.battle_id is None
    )
    bound_cancellation = (
        previous_status == "queued"
        and assignment.status == "cancelled"
        and previous_battle_id is not None
        and assignment.battle_id == previous_battle_id
    )
    if not (binding or cancellation or bound_cancellation):
        raise ValueError("controlled-run assignment lifecycle is write-once")
    if binding:
        battle = (
            connection.execute(
                sql_text(
                    """
                SELECT controlled_run_id, data_stratum, task_id, task_revision,
                       prompt_sha256, category, track, assignment_seed,
                       scheduler_version, track_assignment_probability,
                       model_assignment_probability, side_assignment_probability
                FROM battles WHERE id = :battle_id
                """
                ),
                {"battle_id": assignment.battle_id},
            )
            .mappings()
            .one_or_none()
        )
        expected = {
            "controlled_run_id": assignment.controlled_run_id,
            "data_stratum": "controlled",
            "task_id": assignment.task_id,
            "task_revision": assignment.task_revision,
            "prompt_sha256": assignment.task_prompt_sha256,
            "category": assignment.task_family,
            "track": assignment.track,
            "assignment_seed": assignment.assignment_seed,
            "scheduler_version": "controlled-frozen-schedule-v1",
            "track_assignment_probability": "1/1",
            "model_assignment_probability": "1/1",
            "side_assignment_probability": "1/2",
        }
        if battle is None or any(battle[field] != value for field, value in expected.items()):
            raise ValueError("controlled-run assignment does not match its battle")


@event.listens_for(ReviewerIdentityBinding, "before_insert")
def validate_reviewer_identity_binding_insert(
    _mapper: object,
    connection: Connection,
    binding: ReviewerIdentityBinding,
) -> None:
    if not all(
        _is_sha256(value)
        for value in (
            binding.person_commitment_sha256,
            binding.identity_issuer_sha256,
            binding.identity_evidence_sha256,
        )
    ):
        raise ValueError("reviewer identity bindings require SHA-256 evidence commitments")
    if binding.assurance_level != "server_verified" or (
        binding.verification_method != "season_hmac_issuer_subject_v1"
    ):
        raise ValueError("new reviewer identity bindings must be server verified")
    roles = binding.roles_json if isinstance(binding.roles_json, list) else []
    if not roles or len(roles) != len(set(roles)) or not set(roles).issubset(_REVIEWER_ROLES):
        raise ValueError("reviewer identity binding roles are invalid")
    reviewer = connection.execute(
        select(ExpertReviewer.active).where(ExpertReviewer.id == binding.reviewer_id)
    ).one_or_none()
    season = connection.execute(
        select(Season.id).where(Season.id == binding.season_id)
    ).one_or_none()
    if reviewer is None or reviewer[0] is not True or season is None:
        raise ValueError("reviewer identity binding requires an active reviewer and season")


@event.listens_for(ReviewerIdentityBinding, "before_update")
def prevent_reviewer_identity_binding_update(
    _mapper: object,
    _connection: object,
    binding: ReviewerIdentityBinding,
) -> None:
    changed = _changed_fields(binding, _IMMUTABLE_REVIEWER_IDENTITY_BINDING_FIELDS)
    if changed:
        raise ValueError("reviewer identity bindings are immutable: " + ", ".join(changed))


@event.listens_for(ReviewerIdentityBinding, "before_delete")
def prevent_reviewer_identity_binding_delete(
    _mapper: object,
    _connection: object,
    _binding: ReviewerIdentityBinding,
) -> None:
    raise ValueError("reviewer identity bindings are append-only")


@event.listens_for(ReviewerAccessCredential, "before_insert")
def validate_reviewer_access_credential_insert(
    _mapper: object,
    connection: Connection,
    credential: ReviewerAccessCredential,
) -> None:
    if not _is_sha256(credential.secret_hmac_sha256):
        raise ValueError("reviewer access credentials store only an HMAC-SHA256")
    scopes = credential.scopes_json if isinstance(credential.scopes_json, list) else []
    if (
        not scopes
        or len(scopes) != len(set(scopes))
        or any(not isinstance(scope, str) or not scope or len(scope) > 80 for scope in scopes)
    ):
        raise ValueError("reviewer access credential scopes are invalid")
    if credential.status not in {None, "active"} or (credential.use_count or 0) != 0:
        raise ValueError("new reviewer access credentials must be unused and active")
    if credential.consumed_at is not None or credential.revoked_at is not None:
        raise ValueError("new reviewer access credentials cannot carry terminal timestamps")
    binding = connection.execute(
        select(
            ReviewerIdentityBinding.season_id,
            ReviewerIdentityBinding.reviewer_id,
            ReviewerIdentityBinding.assurance_level,
        ).where(ReviewerIdentityBinding.id == credential.identity_binding_id)
    ).one_or_none()
    if binding is None or tuple(binding) != (
        credential.season_id,
        credential.reviewer_id,
        "server_verified",
    ):
        raise ValueError("reviewer access credential does not match its verified binding")


@event.listens_for(ReviewerAccessCredential, "before_update")
def validate_reviewer_access_credential_update(
    _mapper: object,
    _connection: object,
    credential: ReviewerAccessCredential,
) -> None:
    changed = _changed_fields(credential, _IMMUTABLE_REVIEWER_ACCESS_CREDENTIAL_FIELDS)
    if changed:
        raise ValueError("reviewer access credential contract is immutable: " + ", ".join(changed))
    state = inspect(credential)
    previous_uses = int(_prior_value(credential, "use_count"))
    previous_status = str(_prior_value(credential, "status"))
    previous_consumed = _prior_value(credential, "consumed_at")
    previous_revoked = _prior_value(credential, "revoked_at")
    if credential.use_count < previous_uses or credential.use_count > credential.maximum_uses:
        raise ValueError("reviewer credential use count must be monotone and bounded")
    if credential.use_count - previous_uses > 1:
        raise ValueError("a reviewer credential may be consumed only once per transaction")
    if previous_status in {"consumed", "revoked"} and credential.status != previous_status:
        raise ValueError("terminal reviewer credential status is immutable")
    if credential.status == "active":
        if credential.use_count >= credential.maximum_uses:
            raise ValueError("an exhausted reviewer credential cannot remain active")
        if credential.consumed_at is not None or credential.revoked_at is not None:
            raise ValueError("active reviewer credentials cannot carry terminal timestamps")
    elif credential.status == "consumed":
        if credential.use_count != credential.maximum_uses or credential.consumed_at is None:
            raise ValueError("consumed reviewer credentials require exact use exhaustion")
        if credential.revoked_at is not None:
            raise ValueError("consumed reviewer credentials cannot also be revoked")
    elif credential.status == "revoked":
        if credential.revoked_at is None or credential.consumed_at is not None:
            raise ValueError("revoked reviewer credentials require only a revocation timestamp")
    else:
        raise ValueError("reviewer credential status is invalid")
    if previous_consumed is not None and not _same_instant(
        credential.consumed_at, previous_consumed
    ):
        raise ValueError("reviewer credential consumption timestamp is immutable")
    if previous_revoked is not None and not _same_instant(credential.revoked_at, previous_revoked):
        raise ValueError("reviewer credential revocation timestamp is immutable")
    if not any(
        state.attrs[field].history.has_changes()
        for field in ("use_count", "status", "consumed_at", "revoked_at")
    ):
        raise ValueError("reviewer credential update has no governed lifecycle change")


@event.listens_for(ReviewerAccessCredential, "before_delete")
def prevent_reviewer_access_credential_delete(
    _mapper: object,
    _connection: object,
    _credential: ReviewerAccessCredential,
) -> None:
    raise ValueError("reviewer access credentials cannot be deleted")


@event.listens_for(ReviewerEnrollmentOffer, "before_insert")
def validate_reviewer_enrollment_offer_insert(
    _mapper: object,
    _connection: object,
    offer: ReviewerEnrollmentOffer,
) -> None:
    if not all(
        _is_sha256(value)
        for value in (
            offer.secret_hmac_sha256,
            offer.consent_document_sha256,
            offer.activation_manifest_sha256,
        )
    ):
        raise ValueError("reviewer enrollment offers require SHA-256 commitments")
    if offer.status not in {None, "active"} or any(
        value is not None
        for value in (offer.accepted_at, offer.accepted_request_sha256, offer.revoked_at)
    ):
        raise ValueError("new reviewer enrollment offers must be unused and active")


@event.listens_for(ReviewerEnrollmentOffer, "before_update")
def validate_reviewer_enrollment_offer_update(
    _mapper: object,
    _connection: object,
    offer: ReviewerEnrollmentOffer,
) -> None:
    changed = _changed_fields(offer, _IMMUTABLE_REVIEWER_ENROLLMENT_OFFER_FIELDS)
    if changed:
        raise ValueError("reviewer enrollment offer contract is immutable: " + ", ".join(changed))
    previous_status = str(_prior_value(offer, "status"))
    if previous_status != "active" or offer.status not in {"accepted", "revoked"}:
        raise ValueError("terminal reviewer enrollment offers are immutable")
    if offer.status == "accepted":
        if offer.accepted_at is None or not _is_sha256(offer.accepted_request_sha256):
            raise ValueError("accepted enrollment offers require an acceptance commitment")
        if offer.revoked_at is not None:
            raise ValueError("accepted enrollment offers cannot also be revoked")
    elif offer.revoked_at is None or any(
        value is not None for value in (offer.accepted_at, offer.accepted_request_sha256)
    ):
        raise ValueError("revoked enrollment offers require only a revocation timestamp")


@event.listens_for(ReviewerEnrollmentOffer, "before_delete")
def prevent_reviewer_enrollment_offer_delete(
    _mapper: object, _connection: object, _offer: ReviewerEnrollmentOffer
) -> None:
    raise ValueError("reviewer enrollment offers cannot be deleted")


@event.listens_for(ReviewerConsentAcceptance, "before_insert")
def validate_reviewer_consent_acceptance_insert(
    _mapper: object,
    connection: Connection,
    acceptance: ReviewerConsentAcceptance,
) -> None:
    if not all(
        _is_sha256(value)
        for value in (
            acceptance.consent_document_sha256,
            acceptance.activation_manifest_sha256,
            acceptance.retention_policy_sha256,
            acceptance.acceptance_statement_sha256,
            acceptance.confirmation_set_sha256,
            acceptance.request_sha256,
            acceptance.receipt_secret_hmac_sha256,
            acceptance.receipt_sha256,
        )
    ):
        raise ValueError("reviewer consent acceptance requires SHA-256 commitments")
    offer = connection.execute(
        select(
            ReviewerEnrollmentOffer.season_id,
            ReviewerEnrollmentOffer.consent_document_sha256,
            ReviewerEnrollmentOffer.activation_manifest_sha256,
            ReviewerEnrollmentOffer.status,
            ReviewerEnrollmentOffer.accepted_request_sha256,
        ).where(ReviewerEnrollmentOffer.id == acceptance.enrollment_offer_id)
    ).one_or_none()
    if offer is None or tuple(offer) != (
        acceptance.season_id,
        acceptance.consent_document_sha256,
        acceptance.activation_manifest_sha256,
        "accepted",
        acceptance.request_sha256,
    ):
        raise ValueError("reviewer consent acceptance does not match one consumed offer")


@event.listens_for(ReviewerParticipationLifecycle, "before_insert")
def validate_reviewer_participation_insert(
    _mapper: object,
    connection: Connection,
    lifecycle: ReviewerParticipationLifecycle,
) -> None:
    if not _is_sha256(lifecycle.audit_marker_sha256) or lifecycle.status not in {None, "active"}:
        raise ValueError("new reviewer participation must be active and pseudonymous")
    if any(
        value is not None
        for value in (
            lifecycle.withdrawn_at,
            lifecycle.assignments_stopped_at,
            lifecycle.withdrawal_receipt_sha256,
            lifecycle.redacted_at,
            lifecycle.deletion_receipt_sha256,
        )
    ):
        raise ValueError("new reviewer participation cannot carry terminal state")
    aligned = connection.execute(
        select(
            ReviewerConsentAcceptance.season_id,
            ReviewerIdentityBinding.season_id,
            ReviewerIdentityBinding.reviewer_id,
            ExpertReviewer.active,
        )
        .select_from(ReviewerConsentAcceptance)
        .join(
            ReviewerIdentityBinding,
            ReviewerIdentityBinding.id == lifecycle.identity_binding_id,
        )
        .join(ExpertReviewer, ExpertReviewer.id == lifecycle.reviewer_id)
        .where(ReviewerConsentAcceptance.id == lifecycle.consent_acceptance_id)
    ).one_or_none()
    if aligned is None or tuple(aligned) != (
        lifecycle.season_id,
        lifecycle.season_id,
        lifecycle.reviewer_id,
        True,
    ):
        raise ValueError("reviewer participation consent and identity are misaligned")


@event.listens_for(ReviewerParticipationLifecycle, "before_update")
def validate_reviewer_participation_update(
    _mapper: object,
    connection: Connection,
    lifecycle: ReviewerParticipationLifecycle,
) -> None:
    changed = _changed_fields(lifecycle, _IMMUTABLE_REVIEWER_PARTICIPATION_FIELDS)
    if changed:
        raise ValueError("reviewer participation identity is immutable: " + ", ".join(changed))
    previous_status = str(_prior_value(lifecycle, "status"))
    if previous_status == "active" and lifecycle.status == "withdrawn":
        if (
            lifecycle.withdrawn_at is None
            or lifecycle.assignments_stopped_at is None
            or not _is_sha256(lifecycle.withdrawal_receipt_sha256)
            or lifecycle.redacted_at is not None
            or lifecycle.deletion_receipt_sha256 is not None
        ):
            raise ValueError("withdrawn reviewer participation is incomplete")
        receipt = connection.execute(
            select(ReviewerWithdrawalReceipt.id).where(
                ReviewerWithdrawalReceipt.lifecycle_id == lifecycle.id,
                ReviewerWithdrawalReceipt.receipt_sha256 == lifecycle.withdrawal_receipt_sha256,
            )
        ).one_or_none()
        if receipt is None:
            raise ValueError("withdrawal transition requires its append-only receipt")
    elif previous_status == "withdrawn" and lifecycle.status == "redacted":
        if lifecycle.redacted_at is None or not _is_sha256(lifecycle.deletion_receipt_sha256):
            raise ValueError("redacted reviewer participation is incomplete")
        receipt = connection.execute(
            select(ReviewerDeletionReceipt.id).where(
                ReviewerDeletionReceipt.lifecycle_id == lifecycle.id,
                ReviewerDeletionReceipt.receipt_sha256 == lifecycle.deletion_receipt_sha256,
            )
        ).one_or_none()
        if receipt is None:
            raise ValueError("redaction transition requires its append-only receipt")
    else:
        raise ValueError("reviewer participation lifecycle is monotone and terminal")


@event.listens_for(ReviewerWithdrawalReceipt, "before_insert")
def validate_reviewer_withdrawal_receipt_insert(
    _mapper: object,
    connection: Connection,
    receipt: ReviewerWithdrawalReceipt,
) -> None:
    if not _is_sha256(receipt.request_sha256) or not _is_sha256(receipt.receipt_sha256):
        raise ValueError("reviewer withdrawal receipt requires SHA-256 commitments")
    lifecycle = connection.execute(
        select(
            ReviewerParticipationLifecycle.consent_acceptance_id,
            ReviewerParticipationLifecycle.season_id,
            ReviewerParticipationLifecycle.reviewer_id,
            ReviewerParticipationLifecycle.identity_binding_id,
            ReviewerParticipationLifecycle.status,
        ).where(ReviewerParticipationLifecycle.id == receipt.lifecycle_id)
    ).one_or_none()
    if lifecycle is None or tuple(lifecycle) != (
        receipt.consent_acceptance_id,
        receipt.season_id,
        receipt.reviewer_id,
        receipt.identity_binding_id,
        "active",
    ):
        raise ValueError("reviewer withdrawal receipt does not match active participation")
    if receipt.prior_judgments_preserved is not True:
        raise ValueError("reviewer withdrawal cannot rewrite prior judgments")


@event.listens_for(ReviewerRetentionSchedule, "before_insert")
def validate_reviewer_retention_schedule_insert(
    _mapper: object,
    connection: Connection,
    schedule: ReviewerRetentionSchedule,
) -> None:
    if not _is_sha256(schedule.retention_policy_sha256) or not _is_sha256(schedule.schedule_sha256):
        raise ValueError("reviewer retention schedule requires SHA-256 commitments")
    lifecycle = connection.execute(
        select(
            ReviewerParticipationLifecycle.season_id,
            ReviewerParticipationLifecycle.reviewer_id,
        ).where(ReviewerParticipationLifecycle.id == schedule.lifecycle_id)
    ).one_or_none()
    if lifecycle is None or tuple(lifecycle) != (schedule.season_id, schedule.reviewer_id):
        raise ValueError("reviewer retention schedule scope is misaligned")


@event.listens_for(ReviewerDeletionReceipt, "before_insert")
def validate_reviewer_deletion_receipt_insert(
    _mapper: object,
    connection: Connection,
    receipt: ReviewerDeletionReceipt,
) -> None:
    if not all(
        _is_sha256(value)
        for value in (
            receipt.request_sha256,
            receipt.private_payload_before_sha256,
            receipt.audit_marker_sha256,
            receipt.receipt_sha256,
        )
    ):
        raise ValueError("reviewer deletion receipt requires SHA-256 commitments")
    lifecycle = connection.execute(
        select(
            ReviewerParticipationLifecycle.season_id,
            ReviewerParticipationLifecycle.reviewer_id,
            ReviewerParticipationLifecycle.audit_marker_sha256,
            ReviewerParticipationLifecycle.status,
        ).where(ReviewerParticipationLifecycle.id == receipt.lifecycle_id)
    ).one_or_none()
    schedule = connection.execute(
        select(
            ReviewerRetentionSchedule.lifecycle_id,
            ReviewerRetentionSchedule.direct_payload_delete_due_at,
            ReviewerRetentionSchedule.pseudonymous_audit_retain_until,
        ).where(ReviewerRetentionSchedule.id == receipt.retention_schedule_id)
    ).one_or_none()
    if lifecycle is None or tuple(lifecycle) != (
        receipt.season_id,
        receipt.reviewer_id,
        receipt.audit_marker_sha256,
        "withdrawn",
    ):
        raise ValueError("reviewer deletion receipt requires withdrawn participation")
    if schedule is None or (
        schedule[0] != receipt.lifecycle_id
        or not _same_instant(schedule[1], receipt.direct_payload_delete_due_at)
        or not _same_instant(schedule[2], receipt.pseudonymous_audit_retain_until)
    ):
        raise ValueError("reviewer deletion receipt deadlines are misaligned")
    if receipt.prior_judgments_preserved is not True:
        raise ValueError("reviewer deletion cannot rewrite prior judgments")


def _prevent_participant_receipt_update(
    record: object,
    fields: tuple[str, ...],
    label: str,
) -> None:
    _prevent_immutable_reviewer_evidence_update(record, fields, label)


def _prevent_participant_receipt_delete(
    _mapper: object, _connection: object, _record: object
) -> None:
    raise ValueError("participant lifecycle receipts and schedules are append-only")


def _participant_receipt_update_listener(
    fields: tuple[str, ...],
    label: str,
) -> object:
    def listener(_mapper: object, _connection: object, record: object) -> None:
        _prevent_participant_receipt_update(record, fields, label)

    return listener


for _participant_receipt_model, _participant_receipt_fields, _participant_label in (
    (
        ReviewerConsentAcceptance,
        _IMMUTABLE_REVIEWER_CONSENT_ACCEPTANCE_FIELDS,
        "reviewer consent acceptance",
    ),
    (
        ReviewerWithdrawalReceipt,
        _IMMUTABLE_REVIEWER_WITHDRAWAL_RECEIPT_FIELDS,
        "reviewer withdrawal receipt",
    ),
    (
        ReviewerRetentionSchedule,
        _IMMUTABLE_REVIEWER_RETENTION_SCHEDULE_FIELDS,
        "reviewer retention schedule",
    ),
    (
        ReviewerDeletionReceipt,
        _IMMUTABLE_REVIEWER_DELETION_RECEIPT_FIELDS,
        "reviewer deletion receipt",
    ),
):
    event.listen(
        _participant_receipt_model,
        "before_update",
        _participant_receipt_update_listener(
            _participant_receipt_fields,
            _participant_label,
        ),
    )
    event.listen(
        _participant_receipt_model,
        "before_delete",
        _prevent_participant_receipt_delete,
    )


@event.listens_for(ExpertReviewer, "before_update")
def validate_expert_reviewer_privacy_transition(
    _mapper: object,
    connection: Connection,
    reviewer: ExpertReviewer,
) -> None:
    previous = str(_prior_value(reviewer, "privacy_status"))
    if previous == "redacted" and any(
        inspect(reviewer).attrs[field].history.has_changes()
        for field in (
            "profile_json",
            "qualification_json",
            "qualification_verified",
            "privacy_status",
            "privacy_redacted_at",
            "privacy_redaction_receipt_sha256",
            "active",
            "revoked_at",
        )
    ):
        raise ValueError("redacted reviewer private payload is terminal")
    if reviewer.privacy_status == "redacted" and previous != "redacted":
        receipt = connection.execute(
            select(
                ReviewerDeletionReceipt.audit_marker_sha256,
                ReviewerDeletionReceipt.private_payload_before_sha256,
            ).where(
                ReviewerDeletionReceipt.reviewer_id == reviewer.id,
                ReviewerDeletionReceipt.receipt_sha256 == reviewer.privacy_redaction_receipt_sha256,
            )
        ).one_or_none()
        expected_profile = (
            {
                "schema_version": "flavourbench-reviewer-redacted-profile-v1",
                "privacy_status": "redacted",
                "audit_marker_sha256": receipt[0],
                "private_payload_before_sha256": receipt[1],
            }
            if receipt is not None
            else None
        )
        if (
            previous != "retained"
            or reviewer.active
            or reviewer.revoked_at is None
            or reviewer.qualification_json != []
            or reviewer.qualification_verified
            or reviewer.privacy_redacted_at is None
            or not _is_sha256(reviewer.privacy_redaction_receipt_sha256)
            or not isinstance(reviewer.profile_json, dict)
            or reviewer.profile_json != expected_profile
        ):
            raise ValueError("reviewer private payload redaction is incomplete")


@event.listens_for(ReviewerQualificationEvidence, "before_insert")
def validate_reviewer_qualification_insert(
    _mapper: object,
    connection: Connection,
    evidence: ReviewerQualificationEvidence,
) -> None:
    digests = (
        evidence.qualification_evidence_sha256,
        evidence.independence_evidence_sha256,
        evidence.conflict_disclosure_sha256,
        evidence.consent_document_sha256,
        evidence.training_material_sha256,
        evidence.verifier_principal_sha256,
    )
    if not all(_is_sha256(value) for value in digests):
        raise ValueError("reviewer qualification evidence requires SHA-256 commitments")
    if evidence.family not in _REVIEWER_FAMILIES:
        raise ValueError("reviewer qualification family is invalid")
    expected_cohort = _REVIEWER_COHORT_BY_AFFILIATION.get(evidence.affiliation_class)
    if expected_cohort is None:
        raise ValueError("reviewer qualification affiliation is invalid")
    if evidence.affiliation_class == "independent_external" and not (
        evidence.independence_verified and evidence.conflict_cleared
    ):
        raise ValueError("independent qualification requires verified independence and cleared COI")
    if evidence.affiliation_class != "independent_external" and evidence.independence_verified:
        raise ValueError("affiliated qualification cannot claim independent verification")
    binding = connection.execute(
        select(
            ReviewerIdentityBinding.season_id,
            ReviewerIdentityBinding.reviewer_id,
            ReviewerIdentityBinding.assurance_level,
        ).where(ReviewerIdentityBinding.id == evidence.identity_binding_id)
    ).one_or_none()
    if binding is None or tuple(binding) != (
        evidence.season_id,
        evidence.reviewer_id,
        "server_verified",
    ):
        raise ValueError("reviewer qualification evidence does not match its identity binding")


@event.listens_for(ReviewerCalibrationSet, "before_insert")
def validate_reviewer_calibration_set_insert(
    _mapper: object,
    _connection: object,
    calibration_set: ReviewerCalibrationSet,
) -> None:
    if calibration_set.family not in _REVIEWER_FAMILIES:
        raise ValueError("reviewer calibration family is invalid")
    if not all(
        _is_sha256(value)
        for value in (
            calibration_set.calibration_set_sha256,
            calibration_set.source_artifact_sha256,
            calibration_set.scoring_key_sha256,
        )
    ):
        raise ValueError("reviewer calibration sets require SHA-256 commitments")
    if calibration_set.synthetic_arms != 0:
        raise ValueError("reviewer calibration sets cannot contain synthetic arms")


@event.listens_for(ReviewerCalibrationBallot, "before_insert")
def validate_reviewer_calibration_ballot_insert(
    _mapper: object,
    connection: Connection,
    ballot: ReviewerCalibrationBallot,
) -> None:
    if not _is_sha256(ballot.ballot_sha256) or not _is_sha256(ballot.scoring_result_sha256):
        raise ValueError("reviewer calibration ballots require SHA-256 commitments")
    expected_accuracy = (1000 * ballot.correct_count + ballot.item_count // 2) // ballot.item_count
    if ballot.accuracy_milli != expected_accuracy:
        raise ValueError("reviewer calibration accuracy does not match exact counts")
    rows = connection.execute(
        select(
            ReviewerIdentityBinding.season_id,
            ReviewerIdentityBinding.reviewer_id,
            ReviewerCalibrationSet.season_id,
            ReviewerCalibrationSet.item_count,
            ReviewerCalibrationSet.frozen_at,
        )
        .select_from(ReviewerIdentityBinding)
        .join(
            ReviewerCalibrationSet,
            ReviewerCalibrationSet.id == ballot.calibration_set_id,
        )
        .where(ReviewerIdentityBinding.id == ballot.identity_binding_id)
    ).one_or_none()
    if rows is None or (
        rows[0] != ballot.season_id
        or rows[1] != ballot.reviewer_id
        or rows[2] != ballot.season_id
        or rows[3] != ballot.item_count
        or _as_utc(ballot.completed_at) < _as_utc(rows[4])
    ):
        raise ValueError("reviewer calibration ballot does not match its binding and frozen set")


@event.listens_for(ReviewerFamilyAdmission, "before_insert")
def validate_reviewer_family_admission_insert(
    _mapper: object,
    connection: Connection,
    admission: ReviewerFamilyAdmission,
) -> None:
    if admission.family not in _REVIEWER_FAMILIES or admission.review_role not in _REVIEWER_ROLES:
        raise ValueError("reviewer family admission scope is invalid")
    if not all(
        _is_sha256(value)
        for value in (
            admission.admission_policy_sha256,
            admission.evidence_bundle_sha256,
            admission.decision_reference_sha256,
        )
    ):
        raise ValueError("reviewer family admission requires SHA-256 commitments")
    if admission.admission_policy_sha256 != _json_sha256(admission.admission_policy_json):
        raise ValueError("reviewer family admission policy digest does not match its payload")
    policy = admission.admission_policy_json
    if not isinstance(policy, dict) or policy.get("schema_version") != (
        "flavourbench-reviewer-admission-policy-v1"
    ):
        raise ValueError("reviewer family admission policy is invalid")
    requires_calibration = policy.get("requires_calibration")
    minimum_accuracy = policy.get("minimum_accuracy_milli")
    if (
        not isinstance(requires_calibration, bool)
        or not isinstance(minimum_accuracy, int)
        or not (0 <= minimum_accuracy <= 1000)
    ):
        raise ValueError("reviewer family admission policy thresholds are invalid")
    binding = connection.execute(
        select(
            ReviewerIdentityBinding.season_id,
            ReviewerIdentityBinding.reviewer_id,
            ReviewerIdentityBinding.assurance_level,
            ReviewerIdentityBinding.roles_json,
        ).where(ReviewerIdentityBinding.id == admission.identity_binding_id)
    ).one_or_none()
    evidence = connection.execute(
        select(
            ReviewerQualificationEvidence.season_id,
            ReviewerQualificationEvidence.reviewer_id,
            ReviewerQualificationEvidence.identity_binding_id,
            ReviewerQualificationEvidence.family,
            ReviewerQualificationEvidence.affiliation_class,
            ReviewerQualificationEvidence.independence_verified,
            ReviewerQualificationEvidence.conflict_cleared,
            ReviewerQualificationEvidence.qualification_evidence_sha256,
            ReviewerQualificationEvidence.verified_at,
            ReviewerQualificationEvidence.valid_until,
        ).where(ReviewerQualificationEvidence.id == admission.qualification_evidence_id)
    ).one_or_none()
    if binding is None or (
        binding[0] != admission.season_id
        or binding[1] != admission.reviewer_id
        or binding[2] != "server_verified"
        or admission.review_role not in binding[3]
    ):
        raise ValueError("reviewer family admission does not match its identity binding")
    if evidence is None or (
        evidence[0] != admission.season_id
        or evidence[1] != admission.reviewer_id
        or evidence[2] != admission.identity_binding_id
        or evidence[3] != admission.family
        or _REVIEWER_COHORT_BY_AFFILIATION.get(evidence[4]) != admission.cohort
        or _as_utc(admission.valid_from) < _as_utc(evidence[8])
        or (evidence[9] is not None and _as_utc(admission.valid_until) > _as_utc(evidence[9]))
    ):
        raise ValueError("reviewer family admission does not match qualification evidence")
    if admission.cohort == "expert_independent" and not (evidence[5] and evidence[6]):
        raise ValueError("independent admission requires verified independence and cleared COI")
    calibration_ballot_sha256: str | None = None
    if admission.calibration_ballot_id is not None:
        ballot = connection.execute(
            select(
                ReviewerCalibrationBallot.season_id,
                ReviewerCalibrationBallot.reviewer_id,
                ReviewerCalibrationBallot.identity_binding_id,
                ReviewerCalibrationBallot.ballot_sha256,
                ReviewerCalibrationBallot.accuracy_milli,
                ReviewerCalibrationBallot.passed,
                ReviewerCalibrationBallot.completed_at,
                ReviewerCalibrationSet.family,
            )
            .join(
                ReviewerCalibrationSet,
                ReviewerCalibrationSet.id == ReviewerCalibrationBallot.calibration_set_id,
            )
            .where(ReviewerCalibrationBallot.id == admission.calibration_ballot_id)
        ).one_or_none()
        if ballot is None or (
            ballot[0] != admission.season_id
            or ballot[1] != admission.reviewer_id
            or ballot[2] != admission.identity_binding_id
            or ballot[4] < minimum_accuracy
            or ballot[5] is not True
            or _as_utc(ballot[6]) > _as_utc(admission.valid_from)
            or ballot[7] != admission.family
        ):
            raise ValueError("reviewer family admission calibration is inadmissible")
        calibration_ballot_sha256 = ballot[3]
    elif requires_calibration:
        raise ValueError("reviewer family admission requires a calibration ballot")
    expected_evidence_sha256 = _reviewer_admission_evidence_sha256(
        admission,
        qualification_evidence_sha256=evidence[7],
        calibration_ballot_sha256=calibration_ballot_sha256,
    )
    if admission.evidence_bundle_sha256 != expected_evidence_sha256:
        raise ValueError("reviewer family admission evidence digest does not reconcile")


@event.listens_for(TaskValidationAuditAuthorization, "before_insert")
def validate_task_validation_audit_authorization_insert(
    _mapper: object,
    connection: Connection,
    authorization: TaskValidationAuditAuthorization,
) -> None:
    if authorization.audit_kind != "rights":
        raise ValueError(
            "task-validation contamination authorization requires a five-method successor"
        )
    if authorization.cohort != "expert_independent":
        raise ValueError("task-validation auditors must be independent")
    if not all(
        _is_sha256(value)
        for value in (
            authorization.campaign_sha256,
            authorization.qualification_evidence_sha256,
            authorization.conflict_evidence_sha256,
            authorization.automated_evidence_sha256,
            authorization.audit_plan_sha256,
            authorization.decision_reference_sha256,
            authorization.authorization_sha256,
        )
    ):
        raise ValueError("task-validation audit authorization requires SHA-256 evidence")
    if authorization.audit_plan_sha256 != _json_sha256(authorization.audit_plan_json):
        raise ValueError("task-validation audit plan digest does not match its payload")
    if not (
        authorization.campaign_sha256 == TASK_VALIDATION_V6_CAMPAIGN_SHA256
        and authorization.automated_evidence_sha256 == TASK_VALIDATION_V1_REPLAY_SHA256
        and authorization.audit_plan_json == rights_audit_plan()
        and authorization.audit_plan_sha256 == TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256
    ):
        raise ValueError("task-validation audit replay binding is inadmissible")
    if authorization.authorization_sha256 != _task_validation_audit_authorization_sha256(
        authorization
    ):
        raise ValueError("task-validation audit authorization digest does not reconcile")
    binding = connection.execute(
        select(
            ReviewerIdentityBinding.season_id,
            ReviewerIdentityBinding.reviewer_id,
            ReviewerIdentityBinding.assurance_level,
            ExpertReviewer.active,
            ExpertReviewer.cohort,
        )
        .join(ExpertReviewer, ExpertReviewer.id == ReviewerIdentityBinding.reviewer_id)
        .where(ReviewerIdentityBinding.id == authorization.identity_binding_id)
    ).one_or_none()
    if binding is None or (
        binding[0] != authorization.season_id
        or binding[1] != authorization.reviewer_id
        or binding[2] != "server_verified"
        or binding[3] is not True
        or binding[4] != "expert_independent"
    ):
        raise ValueError("task-validation audit authorization identity is inadmissible")


@event.listens_for(TaskValidationCampaignEvent, "before_insert")
def validate_task_validation_campaign_event_insert(
    _mapper: object,
    connection: Connection,
    campaign_event: TaskValidationCampaignEvent,
) -> None:
    if campaign_event.created_at is None:
        campaign_event.created_at = utcnow()
    allowed_types = {
        "blind_ballot",
        "criterion_pack_confirmation",
        "adjudication",
        "rights_batch_audit",
        "contamination_batch_audit",
    }
    if campaign_event.event_type not in allowed_types or campaign_event.sequence < 1:
        raise ValueError("task-validation campaign event envelope is invalid")
    if not all(
        _is_sha256(value)
        for value in (
            campaign_event.campaign_sha256,
            campaign_event.person_commitment_sha256,
            campaign_event.reviewer_admission_receipt_sha256,
            campaign_event.previous_event_sha256,
            campaign_event.event_sha256,
        )
    ):
        raise ValueError("task-validation campaign event requires SHA-256 commitments")
    if campaign_event.event_sha256 != _task_validation_campaign_event_sha256(campaign_event):
        raise ValueError("task-validation campaign event digest does not reconcile")
    binding = connection.execute(
        select(
            ReviewerIdentityBinding.season_id,
            ReviewerIdentityBinding.reviewer_id,
            ReviewerIdentityBinding.person_commitment_sha256,
            ReviewerIdentityBinding.assurance_level,
        ).where(ReviewerIdentityBinding.id == campaign_event.identity_binding_id)
    ).one_or_none()
    if binding is None or (
        binding[0] != campaign_event.season_id
        or binding[1] != campaign_event.reviewer_id
        or binding[2] != campaign_event.person_commitment_sha256
        or binding[3] != "server_verified"
    ):
        raise ValueError("task-validation campaign event identity is inadmissible")
    if campaign_event.event_type in {
        "blind_ballot",
        "criterion_pack_confirmation",
        "adjudication",
    }:
        expected_role = (
            "task_adjudicator" if campaign_event.event_type == "adjudication" else "task_validator"
        )
        admission = connection.execute(
            select(
                ReviewerFamilyAdmission.season_id,
                ReviewerFamilyAdmission.reviewer_id,
                ReviewerFamilyAdmission.identity_binding_id,
                ReviewerFamilyAdmission.review_role,
                ReviewerFamilyAdmission.cohort,
                ReviewerFamilyAdmission.evidence_bundle_sha256,
                ReviewerFamilyAdmission.valid_from,
                ReviewerFamilyAdmission.valid_until,
            ).where(ReviewerFamilyAdmission.id == campaign_event.family_admission_id)
        ).one_or_none()
        if admission is None or (
            admission[0] != campaign_event.season_id
            or admission[1] != campaign_event.reviewer_id
            or admission[2] != campaign_event.identity_binding_id
            or admission[3] != expected_role
            or admission[4] != "expert_independent"
            or admission[5] != campaign_event.reviewer_admission_receipt_sha256
            or _as_utc(campaign_event.created_at) < _as_utc(admission[6])
            or _as_utc(campaign_event.created_at) > _as_utc(admission[7])
        ):
            raise ValueError("task-validation campaign event admission is inadmissible")
    else:
        authorization = connection.execute(
            select(
                TaskValidationAuditAuthorization.season_id,
                TaskValidationAuditAuthorization.reviewer_id,
                TaskValidationAuditAuthorization.identity_binding_id,
                TaskValidationAuditAuthorization.audit_kind,
                TaskValidationAuditAuthorization.authorization_sha256,
                TaskValidationAuditAuthorization.campaign_sha256,
                TaskValidationAuditAuthorization.automated_evidence_sha256,
                TaskValidationAuditAuthorization.audit_plan_json,
                TaskValidationAuditAuthorization.audit_plan_sha256,
            ).where(TaskValidationAuditAuthorization.id == campaign_event.audit_authorization_id)
        ).one_or_none()
        payload = campaign_event.payload_json
        if authorization is None or (
            authorization[0] != campaign_event.season_id
            or authorization[1] != campaign_event.reviewer_id
            or authorization[2] != campaign_event.identity_binding_id
            or authorization[3] != "rights"
            or authorization[4] != campaign_event.reviewer_admission_receipt_sha256
            or authorization[5] != TASK_VALIDATION_V6_CAMPAIGN_SHA256
            or authorization[6] != TASK_VALIDATION_V1_REPLAY_SHA256
            or authorization[7] != rights_audit_plan()
            or authorization[8] != TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256
            or campaign_event.event_type != "rights_batch_audit"
            or payload.get("audit_kind") != "rights"
            or payload.get("audit_plan_sha256") != TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256
            or payload.get("automated_evidence_sha256") != TASK_VALIDATION_V1_REPLAY_SHA256
            or payload.get("automated_evidence_verified") is not True
            or payload.get("rights_snapshot_integrity_verified") is not True
            or payload.get("local_prompt_risk_replay_verified") is not True
            or payload.get("contamination_campaign_coverage_verified") is not False
            or payload.get("reviewed_candidate_ids") != list(TASK_VALIDATION_RIGHTS_REQUIRED_IDS)
        ):
            raise ValueError("task-validation campaign audit event is inadmissible")


def _prevent_immutable_reviewer_evidence_update(
    record: object,
    fields: tuple[str, ...],
    label: str,
) -> None:
    changed = _changed_fields(record, fields)
    if changed:
        raise ValueError(f"{label} is immutable: " + ", ".join(changed))


@event.listens_for(ReviewerQualificationEvidence, "before_update")
def prevent_reviewer_qualification_update(
    _mapper: object, _connection: object, record: ReviewerQualificationEvidence
) -> None:
    _prevent_immutable_reviewer_evidence_update(
        record, _IMMUTABLE_REVIEWER_QUALIFICATION_FIELDS, "reviewer qualification evidence"
    )


@event.listens_for(ReviewerCalibrationSet, "before_update")
def prevent_reviewer_calibration_set_update(
    _mapper: object, _connection: object, record: ReviewerCalibrationSet
) -> None:
    _prevent_immutable_reviewer_evidence_update(
        record, _IMMUTABLE_REVIEWER_CALIBRATION_SET_FIELDS, "reviewer calibration set"
    )


@event.listens_for(ReviewerCalibrationBallot, "before_update")
def prevent_reviewer_calibration_ballot_update(
    _mapper: object, _connection: object, record: ReviewerCalibrationBallot
) -> None:
    _prevent_immutable_reviewer_evidence_update(
        record, _IMMUTABLE_REVIEWER_CALIBRATION_BALLOT_FIELDS, "reviewer calibration ballot"
    )


@event.listens_for(ReviewerFamilyAdmission, "before_update")
def prevent_reviewer_family_admission_update(
    _mapper: object, _connection: object, record: ReviewerFamilyAdmission
) -> None:
    _prevent_immutable_reviewer_evidence_update(
        record, _IMMUTABLE_REVIEWER_FAMILY_ADMISSION_FIELDS, "reviewer family admission"
    )


@event.listens_for(TaskValidationAuditAuthorization, "before_update")
def prevent_task_validation_audit_authorization_update(
    _mapper: object,
    _connection: object,
    record: TaskValidationAuditAuthorization,
) -> None:
    _prevent_immutable_reviewer_evidence_update(
        record,
        _IMMUTABLE_TASK_VALIDATION_AUDIT_AUTHORIZATION_FIELDS,
        "task-validation audit authorization",
    )


@event.listens_for(TaskValidationCampaignEvent, "before_update")
def prevent_task_validation_campaign_event_update(
    _mapper: object,
    _connection: object,
    record: TaskValidationCampaignEvent,
) -> None:
    _prevent_immutable_reviewer_evidence_update(
        record,
        _IMMUTABLE_TASK_VALIDATION_CAMPAIGN_EVENT_FIELDS,
        "task-validation campaign event",
    )


def _prevent_immutable_reviewer_evidence_delete(
    _mapper: object, _connection: object, _record: object
) -> None:
    raise ValueError("reviewer admission evidence is append-only")


for _immutable_reviewer_model in (
    ReviewerQualificationEvidence,
    ReviewerCalibrationSet,
    ReviewerCalibrationBallot,
    ReviewerFamilyAdmission,
    TaskValidationAuditAuthorization,
    TaskValidationCampaignEvent,
):
    event.listen(
        _immutable_reviewer_model,
        "before_delete",
        _prevent_immutable_reviewer_evidence_delete,
    )


@event.listens_for(Vote, "before_insert")
@event.listens_for(Vote, "before_update")
def validate_vote_domain(_mapper: object, connection: Connection, vote: Vote) -> None:
    if vote.choice not in {"left", "right", "tie", "both_bad"}:
        raise ValueError("vote choice is outside the frozen preference domain")
    if vote.cohort not in {
        "public",
        "expert_independent",
        "expert_product_affiliated",
        "expert_provider_affiliated",
    }:
        raise ValueError("vote cohort is outside the governed rater domain")
    if vote.provenance_status not in {
        "legacy_unverified",
        "public_pseudonymous",
        "expert_verified_v1",
    }:
        raise ValueError("vote provenance status is invalid")
    provenance_ids = (
        vote.reviewer_id,
        vote.reviewer_identity_binding_id,
        vote.reviewer_family_admission_id,
    )
    if vote.provenance_status == "expert_verified_v1":
        if not vote.cohort.startswith("expert_") or any(value is None for value in provenance_ids):
            raise ValueError("verified expert votes require complete reviewer provenance")
        if vote.provenance_sha256 != _verified_vote_provenance_sha256(vote):
            raise ValueError("verified expert vote provenance digest does not reconcile")
        created_at = vote.created_at or utcnow()
        provenance = connection.execute(
            select(
                Battle.season_id,
                Battle.category,
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
            )
            .select_from(Battle)
            .join(
                ReviewerIdentityBinding,
                ReviewerIdentityBinding.id == vote.reviewer_identity_binding_id,
            )
            .join(
                ReviewerFamilyAdmission,
                ReviewerFamilyAdmission.id == vote.reviewer_family_admission_id,
            )
            .join(ExpertReviewer, ExpertReviewer.id == vote.reviewer_id)
            .where(Battle.id == vote.battle_id)
        ).one_or_none()
        if provenance is None or (
            provenance[0] != provenance[2]
            or provenance[0] != provenance[5]
            or provenance[1] != provenance[8]
            or provenance[3] != vote.reviewer_id
            or provenance[4] != "server_verified"
            or provenance[6] != vote.reviewer_id
            or provenance[7] != vote.reviewer_identity_binding_id
            or provenance[9] != "output_rater"
            or provenance[10] != vote.cohort
            or _as_utc(created_at) < _as_utc(provenance[11])
            or _as_utc(created_at) > _as_utc(provenance[12])
            or provenance[13] is not True
        ):
            raise ValueError("verified expert vote provenance is not admitted for this battle")
    else:
        if any(value is not None for value in provenance_ids):
            raise ValueError("unverified votes cannot carry reviewer provenance foreign keys")
        if vote.provenance_status == "public_pseudonymous":
            if vote.cohort != "public" or vote.provenance_sha256 is not None:
                raise ValueError("public pseudonymous vote provenance is malformed")
        elif vote.provenance_sha256 is not None:
            raise ValueError("legacy unverified votes cannot carry a provenance digest")
    state = inspect(vote)
    if not state.persistent:
        admissible = connection.execute(
            sql_text(
                """
                SELECT 1
                FROM battles AS b
                JOIN response_arms AS left_arm
                  ON left_arm.id = b.left_arm_id
                 AND left_arm.battle_id = b.id
                 AND left_arm.side = 'left'
                 AND left_arm.status = 'complete'
                JOIN response_arms AS right_arm
                  ON right_arm.id = b.right_arm_id
                 AND right_arm.battle_id = b.id
                 AND right_arm.side = 'right'
                 AND right_arm.status = 'complete'
                WHERE b.id = :battle_id
                  AND b.status = 'complete'
                  AND b.completed_at IS NOT NULL
                  AND :created_at >= b.completed_at
                  AND b.left_arm_id IS NOT NULL
                  AND b.right_arm_id IS NOT NULL
                  AND b.left_arm_id <> b.right_arm_id
                """
            ),
            {"battle_id": vote.battle_id, "created_at": vote.created_at or utcnow()},
        ).one_or_none()
        if admissible is None:
            raise ValueError("vote does not follow a completed anonymous battle")
    if state.persistent and any(
        state.attrs[field].history.has_changes()
        for field in (
            "battle_id",
            "rater_pseudonym",
            "cohort",
            "choice",
            "reason_tags_json",
            "rubric_json",
            "idempotency_key",
            "reviewer_id",
            "reviewer_identity_binding_id",
            "reviewer_family_admission_id",
            "provenance_status",
            "provenance_sha256",
            "created_at",
        )
    ):
        raise ValueError("votes are append-only and cannot be updated")


@event.listens_for(Vote, "before_delete")
def prevent_vote_delete(_mapper: object, _connection: object, _vote: Vote) -> None:
    raise ValueError("votes are append-only and cannot be deleted")
