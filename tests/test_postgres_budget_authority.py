from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json, Jsonb
from pydantic import ValidationError
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from flavourbench.anonymous_pool_rotation import (
    ROTATION_EVENT_TYPE,
    rotate_anonymous_pool,
)
from flavourbench.author_evaluator_import import import_bundle, load_bundle
from flavourbench.budget_integrity import assert_budget_integrity
from flavourbench.config import get_settings
from flavourbench.database import (
    EXPECTED_SCHEMA_REVISION,
    _assert_postgresql_budget_authority,
    _assert_postgresql_normal_finish_guards,
    _assert_postgresql_participant_lifecycle_guards,
    _assert_postgresql_reviewer_task_validation_guards,
)
from flavourbench.models import (
    Battle,
    CatalogModel,
    CostEvent,
    ExpertReviewer,
    GenerationAttempt,
    ProviderAccountAuthorization,
    ProviderAccountBudget,
    ResponseArm,
    RunEvent,
    Season,
    SeasonProviderBudget,
)
from flavourbench.schemas import BedrockBillingCrosscheckCreate
from flavourbench.task_validation_replay_binding import (
    TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
    TASK_VALIDATION_RIGHTS_REQUIRED_IDS,
    TASK_VALIDATION_V1_REPLAY_SHA256,
    TASK_VALIDATION_V6_CAMPAIGN_SHA256,
    rights_audit_plan,
)

POSTGRES_URL = os.environ.get("FLAVOURBENCH_TEST_POSTGRES_URL", "")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="FLAVOURBENCH_TEST_POSTGRES_URL was not provided",
)

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REVIEW_CANDIDATE_SHA256 = "94e917b6c202eb49953f3a8c22f897301eaa7ffba47116b83c915d17a6850b69"
REVIEW_CANDIDATE_PATH = (
    SERVICE_ROOT
    / "artifacts"
    / "expert-calibration"
    / "candidate-v11"
    / f"candidate-pack-{REVIEW_CANDIDATE_SHA256}.json"
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _psycopg_url(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


@pytest.fixture(scope="module")
def postgres_database_url() -> str:
    """Create and later destroy a migration-managed database for this module."""

    source = make_url(POSTGRES_URL)
    if source.database is None or "test" not in source.database.lower():
        pytest.fail("FLAVOURBENCH_TEST_POSTGRES_URL must name a disposable test database")
    owner_url = source.set(drivername="postgresql+psycopg", database="postgres")
    owner_dsn = _psycopg_url(owner_url.render_as_string(hide_password=False))
    database_name = f"flavourbench_authority_{uuid.uuid4().hex[:12]}"

    with psycopg.connect(owner_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_user, rolsuper OR rolcreatedb "
                "FROM pg_catalog.pg_roles WHERE rolname = current_user"
            )
            identity = cursor.fetchone()
            if identity != ("flavourbench_owner", True):
                pytest.fail(
                    "the PostgreSQL authority suite requires a disposable "
                    "flavourbench_owner URL with CREATEDB"
                )
            cursor.execute(
                sql.SQL("CREATE DATABASE {} OWNER flavourbench_owner").format(
                    sql.Identifier(database_name)
                )
            )

    test_url = source.set(
        drivername="postgresql+psycopg",
        database=database_name,
    ).render_as_string(hide_password=False)
    migration_environment = os.environ.copy()
    migration_environment.update(
        {
            "FLAVOURBENCH_DATABASE_URL": test_url,
            "FLAVOURBENCH_ENVIRONMENT": "test",
            "FLAVOURBENCH_SERVICE_ROLE": "migration",
            "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
        }
    )
    try:
        migration = subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                "alembic.ini",
                "upgrade",
                "head",
            ],
            cwd=SERVICE_ROOT,
            env=migration_environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if migration.returncode:
            pytest.fail(
                f"temporary PostgreSQL migration failed:\n{migration.stdout}\n{migration.stderr}"
            )
        yield test_url
    finally:
        with psycopg.connect(owner_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                        sql.Identifier(database_name)
                    )
                )


def test_qwencloud_backend_is_bound_into_postgresql_budget_authority(
    postgres_database_url: str,
) -> None:
    with psycopg.connect(_psycopg_url(postgres_database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.flavourbench_reserve_battle_budget(text)'"
                "::pg_catalog.regprocedure)"
            )
            reservation_definition = cursor.fetchone()[0]
            assert reservation_definition.count("qwencloud_direct") == 2

            for table_name, constraint_name in (
                (
                    "season_provider_budgets",
                    "ck_season_provider_budgets_backend",
                ),
                (
                    "provider_account_budgets",
                    "ck_provider_account_budgets_backend",
                ),
                (
                    "provider_account_authorizations",
                    "ck_provider_account_authorizations_backend",
                ),
            ):
                cursor.execute(
                    "SELECT pg_catalog.pg_get_constraintdef(c.oid) "
                    "FROM pg_catalog.pg_constraint AS c "
                    "JOIN pg_catalog.pg_class AS r ON r.oid = c.conrelid "
                    "WHERE r.relname = %s AND c.conname = %s",
                    (table_name, constraint_name),
                )
                constraint_definition = cursor.fetchone()
                assert constraint_definition is not None
                assert "qwencloud_direct" in constraint_definition[0]


def test_task_validation_batch_audit_is_singleton_in_postgresql(
    postgres_database_url: str,
) -> None:
    marker = uuid.uuid4().hex
    season_id = str(uuid.uuid4())
    reviewer_id = str(uuid.uuid4())
    binding_id = str(uuid.uuid4())
    authorization_id = str(uuid.uuid4())
    campaign_sha256 = TASK_VALIDATION_V6_CAMPAIGN_SHA256
    person_commitment = _sha256(f"person-{marker}")
    authorization_sha256 = _sha256(f"authorization-{marker}")
    offer_id = str(uuid.uuid4())
    acceptance_id = str(uuid.uuid4())
    lifecycle_id = str(uuid.uuid4())
    consent_sha256 = _sha256(f"consent-{marker}")
    activation_sha256 = _sha256(f"activation-{marker}")
    retention_sha256 = _sha256(f"retention-{marker}")
    acceptance_request_sha256 = _sha256(f"acceptance-request-{marker}")
    acceptance_receipt_sha256 = _sha256(f"acceptance-receipt-{marker}")
    audit_plan = rights_audit_plan()
    audit_payload = {
        "audit_kind": "rights",
        "audit_plan_sha256": TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
        "automated_evidence_sha256": TASK_VALIDATION_V1_REPLAY_SHA256,
        "automated_evidence_verified": True,
        "rights_snapshot_integrity_verified": True,
        "local_prompt_risk_replay_verified": True,
        "contamination_campaign_coverage_verified": False,
        "reviewed_candidate_ids": list(TASK_VALIDATION_RIGHTS_REQUIRED_IDS),
    }
    now = datetime.now(UTC)
    with psycopg.connect(_psycopg_url(postgres_database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT indexdef FROM pg_catalog.pg_indexes "
                "WHERE schemaname = 'public' AND indexname = "
                "'uq_task_validation_campaign_events_audit_authorization_type'"
            )
            index_definition = cursor.fetchone()
            assert index_definition is not None
            assert "UNIQUE INDEX" in index_definition[0]
            assert "audit_authorization_id IS NOT NULL" in index_definition[0]

            cursor.execute(
                """
                INSERT INTO seasons (
                    id, slug, name, status, official, manifest_sha256,
                    prompt_registry_sha256, tool_registry_sha256,
                    epicure_release_id, epicure_bundle_sha256,
                    epicure_application_sha256, budget_cap_micros,
                    budget_used_micros, budget_reserved_micros, created_at,
                    analysis_plan_sha256, protocol_bundle_json,
                    protocol_bundle_sha256
                ) VALUES (
                    %s, %s, %s, 'draft', false, %s, %s, %s, %s, %s, %s,
                    0, 0, 0, %s, %s, '{}'::json, %s
                )
                """,
                (
                    season_id,
                    f"audit-singleton-{marker}",
                    "Audit singleton PostgreSQL regression",
                    _sha256(f"manifest-{marker}"),
                    _sha256(f"prompts-{marker}"),
                    _sha256(f"tools-{marker}"),
                    f"epicure-{marker}",
                    _sha256(f"bundle-{marker}"),
                    _sha256(f"application-{marker}"),
                    now,
                    _sha256(f"analysis-{marker}"),
                    _sha256(f"protocol-{marker}"),
                ),
            )
            cursor.execute(
                """
                INSERT INTO reviewer_enrollment_offers (
                    id, season_id, credential_prefix, secret_hmac_sha256,
                    hmac_key_id, consent_document_sha256,
                    activation_manifest_sha256, status, not_before, expires_at,
                    created_at
                ) VALUES (%s, %s, %s, %s, 'test-key', %s, %s, 'active',
                          %s, %s, %s)
                """,
                (
                    offer_id,
                    season_id,
                    marker[:16],
                    _sha256(f"offer-secret-{marker}"),
                    consent_sha256,
                    activation_sha256,
                    now - timedelta(seconds=1),
                    now + timedelta(hours=1),
                    now,
                ),
            )
            cursor.execute(
                """
                UPDATE reviewer_enrollment_offers
                SET status = 'accepted', accepted_at = %s,
                    accepted_request_sha256 = %s
                WHERE id = %s
                """,
                (now, acceptance_request_sha256, offer_id),
            )
            cursor.execute(
                """
                INSERT INTO reviewer_consent_acceptances (
                    id, enrollment_offer_id, season_id,
                    consent_document_sha256, activation_manifest_sha256,
                    retention_policy_sha256, acceptance_statement_sha256,
                    confirmation_set_sha256, request_sha256, receipt_prefix,
                    receipt_secret_hmac_sha256, hmac_key_id, receipt_sha256,
                    accepted_at, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                          'test-key', %s, %s, %s)
                """,
                (
                    acceptance_id,
                    offer_id,
                    season_id,
                    consent_sha256,
                    activation_sha256,
                    retention_sha256,
                    _sha256(f"acceptance-statement-{marker}"),
                    _sha256(f"confirmation-set-{marker}"),
                    acceptance_request_sha256,
                    _sha256(f"receipt-prefix-{marker}")[:16],
                    _sha256(f"receipt-secret-{marker}"),
                    acceptance_receipt_sha256,
                    now,
                    now,
                ),
            )
            cursor.execute(
                """
                INSERT INTO expert_reviewers (
                    id, reviewer_code, invitation_sha256, qualification_json,
                    active, created_at, qualification_verified, cohort,
                    profile_json, batch_reveal_only
                ) VALUES (%s, %s, %s, '[]'::json, true, %s, true,
                          'expert_independent', %s, true)
                """,
                (
                    reviewer_id,
                    f"audit-singleton-{marker}",
                    _sha256(f"invitation-{marker}"),
                    now,
                    Json(
                        {
                            "consent_acceptance_sha256": acceptance_receipt_sha256,
                            "activation_manifest_sha256": activation_sha256,
                        }
                    ),
                ),
            )
            cursor.execute(
                """
                INSERT INTO reviewer_identity_bindings (
                    id, season_id, reviewer_id, person_commitment_sha256,
                    identity_issuer_sha256, identity_evidence_sha256,
                    hmac_key_id, verification_method, assurance_level,
                    roles_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, 'test-key',
                          'season_hmac_issuer_subject_v1', 'server_verified',
                          '["task_validator"]'::json, %s)
                """,
                (
                    binding_id,
                    season_id,
                    reviewer_id,
                    person_commitment,
                    _sha256(f"issuer-{marker}"),
                    _sha256(f"identity-{marker}"),
                    now,
                ),
            )
            cursor.execute(
                """
                INSERT INTO reviewer_participation_lifecycles (
                    id, consent_acceptance_id, season_id, reviewer_id,
                    identity_binding_id, audit_marker_sha256, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, 'active', %s)
                """,
                (
                    lifecycle_id,
                    acceptance_id,
                    season_id,
                    reviewer_id,
                    binding_id,
                    _sha256(f"audit-marker-{marker}"),
                    now,
                ),
            )
            cursor.execute(
                """
                INSERT INTO task_validation_audit_authorizations (
                    id, season_id, campaign_sha256, reviewer_id,
                    identity_binding_id, audit_kind, cohort,
                    qualification_evidence_sha256, conflict_evidence_sha256,
                    automated_evidence_sha256, audit_plan_json,
                    audit_plan_sha256, decision_reference_sha256,
                    authorization_sha256, created_at
                ) VALUES (%s, %s, %s, %s, %s, 'rights',
                          'expert_independent', %s, %s, %s, %s,
                          %s, %s, %s, %s)
                """,
                (
                    authorization_id,
                    season_id,
                    campaign_sha256,
                    reviewer_id,
                    binding_id,
                    _sha256(f"qualification-{marker}"),
                    _sha256(f"conflict-{marker}"),
                    TASK_VALIDATION_V1_REPLAY_SHA256,
                    Json(audit_plan),
                    TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
                    _sha256(f"decision-{marker}"),
                    authorization_sha256,
                    now,
                ),
            )
            event_statement = """
                INSERT INTO task_validation_campaign_events (
                    id, season_id, campaign_sha256, sequence, event_id,
                    event_type, candidate_id, reviewer_id,
                    identity_binding_id, family_admission_id,
                    audit_authorization_id, reviewer_pseudonym,
                    person_commitment_sha256,
                    reviewer_admission_receipt_sha256, payload_json,
                    previous_event_sha256, event_sha256, created_at
                ) VALUES (%s, %s, %s, %s, %s, 'rights_batch_audit', NULL,
                          %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            first_parameters = (
                str(uuid.uuid4()),
                season_id,
                campaign_sha256,
                1,
                f"audit-event-1-{marker}",
                reviewer_id,
                binding_id,
                authorization_id,
                f"reviewer-{marker}",
                person_commitment,
                authorization_sha256,
                Json(audit_payload),
                "0" * 64,
                _sha256(f"event-1-{marker}"),
                now,
            )
            invalid_payload = dict(audit_payload)
            invalid_payload["reviewed_candidate_ids"] = list(
                reversed(TASK_VALIDATION_RIGHTS_REQUIRED_IDS)
            )
            invalid_parameters = list(first_parameters)
            invalid_parameters[11] = Json(invalid_payload)
            with (
                pytest.raises(
                    psycopg.errors.RaiseException,
                    match="task-validation audit event replay binding is inadmissible",
                ),
                connection.transaction(),
            ):
                cursor.execute(event_statement, invalid_parameters)
            cursor.execute(event_statement, first_parameters)
            second_parameters = (
                str(uuid.uuid4()),
                season_id,
                campaign_sha256,
                2,
                f"audit-event-2-{marker}",
                reviewer_id,
                binding_id,
                authorization_id,
                f"reviewer-{marker}",
                person_commitment,
                authorization_sha256,
                Json(audit_payload),
                _sha256(f"event-1-{marker}"),
                _sha256(f"event-2-{marker}"),
                now,
            )
            with pytest.raises(psycopg.errors.UniqueViolation):
                cursor.execute(event_statement, second_parameters)
        connection.rollback()


def test_postgresql_replay_guard_rejects_contamination_and_changed_plan(
    postgres_database_url: str,
) -> None:
    statement = """
        INSERT INTO task_validation_audit_authorizations (
            id, season_id, campaign_sha256, reviewer_id,
            identity_binding_id, audit_kind, cohort,
            qualification_evidence_sha256, conflict_evidence_sha256,
            automated_evidence_sha256, audit_plan_json,
            audit_plan_sha256, decision_reference_sha256,
            authorization_sha256, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, 'expert_independent',
                  %s, %s, %s, %s, %s, %s, %s, %s)
    """
    for audit_kind, plan in (
        ("contamination", rights_audit_plan()),
        (
            "rights",
            {
                **rights_audit_plan(),
                "sample_candidate_ids": list(reversed(rights_audit_plan()["sample_candidate_ids"])),
            },
        ),
    ):
        marker = uuid.uuid4().hex
        connection = psycopg.connect(_psycopg_url(postgres_database_url))
        try:
            with (
                connection.cursor() as cursor,
                pytest.raises(
                    psycopg.errors.RaiseException,
                    match="task-validation audit replay binding is inadmissible",
                ),
            ):
                cursor.execute(
                    statement,
                    (
                        str(uuid.uuid4()),
                        str(uuid.uuid4()),
                        TASK_VALIDATION_V6_CAMPAIGN_SHA256,
                        str(uuid.uuid4()),
                        str(uuid.uuid4()),
                        audit_kind,
                        _sha256(f"qualification-{marker}"),
                        _sha256(f"conflict-{marker}"),
                        TASK_VALIDATION_V1_REPLAY_SHA256,
                        Json(plan),
                        TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
                        _sha256(f"decision-{marker}"),
                        _sha256(f"authorization-{marker}"),
                        datetime.now(UTC),
                    ),
                )
            connection.rollback()
        finally:
            connection.close()


@dataclass(frozen=True)
class AuthorityFixture:
    database_url: str
    season_id: str
    season_slug: str
    arm_ids: list[str]
    authorization_reference_sha256: str
    authorization_envelope_sha256: str
    generation_request_map_sha256: str
    coverage_start: datetime
    coverage_end: datetime
    baseline_used_micros: int = 1_000
    rate_card_micros: int = 1_000

    def request(
        self,
        label: str,
        *,
        billed_usage_micros: int,
        supersedes_crosscheck_id: str | None = None,
        source_artifact_uri: str | None = None,
        generation_request_map_sha256: str | None = None,
    ) -> dict[str, object]:
        return {
            "arm_ids": self.arm_ids,
            "source_kind": "aws_cur",
            "source_artifact_uri": (source_artifact_uri or f"s3://billing-test/{label}"),
            "source_artifact_sha256": _sha256(f"source:{label}"),
            "statement_sha256": _sha256(f"statement:{label}"),
            "generation_request_map_sha256": (
                generation_request_map_sha256 or self.generation_request_map_sha256
            ),
            "coverage_start": self.coverage_start.isoformat(),
            "coverage_end": self.coverage_end.isoformat(),
            "billed_usage_micros": billed_usage_micros,
            "credits_policy": "gross_usage_before_credits_excluding_tax",
            "authorization_reference_sha256": (self.authorization_reference_sha256),
            "supersedes_crosscheck_id": supersedes_crosscheck_id,
        }


def _authorization_material(
    account: ProviderAccountBudget,
    *,
    suffix: str,
    valid_until: datetime,
) -> dict[str, Any]:
    reference_sha256 = _sha256(f"authorization-reference:{suffix}")
    exposure = {
        "schema_version": "postgres-authority-integration-v1",
        "used_micros": 0,
        "reserved_micros": 0,
    }
    exposure_sha256 = _canonical_sha256(exposure)
    binding = {
        "schema_version": "postgres-authority-binding-v1",
        "credential_scope_sha256": account.account_scope_sha256,
    }
    binding_sha256 = _canonical_sha256(binding)
    envelope = {
        "schema_version": "flavourbench-provider-account-authorization-v3",
        "provider_account_budget_id": account.id,
        "execution_backend": account.execution_backend,
        "currency": account.currency,
        "budget_cap_micros": account.budget_cap_micros,
        "account_scope_sha256": account.account_scope_sha256,
        "authorization_reference_sha256": reference_sha256,
        "ledger_opening_balance_sha256": account.opening_balance_sha256,
        "exposure_attestation_sha256": exposure_sha256,
        "cumulative_used_micros": 0,
        "cumulative_reserved_micros": 0,
        "credential_binding_sha256": binding_sha256,
        "supersedes_authorization_envelope_sha256": None,
        "valid_until": valid_until.isoformat(),
    }
    envelope_sha256 = _canonical_sha256(envelope)
    signature = hmac.new(
        get_settings().budget_authorization_signing_secret.encode(),
        envelope_sha256.encode(),
        hashlib.sha256,
    ).hexdigest()
    return {
        "reference_sha256": reference_sha256,
        "exposure": exposure,
        "exposure_sha256": exposure_sha256,
        "binding": binding,
        "binding_sha256": binding_sha256,
        "envelope": envelope,
        "envelope_sha256": envelope_sha256,
        "signature": signature,
    }


def _seed_authority_fixture(database_url: str, label: str) -> AuthorityFixture:
    engine = create_engine(database_url)
    suffix = f"{label}-{uuid.uuid4().hex[:8]}"
    scope_sha256 = _sha256(f"account-scope:{suffix}")
    valid_until = datetime.now(UTC) + timedelta(days=30)
    opening_balance = {
        "used_micros": 0,
        "reserved_micros": 0,
        "fixture": suffix,
    }
    opening_balance_sha256 = _canonical_sha256(opening_balance)
    model_id = f"postgres-authority/model-{suffix}"

    try:
        with Session(engine) as session:
            season = Season(
                slug=f"postgres-authority-{suffix}",
                name="PostgreSQL budget authority integration fixture",
                status="active",
                official=True,
                epicure_release_id="postgres-authority-integration-release",
                budget_cap_micros=10_000,
                budget_reserved_micros=2_000,
            )
            model = CatalogModel(
                model_id=model_id,
                canonical_slug=model_id,
                name="PostgreSQL authority fixture model",
                family="postgres-authority-integration",
            )
            session.add_all([season, model])
            session.flush()

            account = ProviderAccountBudget(
                id=str(uuid.uuid4()),
                execution_backend="bedrock",
                currency="USD",
                status="active",
                budget_cap_micros=10_000,
                budget_reserved_micros=2_000,
                opening_used_micros=0,
                opening_reserved_micros=0,
                account_scope_sha256=scope_sha256,
                authorization_reference_sha256="0" * 64,
                opening_balance_json=opening_balance,
                opening_balance_sha256=opening_balance_sha256,
                credential_binding_json={"pending": True},
                credential_binding_sha256=_sha256(f"pending-binding:{suffix}"),
                authorization_envelope_json={"pending": True},
                authorization_envelope_sha256=_sha256(f"pending-envelope:{suffix}"),
                authorization_hmac_sha256="0" * 64,
                valid_until=valid_until,
            )
            material = _authorization_material(
                account,
                suffix=suffix,
                valid_until=valid_until,
            )
            account.authorization_reference_sha256 = material["reference_sha256"]
            account.credential_binding_json = material["binding"]
            account.credential_binding_sha256 = material["binding_sha256"]
            account.authorization_envelope_json = material["envelope"]
            account.authorization_envelope_sha256 = material["envelope_sha256"]
            account.authorization_hmac_sha256 = material["signature"]
            provider = SeasonProviderBudget(
                season_id=season.id,
                execution_backend="bedrock",
                currency="USD",
                budget_cap_micros=10_000,
                budget_reserved_micros=2_000,
                account_scope_sha256=scope_sha256,
                authorization_reference_sha256=_sha256(f"provider-authorization:{suffix}"),
                account_authorization_envelope_sha256=material["envelope_sha256"],
                authorization_envelope_json={"fixture": suffix},
                authorization_envelope_sha256=_canonical_sha256({"fixture": suffix}),
                valid_until=valid_until,
            )
            authorization = ProviderAccountAuthorization(
                provider_account_budget_id=account.id,
                execution_backend="bedrock",
                account_scope_sha256=scope_sha256,
                status="active",
                authorization_reference_sha256=material["reference_sha256"],
                exposure_attestation_json=material["exposure"],
                exposure_attestation_sha256=material["exposure_sha256"],
                authorized_used_micros=0,
                authorized_reserved_micros=0,
                credential_binding_json=material["binding"],
                credential_binding_sha256=material["binding_sha256"],
                authorization_envelope_json=material["envelope"],
                authorization_envelope_sha256=material["envelope_sha256"],
                authorization_hmac_sha256=material["signature"],
                valid_until=valid_until,
            )
            session.add(account)
            session.flush()
            session.add_all([provider, authorization])
            session.flush()

            battle = Battle(
                season_id=season.id,
                run_class="official",
                rank_eligible=True,
                data_stratum="public_freeform",
                track="model_arena",
                category="composition",
                prompt=f"PostgreSQL authority fixture {suffix}",
                prompt_sha256=_sha256(f"prompt:{suffix}"),
                client_nonce_sha256=_sha256(f"nonce:{suffix}"),
                requester_pseudonym=_sha256(f"requester:{suffix}"),
                status="queued",
                reserved_cost_micros=2_000,
                provider_reservations_json={"bedrock": 2_000},
                retention_until=datetime.now(UTC) + timedelta(days=30),
            )
            session.add(battle)
            session.flush()
            arms = [
                ResponseArm(
                    battle_id=battle.id,
                    side=side,
                    condition="epicure_on",
                    model_id=model_id,
                    execution_backend="bedrock",
                    provider_slug="bedrock-test-route",
                    status="queued",
                    prompt_sha256=battle.prompt_sha256,
                    schema_sha256=_sha256(f"schema:{suffix}"),
                    tool_schema_sha256=_sha256(f"tools:{suffix}"),
                    epicure_release_id=season.epicure_release_id,
                    epicure_bundle_sha256=_sha256(f"bundle:{suffix}"),
                )
                for side in ("left", "right")
            ]
            session.add_all(arms)
            session.add_all(
                [
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="reserve",
                        amount_micros=2_000,
                        provider="governor",
                    ),
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="provider_reserve",
                        amount_micros=2_000,
                        provider="bedrock",
                        accounting_json={"budget_scope": "provider"},
                    ),
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="provider_account_reserve",
                        amount_micros=2_000,
                        provider="bedrock",
                        accounting_json={
                            "budget_scope": "provider_account",
                            "account_scope_sha256": scope_sha256,
                        },
                    ),
                ]
            )
            session.flush()

            completed_at = datetime.now(UTC)
            generation_map: list[dict[str, object]] = []
            for index, arm in enumerate(sorted(arms, key=lambda item: item.id)):
                generation_id = f"postgres-authority-generation-{suffix}-{index}"
                arm.cost_micros = 500
                arm.cost_reconciled = True
                arm.cost_accounting_basis = "bedrock_rate_card_receipt"
                arm.billing_reconciliation_status = "pending_aws_billing_crosscheck"
                arm.status = "failed"
                arm.actual_provider_slug = arm.provider_slug
                arm.actual_model_id = arm.model_id
                arm.generation_id = generation_id
                arm.provider_generation_ids_json = [generation_id]
                arm.completed_at = completed_at
                generation_set_sha256 = _canonical_sha256({"generation_ids": [generation_id]})
                generation_map.append(
                    {
                        "arm_id": arm.id,
                        "generation_ids": [generation_id],
                        "account_authorization_envelope_sha256": material["envelope_sha256"],
                        "generation_set_sha256": generation_set_sha256,
                    }
                )
                session.add(
                    GenerationAttempt(
                        attempt_id=str(uuid.uuid4()),
                        arm_id=arm.id,
                        request_key_sha256=_sha256(f"request-key:{suffix}:{index}"),
                        phase="generation",
                        attempt_index=1,
                        event_type="request_started",
                        payload_sha256=_sha256(f"payload:{suffix}:{index}"),
                        metadata_json={
                            "verified_provider_account_authorization_envelope_sha256": (
                                material["envelope_sha256"]
                            )
                        },
                        created_at=completed_at,
                    )
                )
                session.add(
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        arm_id=arm.id,
                        kind="actual",
                        amount_micros=500,
                        provider=arm.provider_slug,
                        generation_id=generation_id,
                        accounting_json={
                            "generation_ids": [generation_id],
                            "reconciled": True,
                            "cost_accounting_basis": ("bedrock_rate_card_receipt"),
                            "billing_reconciliation_status": ("pending_aws_billing_crosscheck"),
                        },
                    )
                )

            battle.reserved_cost_micros = 0
            season.budget_reserved_micros = 0
            season.budget_used_micros = 1_000
            provider.budget_reserved_micros = 0
            provider.budget_used_micros = 1_000
            account.budget_reserved_micros = 0
            account.budget_used_micros = 1_000
            session.add_all(
                [
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="release",
                        amount_micros=-2_000,
                        provider="governor",
                    ),
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="reconcile",
                        amount_micros=1_000,
                        provider="governor",
                    ),
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="provider_release",
                        amount_micros=-2_000,
                        provider="bedrock",
                        accounting_json={"budget_scope": "provider"},
                    ),
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="provider_reconcile",
                        amount_micros=1_000,
                        provider="bedrock",
                        accounting_json={"budget_scope": "provider"},
                    ),
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="provider_account_release",
                        amount_micros=-2_000,
                        provider="bedrock",
                        accounting_json={
                            "budget_scope": "provider_account",
                            "account_scope_sha256": scope_sha256,
                        },
                    ),
                    CostEvent(
                        season_id=season.id,
                        battle_id=battle.id,
                        kind="provider_account_reconcile",
                        amount_micros=1_000,
                        provider="bedrock",
                        accounting_json={
                            "budget_scope": "provider_account",
                            "account_scope_sha256": scope_sha256,
                        },
                    ),
                ]
            )
            session.commit()

            return AuthorityFixture(
                database_url=database_url,
                season_id=season.id,
                season_slug=season.slug,
                arm_ids=sorted(arm.id for arm in arms),
                authorization_reference_sha256=material["reference_sha256"],
                authorization_envelope_sha256=material["envelope_sha256"],
                generation_request_map_sha256=_canonical_sha256({"arms": generation_map}),
                coverage_start=completed_at - timedelta(seconds=1),
                coverage_end=completed_at + timedelta(seconds=1),
            )
    finally:
        engine.dispose()


def _seed_normal_finish_guard_fixture(
    database_url: str,
) -> tuple[str, str, tuple[str, str]]:
    engine = create_engine(database_url)
    suffix = uuid.uuid4().hex[:12]
    model_id = f"normal-finish/model-{suffix}"
    now = datetime.now(UTC)
    prompt_sha256 = _sha256(f"normal-finish-prompt:{suffix}")
    try:
        with Session(engine) as session:
            season = Season(
                slug=f"normal-finish-{suffix}",
                name="PostgreSQL normal-finish guard fixture",
                status="active",
                official=False,
                epicure_release_id="normal-finish-test-release",
                budget_cap_micros=0,
            )
            model = CatalogModel(
                model_id=model_id,
                canonical_slug=model_id,
                name="Normal-finish guard model",
                family="normal-finish-integration",
            )
            session.add_all([season, model])
            session.flush()
            battle = Battle(
                season_id=season.id,
                run_class="exploratory",
                rank_eligible=False,
                data_stratum="public_freeform",
                track="model_arena",
                category="composition",
                prompt="Exercise the PostgreSQL normal-finish evidence boundary.",
                prompt_sha256=prompt_sha256,
                client_nonce_sha256=_sha256(f"normal-finish-nonce:{suffix}"),
                requester_pseudonym=_sha256(f"normal-finish-requester:{suffix}"),
                status="queued",
                reserved_cost_micros=0,
                provider_reservations_json={},
                created_at=now,
                retention_until=now + timedelta(days=30),
            )
            session.add(battle)
            session.flush()
            arms = tuple(
                ResponseArm(
                    battle_id=battle.id,
                    side=side,
                    condition="epicure_on" if side == "left" else "epicure_off",
                    model_id=model_id,
                    execution_backend="openrouter",
                    provider_slug="normal-finish-test-route",
                    status="queued",
                    prompt_sha256=prompt_sha256,
                    schema_sha256=_sha256(f"normal-finish-schema:{suffix}"),
                    tool_schema_sha256=_sha256(f"normal-finish-tools:{suffix}"),
                    epicure_release_id=season.epicure_release_id,
                    epicure_bundle_sha256=_sha256(f"normal-finish-bundle:{suffix}"),
                    created_at=now,
                )
                for side in ("left", "right")
            )
            session.add_all(arms)
            session.commit()
            return battle.id, model_id, (arms[0].id, arms[1].id)
    finally:
        engine.dispose()


def _register(
    fixture: AuthorityFixture,
    request: dict[str, object],
) -> dict[str, Any]:
    with psycopg.connect(
        _psycopg_url(fixture.database_url),
        row_factory=dict_row,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE flavourbench_api")
            cursor.execute(
                "SELECT * FROM public.flavourbench_register_bedrock_billing_adjustment(%s, %s)",
                (fixture.season_id, Jsonb(request)),
            )
            result = cursor.fetchone()
        connection.commit()
    assert result is not None
    return dict(result)


def _snapshot(fixture: AuthorityFixture) -> tuple[int, int, int, tuple[int, int, int]]:
    with psycopg.connect(_psycopg_url(fixture.database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM bedrock_billing_crosschecks WHERE season_id = %s",
                (fixture.season_id,),
            )
            crosschecks = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM bedrock_billing_crosscheck_arms AS membership
                JOIN bedrock_billing_crosschecks AS crosscheck
                  ON crosscheck.id = membership.crosscheck_id
                WHERE crosscheck.season_id = %s
                """,
                (fixture.season_id,),
            )
            memberships = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT COUNT(*) FROM cost_events "
                "WHERE season_id = %s AND kind = 'bedrock_billing_adjustment'",
                (fixture.season_id,),
            )
            events = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT season.budget_used_micros,
                       provider.budget_used_micros,
                       account.budget_used_micros
                FROM seasons AS season
                JOIN season_provider_budgets AS provider
                  ON provider.season_id = season.id
                 AND provider.execution_backend = 'bedrock'
                JOIN provider_account_budgets AS account
                  ON account.execution_backend = provider.execution_backend
                 AND account.account_scope_sha256 = provider.account_scope_sha256
                WHERE season.id = %s
                """,
                (fixture.season_id,),
            )
            counters = tuple(int(value) for value in cursor.fetchone())
    return crosschecks, memberships, events, counters  # type: ignore[return-value]


def test_runtime_role_and_function_grants(postgres_database_url: str) -> None:
    engine = create_engine(postgres_database_url)
    try:
        with engine.connect() as connection:
            _assert_postgresql_budget_authority(connection)
            _assert_postgresql_normal_finish_guards(connection)
            _assert_postgresql_participant_lifecycle_guards(connection)
            _assert_postgresql_reviewer_task_validation_guards(connection)
    finally:
        engine.dispose()

    with psycopg.connect(_psycopg_url(postgres_database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                  has_function_privilege(
                    'flavourbench_api',
                    'public.flavourbench_register_bedrock_billing_adjustment(text,jsonb)',
                    'EXECUTE'
                  ),
                  has_function_privilege(
                    'flavourbench_worker',
                    'public.flavourbench_register_bedrock_billing_adjustment(text,jsonb)',
                    'EXECUTE'
                  ),
                  has_function_privilege(
                    'flavourbench_api',
                    'public.flavourbench_apply_bedrock_billing_adjustment(text)',
                    'EXECUTE'
                  ),
                  has_function_privilege(
                    'flavourbench_worker',
                    'public.flavourbench_apply_bedrock_billing_adjustment(text)',
                    'EXECUTE'
                  ),
                  has_table_privilege(
                    'flavourbench_api', 'bedrock_billing_crosschecks', 'INSERT'
                  ),
                  has_table_privilege(
                    'flavourbench_api', 'bedrock_billing_crosscheck_arms', 'INSERT'
                  ),
                  has_table_privilege(
                    'flavourbench_worker', 'bedrock_billing_crosschecks', 'INSERT'
                  ),
                  has_table_privilege(
                    'flavourbench_worker', 'bedrock_billing_crosscheck_arms', 'INSERT'
                  )
                """
            )
            assert cursor.fetchone() == (
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            )


def test_reviewer_task_validation_guard_readiness_detects_postgresql_drift(
    postgres_database_url: str,
) -> None:
    engine = create_engine(postgres_database_url)
    try:
        with engine.connect() as connection:
            _assert_postgresql_reviewer_task_validation_guards(connection)
            connection.rollback()

            transaction = connection.begin()
            connection.execute(
                text(
                    "ALTER FUNCTION "
                    "public.flavourbench_task_validation_event_guard_v1() "
                    "RESET search_path"
                )
            )
            with pytest.raises(
                RuntimeError,
                match="reviewer/task-validation guard metadata is unsafe",
            ):
                _assert_postgresql_reviewer_task_validation_guards(connection)
            transaction.rollback()

            transaction = connection.begin()
            connection.execute(
                text(
                    "ALTER TABLE public.task_validation_campaign_events "
                    "DISABLE TRIGGER "
                    "trg_task_validation_campaign_events_authority_v1"
                )
            )
            with pytest.raises(
                RuntimeError,
                match="reviewer/task-validation guard trigger is unsafe",
            ):
                _assert_postgresql_reviewer_task_validation_guards(connection)
            transaction.rollback()

            transaction = connection.begin()
            connection.execute(
                text(
                    "CREATE OR REPLACE FUNCTION "
                    "public.flavourbench_reviewer_evidence_append_only_v1() "
                    "RETURNS trigger LANGUAGE plpgsql SECURITY INVOKER "
                    "SET search_path = pg_catalog, public AS $$ "
                    "BEGIN RETURN OLD; END; $$"
                )
            )
            with pytest.raises(
                RuntimeError,
                match="reviewer/task-validation guard body is stale or unsafe",
            ):
                _assert_postgresql_reviewer_task_validation_guards(connection)
            transaction.rollback()

            _assert_postgresql_reviewer_task_validation_guards(connection)
    finally:
        engine.dispose()


def test_normal_finish_guard_readiness_detects_postgresql_drift(
    postgres_database_url: str,
) -> None:
    engine = create_engine(postgres_database_url)
    try:
        with engine.connect() as connection:
            _assert_postgresql_normal_finish_guards(connection)
            connection.rollback()

            transaction = connection.begin()
            connection.execute(
                text("ALTER TABLE public.votes DISABLE TRIGGER trg_vote_normal_finish_guard")
            )
            with pytest.raises(RuntimeError, match="normal-finish guard trigger"):
                _assert_postgresql_normal_finish_guards(connection)
            transaction.rollback()

            transaction = connection.begin()
            connection.execute(
                text(
                    """
                    CREATE OR REPLACE FUNCTION
                        public.flavourbench_response_arm_normal_finish_guard()
                    RETURNS trigger
                    LANGUAGE plpgsql
                    SECURITY INVOKER
                    SET search_path = pg_catalog, public
                    AS $$
                    BEGIN
                        RETURN NEW;
                    END;
                    $$
                    """
                )
            )
            with pytest.raises(RuntimeError, match="guard body is stale or unsafe"):
                _assert_postgresql_normal_finish_guards(connection)
            transaction.rollback()

            transaction = connection.begin()
            connection.execute(
                text(
                    "ALTER FUNCTION public.flavourbench_vote_normal_finish_guard() SECURITY DEFINER"
                )
            )
            with pytest.raises(RuntimeError, match="guard metadata is unsafe"):
                _assert_postgresql_normal_finish_guards(connection)
            transaction.rollback()

            _assert_postgresql_normal_finish_guards(connection)
    finally:
        engine.dispose()


def test_normal_finish_guards_accept_and_reject_postgresql_evidence(
    postgres_database_url: str,
) -> None:
    battle_id, model_id, arm_ids = _seed_normal_finish_guard_fixture(postgres_database_url)
    completed_at = datetime.now(UTC) + timedelta(milliseconds=10)
    update_arm = """
        UPDATE public.response_arms
        SET actual_provider_slug = %s,
            actual_model_id = %s,
            generation_id = %s,
            provider_generation_ids_json = %s,
            status = 'complete',
            answer_markdown = %s,
            answer_markdown_sha256 = %s,
            output_json = %s,
            output_json_sha256 = %s,
            cost_micros = 0,
            cost_reconciled = TRUE,
            cost_accounting_basis = 'known_zero_cost',
            billing_reconciliation_status = 'not_applicable',
            latency_ms = 1,
            finish_reason = %s,
            completed_at = %s
        WHERE id = %s
    """

    def arm_parameters(arm_id: str, finish_reason: str | None) -> tuple[object, ...]:
        generation_id = f"normal-finish-generation-{arm_id}"
        answer = f"Normal-finish answer for {arm_id}."
        output = {"answer_markdown": answer}
        return (
            "normal-finish-test-route",
            model_id,
            generation_id,
            Json([generation_id]),
            answer,
            _sha256(answer),
            Json(output),
            _canonical_sha256(output),
            finish_reason,
            completed_at,
            arm_id,
        )

    with psycopg.connect(_psycopg_url(postgres_database_url), autocommit=True) as connection:
        with connection.cursor() as cursor:
            with pytest.raises(
                psycopg.errors.RaiseException,
                match="complete response arm requires a normal provider finish reason",
            ):
                cursor.execute(update_arm, arm_parameters(arm_ids[0], "length"))
            cursor.execute(
                "SELECT status, finish_reason FROM public.response_arms WHERE id = %s",
                (arm_ids[0],),
            )
            assert cursor.fetchone() == ("queued", None)

            cursor.execute(update_arm, arm_parameters(arm_ids[0], " STOP "))

            with pytest.raises(
                psycopg.errors.RaiseException,
                match="complete response arm requires a normal provider finish reason",
            ):
                cursor.execute(update_arm, arm_parameters(arm_ids[1], None))
            cursor.execute(update_arm, arm_parameters(arm_ids[1], "end_turn"))

            battle_completed_at = completed_at + timedelta(milliseconds=10)
            cursor.execute(
                """
                UPDATE public.battles
                SET left_arm_id = %s,
                    right_arm_id = %s,
                    status = 'complete',
                    completed_at = %s
                WHERE id = %s
                """,
                (*arm_ids, battle_completed_at, battle_id),
            )
            cursor.execute(
                """
                INSERT INTO public.votes (
                    id, battle_id, rater_pseudonym, cohort, choice,
                    reason_tags_json, rubric_json, idempotency_key, created_at
                ) VALUES (%s, %s, %s, 'public', 'left', %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    battle_id,
                    _sha256(f"normal-finish-rater:{battle_id}"),
                    Json([]),
                    Json({}),
                    f"normal-finish-vote-{battle_id}",
                    battle_completed_at + timedelta(milliseconds=10),
                ),
            )

            cursor.execute("CREATE TEMP TABLE normal_finish_vote_probe (battle_id varchar)")
            cursor.execute(
                """
                CREATE TRIGGER normal_finish_vote_probe_trigger
                BEFORE INSERT ON normal_finish_vote_probe
                FOR EACH ROW EXECUTE FUNCTION
                    public.flavourbench_vote_normal_finish_guard()
                """
            )
            cursor.execute(
                "INSERT INTO normal_finish_vote_probe (battle_id) VALUES (%s)",
                (battle_id,),
            )
            with pytest.raises(
                psycopg.errors.RaiseException,
                match="vote requires two normally finished response arms",
            ):
                cursor.execute(
                    "INSERT INTO normal_finish_vote_probe (battle_id) VALUES (%s)",
                    (f"missing-{uuid.uuid4()}",),
                )

            cursor.execute(
                "SELECT side, finish_reason FROM public.response_arms "
                "WHERE id = ANY(%s) ORDER BY side",
                (list(arm_ids),),
            )
            assert cursor.fetchall() == [("left", " STOP "), ("right", "end_turn")]


def test_postgresql_downgrade_across_replay_binding_is_blocked(
    postgres_database_url: str,
) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "FLAVOURBENCH_DATABASE_URL": postgres_database_url,
            "FLAVOURBENCH_ENVIRONMENT": "test",
            "FLAVOURBENCH_SERVICE_ROLE": "migration",
            "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
        }
    )
    downgrade = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            "alembic.ini",
            "downgrade",
            "0022_postgresql_finish_guard_coalesce",
        ],
        cwd=SERVICE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert downgrade.returncode != 0
    assert (
        "downgrade across participant consent, withdrawal, and privacy evidence is prohibited"
        in downgrade.stdout + downgrade.stderr
    )
    with psycopg.connect(_psycopg_url(postgres_database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version_num FROM public.alembic_version")
            assert cursor.fetchone() == (EXPECTED_SCHEMA_REVISION,)


def test_atomic_initial_registration_uses_canonical_ascii_evidence(
    postgres_database_url: str,
) -> None:
    fixture = _seed_authority_fixture(postgres_database_url, "initial")
    before = _snapshot(fixture)
    result = _register(
        fixture,
        fixture.request("initial", billed_usage_micros=1_200),
    )
    after = _snapshot(fixture)

    assert result["billing_difference_micros"] == 200
    assert result["ledger_delta_micros"] == 200
    assert result["governed_delta_micros"] == 200
    assert before == (0, 0, 0, (1_000, 1_000, 1_000))
    assert after == (1, 2, 1, (1_200, 1_200, 1_200))

    with psycopg.connect(_psycopg_url(postgres_database_url)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT evidence_json::jsonb FROM bedrock_billing_crosschecks WHERE id = %s",
                (result["crosscheck_id"],),
            )
            evidence = cursor.fetchone()[0]
    assert result["evidence_sha256"] == _canonical_sha256(evidence)


def test_synchronized_conflict_is_fb001_and_charges_exactly_once(
    postgres_database_url: str,
) -> None:
    fixture = _seed_authority_fixture(postgres_database_url, "conflict")
    barrier = threading.Barrier(2)
    mutex = threading.Lock()
    results: list[tuple[str, str | None]] = []

    def contender(index: int) -> None:
        request = fixture.request(
            f"conflict-{index}",
            billed_usage_micros=1_200,
        )
        try:
            with psycopg.connect(_psycopg_url(postgres_database_url)) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SET ROLE flavourbench_api")
                    cursor.execute("SET statement_timeout = '5s'")
                    barrier.wait(timeout=5)
                    cursor.execute(
                        "SELECT * FROM "
                        "public.flavourbench_register_bedrock_billing_adjustment(%s, %s)",
                        (fixture.season_id, Jsonb(request)),
                    )
                    cursor.fetchone()
                connection.commit()
            outcome = ("ok", None)
        except Exception as exc:  # noqa: BLE001 - preserve the PostgreSQL SQLSTATE
            outcome = ("error", getattr(exc, "sqlstate", None))
        with mutex:
            results.append(outcome)

    threads = [threading.Thread(target=contender, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(results) == [("error", "FB001"), ("ok", None)]
    assert _snapshot(fixture) == (1, 2, 1, (1_200, 1_200, 1_200))


def test_correction_chain_preserves_monotone_authority_and_integrity(
    postgres_database_url: str,
) -> None:
    fixture = _seed_authority_fixture(postgres_database_url, "corrections")
    initial = _register(
        fixture,
        fixture.request("correction-initial", billed_usage_micros=1_200),
    )
    downward = _register(
        fixture,
        fixture.request(
            "correction-down",
            billed_usage_micros=800,
            supersedes_crosscheck_id=initial["crosscheck_id"],
        ),
    )
    upward = _register(
        fixture,
        fixture.request(
            "correction-up",
            billed_usage_micros=1_100,
            supersedes_crosscheck_id=downward["crosscheck_id"],
        ),
    )

    assert (
        initial["ledger_delta_micros"],
        initial["governed_delta_micros"],
    ) == (200, 200)
    assert (
        downward["ledger_delta_micros"],
        downward["governed_delta_micros"],
    ) == (-400, 0)
    assert (
        upward["ledger_delta_micros"],
        upward["governed_delta_micros"],
    ) == (300, 300)
    assert _snapshot(fixture) == (3, 6, 3, (1_500, 1_500, 1_500))

    engine = create_engine(postgres_database_url)
    try:
        with Session(engine) as session:
            report = assert_budget_integrity(
                session,
                fixture.season_id,
                lock_aggregates=True,
            )
            assert report.ok
            assert report.violations == ()
            assert report.season_used_micros == 1_500
    finally:
        engine.dispose()


def test_invalid_uri_and_map_roll_back_without_residue(
    postgres_database_url: str,
) -> None:
    fixture = _seed_authority_fixture(postgres_database_url, "rollback")
    unicode_request = fixture.request(
        "unicode",
        billed_usage_micros=1_200,
        source_artifact_uri="s3://billing-test/café",
    )
    with pytest.raises(ValidationError):
        BedrockBillingCrosscheckCreate.model_validate(unicode_request)

    before = _snapshot(fixture)
    with pytest.raises(psycopg.Error) as unicode_error:
        _register(fixture, unicode_request)
    assert unicode_error.value.sqlstate == "P0001"
    assert _snapshot(fixture) == before

    invalid_map = fixture.request(
        "invalid-map",
        billed_usage_micros=1_200,
        generation_request_map_sha256="0" * 64,
    )
    with pytest.raises(psycopg.Error) as map_error:
        _register(fixture, invalid_map)
    assert map_error.value.sqlstate == "P0001"
    assert _snapshot(fixture) == before


@pytest.fixture(scope="module")
def concurrent_review_pool_import(
    postgres_database_url: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = load_bundle(
        candidate_path=REVIEW_CANDIDATE_PATH,
        comparison_manifest_path=(
            SERVICE_ROOT
            / "artifacts"
            / "season0"
            / "comparisons"
            / (
                "season0-comparisons-"
                "c6e9052d19737b39b540dafbd0cea53d1dd0c54b1a04584fd3775ddfe9f35ca7.json"
            )
        ),
        model_manifest_path=(
            SERVICE_ROOT
            / "artifacts"
            / "season0"
            / "manifests"
            / (
                "season0-model-manifest-"
                "3919def66686b4bd939c94cdd89659f63ae2afbbf03288413129e2ea8d6b83d2.json"
            )
        ),
        arm_directory=SERVICE_ROOT / "artifacts" / "season0" / "scored-v1" / "arms",
    )
    engine = create_engine(postgres_database_url, pool_size=4)
    barrier = threading.Barrier(2)

    def execute_import() -> dict[str, Any]:
        with Session(engine) as session:
            barrier.wait(timeout=15)
            result = import_bundle(session, bundle)
            session.commit()
            return result

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(execute_import) for _ in range(2)]
            results = tuple(future.result(timeout=60) for future in futures)
    finally:
        engine.dispose()
    assert len(results) == 2
    return results[0], results[1]


def test_same_candidate_postgresql_import_retries_serialize_without_duplicates(
    postgres_database_url: str,
    concurrent_review_pool_import: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    assert sorted(result["idempotent"] for result in concurrent_review_pool_import) == [
        False,
        True,
    ]
    engine = create_engine(postgres_database_url)
    try:
        with engine.connect() as connection:
            counts = connection.execute(
                text(
                    """
                    WITH candidate_season AS (
                        SELECT id FROM seasons WHERE manifest_sha256 = :candidate_sha256
                    )
                    SELECT
                        (SELECT count(*) FROM run_events
                         WHERE entity_type = 'author_evaluator_pool'
                           AND entity_id = :candidate_sha256
                           AND event_type = 'author_evaluator_pool_imported'),
                        (SELECT count(*) FROM battles
                         WHERE season_id IN (SELECT id FROM candidate_season)),
                        (SELECT count(*) FROM response_arms AS arm
                         JOIN battles AS battle ON battle.id = arm.battle_id
                         WHERE battle.season_id IN (SELECT id FROM candidate_season)),
                        (SELECT count(*) FROM tool_calls AS tool
                         JOIN response_arms AS arm ON arm.id = tool.arm_id
                         JOIN battles AS battle ON battle.id = arm.battle_id
                         WHERE battle.season_id IN (SELECT id FROM candidate_season))
                    """
                ),
                {"candidate_sha256": REVIEW_CANDIDATE_SHA256},
            ).one()
            assert counts == (1, 32, 64, 89)
    finally:
        engine.dispose()


def test_same_anonymous_pool_postgresql_rotations_serialize_without_duplicates(
    postgres_database_url: str,
    concurrent_review_pool_import: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    assert concurrent_review_pool_import
    reviewer_id = str(uuid.uuid4())
    prior_pool_sha256 = _sha256(f"prior-pool:{reviewer_id}")
    engine = create_engine(postgres_database_url, pool_size=4)
    try:
        with Session(engine) as session:
            session.add(
                ExpertReviewer(
                    id=reviewer_id,
                    reviewer_code=f"fbr-anon-{uuid.uuid4().hex[:16]}",
                    invitation_sha256=_sha256(f"invitation:{reviewer_id}"),
                    qualification_json=[
                        "substitution",
                        "composition",
                        "cookability",
                        "evidence",
                    ],
                    qualification_verified=False,
                    cohort="expert_independent",
                    profile_json={
                        "admission_pathway": "anonymous_external_rater",
                        "anonymous_external_admission_status": "active",
                        "anonymous_external_pool_sha256": prior_pool_sha256,
                        "anonymous_external_pool_activation_sha256": _sha256(
                            f"prior-activation:{reviewer_id}"
                        ),
                        "consent_document_sha256": (
                            get_settings().active_expert_consent_sha256s[0]
                        ),
                        "identity_collection_prohibited": True,
                        "independence_basis": "reviewer_self_attestation",
                        "qualification_basis": "reviewer_self_attestation_unverified",
                        "independent_expert_validation_claim": False,
                    },
                    batch_reveal_only=True,
                    active=True,
                )
            )
            session.commit()

        barrier = threading.Barrier(2)

        def execute_rotation() -> dict[str, Any]:
            with Session(engine) as session:
                barrier.wait(timeout=15)
                return rotate_anonymous_pool(
                    session,
                    candidate_path=REVIEW_CANDIDATE_PATH,
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(execute_rotation) for _ in range(2)]
            results = tuple(future.result(timeout=30) for future in futures)

        assert sorted(result["idempotent"] for result in results) == [False, True]
        with Session(engine) as session:
            rotation_events = session.scalars(
                select(RunEvent).where(
                    RunEvent.entity_type == "expert_reviewer",
                    RunEvent.entity_id == reviewer_id,
                    RunEvent.event_type == ROTATION_EVENT_TYPE,
                )
            ).all()
            assert len(rotation_events) == 1
            reviewer = session.get(ExpertReviewer, reviewer_id)
            assert reviewer is not None
            assert (
                reviewer.profile_json["anonymous_external_pool_sha256"] == REVIEW_CANDIDATE_SHA256
            )
            assert rotation_events[0].payload_json["prior_review_session_id"] is None
    finally:
        engine.dispose()
