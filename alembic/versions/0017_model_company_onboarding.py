"""Add tenant-scoped model-company onboarding and evaluation orders.

Revision ID: 0017_model_company_onboarding
Revises: 0016_runtime_budget_authority
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0017_model_company_onboarding"
down_revision = "0016_runtime_budget_authority"
branch_labels = None
depends_on = None


def _index(table: str, name: str, columns: list[str], *, unique: bool = False) -> None:
    observed = {row["name"] for row in sa.inspect(op.get_bind()).get_indexes(table)}
    if name not in observed:
        op.create_index(name, table, columns, unique=unique)


def _add_runtime_grants() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'flavourbench_api'
            ) THEN
                GRANT SELECT, INSERT, UPDATE ON TABLE
                    public.organizations,
                    public.organization_api_keys,
                    public.governance_acceptances,
                    public.model_submissions,
                    public.model_route_revisions,
                    public.route_contract_tests,
                    public.evaluation_orders,
                    public.evidence_bundles,
                    public.api_idempotency_keys
                TO flavourbench_api;
            END IF;
            IF EXISTS (
                SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'flavourbench_worker'
            ) THEN
                GRANT SELECT ON TABLE
                    public.organizations,
                    public.model_submissions,
                    public.model_route_revisions,
                    public.evaluation_orders,
                    public.evidence_bundles
                TO flavourbench_worker;
                GRANT SELECT, INSERT, UPDATE ON TABLE public.route_contract_tests
                TO flavourbench_worker;
            END IF;
        END;
        $$;
        """
    )


def upgrade() -> None:
    bind = op.get_bind()
    tables = set(sa.inspect(bind).get_table_names())

    if "organizations" not in tables:
        op.create_table(
            "organizations",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("slug", sa.String(length=80), nullable=False),
            sa.Column("legal_name", sa.String(length=240), nullable=False),
            sa.Column("display_name", sa.String(length=160), nullable=False),
            sa.Column(
                "status",
                sa.String(length=32),
                nullable=False,
                server_default="pending_verification",
            ),
            sa.Column("idp_tenant_reference_sha256", sa.String(length=64), nullable=False),
            sa.Column("billing_reference_sha256", sa.String(length=64), nullable=True),
            sa.Column("data_region", sa.String(length=32), nullable=False),
            sa.Column("retention_policy_json", sa.JSON(), nullable=False),
            sa.Column("retention_policy_sha256", sa.String(length=64), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint(
                "status IN ('pending_verification', 'active', 'suspended', 'closed')",
                name="ck_organizations_status",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("slug"),
            sa.UniqueConstraint("idp_tenant_reference_sha256"),
        )
    for name, columns in {
        "ix_organizations_slug": ["slug"],
        "ix_organizations_status": ["status"],
        "ix_organizations_billing_reference_sha256": ["billing_reference_sha256"],
    }.items():
        _index("organizations", name, columns)

    if "organization_api_keys" not in tables:
        op.create_table(
            "organization_api_keys",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("organization_id", sa.String(length=36), nullable=False),
            sa.Column("key_prefix", sa.String(length=32), nullable=False),
            sa.Column("secret_hmac_sha256", sa.String(length=64), nullable=False),
            sa.Column(
                "hmac_key_id", sa.String(length=64), nullable=False, server_default="primary"
            ),
            sa.Column("label", sa.String(length=120), nullable=False),
            sa.Column("scopes_json", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False, server_default="active"),
            sa.Column(
                "rate_limit_profile",
                sa.String(length=64),
                nullable=False,
                server_default="standard",
            ),
            sa.Column("network_policy_json", sa.JSON(), nullable=False),
            sa.Column("created_by_principal_ref_sha256", sa.String(length=64), nullable=False),
            sa.Column("not_before", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("supersedes_key_id", sa.String(length=36), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "status IN ('active', 'revoked')",
                name="ck_organization_api_keys_status",
            ),
            sa.CheckConstraint(
                "(status = 'active' AND revoked_at IS NULL) OR "
                "(status = 'revoked' AND revoked_at IS NOT NULL)",
                name="ck_organization_api_keys_revocation",
            ),
            sa.CheckConstraint(
                "expires_at > not_before",
                name="ck_organization_api_keys_validity_window",
            ),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
            sa.ForeignKeyConstraint(["supersedes_key_id"], ["organization_api_keys.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("key_prefix"),
            sa.UniqueConstraint("secret_hmac_sha256"),
            sa.UniqueConstraint("supersedes_key_id"),
        )
    for name, columns in {
        "ix_organization_api_keys_organization_id": ["organization_id"],
        "ix_organization_api_keys_key_prefix": ["key_prefix"],
        "ix_organization_api_keys_status": ["status"],
        "ix_organization_api_keys_expires_at": ["expires_at"],
    }.items():
        _index("organization_api_keys", name, columns)

    if "model_submissions" not in tables:
        op.create_table(
            "model_submissions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("organization_id", sa.String(length=36), nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("supersedes_submission_id", sa.String(length=36), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
            sa.Column("display_name", sa.String(length=240), nullable=False),
            sa.Column("publisher", sa.String(length=240), nullable=False),
            sa.Column("requested_canonical_model_id", sa.String(length=240), nullable=False),
            sa.Column("exact_model_version", sa.String(length=240), nullable=False),
            sa.Column("release_date", sa.String(length=32), nullable=False),
            sa.Column("model_card_uri", sa.Text(), nullable=False),
            sa.Column("model_card_sha256", sa.String(length=64), nullable=False),
            sa.Column("license_uri", sa.Text(), nullable=False),
            sa.Column("license_document_sha256", sa.String(length=64), nullable=False),
            sa.Column("capability_claims_json", sa.JSON(), nullable=False),
            sa.Column("capability_claims_sha256", sa.String(length=64), nullable=False),
            sa.Column("contamination_disclosure_json", sa.JSON(), nullable=False),
            sa.Column("contamination_disclosure_sha256", sa.String(length=64), nullable=False),
            sa.Column("submission_payload_json", sa.JSON(), nullable=False),
            sa.Column("submission_payload_sha256", sa.String(length=64), nullable=False),
            sa.Column("catalog_model_id", sa.String(length=200), nullable=True),
            sa.Column("submitted_by_key_id", sa.String(length=36), nullable=False),
            sa.Column("decision_reference_sha256", sa.String(length=64), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint("revision >= 1", name="ck_model_submissions_revision"),
            sa.CheckConstraint(
                "status IN ('draft', 'submitted', 'changes_requested', 'approved', "
                "'rejected', 'withdrawn', 'suspended', 'retired')",
                name="ck_model_submissions_status",
            ),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
            sa.ForeignKeyConstraint(["supersedes_submission_id"], ["model_submissions.id"]),
            sa.ForeignKeyConstraint(["catalog_model_id"], ["catalog_models.model_id"]),
            sa.ForeignKeyConstraint(["submitted_by_key_id"], ["organization_api_keys.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "organization_id",
                "requested_canonical_model_id",
                "revision",
                name="uq_model_submissions_org_model_revision",
            ),
            sa.UniqueConstraint("supersedes_submission_id"),
            sa.UniqueConstraint("submission_payload_sha256"),
        )
    for name, columns in {
        "ix_model_submissions_organization_id": ["organization_id"],
        "ix_model_submissions_status": ["status"],
        "ix_model_submissions_requested_canonical_model_id": ["requested_canonical_model_id"],
        "ix_model_submissions_submission_payload_sha256": ["submission_payload_sha256"],
        "ix_model_submissions_catalog_model_id": ["catalog_model_id"],
        "ix_model_submissions_submitted_by_key_id": ["submitted_by_key_id"],
    }.items():
        _index("model_submissions", name, columns)

    if "model_route_revisions" not in tables:
        op.create_table(
            "model_route_revisions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("model_submission_id", sa.String(length=36), nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("route_kind", sa.String(length=32), nullable=False),
            sa.Column("execution_backend", sa.String(length=32), nullable=False),
            sa.Column("managed_route_reference_sha256", sa.String(length=64), nullable=False),
            sa.Column("requested_model_id", sa.String(length=240), nullable=False),
            sa.Column("expected_actual_model_id", sa.String(length=240), nullable=False),
            sa.Column("expected_actual_provider_slug", sa.String(length=160), nullable=False),
            sa.Column("supported_parameters_json", sa.JSON(), nullable=False),
            sa.Column("supported_parameters_sha256", sa.String(length=64), nullable=False),
            sa.Column("decoding_bounds_json", sa.JSON(), nullable=False),
            sa.Column("decoding_bounds_sha256", sa.String(length=64), nullable=False),
            sa.Column("endpoint_document_sha256", sa.String(length=64), nullable=False),
            sa.Column("data_policy_json", sa.JSON(), nullable=False),
            sa.Column("data_policy_sha256", sa.String(length=64), nullable=False),
            sa.Column("rate_card_json", sa.JSON(), nullable=False),
            sa.Column("rate_card_sha256", sa.String(length=64), nullable=False),
            sa.Column("descriptor_json", sa.JSON(), nullable=False),
            sa.Column("descriptor_sha256", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
            sa.Column("approved_contract_test_id", sa.String(length=36), nullable=True),
            sa.Column("approved_season_id", sa.String(length=36), nullable=True),
            sa.Column("approved_season_manifest_sha256", sa.String(length=64), nullable=True),
            sa.Column("approved_endpoint_contract_sha256", sa.String(length=64), nullable=True),
            sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint("revision >= 1", name="ck_model_route_revisions_revision"),
            sa.CheckConstraint(
                "route_kind IN ('managed_bedrock', 'managed_openrouter')",
                name="ck_model_route_revisions_kind",
            ),
            sa.CheckConstraint(
                "(route_kind = 'managed_bedrock' AND execution_backend = 'bedrock') OR "
                "(route_kind = 'managed_openrouter' AND execution_backend = 'openrouter')",
                name="ck_model_route_revisions_backend",
            ),
            sa.CheckConstraint(
                "status IN ('draft', 'submitted', 'contract_testing', 'approved', "
                "'changes_requested', 'rejected', 'withdrawn', 'suspended', 'retired')",
                name="ck_model_route_revisions_status",
            ),
            sa.ForeignKeyConstraint(["model_submission_id"], ["model_submissions.id"]),
            sa.ForeignKeyConstraint(["approved_season_id"], ["seasons.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "model_submission_id",
                "revision",
                name="uq_model_route_revisions_submission_revision",
            ),
            sa.UniqueConstraint("descriptor_sha256"),
            sa.UniqueConstraint("approved_contract_test_id"),
        )
    for name, columns in {
        "ix_model_route_revisions_model_submission_id": ["model_submission_id"],
        "ix_model_route_revisions_execution_backend": ["execution_backend"],
        "ix_model_route_revisions_descriptor_sha256": ["descriptor_sha256"],
        "ix_model_route_revisions_status": ["status"],
        "ix_model_route_revisions_approved_season_id": ["approved_season_id"],
    }.items():
        _index("model_route_revisions", name, columns)

    if "route_contract_tests" not in tables:
        op.create_table(
            "route_contract_tests",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("route_revision_id", sa.String(length=36), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False, server_default="queued"),
            sa.Column("suite_version", sa.String(length=80), nullable=False),
            sa.Column("protocol_bundle_sha256", sa.String(length=64), nullable=False),
            sa.Column("worker_build_digest", sa.String(length=160), nullable=False),
            sa.Column("request_sha256", sa.String(length=64), nullable=True),
            sa.Column("response_sha256", sa.String(length=64), nullable=True),
            sa.Column("tool_trace_sha256", sa.String(length=64), nullable=True),
            sa.Column("structured_output_sha256", sa.String(length=64), nullable=True),
            sa.Column("observed_model_id", sa.String(length=240), nullable=True),
            sa.Column("observed_provider_slug", sa.String(length=160), nullable=True),
            sa.Column("check_results_json", sa.JSON(), nullable=False),
            sa.Column("check_results_sha256", sa.String(length=64), nullable=True),
            sa.Column("generation_id", sa.String(length=240), nullable=True),
            sa.Column("usage_json", sa.JSON(), nullable=False),
            sa.Column("cost_micros", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column(
                "cost_accounting_basis",
                sa.String(length=80),
                nullable=False,
                server_default="unrecorded",
            ),
            sa.Column("latency_ms", sa.Integer(), nullable=True),
            sa.Column("failure_code", sa.String(length=80), nullable=True),
            sa.Column("incident_id", sa.String(length=36), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "status IN ('queued', 'running', 'passed', 'failed', 'inconclusive', 'cancelled')",
                name="ck_route_contract_tests_status",
            ),
            sa.ForeignKeyConstraint(["route_revision_id"], ["model_route_revisions.id"]),
            sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
    for name, columns in {
        "ix_route_contract_tests_route_revision_id": ["route_revision_id"],
        "ix_route_contract_tests_status": ["status"],
        "ix_route_contract_tests_valid_until": ["valid_until"],
    }.items():
        _index("route_contract_tests", name, columns)
    controlled_columns = {row["name"] for row in sa.inspect(bind).get_columns("controlled_runs")}
    if "organization_id" not in controlled_columns:
        if bind.dialect.name == "sqlite":
            op.execute(
                "ALTER TABLE controlled_runs "
                "ADD COLUMN organization_id VARCHAR(36) "
                "REFERENCES organizations(id)"
            )
        else:
            op.add_column(
                "controlled_runs",
                sa.Column("organization_id", sa.String(length=36), nullable=True),
            )
            op.create_foreign_key(
                "fk_controlled_runs_organization_id_organizations",
                "controlled_runs",
                "organizations",
                ["organization_id"],
                ["id"],
            )
        _index(
            "controlled_runs",
            "ix_controlled_runs_organization_id",
            ["organization_id"],
        )
    controlled_columns = {row["name"] for row in sa.inspect(bind).get_columns("controlled_runs")}
    if "route_revision_id" not in controlled_columns:
        if bind.dialect.name == "sqlite":
            op.execute(
                "ALTER TABLE controlled_runs "
                "ADD COLUMN route_revision_id VARCHAR(36) "
                "REFERENCES model_route_revisions(id)"
            )
        else:
            op.add_column(
                "controlled_runs",
                sa.Column("route_revision_id", sa.String(length=36), nullable=True),
            )
    if "endpoint_descriptor_sha256" not in controlled_columns:
        op.add_column(
            "controlled_runs",
            sa.Column("endpoint_descriptor_sha256", sa.String(length=64), nullable=True),
        )
    if bind.dialect.name != "sqlite":
        controlled_fks = sa.inspect(bind).get_foreign_keys("controlled_runs")
        if not any(
            row.get("constrained_columns") == ["route_revision_id"] for row in controlled_fks
        ):
            op.create_foreign_key(
                "fk_controlled_runs_route_revision_id_model_route_revisions",
                "controlled_runs",
                "model_route_revisions",
                ["route_revision_id"],
                ["id"],
            )
    for name, columns, unique in (
        ("ix_controlled_runs_route_revision_id", ["route_revision_id"], False),
        (
            "ix_controlled_runs_endpoint_descriptor_sha256",
            ["endpoint_descriptor_sha256"],
            False,
        ),
    ):
        _index("controlled_runs", name, columns, unique=unique)

    if "evaluation_orders" not in tables:
        op.create_table(
            "evaluation_orders",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("organization_id", sa.String(length=36), nullable=False),
            sa.Column("model_submission_id", sa.String(length=36), nullable=False),
            sa.Column("route_revision_id", sa.String(length=36), nullable=False),
            sa.Column("season_id", sa.String(length=36), nullable=False),
            sa.Column("evaluation_profile_id", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
            sa.Column(
                "billing_status",
                sa.String(length=24),
                nullable=False,
                server_default="unquoted",
            ),
            sa.Column(
                "publication_status",
                sa.String(length=24),
                nullable=False,
                server_default="private",
            ),
            sa.Column(
                "requested_visibility",
                sa.String(length=24),
                nullable=False,
                server_default="private",
            ),
            sa.Column("comparison_plan_json", sa.JSON(), nullable=False),
            sa.Column("comparison_plan_sha256", sa.String(length=64), nullable=False),
            sa.Column("rater_plan_sha256", sa.String(length=64), nullable=False),
            sa.Column("analysis_plan_sha256", sa.String(length=64), nullable=False),
            sa.Column("forecast_cost_micros", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column("budget_cap_micros", sa.BigInteger(), nullable=False),
            sa.Column("currency", sa.String(length=8), nullable=False, server_default="USD"),
            sa.Column("quote_reference_sha256", sa.String(length=64), nullable=True),
            sa.Column("client_reference_sha256", sa.String(length=64), nullable=False),
            sa.Column("order_card_json", sa.JSON(), nullable=False),
            sa.Column("order_card_sha256", sa.String(length=64), nullable=False),
            sa.Column("order_card_signature", sa.String(length=64), nullable=False),
            sa.Column("submitted_by_key_id", sa.String(length=36), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint(
                "status IN ('draft', 'submitted', 'approved', 'provisioning', 'ready', "
                "'running', 'collection_complete', 'analysis_complete', 'delivered', "
                "'rejected', 'cancelling', 'cancelled', 'failed')",
                name="ck_evaluation_orders_status",
            ),
            sa.CheckConstraint(
                "billing_status IN ('unquoted', 'quoted', 'authorized', 'reconciled', "
                "'disputed', 'void')",
                name="ck_evaluation_orders_billing_status",
            ),
            sa.CheckConstraint(
                "publication_status IN ('private', 'authorized', 'published', 'withdrawn')",
                name="ck_evaluation_orders_publication_status",
            ),
            sa.CheckConstraint(
                "requested_visibility IN ('private', 'public_candidate')",
                name="ck_evaluation_orders_visibility",
            ),
            sa.CheckConstraint(
                "forecast_cost_micros >= 0 AND budget_cap_micros > 0",
                name="ck_evaluation_orders_budget",
            ),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
            sa.ForeignKeyConstraint(["model_submission_id"], ["model_submissions.id"]),
            sa.ForeignKeyConstraint(["route_revision_id"], ["model_route_revisions.id"]),
            sa.ForeignKeyConstraint(["season_id"], ["seasons.id"]),
            sa.ForeignKeyConstraint(["submitted_by_key_id"], ["organization_api_keys.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "organization_id",
                "client_reference_sha256",
                name="uq_evaluation_orders_org_client_reference",
            ),
            sa.UniqueConstraint("order_card_sha256"),
        )
    for name, columns in {
        "ix_evaluation_orders_organization_id": ["organization_id"],
        "ix_evaluation_orders_model_submission_id": ["model_submission_id"],
        "ix_evaluation_orders_route_revision_id": ["route_revision_id"],
        "ix_evaluation_orders_season_id": ["season_id"],
        "ix_evaluation_orders_evaluation_profile_id": ["evaluation_profile_id"],
        "ix_evaluation_orders_status": ["status"],
        "ix_evaluation_orders_billing_status": ["billing_status"],
        "ix_evaluation_orders_publication_status": ["publication_status"],
        "ix_evaluation_orders_order_card_sha256": ["order_card_sha256"],
        "ix_evaluation_orders_submitted_by_key_id": ["submitted_by_key_id"],
    }.items():
        _index("evaluation_orders", name, columns)

    controlled_columns = {row["name"] for row in sa.inspect(bind).get_columns("controlled_runs")}
    if "evaluation_order_id" not in controlled_columns:
        if bind.dialect.name == "sqlite":
            op.execute(
                "ALTER TABLE controlled_runs "
                "ADD COLUMN evaluation_order_id VARCHAR(36) "
                "REFERENCES evaluation_orders(id)"
            )
        else:
            op.add_column(
                "controlled_runs",
                sa.Column("evaluation_order_id", sa.String(length=36), nullable=True),
            )
            op.create_foreign_key(
                "fk_controlled_runs_evaluation_order_id_evaluation_orders",
                "controlled_runs",
                "evaluation_orders",
                ["evaluation_order_id"],
                ["id"],
            )
        _index(
            "controlled_runs",
            "ix_controlled_runs_evaluation_order_id",
            ["evaluation_order_id"],
            unique=True,
        )
    if bind.dialect.name != "sqlite":
        controlled_checks = {
            row["name"] for row in sa.inspect(bind).get_check_constraints("controlled_runs")
        }
        if "ck_controlled_runs_commercial_binding" not in controlled_checks:
            op.create_check_constraint(
                "ck_controlled_runs_commercial_binding",
                "controlled_runs",
                "(organization_id IS NULL AND evaluation_order_id IS NULL AND "
                "route_revision_id IS NULL AND endpoint_descriptor_sha256 IS NULL) OR "
                "(organization_id IS NOT NULL AND evaluation_order_id IS NOT NULL AND "
                "route_revision_id IS NOT NULL AND endpoint_descriptor_sha256 IS NOT NULL)",
            )
    else:
        op.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_controlled_runs_commercial_binding_insert
            BEFORE INSERT ON controlled_runs
            FOR EACH ROW
            WHEN NOT (
                (NEW.organization_id IS NULL AND NEW.evaluation_order_id IS NULL AND
                 NEW.route_revision_id IS NULL AND NEW.endpoint_descriptor_sha256 IS NULL)
                OR
                (NEW.organization_id IS NOT NULL AND NEW.evaluation_order_id IS NOT NULL AND
                 NEW.route_revision_id IS NOT NULL AND
                 NEW.endpoint_descriptor_sha256 IS NOT NULL AND EXISTS (
                    SELECT 1
                    FROM evaluation_orders AS orders
                    JOIN model_route_revisions AS route
                      ON route.id = orders.route_revision_id
                     AND route.id = NEW.route_revision_id
                     AND route.descriptor_sha256 = NEW.endpoint_descriptor_sha256
                     AND route.status = 'approved'
                     AND route.approved_season_id = NEW.season_id
                    JOIN model_submissions AS submission
                      ON submission.id = orders.model_submission_id
                     AND submission.id = route.model_submission_id
                     AND submission.status = 'approved'
                     AND submission.catalog_model_id = NEW.submitted_endpoint_model_id
                     AND submission.model_card_sha256 = NEW.submitted_model_card_sha256
                    JOIN seasons AS season
                      ON season.id = orders.season_id
                     AND season.id = NEW.season_id
                     AND season.status = 'active'
                     AND season.official = 1
                     AND season.frozen_at IS NOT NULL
                     AND season.manifest_sha256 = route.approved_season_manifest_sha256
                    WHERE orders.id = NEW.evaluation_order_id
                      AND orders.organization_id = NEW.organization_id
                      AND orders.status = 'approved'
                      AND orders.billing_status = 'authorized'
                      AND orders.quote_reference_sha256 IS NOT NULL
                      AND orders.rater_plan_sha256 = NEW.rater_plan_sha256
                      AND orders.analysis_plan_sha256 = NEW.analysis_plan_sha256
                      AND orders.budget_cap_micros = NEW.budget_cap_micros
                      AND route.data_policy_sha256 = NEW.data_policy_sha256
                 ))
            )
            BEGIN
                SELECT RAISE(ABORT, 'commercial controlled-run binding is invalid');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_controlled_runs_commercial_binding_update
            BEFORE UPDATE OF organization_id, evaluation_order_id, route_revision_id,
                             endpoint_descriptor_sha256
            ON controlled_runs
            FOR EACH ROW
            WHEN OLD.organization_id IS NOT NEW.organization_id
              OR OLD.evaluation_order_id IS NOT NEW.evaluation_order_id
              OR OLD.route_revision_id IS NOT NEW.route_revision_id
              OR OLD.endpoint_descriptor_sha256 IS NOT NEW.endpoint_descriptor_sha256
            BEGIN
                SELECT RAISE(ABORT, 'commercial controlled-run binding is immutable');
            END
            """
        )

    if "governance_acceptances" not in tables:
        op.create_table(
            "governance_acceptances",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("organization_id", sa.String(length=36), nullable=False),
            sa.Column("model_submission_id", sa.String(length=36), nullable=True),
            sa.Column("route_revision_id", sa.String(length=36), nullable=True),
            sa.Column("evaluation_order_id", sa.String(length=36), nullable=True),
            sa.Column("agreement_type", sa.String(length=80), nullable=False),
            sa.Column("agreement_version", sa.String(length=80), nullable=False),
            sa.Column("document_sha256", sa.String(length=64), nullable=False),
            sa.Column(
                "external_envelope_reference_sha256",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column(
                "signatory_principal_reference_sha256",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column("authority_basis", sa.String(length=160), nullable=False),
            sa.Column("binding_json", sa.JSON(), nullable=False),
            sa.Column("binding_sha256", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False, server_default="active"),
            sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("supersedes_acceptance_id", sa.String(length=36), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "status IN ('active', 'revoked', 'expired', 'superseded')",
                name="ck_governance_acceptances_status",
            ),
            sa.CheckConstraint(
                "(CASE WHEN model_submission_id IS NULL THEN 0 ELSE 1 END + "
                "CASE WHEN route_revision_id IS NULL THEN 0 ELSE 1 END + "
                "CASE WHEN evaluation_order_id IS NULL THEN 0 ELSE 1 END) <= 1",
                name="ck_governance_acceptances_subject",
            ),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
            sa.ForeignKeyConstraint(["model_submission_id"], ["model_submissions.id"]),
            sa.ForeignKeyConstraint(["route_revision_id"], ["model_route_revisions.id"]),
            sa.ForeignKeyConstraint(["evaluation_order_id"], ["evaluation_orders.id"]),
            sa.ForeignKeyConstraint(["supersedes_acceptance_id"], ["governance_acceptances.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("supersedes_acceptance_id"),
        )
    for name, columns in {
        "ix_governance_acceptances_organization_id": ["organization_id"],
        "ix_governance_acceptances_model_submission_id": ["model_submission_id"],
        "ix_governance_acceptances_route_revision_id": ["route_revision_id"],
        "ix_governance_acceptances_evaluation_order_id": ["evaluation_order_id"],
        "ix_governance_acceptances_agreement_type": ["agreement_type"],
        "ix_governance_acceptances_status": ["status"],
        "ix_governance_acceptances_expires_at": ["expires_at"],
    }.items():
        _index("governance_acceptances", name, columns)

    if "evidence_bundles" not in tables:
        op.create_table(
            "evidence_bundles",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("organization_id", sa.String(length=36), nullable=False),
            sa.Column("evaluation_order_id", sa.String(length=36), nullable=False),
            sa.Column("bundle_class", sa.String(length=32), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False, server_default="building"),
            sa.Column("schema_version", sa.String(length=80), nullable=False),
            sa.Column("manifest_json", sa.JSON(), nullable=False),
            sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
            sa.Column("archive_sha256", sa.String(length=64), nullable=True),
            sa.Column("storage_object_key", sa.Text(), nullable=True),
            sa.Column("size_bytes", sa.BigInteger(), nullable=True),
            sa.Column("signature_algorithm", sa.String(length=32), nullable=True),
            sa.Column("signing_key_id", sa.String(length=160), nullable=True),
            sa.Column("signature_base64", sa.Text(), nullable=True),
            sa.Column("publication_authorization_id", sa.String(length=36), nullable=True),
            sa.Column("supersedes_bundle_id", sa.String(length=36), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("sealed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("available_until", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.CheckConstraint(
                "bundle_class IN ('private_customer', 'public_release')",
                name="ck_evidence_bundles_class",
            ),
            sa.CheckConstraint(
                "status IN ('building', 'sealed', 'available', 'superseded', 'revoked', 'failed')",
                name="ck_evidence_bundles_status",
            ),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
            sa.ForeignKeyConstraint(["evaluation_order_id"], ["evaluation_orders.id"]),
            sa.ForeignKeyConstraint(
                ["publication_authorization_id"], ["governance_acceptances.id"]
            ),
            sa.ForeignKeyConstraint(["supersedes_bundle_id"], ["evidence_bundles.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("manifest_sha256"),
            sa.UniqueConstraint("archive_sha256"),
            sa.UniqueConstraint("supersedes_bundle_id"),
        )
    for name, columns in {
        "ix_evidence_bundles_organization_id": ["organization_id"],
        "ix_evidence_bundles_evaluation_order_id": ["evaluation_order_id"],
        "ix_evidence_bundles_bundle_class": ["bundle_class"],
        "ix_evidence_bundles_status": ["status"],
    }.items():
        _index("evidence_bundles", name, columns)

    if "api_idempotency_keys" not in tables:
        op.create_table(
            "api_idempotency_keys",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("organization_id", sa.String(length=36), nullable=False),
            sa.Column("api_key_id", sa.String(length=36), nullable=False),
            sa.Column("method", sa.String(length=12), nullable=False),
            sa.Column("route_template", sa.String(length=160), nullable=False),
            sa.Column("idempotency_key_sha256", sa.String(length=64), nullable=False),
            sa.Column("request_sha256", sa.String(length=64), nullable=False),
            sa.Column("response_status", sa.Integer(), nullable=False),
            sa.Column("resource_type", sa.String(length=80), nullable=False),
            sa.Column("resource_id", sa.String(length=80), nullable=False),
            sa.Column("response_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
            sa.ForeignKeyConstraint(["api_key_id"], ["organization_api_keys.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "organization_id",
                "method",
                "route_template",
                "idempotency_key_sha256",
                name="uq_api_idempotency_scope",
            ),
        )
    for name, columns in {
        "ix_api_idempotency_keys_organization_id": ["organization_id"],
        "ix_api_idempotency_keys_api_key_id": ["api_key_id"],
        "ix_api_idempotency_keys_expires_at": ["expires_at"],
    }.items():
        _index("api_idempotency_keys", name, columns)

    arm_columns = {row["name"] for row in sa.inspect(bind).get_columns("response_arms")}
    if "route_revision_id" not in arm_columns:
        if bind.dialect.name == "sqlite":
            op.execute(
                "ALTER TABLE response_arms "
                "ADD COLUMN route_revision_id VARCHAR(36) "
                "REFERENCES model_route_revisions(id)"
            )
            op.add_column(
                "response_arms",
                sa.Column("endpoint_descriptor_sha256", sa.String(length=64), nullable=True),
            )
        else:
            op.add_column(
                "response_arms",
                sa.Column("route_revision_id", sa.String(length=36), nullable=True),
            )
            op.add_column(
                "response_arms",
                sa.Column("endpoint_descriptor_sha256", sa.String(length=64), nullable=True),
            )
            op.create_foreign_key(
                "fk_response_arms_route_revision_id_model_route_revisions",
                "response_arms",
                "model_route_revisions",
                ["route_revision_id"],
                ["id"],
            )
        _index(
            "response_arms",
            "ix_response_arms_route_revision_id",
            ["route_revision_id"],
        )
        _index(
            "response_arms",
            "ix_response_arms_endpoint_descriptor_sha256",
            ["endpoint_descriptor_sha256"],
        )
    if bind.dialect.name != "sqlite":
        arm_checks = {
            row["name"] for row in sa.inspect(bind).get_check_constraints("response_arms")
        }
        if "ck_response_arms_route_binding" not in arm_checks:
            op.create_check_constraint(
                "ck_response_arms_route_binding",
                "response_arms",
                "(route_revision_id IS NULL AND endpoint_descriptor_sha256 IS NULL) OR "
                "(route_revision_id IS NOT NULL AND endpoint_descriptor_sha256 IS NOT NULL)",
            )
    else:
        op.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_response_arms_commercial_binding_insert
            BEFORE INSERT ON response_arms
            FOR EACH ROW
            WHEN (
                EXISTS (
                    SELECT 1
                    FROM battles AS battle
                    JOIN controlled_runs AS run ON run.id = battle.controlled_run_id
                    WHERE battle.id = NEW.battle_id
                      AND run.evaluation_order_id IS NOT NULL
                      AND run.submitted_endpoint_model_id = NEW.model_id
                      AND (
                          NEW.route_revision_id IS NOT run.route_revision_id OR
                          NEW.endpoint_descriptor_sha256 IS NOT
                              run.endpoint_descriptor_sha256
                      )
                )
                OR
                (
                    NOT EXISTS (
                        SELECT 1
                        FROM battles AS battle
                        JOIN controlled_runs AS run ON run.id = battle.controlled_run_id
                        WHERE battle.id = NEW.battle_id
                          AND run.evaluation_order_id IS NOT NULL
                          AND run.submitted_endpoint_model_id = NEW.model_id
                    )
                    AND (
                        NEW.route_revision_id IS NOT NULL OR
                        NEW.endpoint_descriptor_sha256 IS NOT NULL
                    )
                )
            )
            BEGIN
                SELECT RAISE(ABORT, 'response-arm commercial binding is invalid');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_response_arms_commercial_binding_update
            BEFORE UPDATE OF route_revision_id, endpoint_descriptor_sha256
            ON response_arms
            FOR EACH ROW
            WHEN OLD.route_revision_id IS NOT NEW.route_revision_id
              OR OLD.endpoint_descriptor_sha256 IS NOT NEW.endpoint_descriptor_sha256
            BEGIN
                SELECT RAISE(ABORT, 'response-arm commercial binding is immutable');
            END
            """
        )

    _add_runtime_grants()


def downgrade() -> None:
    bind = op.get_bind()
    sqlite_triggers: list[str] = []
    if bind.dialect.name == "sqlite":
        rows = bind.execute(
            sa.text(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'trigger' "
                "AND (sql LIKE '%response_arms%' OR sql LIKE '%controlled_runs%')"
            )
        ).mappings()
        for row in rows:
            if row["sql"] and row["name"] not in {
                "trg_controlled_runs_commercial_binding_insert",
                "trg_controlled_runs_commercial_binding_update",
                "trg_response_arms_commercial_binding_insert",
                "trg_response_arms_commercial_binding_update",
            }:
                sqlite_triggers.append(str(row["sql"]))
            op.execute(f'DROP TRIGGER IF EXISTS "{row["name"]}"')

    arm_columns = {row["name"] for row in sa.inspect(bind).get_columns("response_arms")}
    if "route_revision_id" in arm_columns:
        arm_indexes = {row["name"] for row in sa.inspect(bind).get_indexes("response_arms")}
        for index_name in (
            "ix_response_arms_route_revision_id",
            "ix_response_arms_endpoint_descriptor_sha256",
        ):
            if index_name in arm_indexes:
                op.drop_index(index_name, table_name="response_arms")
        with op.batch_alter_table("response_arms") as batch:
            fks = sa.inspect(bind).get_foreign_keys("response_arms")
            checks = sa.inspect(bind).get_check_constraints("response_arms")
            if bind.dialect.name != "sqlite" and any(
                row.get("name") == "ck_response_arms_route_binding" for row in checks
            ):
                batch.drop_constraint("ck_response_arms_route_binding", type_="check")
            if bind.dialect.name != "sqlite" and any(
                row.get("constrained_columns") == ["route_revision_id"] for row in fks
            ):
                batch.drop_constraint(
                    "fk_response_arms_route_revision_id_model_route_revisions",
                    type_="foreignkey",
                )
            batch.drop_column("endpoint_descriptor_sha256")
            batch.drop_column("route_revision_id")

    controlled_columns = {row["name"] for row in sa.inspect(bind).get_columns("controlled_runs")}
    if "organization_id" in controlled_columns:
        controlled_indexes = {
            row["name"] for row in sa.inspect(bind).get_indexes("controlled_runs")
        }
        for index_name in (
            "ix_controlled_runs_organization_id",
            "ix_controlled_runs_evaluation_order_id",
            "ix_controlled_runs_route_revision_id",
            "ix_controlled_runs_endpoint_descriptor_sha256",
        ):
            if index_name in controlled_indexes:
                op.drop_index(index_name, table_name="controlled_runs")
        with op.batch_alter_table("controlled_runs") as batch:
            fks = sa.inspect(bind).get_foreign_keys("controlled_runs")
            checks = sa.inspect(bind).get_check_constraints("controlled_runs")
            if bind.dialect.name != "sqlite" and any(
                row.get("name") == "ck_controlled_runs_commercial_binding" for row in checks
            ):
                batch.drop_constraint(
                    "ck_controlled_runs_commercial_binding",
                    type_="check",
                )
            if bind.dialect.name != "sqlite" and any(
                row.get("constrained_columns") == ["organization_id"] for row in fks
            ):
                batch.drop_constraint(
                    "fk_controlled_runs_organization_id_organizations",
                    type_="foreignkey",
                )
            if bind.dialect.name != "sqlite" and any(
                row.get("constrained_columns") == ["route_revision_id"] for row in fks
            ):
                batch.drop_constraint(
                    "fk_controlled_runs_route_revision_id_model_route_revisions",
                    type_="foreignkey",
                )
            if bind.dialect.name != "sqlite" and any(
                row.get("constrained_columns") == ["evaluation_order_id"] for row in fks
            ):
                batch.drop_constraint(
                    "fk_controlled_runs_evaluation_order_id_evaluation_orders",
                    type_="foreignkey",
                )
            for column_name in (
                "endpoint_descriptor_sha256",
                "route_revision_id",
                "evaluation_order_id",
            ):
                if column_name in controlled_columns:
                    batch.drop_column(column_name)
            batch.drop_column("organization_id")

    for trigger_sql in sqlite_triggers:
        op.execute(trigger_sql)

    for table in (
        "api_idempotency_keys",
        "evidence_bundles",
        "governance_acceptances",
        "evaluation_orders",
        "route_contract_tests",
        "model_route_revisions",
        "model_submissions",
        "organization_api_keys",
        "organizations",
    ):
        if table in sa.inspect(bind).get_table_names():
            op.drop_table(table)
