"""Bind commercial execution and publication to exact governance authorities.

Revision ID: 0019_commercial_authority_binding
Revises: 0018_commercial_integrity_guards
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0019_commercial_authority_binding"
down_revision = "0018_commercial_integrity_guards"
branch_labels = None
depends_on = None


AUTHORITY_GUARD_FUNCTION = "flavourbench_commercial_authority_guard"

COMMERCIAL_BINDING_CHECK = (
    "(organization_id IS NULL AND evaluation_order_id IS NULL AND "
    "route_revision_id IS NULL AND endpoint_descriptor_sha256 IS NULL AND "
    "spend_authorization_id IS NULL AND "
    "spend_authorization_binding_sha256 IS NULL) OR "
    "(organization_id IS NOT NULL AND evaluation_order_id IS NOT NULL AND "
    "route_revision_id IS NOT NULL AND endpoint_descriptor_sha256 IS NOT NULL AND "
    "spend_authorization_id IS NOT NULL AND "
    "spend_authorization_binding_sha256 IS NOT NULL)"
)


def _add_columns() -> None:
    if op.get_bind().dialect.name == "sqlite":
        op.add_column(
            "controlled_runs",
            sa.Column("spend_authorization_id", sa.String(36), nullable=True),
        )
        op.add_column(
            "controlled_runs",
            sa.Column("spend_authorization_binding_sha256", sa.String(64), nullable=True),
        )
        op.add_column(
            "controlled_runs",
            sa.Column("publication_authorization_id", sa.String(36), nullable=True),
        )
        op.add_column(
            "controlled_runs",
            sa.Column(
                "publication_authorization_binding_sha256",
                sa.String(64),
                nullable=True,
            ),
        )
    else:
        with op.batch_alter_table("controlled_runs") as batch:
            batch.add_column(sa.Column("spend_authorization_id", sa.String(36), nullable=True))
            batch.add_column(
                sa.Column("spend_authorization_binding_sha256", sa.String(64), nullable=True)
            )
            batch.add_column(
                sa.Column("publication_authorization_id", sa.String(36), nullable=True)
            )
            batch.add_column(
                sa.Column(
                    "publication_authorization_binding_sha256",
                    sa.String(64),
                    nullable=True,
                )
            )
            batch.create_foreign_key(
                "fk_cr_spend_acceptance",
                "governance_acceptances",
                ["spend_authorization_id"],
                ["id"],
            )
            batch.create_foreign_key(
                "fk_cr_publication_acceptance",
                "governance_acceptances",
                ["publication_authorization_id"],
                ["id"],
            )
    op.create_index(
        "ix_controlled_runs_spend_authorization_id",
        "controlled_runs",
        ["spend_authorization_id"],
    )
    op.create_index(
        "ix_controlled_runs_publication_authorization_id",
        "controlled_runs",
        ["publication_authorization_id"],
    )


def _backfill_legacy_commercial_runs() -> None:
    op.execute(
        """
        UPDATE controlled_runs
        SET spend_authorization_id = (
                SELECT acceptance.id
                FROM governance_acceptances AS acceptance
                WHERE acceptance.organization_id = controlled_runs.organization_id
                  AND acceptance.evaluation_order_id = controlled_runs.evaluation_order_id
                  AND acceptance.agreement_type = 'spend_authorization'
                  AND acceptance.status = 'active'
                  AND NOT EXISTS (
                      SELECT 1 FROM governance_acceptances AS successor
                      WHERE successor.supersedes_acceptance_id = acceptance.id
                        AND successor.status = 'active'
                  )
                ORDER BY acceptance.accepted_at DESC, acceptance.id DESC
                LIMIT 1
            ),
            spend_authorization_binding_sha256 = (
                SELECT acceptance.binding_sha256
                FROM governance_acceptances AS acceptance
                WHERE acceptance.organization_id = controlled_runs.organization_id
                  AND acceptance.evaluation_order_id = controlled_runs.evaluation_order_id
                  AND acceptance.agreement_type = 'spend_authorization'
                  AND acceptance.status = 'active'
                  AND NOT EXISTS (
                      SELECT 1 FROM governance_acceptances AS successor
                      WHERE successor.supersedes_acceptance_id = acceptance.id
                        AND successor.status = 'active'
                  )
                ORDER BY acceptance.accepted_at DESC, acceptance.id DESC
                LIMIT 1
            )
        WHERE organization_id IS NOT NULL
        """
    )
    unresolved = op.get_bind().execute(
        sa.text(
            "SELECT COUNT(*) FROM controlled_runs "
            "WHERE organization_id IS NOT NULL AND spend_authorization_id IS NULL"
        )
    ).scalar_one()
    if unresolved:
        raise RuntimeError(
            "0019 preflight requires one active order-bound spend authorization "
            f"for each legacy commercial run; unresolved={unresolved}"
        )


def _replace_binding_constraint() -> None:
    if op.get_bind().dialect.name == "sqlite":
        op.execute(
            """
            CREATE TRIGGER trg_controlled_runs_commercial_authority_binding_insert
            BEFORE INSERT ON controlled_runs
            WHEN NOT (
                (NEW.organization_id IS NULL AND NEW.evaluation_order_id IS NULL AND
                 NEW.route_revision_id IS NULL AND NEW.endpoint_descriptor_sha256 IS NULL AND
                 NEW.spend_authorization_id IS NULL AND
                 NEW.spend_authorization_binding_sha256 IS NULL) OR
                (NEW.organization_id IS NOT NULL AND NEW.evaluation_order_id IS NOT NULL AND
                 NEW.route_revision_id IS NOT NULL AND
                 NEW.endpoint_descriptor_sha256 IS NOT NULL AND
                 NEW.spend_authorization_id IS NOT NULL AND
                 NEW.spend_authorization_binding_sha256 IS NOT NULL)
            )
            BEGIN
                SELECT RAISE(ABORT, 'commercial controlled-run authority binding is incomplete');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER trg_controlled_runs_commercial_authority_binding_update
            BEFORE UPDATE ON controlled_runs
            WHEN OLD.spend_authorization_id IS NOT NEW.spend_authorization_id OR
                 OLD.spend_authorization_binding_sha256 IS NOT
                    NEW.spend_authorization_binding_sha256
            BEGIN
                SELECT RAISE(ABORT, 'commercial spend-authority binding is immutable');
            END
            """
        )
        return
    with op.batch_alter_table("controlled_runs") as batch:
        batch.drop_constraint("ck_controlled_runs_commercial_binding", type_="check")
        batch.create_check_constraint(
            "ck_controlled_runs_commercial_binding",
            COMMERCIAL_BINDING_CHECK,
        )


def _postgres_authority_guard() -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION public.flavourbench_prevent_controlled_run_contract_update()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY INVOKER
        SET search_path = pg_catalog, public
        AS $$
        DECLARE
            mutable_fields text[] := ARRAY[
                'access_token_sha256', 'token_version', 'status',
                'budget_used_micros', 'budget_reserved_micros', 'release_authorized',
                'release_authorization_reference_sha256', 'release_authorized_at',
                'publication_authorization_id',
                'publication_authorization_binding_sha256',
                'collection_completed_at', 'closed_at', 'revoked_at'
            ];
        BEGIN
            IF (pg_catalog.to_jsonb(OLD) - mutable_fields) IS DISTINCT FROM
               (pg_catalog.to_jsonb(NEW) - mutable_fields) THEN
                RAISE EXCEPTION 'controlled-run contract is immutable; create a new run';
            END IF;
            RETURN NEW;
        END;
        $$;

        CREATE OR REPLACE FUNCTION public.{AUTHORITY_GUARD_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            spend_matches integer;
            publication_matches integer;
        BEGIN
            IF TG_OP = 'UPDATE' AND (
                OLD.spend_authorization_id IS DISTINCT FROM NEW.spend_authorization_id OR
                OLD.spend_authorization_binding_sha256 IS DISTINCT FROM
                    NEW.spend_authorization_binding_sha256
            ) THEN
                RAISE EXCEPTION 'commercial spend-authority binding is immutable';
            END IF;

            IF NEW.organization_id IS NULL THEN
                IF NEW.spend_authorization_id IS NOT NULL OR
                   NEW.spend_authorization_binding_sha256 IS NOT NULL OR
                   NEW.publication_authorization_id IS NOT NULL OR
                   NEW.publication_authorization_binding_sha256 IS NOT NULL THEN
                    RAISE EXCEPTION 'noncommercial run cannot claim commercial authority';
                END IF;
                RETURN NEW;
            END IF;

            SELECT COUNT(*) INTO spend_matches
            FROM public.governance_acceptances AS acceptance
            JOIN public.evaluation_orders AS orders
              ON orders.id = NEW.evaluation_order_id
             AND orders.organization_id = NEW.organization_id
            WHERE acceptance.id = NEW.spend_authorization_id
              AND acceptance.organization_id = NEW.organization_id
              AND acceptance.evaluation_order_id = NEW.evaluation_order_id
              AND acceptance.model_submission_id IS NULL
              AND acceptance.route_revision_id IS NULL
              AND acceptance.agreement_type = 'spend_authorization'
              AND acceptance.status = 'active'
              AND acceptance.accepted_at <= CURRENT_TIMESTAMP
              AND (acceptance.expires_at IS NULL OR acceptance.expires_at > CURRENT_TIMESTAMP)
              AND acceptance.binding_sha256 = NEW.spend_authorization_binding_sha256
              AND acceptance.binding_json::jsonb = jsonb_build_object(
                    'orderCardSha256', orders.order_card_sha256,
                    'budgetCapMicros', orders.budget_cap_micros,
                    'currency', orders.currency,
                    'forecastCostMicros', orders.forecast_cost_micros,
                    'routeRevisionId', orders.route_revision_id,
                    'seasonId', orders.season_id,
                    'quoteReferenceSha256', orders.quote_reference_sha256
              )
              AND NOT EXISTS (
                    SELECT 1 FROM public.governance_acceptances AS successor
                    WHERE successor.supersedes_acceptance_id = acceptance.id
                      AND successor.status = 'active'
                      AND successor.accepted_at <= CURRENT_TIMESTAMP
              );
            IF spend_matches <> 1 THEN
                RAISE EXCEPTION 'commercial spend authority is inactive or mismatched';
            END IF;

            IF NEW.release_authorized THEN
                SELECT COUNT(*) INTO publication_matches
                FROM public.governance_acceptances AS acceptance
                JOIN public.evaluation_orders AS orders
                  ON orders.id = NEW.evaluation_order_id
                 AND orders.organization_id = NEW.organization_id
                WHERE acceptance.id = NEW.publication_authorization_id
                  AND acceptance.organization_id = NEW.organization_id
                  AND acceptance.evaluation_order_id = NEW.evaluation_order_id
                  AND acceptance.model_submission_id IS NULL
                  AND acceptance.route_revision_id IS NULL
                  AND acceptance.agreement_type = 'publication_authorization'
                  AND acceptance.status = 'active'
                  AND acceptance.accepted_at <= CURRENT_TIMESTAMP
                  AND (acceptance.expires_at IS NULL OR acceptance.expires_at > CURRENT_TIMESTAMP)
                  AND acceptance.binding_sha256 =
                      NEW.publication_authorization_binding_sha256
                  AND acceptance.binding_json::jsonb = jsonb_build_object(
                        'evaluationOrderId', orders.id,
                        'organizationId', orders.organization_id,
                        'orderCardSha256', orders.order_card_sha256,
                        'publicationScope', 'controlled_run_results_and_evidence',
                        'requestedVisibility', orders.requested_visibility,
                        'runCardSha256', NEW.run_card_sha256,
                        'seasonId', orders.season_id
                  )
                  AND NOT EXISTS (
                        SELECT 1 FROM public.governance_acceptances AS successor
                        WHERE successor.supersedes_acceptance_id = acceptance.id
                          AND successor.status = 'active'
                          AND successor.accepted_at <= CURRENT_TIMESTAMP
                  );
                IF publication_matches <> 1 THEN
                    RAISE EXCEPTION 'commercial publication authority is inactive or mismatched';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$;

        DROP TRIGGER IF EXISTS trg_controlled_runs_commercial_authority_guard
            ON public.controlled_runs;
        CREATE TRIGGER trg_controlled_runs_commercial_authority_guard
        BEFORE INSERT OR UPDATE ON public.controlled_runs
        FOR EACH ROW EXECUTE FUNCTION public.{AUTHORITY_GUARD_FUNCTION}();

        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'flavourbench_api') THEN
                GRANT UPDATE (
                    release_authorized,
                    release_authorization_reference_sha256,
                    release_authorized_at,
                    publication_authorization_id,
                    publication_authorization_binding_sha256
                ) ON TABLE public.controlled_runs TO flavourbench_api;
            END IF;
        END;
        $$;
        """
    )


def upgrade() -> None:
    _add_columns()
    _backfill_legacy_commercial_runs()
    _replace_binding_constraint()
    if op.get_bind().dialect.name == "postgresql":
        _postgres_authority_guard()


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute(
            "DROP TRIGGER IF EXISTS trg_controlled_runs_commercial_authority_guard "
            "ON public.controlled_runs"
        )
        op.execute(
            f"DROP FUNCTION IF EXISTS public.{AUTHORITY_GUARD_FUNCTION}()"
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION public.flavourbench_prevent_controlled_run_contract_update()
            RETURNS trigger
            LANGUAGE plpgsql
            SECURITY INVOKER
            SET search_path = pg_catalog, public
            AS $$
            DECLARE
                mutable_fields text[] := ARRAY[
                    'access_token_sha256', 'token_version', 'status',
                    'budget_used_micros', 'budget_reserved_micros', 'release_authorized',
                    'release_authorization_reference_sha256', 'release_authorized_at',
                    'collection_completed_at', 'closed_at', 'revoked_at'
                ];
            BEGIN
                IF (pg_catalog.to_jsonb(OLD) - mutable_fields) IS DISTINCT FROM
                   (pg_catalog.to_jsonb(NEW) - mutable_fields) THEN
                    RAISE EXCEPTION 'controlled-run contract is immutable; create a new run';
                END IF;
                RETURN NEW;
            END;
            $$;
            """
        )
    else:
        op.execute(
            "DROP TRIGGER IF EXISTS trg_controlled_runs_commercial_authority_binding_insert"
        )
        op.execute(
            "DROP TRIGGER IF EXISTS trg_controlled_runs_commercial_authority_binding_update"
        )
    if dialect != "sqlite":
        with op.batch_alter_table("controlled_runs") as batch:
            batch.drop_constraint("ck_controlled_runs_commercial_binding", type_="check")
            batch.create_check_constraint(
                "ck_controlled_runs_commercial_binding",
                "(organization_id IS NULL AND evaluation_order_id IS NULL AND "
                "route_revision_id IS NULL AND endpoint_descriptor_sha256 IS NULL) OR "
                "(organization_id IS NOT NULL AND evaluation_order_id IS NOT NULL AND "
                "route_revision_id IS NOT NULL AND endpoint_descriptor_sha256 IS NOT NULL)",
            )
    op.drop_index("ix_controlled_runs_publication_authorization_id", table_name="controlled_runs")
    op.drop_index("ix_controlled_runs_spend_authorization_id", table_name="controlled_runs")
    if dialect == "sqlite":
        op.drop_column("controlled_runs", "publication_authorization_binding_sha256")
        op.drop_column("controlled_runs", "publication_authorization_id")
        op.drop_column("controlled_runs", "spend_authorization_binding_sha256")
        op.drop_column("controlled_runs", "spend_authorization_id")
    else:
        with op.batch_alter_table("controlled_runs") as batch:
            batch.drop_constraint(
                "fk_cr_publication_acceptance",
                type_="foreignkey",
            )
            batch.drop_constraint(
                "fk_cr_spend_acceptance",
                type_="foreignkey",
            )
            batch.drop_column("publication_authorization_binding_sha256")
            batch.drop_column("publication_authorization_id")
            batch.drop_column("spend_authorization_binding_sha256")
            batch.drop_column("spend_authorization_id")
