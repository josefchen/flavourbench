"""Move governed budget mutations behind PostgreSQL owner authority.

Revision ID: 0016_runtime_budget_authority
Revises: 0015_release_integrity_controls
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0016_runtime_budget_authority"
down_revision = "0015_release_integrity_controls"
branch_labels = None
depends_on = None


COMMON_PARTIAL_INDEXES = {
    "uq_cost_events_battle_governor_reserve": ("battle_id", "kind = 'reserve'"),
    "uq_cost_events_battle_governor_release": ("battle_id", "kind = 'release'"),
    "uq_cost_events_battle_provider_release": (
        "battle_id, provider",
        "kind = 'provider_release'",
    ),
    "uq_cost_events_battle_provider_reconcile": (
        "battle_id, provider",
        "kind = 'provider_reconcile'",
    ),
    "uq_cost_events_battle_provider_account_release": (
        "battle_id, provider",
        "kind = 'provider_account_release'",
    ),
    "uq_cost_events_battle_provider_account_reconcile": (
        "battle_id, provider",
        "kind = 'provider_account_reconcile'",
    ),
    "uq_cost_events_arm_actual": ("arm_id", "kind = 'actual'"),
    "uq_cost_events_arm_actual_settlement": (
        "arm_id",
        "kind = 'actual_settlement'",
    ),
}

COUNTER_GUARD_FUNCTION = "flavourbench_budget_counter_write_guard"
BATTLE_GUARD_FUNCTION = "flavourbench_battle_reservation_write_guard"
COST_EVENT_GUARD_FUNCTION = "flavourbench_cost_event_authority_guard"
BEDROCK_MEMBERSHIP_GUARD_FUNCTION = "flavourbench_bedrock_membership_seal_guard"
ATTEMPT_ARM_GUARD_FUNCTION = "flavourbench_generation_attempt_arm_guard"
RESERVE_FUNCTION = "flavourbench_reserve_battle_budget"
SETTLE_FUNCTION = "flavourbench_settle_battle_budget"
BILLING_FUNCTION = "flavourbench_apply_bedrock_billing_adjustment"
REGISTER_BILLING_FUNCTION = "flavourbench_register_bedrock_billing_adjustment"
CANONICAL_JSON_FUNCTION = "flavourbench_canonical_jsonb_text"


POSTGRESQL_LEGACY_TRIGGER_REPAIRS = """
CREATE OR REPLACE FUNCTION public.flavourbench_0010_execution_contract_immutable()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    old_json jsonb := pg_catalog.to_jsonb(OLD);
    new_json jsonb := pg_catalog.to_jsonb(NEW);
BEGIN
    IF TG_TABLE_NAME = 'season_models'
       AND old_json ->> 'manifest_sha256' NOT IN ('', 'unfrozen', 'unresolved')
       AND (
           old_json ->> 'execution_backend' IS DISTINCT FROM
               new_json ->> 'execution_backend'
           OR old_json -> 'backend_contract_json' IS DISTINCT FROM
               new_json -> 'backend_contract_json'
           OR old_json ->> 'backend_contract_sha256' IS DISTINCT FROM
               new_json ->> 'backend_contract_sha256'
       ) THEN
        RAISE EXCEPTION 'frozen season execution backend is immutable';
    ELSIF TG_TABLE_NAME = 'battles'
       AND old_json -> 'provider_reservations_json' IS DISTINCT FROM
           new_json -> 'provider_reservations_json' THEN
        RAISE EXCEPTION 'battle provider reservation contract is immutable';
    ELSIF TG_TABLE_NAME = 'response_arms'
       AND old_json ->> 'execution_backend' IS DISTINCT FROM
           new_json ->> 'execution_backend' THEN
        RAISE EXCEPTION 'response-arm execution backend is immutable';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.flavourbench_child_retention_authorization()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    authorized boolean := false;
    row_json jsonb := pg_catalog.to_jsonb(NEW);
    parent_battle_id text;
BEGIN
    IF TG_TABLE_NAME = 'response_arms' THEN
        parent_battle_id := row_json ->> 'battle_id';
    ELSIF TG_TABLE_NAME IN ('tool_calls', 'validator_results') THEN
        SELECT a.battle_id INTO parent_battle_id
          FROM public.response_arms AS a
         WHERE a.id = row_json ->> 'arm_id';
    ELSIF TG_TABLE_NAME = 'run_events' THEN
        IF row_json ->> 'entity_type' = 'battle' THEN
            parent_battle_id := row_json ->> 'entity_id';
        ELSIF row_json ->> 'entity_type' = 'response_arm' THEN
            SELECT a.battle_id INTO parent_battle_id
              FROM public.response_arms AS a
             WHERE a.id = row_json ->> 'entity_id';
        END IF;
    ELSIF TG_TABLE_NAME IN ('incidents', 'jobs') THEN
        parent_battle_id := row_json ->> 'battle_id';
    END IF;
    IF parent_battle_id IS NOT NULL THEN
        SELECT EXISTS(
            SELECT 1 FROM public.battles AS b
            WHERE b.id = parent_battle_id
              AND b.prompt IS NULL
              AND b.prompt_redacted IS TRUE
              AND b.research_consent IS FALSE
              AND b.retention_until <= CURRENT_TIMESTAMP
        ) INTO authorized;
    END IF;
    IF NOT authorized THEN
        RAISE EXCEPTION 'retention redaction lacks an expired non-consent basis';
    END IF;
    RETURN NEW;
END;
$$;

-- Migration 0014 supersedes this narrower 0009 trigger. The old function
-- compares PostgreSQL json values directly and makes every arm update fail.
DROP TRIGGER IF EXISTS trg_response_arm_contract_immutable ON public.response_arms;

CREATE OR REPLACE FUNCTION public.flavourbench_prevent_endpoint_contract_update()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF OLD.manifest_sha256 NOT IN ('', 'unfrozen', 'unresolved')
       AND pg_catalog.to_jsonb(OLD) IS DISTINCT FROM pg_catalog.to_jsonb(NEW) THEN
        RAISE EXCEPTION 'frozen season endpoint contract is immutable';
    END IF;
    RETURN NEW;
END;
$$;

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

CREATE OR REPLACE FUNCTION public.flavourbench_prevent_controlled_assignment_mutation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'controlled-run assignments are append-only';
    END IF;
    IF (pg_catalog.to_jsonb(OLD) - ARRAY['status', 'battle_id']::text[])
       IS DISTINCT FROM
       (pg_catalog.to_jsonb(NEW) - ARRAY['status', 'battle_id']::text[]) THEN
        RAISE EXCEPTION 'controlled-run assignment content is immutable';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.flavourbench_prevent_snapshot_content_update()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    mutable_fields text[] := ARRAY[
        'publication_status', 'publication_reference_sha256', 'published_at'
    ];
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'leaderboard snapshots are append-only; withdraw or supersede';
    END IF;
    IF (pg_catalog.to_jsonb(OLD) - mutable_fields) IS DISTINCT FROM
       (pg_catalog.to_jsonb(NEW) - mutable_fields) THEN
        RAISE EXCEPTION 'leaderboard snapshot content is immutable';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.flavourbench_prevent_frozen_season_mutation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF OLD.frozen_at IS NOT NULL AND (
        OLD.manifest_sha256 IS DISTINCT FROM NEW.manifest_sha256 OR
        OLD.prompt_registry_sha256 IS DISTINCT FROM NEW.prompt_registry_sha256 OR
        OLD.tool_registry_sha256 IS DISTINCT FROM NEW.tool_registry_sha256 OR
        OLD.epicure_release_id IS DISTINCT FROM NEW.epicure_release_id OR
        OLD.epicure_bundle_sha256 IS DISTINCT FROM NEW.epicure_bundle_sha256 OR
        OLD.epicure_application_sha256 IS DISTINCT FROM
            NEW.epicure_application_sha256 OR
        OLD.analysis_plan_sha256 IS DISTINCT FROM NEW.analysis_plan_sha256 OR
        OLD.protocol_bundle_json::jsonb IS DISTINCT FROM
            NEW.protocol_bundle_json::jsonb OR
        OLD.protocol_bundle_sha256 IS DISTINCT FROM NEW.protocol_bundle_sha256 OR
        OLD.frozen_at IS DISTINCT FROM NEW.frozen_at
    ) THEN
        RAISE EXCEPTION 'frozen season contract is immutable';
    END IF;
    IF OLD.official AND NOT NEW.official THEN
        RAISE EXCEPTION 'official season state cannot be revoked in place';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.flavourbench_provider_budget_contract_immutable()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'provider budget authorizations cannot be deleted';
    END IF;
    IF (pg_catalog.to_jsonb(OLD) - ARRAY[
            'budget_used_micros', 'budget_reserved_micros'
        ]::text[]) IS DISTINCT FROM
       (pg_catalog.to_jsonb(NEW) - ARRAY[
            'budget_used_micros', 'budget_reserved_micros'
        ]::text[]) THEN
        RAISE EXCEPTION 'provider budget authorization is immutable';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.flavourbench_account_budget_contract_immutable()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    mutable_fields text[] := ARRAY[
        'status', 'revoked_at', 'budget_used_micros', 'budget_reserved_micros'
    ];
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'provider account ledgers cannot be deleted';
    END IF;
    IF (pg_catalog.to_jsonb(OLD) - mutable_fields) IS DISTINCT FROM
       (pg_catalog.to_jsonb(NEW) - mutable_fields) THEN
        RAISE EXCEPTION 'provider account ledger contract is immutable';
    END IF;
    IF OLD.status = 'revoked' AND (
        NEW.status IS DISTINCT FROM OLD.status OR
        NEW.revoked_at IS DISTINCT FROM OLD.revoked_at
    ) THEN
        RAISE EXCEPTION 'provider account ledger revocation is irreversible';
    ELSIF NEW.status IS DISTINCT FROM OLD.status OR
          NEW.revoked_at IS DISTINCT FROM OLD.revoked_at THEN
        IF NOT (
            (OLD.status = 'pending_verification' AND NEW.status = 'active'
             AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NULL)
            OR (OLD.status IN ('pending_verification', 'active')
                AND NEW.status = 'revoked'
                AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NOT NULL)
        ) THEN
            RAISE EXCEPTION 'invalid provider account ledger status transition';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.flavourbench_account_authorization_immutable()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'provider account authorization epochs are append-only';
    END IF;
    IF (pg_catalog.to_jsonb(OLD) - ARRAY['status', 'revoked_at']::text[])
       IS DISTINCT FROM
       (pg_catalog.to_jsonb(NEW) - ARRAY['status', 'revoked_at']::text[]) THEN
        RAISE EXCEPTION 'provider account authorization epoch is immutable';
    END IF;
    IF OLD.status = 'active' AND NEW.status = 'revoked'
       AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NOT NULL THEN
        RETURN NEW;
    END IF;
    IF NEW.status IS DISTINCT FROM OLD.status OR
       NEW.revoked_at IS DISTINCT FROM OLD.revoked_at THEN
        RAISE EXCEPTION 'provider account authorization revocation is invalid';
    END IF;
    RETURN NEW;
END;
$$;
"""


def _preflight(bind: sa.Connection) -> None:
    inspector = sa.inspect(bind)
    required = {
        "seasons",
        "season_provider_budgets",
        "provider_account_budgets",
        "controlled_runs",
        "battles",
        "response_arms",
        "season_models",
        "cost_events",
        "generation_attempts",
        "bedrock_billing_crosschecks",
        "bedrock_billing_crosscheck_arms",
    }
    missing = sorted(table for table in required if not inspector.has_table(table))
    if missing:
        raise RuntimeError(f"0016 preflight: missing governed tables: {missing}")
    for table in (
        "seasons",
        "season_provider_budgets",
        "provider_account_budgets",
        "controlled_runs",
    ):
        negative = bind.execute(
            sa.text(
                f"SELECT COUNT(*) FROM {table} "
                "WHERE budget_used_micros < 0 OR budget_reserved_micros < 0"
            )
        ).scalar_one()
        if int(negative):
            raise RuntimeError(f"0016 preflight: {table} contains negative counters")
    negative_arm_costs = int(
        bind.execute(
            sa.text("SELECT COUNT(*) FROM response_arms WHERE cost_micros < 0")
        ).scalar_one()
    )
    if negative_arm_costs:
        raise RuntimeError("0016 preflight: response arms contain negative costs")
    invalid_receipt_amounts = int(
        bind.execute(
            sa.text(
                "SELECT COUNT(*) FROM cost_events "
                "WHERE kind IN ('actual', 'actual_settlement') AND amount_micros < 0"
            )
        ).scalar_one()
    )
    if invalid_receipt_amounts:
        raise RuntimeError("0016 preflight: endpoint receipts contain negative costs")
    if bind.dialect.name != "postgresql":
        return
    role = str(bind.execute(sa.text("SELECT current_user")).scalar_one())
    if role in {"flavourbench_api", "flavourbench_worker"}:
        raise RuntimeError("0016 preflight: a runtime role cannot own ledger authority")
    ownership_mismatches = bind.execute(
        sa.text(
            """
            SELECT c.relname
            FROM pg_catalog.pg_class AS c
            JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relname = ANY(:tables)
              AND pg_catalog.pg_get_userbyid(c.relowner) <> current_user
            """
        ),
        {"tables": list(required)},
    ).scalars()
    mismatches = sorted(ownership_mismatches)
    if mismatches:
        raise RuntimeError(
            f"0016 preflight: migration role does not own governed tables: {mismatches}"
        )
    invalid_events = int(
        bind.execute(
            sa.text(
                """
                SELECT COUNT(*)
                FROM cost_events AS ce
                LEFT JOIN battles AS b ON b.id = ce.battle_id
                WHERE ce.kind IN (
                    'reserve', 'release', 'reconcile',
                    'provider_reserve', 'provider_release', 'provider_reconcile',
                    'provider_account_reserve', 'provider_account_release',
                    'provider_account_reconcile'
                )
                  AND (b.id IS NULL OR b.season_id <> ce.season_id)
                """
            )
        ).scalar_one()
    )
    if invalid_events:
        raise RuntimeError("0016 preflight: governed events violate battle ownership")
    missing_crosscheck = int(
        bind.execute(
            sa.text(
                """
                SELECT COUNT(*) FROM cost_events
                WHERE kind = 'bedrock_billing_adjustment'
                  AND COALESCE(accounting_json::jsonb ->> 'crosscheck_id', '') = ''
                """
            )
        ).scalar_one()
    )
    if missing_crosscheck:
        raise RuntimeError("0016 preflight: billing adjustment lacks a crosscheck")
    cross_season_runs = int(
        bind.execute(
            sa.text(
                """
                SELECT COUNT(*)
                FROM battles AS b
                JOIN controlled_runs AS cr ON cr.id = b.controlled_run_id
                WHERE b.season_id <> cr.season_id
                """
            )
        ).scalar_one()
    )
    if cross_season_runs:
        raise RuntimeError("0016 preflight: battle is bound to a foreign-season run")
    self_superseding = int(
        bind.execute(
            sa.text(
                "SELECT COUNT(*) FROM bedrock_billing_crosschecks "
                "WHERE supersedes_crosscheck_id = id"
            )
        ).scalar_one()
    )
    if self_superseding:
        raise RuntimeError("0016 preflight: Bedrock crosscheck supersedes itself")
    partial_crosschecks = int(
        bind.execute(
            sa.text(
                """
                SELECT COUNT(*)
                FROM bedrock_billing_crosschecks AS c
                WHERE NOT EXISTS (
                    SELECT 1 FROM bedrock_billing_crosscheck_arms AS m
                    WHERE m.crosscheck_id = c.id
                )
                   OR (SELECT COUNT(*) FROM cost_events AS ce
                       WHERE ce.kind = 'bedrock_billing_adjustment'
                         AND ce.accounting_json::jsonb ->> 'crosscheck_id' = c.id) <> 1
                """
            )
        ).scalar_one()
    )
    if partial_crosschecks:
        raise RuntimeError(
            "0016 preflight: Bedrock crosschecks are not atomically applied"
        )
    multiply_active_arms = int(
        bind.execute(
            sa.text(
                """
                SELECT COUNT(*) FROM (
                    SELECT m.arm_id
                    FROM bedrock_billing_crosscheck_arms AS m
                    JOIN bedrock_billing_crosschecks AS c ON c.id = m.crosscheck_id
                    WHERE NOT EXISTS (
                        SELECT 1 FROM bedrock_billing_crosschecks AS successor
                        WHERE successor.supersedes_crosscheck_id = c.id
                    )
                    GROUP BY m.arm_id
                    HAVING COUNT(*) > 1
                ) AS conflicts
                """
            )
        ).scalar_one()
    )
    if multiply_active_arms:
        raise RuntimeError(
            "0016 preflight: a Bedrock arm has multiple active crosschecks"
        )


def _create_postgresql_checks() -> None:
    op.create_check_constraint(
        "ck_response_arms_nonnegative_cost",
        "response_arms",
        "cost_micros >= 0",
    )
    op.create_check_constraint(
        "ck_cost_events_nonnegative_charge",
        "cost_events",
        """
        kind NOT IN (
            'reserve', 'reconcile', 'provider_reserve', 'provider_reconcile',
            'provider_account_reserve', 'provider_account_reconcile',
            'actual', 'actual_settlement'
        ) OR amount_micros >= 0
        """,
    )
    op.create_check_constraint(
        "ck_cost_events_nonpositive_release",
        "cost_events",
        """
        kind NOT IN ('release', 'provider_release', 'provider_account_release')
        OR amount_micros <= 0
        """,
    )
    op.create_check_constraint(
        "ck_bedrock_billing_crosschecks_not_self_superseding",
        "bedrock_billing_crosschecks",
        "supersedes_crosscheck_id IS NULL OR supersedes_crosscheck_id <> id",
    )


def _create_partial_indexes(bind: sa.Connection) -> None:
    for name, (columns, predicate) in COMMON_PARTIAL_INDEXES.items():
        if bind.dialect.name == "postgresql":
            op.execute(
                f"CREATE UNIQUE INDEX {name} ON cost_events ({columns}) "
                f"WHERE {predicate}"
            )
        elif bind.dialect.name == "sqlite":
            op.execute(
                f"CREATE UNIQUE INDEX {name} ON cost_events ({columns}) "
                f"WHERE {predicate}"
            )
        else:
            raise RuntimeError(f"unsupported database dialect for 0016: {bind.dialect.name}")
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE UNIQUE INDEX uq_cost_events_bedrock_crosscheck_adjustment
            ON cost_events ((accounting_json::jsonb ->> 'crosscheck_id'))
            WHERE kind = 'bedrock_billing_adjustment'
            """
        )


POSTGRESQL_GUARDS = f"""
CREATE OR REPLACE FUNCTION public.{COUNTER_GUARD_FUNCTION}()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    table_owner name;
    row_json jsonb;
    old_json jsonb;
BEGIN
    SELECT pg_catalog.pg_get_userbyid(c.relowner)
      INTO table_owner
      FROM pg_catalog.pg_class AS c
     WHERE c.oid = TG_RELID;
    row_json := pg_catalog.to_jsonb(NEW);
    IF TG_OP = 'INSERT' AND current_user <> table_owner THEN
        IF TG_TABLE_NAME = 'provider_account_budgets' THEN
            IF NEW.budget_used_micros <> (row_json ->> 'opening_used_micros')::bigint
               OR NEW.budget_reserved_micros <>
                  (row_json ->> 'opening_reserved_micros')::bigint THEN
                RAISE EXCEPTION
                    'initial account counters must equal the attested opening balance'
                    USING ERRCODE = '42501';
            END IF;
        ELSIF NEW.budget_used_micros <> 0 OR NEW.budget_reserved_micros <> 0 THEN
            RAISE EXCEPTION 'runtime roles cannot insert nonzero governed counters'
                USING ERRCODE = '42501';
        END IF;
    ELSIF TG_OP = 'UPDATE'
      AND (
          NEW.budget_used_micros IS DISTINCT FROM OLD.budget_used_micros
          OR NEW.budget_reserved_micros IS DISTINCT FROM OLD.budget_reserved_micros
      )
      AND current_user <> table_owner THEN
        RAISE EXCEPTION 'governed counters may be changed only by ledger authority'
            USING ERRCODE = '42501';
    END IF;
    IF TG_OP = 'UPDATE' THEN
        old_json := pg_catalog.to_jsonb(OLD);
        IF TG_TABLE_NAME = 'seasons'
           AND old_json ->> 'status' = 'cost_halted'
           AND row_json ->> 'status' IS DISTINCT FROM old_json ->> 'status'
           AND current_user <> table_owner THEN
            RAISE EXCEPTION 'a cost-halted season cannot be reopened by a runtime role'
                USING ERRCODE = '42501';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.{BATTLE_GUARD_FUNCTION}()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE table_owner name;
BEGIN
    SELECT pg_catalog.pg_get_userbyid(c.relowner)
      INTO table_owner
      FROM pg_catalog.pg_class AS c
     WHERE c.oid = TG_RELID;
    IF NEW.controlled_run_id IS NOT NULL
       AND NOT EXISTS (
           SELECT 1 FROM public.controlled_runs AS cr
           WHERE cr.id = NEW.controlled_run_id
             AND cr.season_id = NEW.season_id
       ) THEN
        RAISE EXCEPTION 'battle controlled run belongs to another season'
            USING ERRCODE = '23514';
    END IF;
    IF TG_OP = 'UPDATE'
       AND NEW.reserved_cost_micros IS DISTINCT FROM OLD.reserved_cost_micros
       AND current_user <> table_owner THEN
        RAISE EXCEPTION 'battle reservation may be changed only by ledger authority'
            USING ERRCODE = '42501';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.{COST_EVENT_GUARD_FUNCTION}()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE table_owner name;
BEGIN
    SELECT pg_catalog.pg_get_userbyid(c.relowner)
      INTO table_owner
      FROM pg_catalog.pg_class AS c
     WHERE c.oid = TG_RELID;
    IF current_user = table_owner THEN
        RETURN NEW;
    ELSIF current_user = 'flavourbench_worker' AND NEW.kind = 'actual' THEN
        RETURN NEW;
    ELSIF current_user = 'flavourbench_api' AND NEW.kind = 'actual_settlement' THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'runtime role cannot insert this governed cost-event kind'
        USING ERRCODE = '42501';
END;
$$;

CREATE OR REPLACE FUNCTION public.{BEDROCK_MEMBERSHIP_GUARD_FUNCTION}()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_season_id text;
    v_account_id text;
    v_supersedes text;
    v_active_ids text[];
BEGIN
    SELECT c.season_id::text, c.provider_account_budget_id::text,
           c.supersedes_crosscheck_id
      INTO v_season_id, v_account_id, v_supersedes
      FROM public.bedrock_billing_crosschecks AS c
     WHERE c.id = NEW.crosscheck_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'billing membership has no crosscheck';
    END IF;
    -- Billing evidence is infrequent. Serializing it per governed account
    -- gives every initial record and correction one stable lock before any
    -- crosscheck- or arm-specific lock, avoiding cross-transaction cycles.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(v_season_id || ':' || v_account_id, 7212029)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(NEW.crosscheck_id, 7212026)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(NEW.arm_id, 7212028)
    );
    SELECT pg_catalog.array_agg(DISTINCT c.id ORDER BY c.id)
      INTO v_active_ids
      FROM public.bedrock_billing_crosscheck_arms AS m
      JOIN public.bedrock_billing_crosschecks AS c ON c.id = m.crosscheck_id
     WHERE m.arm_id = NEW.arm_id
       AND c.id <> NEW.crosscheck_id
       AND NOT EXISTS (
           SELECT 1 FROM public.bedrock_billing_crosschecks AS successor
           WHERE successor.supersedes_crosscheck_id = c.id
             AND successor.id <> NEW.crosscheck_id
       );
    IF (v_supersedes IS NULL AND v_active_ids IS NOT NULL)
       OR (v_supersedes IS NOT NULL AND v_active_ids IS DISTINCT FROM ARRAY[v_supersedes]) THEN
        RAISE EXCEPTION 'billing arm already belongs to another active crosscheck';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.cost_events AS ce
        WHERE ce.kind = 'bedrock_billing_adjustment'
          AND ce.accounting_json::jsonb ->> 'crosscheck_id' = NEW.crosscheck_id
    ) THEN
        RAISE EXCEPTION 'a sealed billing crosscheck cannot gain new arms';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.{ATTEMPT_ARM_GUARD_FUNCTION}()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(NEW.attempt_id, 7212027)
    );
    IF EXISTS (
        SELECT 1 FROM public.generation_attempts AS ga
        WHERE ga.attempt_id = NEW.attempt_id AND ga.arm_id <> NEW.arm_id
    ) THEN
        RAISE EXCEPTION 'one generation attempt id cannot span response arms';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_budget_counter_authority_seasons ON public.seasons;
CREATE TRIGGER trg_budget_counter_authority_seasons
BEFORE INSERT OR UPDATE ON public.seasons
FOR EACH ROW EXECUTE FUNCTION public.{COUNTER_GUARD_FUNCTION}();
DROP TRIGGER IF EXISTS trg_budget_counter_authority_season_provider_budgets
ON public.season_provider_budgets;
CREATE TRIGGER trg_budget_counter_authority_season_provider_budgets
BEFORE INSERT OR UPDATE ON public.season_provider_budgets
FOR EACH ROW EXECUTE FUNCTION public.{COUNTER_GUARD_FUNCTION}();
DROP TRIGGER IF EXISTS trg_budget_counter_authority_provider_account_budgets
ON public.provider_account_budgets;
CREATE TRIGGER trg_budget_counter_authority_provider_account_budgets
BEFORE INSERT OR UPDATE ON public.provider_account_budgets
FOR EACH ROW EXECUTE FUNCTION public.{COUNTER_GUARD_FUNCTION}();
DROP TRIGGER IF EXISTS trg_budget_counter_authority_controlled_runs ON public.controlled_runs;
CREATE TRIGGER trg_budget_counter_authority_controlled_runs
BEFORE INSERT OR UPDATE ON public.controlled_runs
FOR EACH ROW EXECUTE FUNCTION public.{COUNTER_GUARD_FUNCTION}();
DROP TRIGGER IF EXISTS trg_battle_reservation_authority ON public.battles;
CREATE TRIGGER trg_battle_reservation_authority
BEFORE INSERT OR UPDATE ON public.battles
FOR EACH ROW EXECUTE FUNCTION public.{BATTLE_GUARD_FUNCTION}();
DROP TRIGGER IF EXISTS trg_cost_event_authority ON public.cost_events;
CREATE TRIGGER trg_cost_event_authority
BEFORE INSERT ON public.cost_events
FOR EACH ROW EXECUTE FUNCTION public.{COST_EVENT_GUARD_FUNCTION}();
DROP TRIGGER IF EXISTS trg_bedrock_membership_seal
ON public.bedrock_billing_crosscheck_arms;
CREATE TRIGGER trg_bedrock_membership_seal
BEFORE INSERT ON public.bedrock_billing_crosscheck_arms
FOR EACH ROW EXECUTE FUNCTION public.{BEDROCK_MEMBERSHIP_GUARD_FUNCTION}();
DROP TRIGGER IF EXISTS trg_generation_attempt_arm_authority
ON public.generation_attempts;
CREATE TRIGGER trg_generation_attempt_arm_authority
BEFORE INSERT ON public.generation_attempts
FOR EACH ROW EXECUTE FUNCTION public.{ATTEMPT_ARM_GUARD_FUNCTION}();
"""


POSTGRESQL_RESERVE_FUNCTION = f"""
CREATE OR REPLACE FUNCTION public.{RESERVE_FUNCTION}(p_battle_id text)
RETURNS TABLE(
    reserved_cost_micros bigint,
    provider_reservations jsonb,
    idempotent boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_battle public.battles%ROWTYPE;
    v_season public.seasons%ROWTYPE;
    v_run public.controlled_runs%ROWTYPE;
    v_provider jsonb;
    v_total bigint;
    v_arm_count integer;
    v_matched_arm_count integer;
    v_existing integer;
    v_backend text;
    v_amount bigint;
    v_pb public.season_provider_budgets%ROWTYPE;
    v_ab public.provider_account_budgets%ROWTYPE;
BEGIN
    SELECT * INTO v_battle FROM public.battles
     WHERE id = p_battle_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'battle does not exist';
    END IF;
    SELECT * INTO v_season FROM public.seasons
     WHERE id = v_battle.season_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'battle season does not exist';
    END IF;
    IF v_battle.controlled_run_id IS NOT NULL THEN
        SELECT * INTO v_run FROM public.controlled_runs
         WHERE id = v_battle.controlled_run_id FOR UPDATE;
    END IF;
    SELECT * INTO v_battle FROM public.battles
     WHERE id = p_battle_id FOR UPDATE;
    PERFORM id FROM public.response_arms
     WHERE battle_id = p_battle_id ORDER BY id FOR UPDATE;
    SELECT COUNT(*) INTO v_arm_count
      FROM public.response_arms AS a
     WHERE a.battle_id = p_battle_id
       AND (
           (a.id = v_battle.left_arm_id AND a.side = 'left')
           OR (a.id = v_battle.right_arm_id AND a.side = 'right')
       );
    IF v_battle.status <> 'queued'
       OR v_battle.left_arm_id IS NULL OR v_battle.right_arm_id IS NULL
       OR v_battle.left_arm_id = v_battle.right_arm_id
       OR v_arm_count <> 2
       OR (SELECT COUNT(*) FROM public.response_arms
            WHERE battle_id = p_battle_id) <> 2
       OR EXISTS (
            SELECT 1 FROM public.generation_attempts AS ga
            JOIN public.response_arms AS a ON a.id = ga.arm_id
            WHERE a.battle_id = p_battle_id
       ) THEN
        RAISE EXCEPTION 'battle is not an unstarted two-arm reservation candidate';
    END IF;
    SELECT COUNT(*) INTO v_matched_arm_count
      FROM public.response_arms AS a
      JOIN public.season_models AS sm
        ON sm.season_id = v_battle.season_id
       AND sm.model_id = a.model_id
       AND sm.execution_backend = a.execution_backend
       AND sm.provider_slug = a.provider_slug
     WHERE a.battle_id = p_battle_id
       AND a.id IN (v_battle.left_arm_id, v_battle.right_arm_id)
       AND a.execution_backend IN ('openrouter', 'bedrock')
       AND sm.eligible IS TRUE
       AND sm.worst_case_cost_micros > 0;
    SELECT pg_catalog.jsonb_object_agg(x.execution_backend, x.amount),
           SUM(x.amount)::bigint
      INTO v_provider, v_total
      FROM (
          SELECT a.execution_backend, SUM(sm.worst_case_cost_micros)::bigint AS amount
          FROM public.response_arms AS a
          JOIN public.season_models AS sm
            ON sm.season_id = v_battle.season_id
           AND sm.model_id = a.model_id
           AND sm.execution_backend = a.execution_backend
           AND sm.provider_slug = a.provider_slug
          WHERE a.battle_id = p_battle_id
            AND a.id IN (v_battle.left_arm_id, v_battle.right_arm_id)
            AND a.execution_backend IN ('openrouter', 'bedrock')
            AND sm.eligible IS TRUE
            AND sm.worst_case_cost_micros > 0
          GROUP BY a.execution_backend
      ) AS x;
    IF v_matched_arm_count <> 2
       OR v_provider IS NULL OR v_total IS NULL OR v_total <= 0
       OR v_provider <> v_battle.provider_reservations_json::jsonb
       OR v_total <> v_battle.reserved_cost_micros THEN
        RAISE EXCEPTION 'battle reservation contract does not match frozen endpoint costs';
    END IF;
    SELECT COUNT(*) INTO v_existing FROM public.cost_events
     WHERE battle_id = p_battle_id
       AND kind IN ('reserve', 'provider_reserve', 'provider_account_reserve');
    IF v_existing > 0 THEN
        IF (SELECT COUNT(*) FROM public.cost_events
             WHERE battle_id = p_battle_id AND kind = 'reserve'
               AND provider = 'governor' AND amount_micros = v_total) <> 1 THEN
            RAISE EXCEPTION 'partial or inconsistent governor reservation evidence';
        END IF;
        FOR v_backend, v_amount IN
            SELECT key, value::bigint FROM pg_catalog.jsonb_each_text(v_provider)
        LOOP
            IF (SELECT COUNT(*) FROM public.cost_events
                 WHERE battle_id = p_battle_id AND kind = 'provider_reserve'
                   AND provider = v_backend AND amount_micros = v_amount) <> 1
               OR (SELECT COUNT(*) FROM public.cost_events
                 WHERE battle_id = p_battle_id AND kind = 'provider_account_reserve'
                   AND provider = v_backend AND amount_micros = v_amount) <> 1 THEN
                RAISE EXCEPTION 'partial or inconsistent provider reservation evidence';
            END IF;
        END LOOP;
        reserved_cost_micros := v_total;
        provider_reservations := v_provider;
        idempotent := true;
        RETURN NEXT;
        RETURN;
    END IF;
    IF v_season.status NOT IN ('pilot', 'active')
       OR v_season.budget_cap_micros <= 0
       OR (v_season.budget_used_micros + v_season.budget_reserved_micros + v_total)
          * 10000 >= v_season.budget_cap_micros * 8500 THEN
        RAISE EXCEPTION 'season budget admission is closed';
    END IF;
    PERFORM id FROM public.season_provider_budgets
     WHERE season_id = v_season.id ORDER BY execution_backend FOR UPDATE;
    PERFORM ab.id
      FROM public.provider_account_budgets AS ab
      JOIN public.season_provider_budgets AS pb
        ON pb.execution_backend = ab.execution_backend
       AND pb.account_scope_sha256 = ab.account_scope_sha256
     WHERE pb.season_id = v_season.id
     ORDER BY ab.execution_backend, ab.account_scope_sha256 FOR UPDATE OF ab;
    IF v_battle.controlled_run_id IS NOT NULL THEN
        IF v_run.id IS NULL OR v_run.status <> 'active'
           OR v_run.season_id <> v_battle.season_id
           OR (v_run.budget_used_micros + v_run.budget_reserved_micros + v_total)
              * 10000 >= v_run.budget_cap_micros * 8500 THEN
            RAISE EXCEPTION 'controlled-run budget admission is closed';
        END IF;
    END IF;
    FOR v_backend, v_amount IN
        SELECT key, value::bigint FROM pg_catalog.jsonb_each_text(v_provider)
    LOOP
        SELECT * INTO v_pb FROM public.season_provider_budgets
         WHERE season_id = v_season.id AND execution_backend = v_backend;
        SELECT * INTO v_ab FROM public.provider_account_budgets
         WHERE execution_backend = v_backend
           AND account_scope_sha256 = v_pb.account_scope_sha256;
        IF v_pb.id IS NULL OR v_pb.valid_until <= CURRENT_TIMESTAMP
           OR (v_pb.budget_used_micros + v_pb.budget_reserved_micros + v_amount)
              * 10000 >= v_pb.budget_cap_micros * 8500
           OR v_ab.id IS NULL OR v_ab.status <> 'active'
           OR v_ab.valid_until <= CURRENT_TIMESTAMP
           OR (v_ab.budget_used_micros + v_ab.budget_reserved_micros + v_amount)
              * 10000 >= v_ab.budget_cap_micros * 8500 THEN
            RAISE EXCEPTION '% provider or account budget admission is closed', v_backend;
        END IF;
    END LOOP;
    UPDATE public.seasons SET budget_reserved_micros = budget_reserved_micros + v_total
     WHERE id = v_season.id;
    IF v_battle.controlled_run_id IS NOT NULL THEN
        UPDATE public.controlled_runs
           SET budget_reserved_micros = budget_reserved_micros + v_total
         WHERE id = v_battle.controlled_run_id;
    END IF;
    INSERT INTO public.cost_events(
        id, season_id, battle_id, kind, amount_micros, provider,
        accounting_json, created_at
    ) VALUES (
        pg_catalog.gen_random_uuid()::text, v_season.id, p_battle_id,
        'reserve', v_total, 'governor', '{{}}'::jsonb, CURRENT_TIMESTAMP
    );
    FOR v_backend, v_amount IN
        SELECT key, value::bigint FROM pg_catalog.jsonb_each_text(v_provider)
    LOOP
        SELECT * INTO v_pb FROM public.season_provider_budgets
         WHERE season_id = v_season.id AND execution_backend = v_backend;
        UPDATE public.season_provider_budgets
           SET budget_reserved_micros = budget_reserved_micros + v_amount
         WHERE id = v_pb.id;
        UPDATE public.provider_account_budgets
           SET budget_reserved_micros = budget_reserved_micros + v_amount
         WHERE execution_backend = v_backend
           AND account_scope_sha256 = v_pb.account_scope_sha256;
        INSERT INTO public.cost_events(
            id, season_id, battle_id, kind, amount_micros, provider,
            accounting_json, created_at
        ) VALUES
        (pg_catalog.gen_random_uuid()::text, v_season.id, p_battle_id,
         'provider_reserve', v_amount, v_backend,
         pg_catalog.jsonb_build_object('budget_scope', 'provider'), CURRENT_TIMESTAMP),
        (pg_catalog.gen_random_uuid()::text, v_season.id, p_battle_id,
         'provider_account_reserve', v_amount, v_backend,
         pg_catalog.jsonb_build_object(
             'budget_scope', 'provider_account',
             'account_scope_sha256', v_pb.account_scope_sha256
         ), CURRENT_TIMESTAMP);
    END LOOP;
    reserved_cost_micros := v_total;
    provider_reservations := v_provider;
    idempotent := false;
    RETURN NEXT;
END;
$$;
"""


POSTGRESQL_SETTLE_FUNCTION = f"""
CREATE OR REPLACE FUNCTION public.{SETTLE_FUNCTION}(p_battle_id text)
RETURNS TABLE(
    released_micros bigint,
    actual_micros bigint,
    cost_halted boolean,
    halt_reasons jsonb,
    idempotent boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_battle public.battles%ROWTYPE;
    v_season public.seasons%ROWTYPE;
    v_provider_reserved jsonb;
    v_provider_actual jsonb;
    v_actual bigint;
    v_reserved bigint;
    v_receipts integer;
    v_unresolved integer;
    v_settlement_events integer;
    v_existing integer;
    v_backend text;
    v_amount bigint;
    v_backend_actual bigint;
    v_pb public.season_provider_budgets%ROWTYPE;
    v_ab public.provider_account_budgets%ROWTYPE;
    v_run public.controlled_runs%ROWTYPE;
    v_halt boolean := false;
    v_reasons jsonb := '[]'::jsonb;
BEGIN
    SELECT * INTO v_battle FROM public.battles
     WHERE id = p_battle_id;
    IF NOT FOUND THEN RAISE EXCEPTION 'battle does not exist'; END IF;
    SELECT * INTO v_season FROM public.seasons
     WHERE id = v_battle.season_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'battle season does not exist'; END IF;
    IF v_battle.controlled_run_id IS NOT NULL THEN
        SELECT * INTO v_run FROM public.controlled_runs
         WHERE id = v_battle.controlled_run_id FOR UPDATE;
        IF NOT FOUND OR v_run.season_id <> v_battle.season_id THEN
            RAISE EXCEPTION 'controlled run does not belong to battle season';
        END IF;
    END IF;
    SELECT * INTO v_battle FROM public.battles
     WHERE id = p_battle_id FOR UPDATE;
    PERFORM id FROM public.response_arms
     WHERE battle_id = p_battle_id ORDER BY id FOR UPDATE;
    IF v_battle.status NOT IN ('complete', 'failed') OR v_battle.completed_at IS NULL
       OR (SELECT COUNT(*) FROM public.response_arms
            WHERE battle_id = p_battle_id
              AND (
                  (id = v_battle.left_arm_id AND side = 'left')
                  OR (id = v_battle.right_arm_id AND side = 'right')
              )
              AND status IN ('complete', 'failed')
              AND completed_at IS NOT NULL
              AND cost_micros >= 0
              AND cost_reconciled IS TRUE
              AND cost_accounting_basis <> 'unrecorded'
              AND billing_reconciliation_status <> 'unrecorded') <> 2 THEN
        RAISE EXCEPTION 'battle arms are not terminal and cost reconciled';
    END IF;
    IF v_battle.left_arm_id IS NULL OR v_battle.right_arm_id IS NULL
       OR v_battle.left_arm_id = v_battle.right_arm_id
       OR (SELECT COUNT(*) FROM public.response_arms
            WHERE battle_id = p_battle_id) <> 2 THEN
        RAISE EXCEPTION 'battle does not own exactly its linked left/right arms';
    END IF;
    SELECT COUNT(*) INTO v_receipts
      FROM public.response_arms AS a
     WHERE a.battle_id = p_battle_id
       AND EXISTS (
           SELECT 1 FROM public.cost_events AS ce
            WHERE ce.id = (
                SELECT x.id FROM public.cost_events AS x
                 WHERE x.arm_id = a.id AND x.kind IN ('actual', 'actual_settlement')
                 ORDER BY CASE WHEN x.kind = 'actual_settlement' THEN 0 ELSE 1 END,
                          x.created_at DESC, x.id DESC
                 LIMIT 1
            )
              AND ce.season_id = v_battle.season_id
              AND ce.battle_id = p_battle_id
              AND ce.amount_micros = a.cost_micros
              AND ce.amount_micros >= 0
              AND ce.provider = COALESCE(a.actual_provider_slug, a.provider_slug)
              AND ce.generation_id IS NOT DISTINCT FROM a.generation_id
              AND (
                  (
                      ce.kind = 'actual'
                      AND COALESCE(
                          ce.accounting_json::jsonb ->> 'reconciled', 'false'
                      ) = 'true'
                      AND COALESCE(
                          ce.accounting_json::jsonb ->> 'cost_accounting_basis',
                          ce.accounting_json::jsonb ->> 'basis'
                      ) = a.cost_accounting_basis
                      AND ce.accounting_json::jsonb ->>
                          'billing_reconciliation_status' =
                          a.billing_reconciliation_status
                  )
                  OR (
                      ce.kind = 'actual_settlement'
                      AND a.cost_accounting_basis = 'manual_authorized_settlement'
                      AND a.billing_reconciliation_status =
                          'manual_authorized_settlement'
                      AND ce.accounting_json::jsonb ->> 'settlement' =
                          'manual_authorized'
                      AND ce.accounting_json::jsonb ->> 'prior_cost_state' =
                          'unresolved_attempt_journal'
                      AND length(COALESCE(
                          ce.accounting_json::jsonb ->>
                              'authorization_reference_sha256', ''
                      )) = 64
                      AND (
                          (
                              EXISTS (
                                  SELECT 1 FROM public.cost_events AS prior
                                  WHERE prior.arm_id = a.id
                                    AND prior.kind = 'actual'
                                    AND prior.id = ce.accounting_json::jsonb ->>
                                        'supersedes_cost_event_id'
                                    AND COALESCE(
                                        prior.accounting_json::jsonb ->>
                                            'reconciled', 'false'
                                    ) = 'false'
                              )
                          )
                          OR (
                              NOT EXISTS (
                                  SELECT 1 FROM public.cost_events AS prior
                                  WHERE prior.arm_id = a.id
                                    AND prior.kind = 'actual'
                              )
                              AND ce.accounting_json::jsonb ->
                                  'supersedes_cost_event_id' = 'null'::jsonb
                              AND EXISTS (
                                  SELECT 1
                                  FROM (
                                      SELECT DISTINCT ON (ga.arm_id, ga.attempt_id)
                                          ga.event_type
                                      FROM public.generation_attempts AS ga
                                      WHERE ga.arm_id = a.id
                                        AND ga.event_type NOT LIKE 'mcp_%'
                                      ORDER BY ga.arm_id, ga.attempt_id,
                                               ga.created_at DESC, ga.id DESC
                                  ) AS latest_prior
                                  WHERE latest_prior.event_type NOT IN (
                                      'pre_send_failure', 'request_rejected',
                                      'accounting_reconciled'
                                  )
                              )
                          )
                      )
                  )
              )
       );
    IF v_receipts <> 2 THEN
        RAISE EXCEPTION 'endpoint-generation receipts are incomplete';
    END IF;
    SELECT COUNT(*) INTO v_settlement_events
      FROM public.run_events AS re
     WHERE re.entity_type = 'battle'
       AND re.entity_id = p_battle_id
       AND re.event_type = 'generation_cost_exposure_settled';
    IF EXISTS (
        SELECT 1 FROM public.cost_events
        WHERE battle_id = p_battle_id AND kind = 'actual_settlement'
    ) AND (
        v_battle.controlled_run_id IS NULL
        OR v_settlement_events <> 1
        OR NOT EXISTS (
            SELECT 1 FROM public.run_events AS re
            WHERE re.entity_type = 'battle'
              AND re.entity_id = p_battle_id
              AND re.event_type = 'generation_cost_exposure_settled'
              AND re.payload_json::jsonb ->> 'controlled_run_id' =
                  v_battle.controlled_run_id
              AND (re.payload_json::jsonb - ARRAY[
                  'controlled_run_id', 'authorization_reference_sha256',
                  'arm_costs_micros'
              ]::text[]) = '{{}}'::jsonb
              AND jsonb_typeof(re.payload_json::jsonb -> 'arm_costs_micros') =
                  'object'
              AND (SELECT COUNT(*) FROM pg_catalog.jsonb_object_keys(
                  re.payload_json::jsonb -> 'arm_costs_micros'
              )) = (SELECT COUNT(*) FROM public.cost_events
                  WHERE battle_id = p_battle_id AND kind = 'actual_settlement')
              AND NOT EXISTS (
                  SELECT 1 FROM public.cost_events AS ce
                  WHERE ce.battle_id = p_battle_id
                    AND ce.kind = 'actual_settlement'
                    AND (
                        re.payload_json::jsonb ->>
                            'authorization_reference_sha256'
                            <> ce.accounting_json::jsonb ->>
                                'authorization_reference_sha256'
                        OR re.payload_json::jsonb -> 'arm_costs_micros' ->>
                            ce.arm_id <> ce.amount_micros::text
                    )
              )
        )
    ) THEN
        RAISE EXCEPTION 'manual settlement authorization evidence is invalid';
    END IF;
    WITH latest AS (
        SELECT DISTINCT ON (ga.arm_id, ga.attempt_id) ga.arm_id, ga.event_type
        FROM public.generation_attempts AS ga
        JOIN public.response_arms AS a ON a.id = ga.arm_id
        WHERE a.battle_id = p_battle_id
          AND ga.event_type NOT LIKE 'mcp_%'
        ORDER BY ga.arm_id, ga.attempt_id, ga.created_at DESC, ga.id DESC
    )
    SELECT COUNT(*) INTO v_unresolved FROM latest
     WHERE event_type NOT IN ('pre_send_failure', 'request_rejected', 'accounting_reconciled')
       AND NOT EXISTS (
           SELECT 1 FROM public.cost_events AS ce
           WHERE ce.arm_id = latest.arm_id AND ce.kind = 'actual_settlement'
       );
    IF v_unresolved > 0 THEN
        RAISE EXCEPTION 'battle has unresolved paid-attempt evidence';
    END IF;
    SELECT COALESCE(SUM(cost_micros), 0)::bigint INTO v_actual
      FROM public.response_arms
     WHERE battle_id = p_battle_id
       AND id IN (v_battle.left_arm_id, v_battle.right_arm_id);
    SELECT pg_catalog.jsonb_object_agg(x.execution_backend, x.amount)
      INTO v_provider_actual
      FROM (
          SELECT execution_backend, SUM(cost_micros)::bigint AS amount
          FROM public.response_arms
          WHERE battle_id = p_battle_id
            AND id IN (v_battle.left_arm_id, v_battle.right_arm_id)
          GROUP BY execution_backend
      ) AS x;
    v_provider_reserved := v_battle.provider_reservations_json::jsonb;
    v_reserved := v_battle.reserved_cost_micros;
    SELECT COUNT(*) INTO v_existing FROM public.cost_events
     WHERE battle_id = p_battle_id
       AND kind IN (
           'release', 'reconcile', 'provider_release', 'provider_reconcile',
           'provider_account_release', 'provider_account_reconcile'
       );
    IF EXISTS (SELECT 1 FROM public.cost_events
                WHERE battle_id = p_battle_id AND kind = 'reconcile') THEN
        IF v_reserved <> 0
           OR (SELECT COUNT(*) FROM public.cost_events
                WHERE battle_id = p_battle_id AND kind = 'reconcile'
                  AND provider = 'governor' AND amount_micros = v_actual) <> 1 THEN
            RAISE EXCEPTION 'settlement idempotency evidence is inconsistent';
        END IF;
        released_micros := COALESCE((SELECT -amount_micros FROM public.cost_events
            WHERE battle_id = p_battle_id AND kind = 'release'), 0);
        actual_micros := v_actual;
        cost_halted := EXISTS (
            SELECT 1 FROM public.seasons
            WHERE id = v_battle.season_id AND status = 'cost_halted'
        );
        halt_reasons := CASE
            WHEN cost_halted
            THEN '["season_already_cost_halted"]'::jsonb
            ELSE '[]'::jsonb
        END;
        idempotent := true;
        RETURN NEXT;
        RETURN;
    ELSIF v_existing > 0 THEN
        RAISE EXCEPTION 'partial settlement evidence exists';
    END IF;
    IF v_reserved <= 0 OR v_provider_reserved IS NULL
       OR (SELECT COALESCE(SUM(value::bigint), 0)
             FROM pg_catalog.jsonb_each_text(v_provider_reserved)) <> v_reserved
       OR (SELECT COUNT(*) FROM public.cost_events
            WHERE battle_id = p_battle_id AND kind = 'reserve'
              AND provider = 'governor' AND amount_micros = v_reserved) <> 1 THEN
        RAISE EXCEPTION 'original reservation evidence is invalid';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_catalog.jsonb_object_keys(v_provider_actual) AS actual(key)
        WHERE NOT (v_provider_reserved ? actual.key)
    ) OR EXISTS (
        SELECT 1 FROM pg_catalog.jsonb_object_keys(v_provider_reserved) AS reserved(key)
        WHERE NOT (v_provider_actual ? reserved.key)
    ) THEN
        RAISE EXCEPTION 'actual provider set differs from the frozen reservation contract';
    END IF;
    PERFORM id FROM public.season_provider_budgets
     WHERE season_id = v_season.id ORDER BY execution_backend FOR UPDATE;
    PERFORM ab.id
      FROM public.provider_account_budgets AS ab
      JOIN public.season_provider_budgets AS pb
        ON pb.execution_backend = ab.execution_backend
       AND pb.account_scope_sha256 = ab.account_scope_sha256
     WHERE pb.season_id = v_season.id
     ORDER BY ab.execution_backend, ab.account_scope_sha256 FOR UPDATE OF ab;
    IF v_season.budget_reserved_micros < v_reserved THEN
        RAISE EXCEPTION 'season reservation underflow';
    END IF;
    IF v_actual > v_reserved THEN
        v_halt := true;
        v_reasons := v_reasons || pg_catalog.jsonb_build_array('actual_exceeds_reservation');
    END IF;
    UPDATE public.seasons
       SET budget_reserved_micros = budget_reserved_micros - v_reserved,
           budget_used_micros = budget_used_micros + v_actual
     WHERE id = v_season.id;
    IF v_battle.controlled_run_id IS NOT NULL THEN
        IF NOT EXISTS (
            SELECT 1 FROM public.controlled_runs
            WHERE id = v_battle.controlled_run_id
              AND season_id = v_battle.season_id
              AND budget_reserved_micros >= v_reserved
        ) THEN
            RAISE EXCEPTION 'controlled-run reservation underflow';
        END IF;
        UPDATE public.controlled_runs
           SET budget_reserved_micros = budget_reserved_micros - v_reserved,
               budget_used_micros = budget_used_micros + v_actual
         WHERE id = v_battle.controlled_run_id;
    END IF;
    FOR v_backend, v_amount IN
        SELECT key, value::bigint FROM pg_catalog.jsonb_each_text(v_provider_reserved)
    LOOP
        v_backend_actual := COALESCE((v_provider_actual ->> v_backend)::bigint, 0);
        SELECT * INTO v_pb FROM public.season_provider_budgets
         WHERE season_id = v_season.id AND execution_backend = v_backend;
        SELECT * INTO v_ab FROM public.provider_account_budgets
         WHERE execution_backend = v_backend
           AND account_scope_sha256 = v_pb.account_scope_sha256;
        IF v_pb.id IS NULL OR v_ab.id IS NULL
           OR v_pb.budget_reserved_micros < v_amount
           OR v_ab.budget_reserved_micros < v_amount THEN
            RAISE EXCEPTION '% provider/account reservation underflow', v_backend;
        END IF;
        IF v_backend_actual > v_amount THEN
            v_halt := true;
            v_reasons := v_reasons ||
                pg_catalog.jsonb_build_array(v_backend || '_actual_exceeds_reservation');
        END IF;
        UPDATE public.season_provider_budgets
           SET budget_reserved_micros = budget_reserved_micros - v_amount,
               budget_used_micros = budget_used_micros + v_backend_actual
         WHERE id = v_pb.id;
        UPDATE public.provider_account_budgets
           SET budget_reserved_micros = budget_reserved_micros - v_amount,
               budget_used_micros = budget_used_micros + v_backend_actual
         WHERE id = v_ab.id;
        INSERT INTO public.cost_events(
            id, season_id, battle_id, kind, amount_micros, provider,
            accounting_json, created_at
        ) VALUES
        (pg_catalog.gen_random_uuid()::text, v_season.id, p_battle_id,
         'provider_release', -v_amount, v_backend,
         pg_catalog.jsonb_build_object('budget_scope', 'provider'), CURRENT_TIMESTAMP),
        (pg_catalog.gen_random_uuid()::text, v_season.id, p_battle_id,
         'provider_reconcile', v_backend_actual, v_backend,
         pg_catalog.jsonb_build_object('budget_scope', 'provider'), CURRENT_TIMESTAMP),
        (pg_catalog.gen_random_uuid()::text, v_season.id, p_battle_id,
         'provider_account_release', -v_amount, v_backend,
         pg_catalog.jsonb_build_object(
             'budget_scope', 'provider_account',
             'account_scope_sha256', v_pb.account_scope_sha256
         ), CURRENT_TIMESTAMP),
        (pg_catalog.gen_random_uuid()::text, v_season.id, p_battle_id,
         'provider_account_reconcile', v_backend_actual, v_backend,
         pg_catalog.jsonb_build_object(
             'budget_scope', 'provider_account',
             'account_scope_sha256', v_pb.account_scope_sha256
         ), CURRENT_TIMESTAMP);
    END LOOP;
    UPDATE public.battles SET reserved_cost_micros = 0 WHERE id = p_battle_id;
    INSERT INTO public.cost_events(
        id, season_id, battle_id, kind, amount_micros, provider,
        accounting_json, created_at
    ) VALUES
    (pg_catalog.gen_random_uuid()::text, v_season.id, p_battle_id,
     'release', -v_reserved, 'governor', '{{}}'::jsonb, CURRENT_TIMESTAMP),
    (pg_catalog.gen_random_uuid()::text, v_season.id, p_battle_id,
     'reconcile', v_actual, 'governor', '{{}}'::jsonb, CURRENT_TIMESTAMP);
    IF EXISTS (SELECT 1 FROM public.seasons
                WHERE id = v_season.id
                  AND budget_used_micros + budget_reserved_micros > budget_cap_micros)
       OR EXISTS (SELECT 1 FROM public.season_provider_budgets
                   WHERE season_id = v_season.id
                     AND budget_used_micros + budget_reserved_micros > budget_cap_micros)
       OR EXISTS (SELECT 1 FROM public.provider_account_budgets AS ab
                   JOIN public.season_provider_budgets AS pb
                     ON pb.execution_backend = ab.execution_backend
                    AND pb.account_scope_sha256 = ab.account_scope_sha256
                  WHERE pb.season_id = v_season.id
                    AND ab.budget_used_micros + ab.budget_reserved_micros >
                        ab.budget_cap_micros) THEN
        v_halt := true;
        v_reasons := v_reasons || pg_catalog.jsonb_build_array('governed_cap_exceeded');
    END IF;
    IF v_halt THEN
        UPDATE public.seasons SET status = 'cost_halted' WHERE id = v_season.id;
    END IF;
    released_micros := v_reserved;
    actual_micros := v_actual;
    cost_halted := v_halt;
    halt_reasons := v_reasons;
    idempotent := false;
    RETURN NEXT;
END;
$$;
"""


POSTGRESQL_BILLING_FUNCTION = f"""
CREATE OR REPLACE FUNCTION public.{BILLING_FUNCTION}(p_crosscheck_id text)
RETURNS TABLE(
    ledger_delta_micros bigint,
    governed_delta_micros bigint,
    cost_halted boolean,
    idempotent boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_cross public.bedrock_billing_crosschecks%ROWTYPE;
    v_season public.seasons%ROWTYPE;
    v_pb public.season_provider_budgets%ROWTYPE;
    v_ab public.provider_account_budgets%ROWTYPE;
    v_rate bigint;
    v_difference bigint;
    v_predecessor_difference bigint := 0;
    v_tolerance bigint;
    v_expected_status text;
    v_expected_arm_hash text;
    v_delta bigint;
    v_governed bigint;
    v_arm_ids text[];
    v_prior_arm_ids text[];
    v_arm_id text;
    v_halt boolean := false;
    v_event public.cost_events%ROWTYPE;
    v_predecessor public.bedrock_billing_crosschecks%ROWTYPE;
    v_predecessor_event public.cost_events%ROWTYPE;
BEGIN
    SELECT * INTO v_cross FROM public.bedrock_billing_crosschecks
     WHERE id = p_crosscheck_id;
    IF NOT FOUND THEN RAISE EXCEPTION 'billing crosscheck does not exist'; END IF;
    IF v_cross.supersedes_crosscheck_id = v_cross.id THEN
        RAISE EXCEPTION 'billing crosscheck cannot supersede itself';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            v_cross.season_id::text || ':' ||
            v_cross.provider_account_budget_id::text,
            7212029
        )
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_crosscheck_id, 7212026)
    );
    SELECT * INTO v_cross FROM public.bedrock_billing_crosschecks
     WHERE id = p_crosscheck_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'billing crosscheck disappeared'; END IF;
    SELECT * INTO v_season FROM public.seasons
     WHERE id = v_cross.season_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'billing crosscheck season does not exist'; END IF;
    SELECT pg_catalog.array_agg(m.arm_id ORDER BY m.arm_id) INTO v_arm_ids
      FROM public.bedrock_billing_crosscheck_arms AS m
     WHERE m.crosscheck_id = p_crosscheck_id;
    IF v_arm_ids IS NULL THEN RAISE EXCEPTION 'billing crosscheck has no arms'; END IF;
    FOREACH v_arm_id IN ARRAY v_arm_ids
    LOOP
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(v_arm_id, 7212028)
        );
        IF (SELECT pg_catalog.array_agg(DISTINCT c.id::text ORDER BY c.id::text)
              FROM public.bedrock_billing_crosscheck_arms AS m
              JOIN public.bedrock_billing_crosschecks AS c
                ON c.id = m.crosscheck_id
             WHERE m.arm_id = v_arm_id
               AND NOT EXISTS (
                   SELECT 1 FROM public.bedrock_billing_crosschecks AS successor
                   WHERE successor.supersedes_crosscheck_id = c.id
               )) IS DISTINCT FROM ARRAY[p_crosscheck_id] THEN
            RAISE EXCEPTION 'billing crosscheck is not the sole active record for its arms';
        END IF;
    END LOOP;
    PERFORM b.id FROM public.battles AS b
     JOIN public.response_arms AS a ON a.battle_id = b.id
     WHERE a.id = ANY(v_arm_ids) ORDER BY b.id FOR UPDATE OF b;
    PERFORM a.id FROM public.response_arms AS a
     WHERE a.id = ANY(v_arm_ids) ORDER BY a.id FOR UPDATE;
    PERFORM id FROM public.season_provider_budgets
     WHERE season_id = v_season.id ORDER BY execution_backend FOR UPDATE;
    SELECT * INTO v_ab FROM public.provider_account_budgets
     WHERE id = v_cross.provider_account_budget_id FOR UPDATE;
    PERFORM arm_id FROM public.bedrock_billing_crosscheck_arms
     WHERE crosscheck_id = p_crosscheck_id ORDER BY arm_id FOR UPDATE;
    SELECT * INTO v_pb FROM public.season_provider_budgets
     WHERE season_id = v_season.id AND execution_backend = 'bedrock';
    IF v_ab.id IS NULL OR v_pb.id IS NULL OR v_ab.id <> v_cross.provider_account_budget_id
       OR v_ab.execution_backend <> 'bedrock'
       OR v_ab.account_scope_sha256 <> v_pb.account_scope_sha256 THEN
        RAISE EXCEPTION 'billing crosscheck is not linked to its Bedrock account';
    END IF;
    IF (SELECT COUNT(*) FROM public.response_arms AS a
        JOIN public.battles AS b ON b.id = a.battle_id
        WHERE a.id = ANY(v_arm_ids) AND b.season_id = v_season.id
          AND a.execution_backend = 'bedrock'
          AND a.cost_reconciled IS TRUE
          AND a.cost_micros >= 0
          AND a.completed_at IS NOT NULL
          AND a.completed_at >= v_cross.coverage_start
          AND a.completed_at <= v_cross.coverage_end
          AND EXISTS (
              SELECT 1 FROM public.cost_events AS ce
              WHERE ce.arm_id = a.id
                AND ce.season_id = v_season.id
                AND ce.battle_id = a.battle_id
                AND ce.kind = 'actual'
                AND ce.amount_micros = a.cost_micros
                AND ce.amount_micros >= 0
          ))
       <> pg_catalog.array_length(v_arm_ids, 1) THEN
        RAISE EXCEPTION 'billing crosscheck arms are not exact reconciled Bedrock arms';
    END IF;
    SELECT SUM(cost_micros)::bigint INTO v_rate FROM public.response_arms
     WHERE id = ANY(v_arm_ids);
    v_difference := v_cross.billed_usage_micros - v_rate;
    v_tolerance := GREATEST(10000::bigint, (v_rate + 49) / 50);
    v_expected_status := CASE
        WHEN abs(v_difference) <= v_tolerance THEN 'accepted'
        ELSE 'discrepant'
    END;
    v_expected_arm_hash := pg_catalog.encode(
        pg_catalog.sha256(
            pg_catalog.convert_to(
                '{{"arm_ids":' || pg_catalog.array_to_json(v_arm_ids)::text || '}}',
                'UTF8'
            )
        ),
        'hex'
    );
    IF v_cross.supersedes_crosscheck_id IS NOT NULL THEN
        SELECT * INTO v_predecessor
          FROM public.bedrock_billing_crosschecks AS c
         WHERE c.id = v_cross.supersedes_crosscheck_id
           AND c.season_id = v_season.id
           AND c.provider_account_budget_id = v_ab.id
           AND c.created_at < v_cross.created_at
         FOR UPDATE;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'billing correction predecessor is invalid or not earlier';
        END IF;
        SELECT pg_catalog.array_agg(m.arm_id ORDER BY m.arm_id)
          INTO v_prior_arm_ids
          FROM public.bedrock_billing_crosscheck_arms AS m
         WHERE m.crosscheck_id = v_predecessor.id;
        IF v_prior_arm_ids IS NULL OR v_prior_arm_ids <> v_arm_ids THEN
            RAISE EXCEPTION 'billing correction predecessor does not match exact arm set';
        END IF;
        SELECT * INTO v_predecessor_event
          FROM public.cost_events AS ce
         WHERE ce.kind = 'bedrock_billing_adjustment'
           AND ce.accounting_json::jsonb ->> 'crosscheck_id' = v_predecessor.id
         FOR UPDATE;
        IF NOT FOUND
           OR v_predecessor_event.season_id <> v_predecessor.season_id
           OR v_predecessor_event.amount_micros <> v_predecessor.ledger_delta_micros
           OR (v_predecessor_event.accounting_json::jsonb ->>
               'governed_budget_delta_micros')::bigint <>
              GREATEST(0::bigint, v_predecessor.ledger_delta_micros) THEN
            RAISE EXCEPTION 'billing correction predecessor was not authoritatively applied';
        END IF;
        v_predecessor_difference := v_predecessor.billing_difference_micros;
    END IF;
    v_delta := v_difference - v_predecessor_difference;
    v_governed := GREATEST(0::bigint, v_delta);
    IF v_rate IS NULL OR v_rate < 0 OR v_cross.billed_usage_micros < 0
       OR v_cross.coverage_end <= v_cross.coverage_start
       OR v_cross.rate_card_estimated_micros <> v_rate
       OR v_cross.billing_difference_micros <> v_difference
       OR v_cross.ledger_delta_micros <> v_delta
       OR v_cross.tolerance_micros <> v_tolerance
       OR v_cross.status <> v_expected_status
       OR v_cross.arm_set_sha256 <> v_expected_arm_hash
       OR v_cross.credits_policy <> 'gross_usage_before_credits_excluding_tax'
       OR NOT (v_cross.evidence_json::jsonb ?& ARRAY[
          'schema_version', 'season_slug', 'account_scope_sha256', 'source_kind',
          'source_artifact_uri', 'source_artifact_sha256', 'statement_sha256',
          'coverage_start', 'coverage_end', 'arm_set_sha256',
          'generation_request_map_sha256', 'rate_card_estimated_micros',
          'billed_usage_micros', 'billing_difference_micros', 'crosscheck_status',
          'supersedes_crosscheck_id', 'tolerance_micros', 'credits_policy',
          'authorization_reference_sha256', 'ledger_delta_micros',
          'governed_budget_delta_micros'
       ]::text[])
       OR (v_cross.evidence_json::jsonb - ARRAY[
          'schema_version', 'season_slug', 'account_scope_sha256', 'source_kind',
          'source_artifact_uri', 'source_artifact_sha256', 'statement_sha256',
          'coverage_start', 'coverage_end', 'arm_set_sha256',
          'generation_request_map_sha256', 'rate_card_estimated_micros',
          'billed_usage_micros', 'billing_difference_micros', 'crosscheck_status',
          'supersedes_crosscheck_id', 'tolerance_micros', 'credits_policy',
          'authorization_reference_sha256', 'ledger_delta_micros',
          'governed_budget_delta_micros'
       ]::text[]) <> '{{}}'::jsonb
       OR (v_cross.evidence_json::jsonb ->> 'rate_card_estimated_micros')::bigint
          IS DISTINCT FROM v_rate
       OR (v_cross.evidence_json::jsonb ->> 'billing_difference_micros')::bigint
          IS DISTINCT FROM v_difference
       OR (v_cross.evidence_json::jsonb ->> 'ledger_delta_micros')::bigint
          IS DISTINCT FROM v_delta
       OR (v_cross.evidence_json::jsonb ->> 'governed_budget_delta_micros')::bigint
          IS DISTINCT FROM v_governed
       OR v_cross.evidence_json::jsonb ->> 'crosscheck_status'
          IS DISTINCT FROM v_expected_status
       OR (v_cross.evidence_json::jsonb ->> 'tolerance_micros')::bigint
          IS DISTINCT FROM v_tolerance
       OR v_cross.evidence_json::jsonb ->> 'arm_set_sha256'
          IS DISTINCT FROM v_expected_arm_hash
       OR v_cross.evidence_json::jsonb ->> 'schema_version'
          IS DISTINCT FROM 'flavourbench-bedrock-billing-crosscheck-v1'
       OR v_cross.evidence_json::jsonb ->> 'season_slug'
          IS DISTINCT FROM v_season.slug
       OR v_cross.evidence_json::jsonb ->> 'account_scope_sha256'
          IS DISTINCT FROM v_ab.account_scope_sha256
       OR v_cross.evidence_json::jsonb ->> 'source_kind'
          IS DISTINCT FROM v_cross.source_kind
       OR v_cross.evidence_json::jsonb ->> 'source_artifact_uri'
          IS DISTINCT FROM v_cross.source_artifact_uri
       OR v_cross.evidence_json::jsonb ->> 'source_artifact_sha256'
          IS DISTINCT FROM v_cross.source_artifact_sha256
       OR v_cross.evidence_json::jsonb ->> 'statement_sha256'
          IS DISTINCT FROM v_cross.statement_sha256
       OR (v_cross.evidence_json::jsonb ->> 'coverage_start')::timestamptz
          IS DISTINCT FROM v_cross.coverage_start
       OR (v_cross.evidence_json::jsonb ->> 'coverage_end')::timestamptz
          IS DISTINCT FROM v_cross.coverage_end
       OR v_cross.evidence_json::jsonb ->> 'generation_request_map_sha256'
          IS DISTINCT FROM v_cross.generation_request_map_sha256
       OR (v_cross.evidence_json::jsonb ->> 'billed_usage_micros')::bigint
          IS DISTINCT FROM v_cross.billed_usage_micros
       OR v_cross.evidence_json::jsonb ->> 'credits_policy'
          IS DISTINCT FROM v_cross.credits_policy
       OR v_cross.evidence_json::jsonb ->> 'authorization_reference_sha256'
          IS DISTINCT FROM v_cross.authorization_reference_sha256
       OR v_cross.evidence_json::jsonb ->> 'supersedes_crosscheck_id'
          IS DISTINCT FROM v_cross.supersedes_crosscheck_id THEN
        RAISE EXCEPTION 'billing crosscheck semantics are inconsistent';
    END IF;
    SELECT * INTO v_event FROM public.cost_events
     WHERE kind = 'bedrock_billing_adjustment'
       AND accounting_json::jsonb ->> 'crosscheck_id' = p_crosscheck_id
     FOR UPDATE;
    IF FOUND THEN
        IF v_event.amount_micros <> v_delta
           OR (v_event.accounting_json::jsonb ->> 'governed_budget_delta_micros')::bigint <>
              v_governed THEN
            RAISE EXCEPTION 'billing adjustment idempotency evidence is inconsistent';
        END IF;
        ledger_delta_micros := v_delta;
        governed_delta_micros := v_governed;
        cost_halted := v_season.status = 'cost_halted';
        idempotent := true;
        RETURN NEXT;
        RETURN;
    END IF;
    UPDATE public.seasons SET budget_used_micros = budget_used_micros + v_governed
     WHERE id = v_season.id;
    UPDATE public.season_provider_budgets
       SET budget_used_micros = budget_used_micros + v_governed WHERE id = v_pb.id;
    UPDATE public.provider_account_budgets
       SET budget_used_micros = budget_used_micros + v_governed WHERE id = v_ab.id;
    IF v_cross.status = 'discrepant'
       OR EXISTS (SELECT 1 FROM public.seasons WHERE id = v_season.id
                   AND budget_used_micros + budget_reserved_micros > budget_cap_micros)
       OR EXISTS (SELECT 1 FROM public.season_provider_budgets WHERE id = v_pb.id
                   AND budget_used_micros + budget_reserved_micros > budget_cap_micros)
       OR EXISTS (SELECT 1 FROM public.provider_account_budgets WHERE id = v_ab.id
                   AND budget_used_micros + budget_reserved_micros > budget_cap_micros) THEN
        v_halt := true;
        UPDATE public.seasons SET status = 'cost_halted' WHERE id = v_season.id;
    END IF;
    INSERT INTO public.cost_events(
        id, season_id, kind, amount_micros, provider, accounting_json, created_at
    ) VALUES (
        pg_catalog.gen_random_uuid()::text, v_season.id,
        'bedrock_billing_adjustment', v_delta, 'bedrock',
        pg_catalog.jsonb_build_object(
            'crosscheck_id', v_cross.id,
            'evidence_sha256', v_cross.evidence_sha256,
            'arm_set_sha256', v_cross.arm_set_sha256,
            'account_scope_sha256', v_ab.account_scope_sha256,
            'governed_budget_delta_micros', v_governed
        ), CURRENT_TIMESTAMP
    );
    ledger_delta_micros := v_delta;
    governed_delta_micros := v_governed;
    cost_halted := v_halt;
    idempotent := false;
    RETURN NEXT;
END;
$$;
"""


POSTGRESQL_REGISTER_BILLING_FUNCTION = f"""
CREATE OR REPLACE FUNCTION public.{CANONICAL_JSON_FUNCTION}(p_value jsonb)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
STRICT
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_result text;
BEGIN
    CASE pg_catalog.jsonb_typeof(p_value)
        WHEN 'object' THEN
            SELECT '{{' || COALESCE(
                pg_catalog.string_agg(
                    pg_catalog.to_jsonb(item.key)::text || ':' ||
                    public.{CANONICAL_JSON_FUNCTION}(item.value),
                    ',' ORDER BY item.key COLLATE "C"
                ),
                ''
            ) || '}}'
              INTO v_result
              FROM pg_catalog.jsonb_each(p_value) AS item(key, value);
        WHEN 'array' THEN
            SELECT '[' || COALESCE(
                pg_catalog.string_agg(
                    public.{CANONICAL_JSON_FUNCTION}(item.value),
                    ',' ORDER BY item.ordinality
                ),
                ''
            ) || ']'
              INTO v_result
              FROM pg_catalog.jsonb_array_elements(p_value)
                   WITH ORDINALITY AS item(value, ordinality);
        ELSE
            v_result := p_value::text;
    END CASE;
    RETURN v_result;
END;
$$;

CREATE OR REPLACE FUNCTION public.{REGISTER_BILLING_FUNCTION}(
    p_season_id text,
    p_request jsonb
)
RETURNS TABLE(
    crosscheck_id text,
    evidence_sha256 text,
    arm_set_sha256 text,
    generation_request_map_sha256 text,
    rate_card_estimated_micros bigint,
    billed_usage_micros bigint,
    billing_difference_micros bigint,
    ledger_delta_micros bigint,
    governed_delta_micros bigint,
    tolerance_micros bigint,
    crosscheck_status text,
    cost_halted boolean
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_required_keys text[] := ARRAY[
        'arm_ids', 'source_kind', 'source_artifact_uri',
        'source_artifact_sha256', 'statement_sha256',
        'generation_request_map_sha256', 'coverage_start', 'coverage_end',
        'billed_usage_micros', 'credits_policy',
        'authorization_reference_sha256', 'supersedes_crosscheck_id'
    ];
    v_season public.seasons%ROWTYPE;
    v_pb public.season_provider_budgets%ROWTYPE;
    v_ab public.provider_account_budgets%ROWTYPE;
    v_arm public.response_arms%ROWTYPE;
    v_battle public.battles%ROWTYPE;
    v_authorization public.provider_account_authorizations%ROWTYPE;
    v_predecessor public.bedrock_billing_crosschecks%ROWTYPE;
    v_predecessor_event public.cost_events%ROWTYPE;
    v_arm_ids text[];
    v_prior_arm_ids text[];
    v_generation_ids text[];
    v_generation_map jsonb := '[]'::jsonb;
    v_memberships jsonb := '[]'::jsonb;
    v_evidence jsonb;
    v_arm_id text;
    v_generation_set_hash text;
    v_map_hash text;
    v_arm_hash text;
    v_evidence_hash text;
    v_epoch_hash text;
    v_arm_epoch_hash text;
    v_supersedes text;
    v_source_kind text;
    v_source_artifact_uri text;
    v_source_artifact_sha text;
    v_statement_sha text;
    v_expected_map_sha text;
    v_credits_policy text;
    v_authorization_reference text;
    v_coverage_start timestamptz;
    v_coverage_end timestamptz;
    v_billed bigint;
    v_rate bigint := 0;
    v_difference bigint;
    v_prior_difference bigint := 0;
    v_delta bigint;
    v_governed bigint;
    v_tolerance bigint;
    v_status text;
    v_crosscheck_id text;
    v_receipt_count integer;
    v_attempt_count integer;
    v_bound_attempt_count integer;
    v_distinct_epoch_count integer;
    v_request_arm_count integer;
    v_distinct_arm_count integer;
    v_arm_attempt_min timestamptz;
    v_arm_attempt_max timestamptz;
    v_attempt_min timestamptz;
    v_attempt_max timestamptz;
    v_chain_reaches_root boolean;
    v_chain_consistent boolean;
    v_apply_delta bigint;
    v_apply_governed bigint;
    v_apply_halted boolean;
    v_apply_idempotent boolean;
    v_membership jsonb;
BEGIN
    IF p_request IS NULL
       OR pg_catalog.jsonb_typeof(p_request) <> 'object'
       OR NOT (p_request ?& v_required_keys)
       OR (p_request - v_required_keys) <> '{{}}'::jsonb
       OR pg_catalog.jsonb_typeof(p_request -> 'arm_ids') <> 'array' THEN
        RAISE EXCEPTION 'billing registration request has an invalid schema';
    END IF;
    SELECT pg_catalog.array_agg(x.arm_id ORDER BY x.arm_id),
           (SELECT COUNT(*) FROM pg_catalog.jsonb_array_elements_text(
               p_request -> 'arm_ids'
           )),
           COUNT(*)
      INTO v_arm_ids, v_request_arm_count, v_distinct_arm_count
      FROM (
          SELECT DISTINCT value AS arm_id
          FROM pg_catalog.jsonb_array_elements_text(p_request -> 'arm_ids')
      ) AS x;
    IF v_arm_ids IS NULL OR v_request_arm_count < 1
       OR v_request_arm_count > 20000
       OR v_request_arm_count <> v_distinct_arm_count THEN
        RAISE EXCEPTION 'billing registration arm set is empty or non-unique';
    END IF;
    BEGIN
        v_coverage_start := (p_request ->> 'coverage_start')::timestamptz;
        v_coverage_end := (p_request ->> 'coverage_end')::timestamptz;
        v_billed := (p_request ->> 'billed_usage_micros')::bigint;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'billing registration interval or usage is invalid';
    END;
    v_supersedes := p_request ->> 'supersedes_crosscheck_id';
    v_source_kind := p_request ->> 'source_kind';
    v_source_artifact_uri := p_request ->> 'source_artifact_uri';
    v_source_artifact_sha := p_request ->> 'source_artifact_sha256';
    v_statement_sha := p_request ->> 'statement_sha256';
    v_expected_map_sha := p_request ->> 'generation_request_map_sha256';
    v_credits_policy := p_request ->> 'credits_policy';
    v_authorization_reference := p_request ->> 'authorization_reference_sha256';
    IF v_source_kind NOT IN ('aws_cur', 'aws_data_export')
       OR v_source_artifact_uri !~ '^(s3|https)://'
       OR char_length(v_source_artifact_uri) < 12
       OR char_length(v_source_artifact_uri) > 1000
       OR octet_length(v_source_artifact_uri) <> char_length(v_source_artifact_uri)
       OR v_source_artifact_uri ~ '[[:space:]]'
       OR v_source_artifact_sha !~ '^[0-9a-f]{{64}}$'
       OR v_statement_sha !~ '^[0-9a-f]{{64}}$'
       OR v_expected_map_sha !~ '^[0-9a-f]{{64}}$'
       OR v_authorization_reference !~ '^[0-9a-f]{{64}}$'
       OR v_coverage_end <= v_coverage_start
       OR v_billed < 0
       OR v_credits_policy <> 'gross_usage_before_credits_excluding_tax' THEN
        RAISE EXCEPTION 'billing registration fields are invalid';
    END IF;

    SELECT * INTO v_pb FROM public.season_provider_budgets
     WHERE season_id = p_season_id AND execution_backend = 'bedrock';
    IF NOT FOUND THEN RAISE EXCEPTION 'season has no Bedrock budget'; END IF;
    SELECT * INTO v_ab FROM public.provider_account_budgets
     WHERE execution_backend = 'bedrock'
       AND account_scope_sha256 = v_pb.account_scope_sha256;
    IF NOT FOUND THEN RAISE EXCEPTION 'season has no Bedrock account budget'; END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_season_id || ':' || v_ab.id::text, 7212029)
    );
    SELECT * INTO v_season FROM public.seasons
     WHERE id = p_season_id FOR UPDATE;
    IF NOT FOUND OR v_season.status NOT IN ('pilot', 'active', 'cost_halted') THEN
        RAISE EXCEPTION 'season cannot accept Bedrock billing evidence';
    END IF;
    SELECT * INTO v_pb FROM public.season_provider_budgets
     WHERE season_id = p_season_id AND execution_backend = 'bedrock' FOR UPDATE;
    SELECT * INTO v_ab FROM public.provider_account_budgets
     WHERE execution_backend = 'bedrock'
       AND account_scope_sha256 = v_pb.account_scope_sha256 FOR UPDATE;
    IF v_pb.id IS NULL OR v_ab.id IS NULL
       OR v_ab.execution_backend <> v_pb.execution_backend
       OR v_ab.account_scope_sha256 <> v_pb.account_scope_sha256 THEN
        RAISE EXCEPTION 'Bedrock provider and account budgets do not agree';
    END IF;
    PERFORM b.id FROM public.battles AS b
     JOIN public.response_arms AS a ON a.battle_id = b.id
     WHERE a.id = ANY(v_arm_ids) ORDER BY b.id FOR UPDATE OF b;
    PERFORM a.id FROM public.response_arms AS a
     WHERE a.id = ANY(v_arm_ids) ORDER BY a.id FOR UPDATE;

    FOREACH v_arm_id IN ARRAY v_arm_ids
    LOOP
        SELECT * INTO v_arm FROM public.response_arms WHERE id = v_arm_id;
        IF NOT FOUND THEN RAISE EXCEPTION 'billing arm does not exist'; END IF;
        SELECT * INTO v_battle FROM public.battles WHERE id = v_arm.battle_id;
        IF NOT FOUND OR v_battle.season_id <> p_season_id
           OR v_arm.execution_backend <> 'bedrock'
           OR v_arm.cost_reconciled IS NOT TRUE
           OR v_arm.billing_reconciliation_status <>
              'pending_aws_billing_crosscheck'
           OR v_arm.cost_micros < 0
           OR v_arm.completed_at IS NULL
           OR v_arm.completed_at < v_coverage_start
           OR v_arm.completed_at > v_coverage_end THEN
            RAISE EXCEPTION 'billing arm is outside the exact governed population';
        END IF;
        SELECT COUNT(*) INTO v_receipt_count
          FROM public.cost_events AS ce
         WHERE ce.arm_id = v_arm.id
           AND ce.battle_id = v_battle.id
           AND ce.season_id = p_season_id
           AND ce.kind = 'actual'
           AND ce.amount_micros = v_arm.cost_micros
           AND ce.amount_micros >= 0
           AND ce.accounting_json::jsonb ->> 'billing_reconciliation_status' =
               'pending_aws_billing_crosscheck';
        IF v_receipt_count <> 1 THEN
            RAISE EXCEPTION 'billing arm lacks one exact rate-card receipt';
        END IF;
        SELECT pg_catalog.array_agg(DISTINCT value ORDER BY value)
          INTO v_generation_ids
          FROM pg_catalog.jsonb_array_elements_text(
              COALESCE(v_arm.provider_generation_ids_json::jsonb, '[]'::jsonb)
          );
        IF v_generation_ids IS NULL THEN
            RAISE EXCEPTION 'billing arm lacks a provider generation identity';
        END IF;
        SELECT COUNT(*),
               COUNT(*) FILTER (
                   WHERE COALESCE(
                       ga.metadata_json::jsonb ->>
                       'verified_provider_account_authorization_envelope_sha256',
                       ''
                   ) <> ''
               ),
               COUNT(DISTINCT ga.metadata_json::jsonb ->>
                   'verified_provider_account_authorization_envelope_sha256'),
               MIN(ga.metadata_json::jsonb ->>
                   'verified_provider_account_authorization_envelope_sha256'),
               MIN(ga.created_at), MAX(ga.created_at)
          INTO v_attempt_count, v_bound_attempt_count, v_distinct_epoch_count,
               v_arm_epoch_hash, v_arm_attempt_min, v_arm_attempt_max
          FROM public.generation_attempts AS ga
         WHERE ga.arm_id = v_arm.id AND ga.event_type = 'request_started';
        IF v_attempt_count < 1 OR v_bound_attempt_count <> v_attempt_count
           OR v_distinct_epoch_count <> 1
           OR v_arm_epoch_hash !~ '^[0-9a-f]{{64}}$' THEN
            RAISE EXCEPTION 'billing arm lacks one exact credential epoch';
        END IF;
        IF v_epoch_hash IS NULL THEN
            v_epoch_hash := v_arm_epoch_hash;
        ELSIF v_epoch_hash <> v_arm_epoch_hash THEN
            RAISE EXCEPTION 'one crosscheck cannot span credential epochs';
        END IF;
        v_attempt_min := LEAST(v_attempt_min, v_arm_attempt_min);
        v_attempt_max := GREATEST(v_attempt_max, v_arm_attempt_max);
        v_generation_set_hash := pg_catalog.encode(
            pg_catalog.sha256(pg_catalog.convert_to(
                public.{CANONICAL_JSON_FUNCTION}(
                    pg_catalog.jsonb_build_object(
                        'generation_ids', pg_catalog.to_jsonb(v_generation_ids)
                    )
                ),
                'UTF8'
            )),
            'hex'
        );
        v_generation_map := v_generation_map || pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'arm_id', v_arm.id,
                'generation_ids', pg_catalog.to_jsonb(v_generation_ids),
                'account_authorization_envelope_sha256', v_arm_epoch_hash,
                'generation_set_sha256', v_generation_set_hash
            )
        );
        v_memberships := v_memberships || pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'arm_id', v_arm.id,
                'generation_set_sha256', v_generation_set_hash
            )
        );
        v_rate := v_rate + v_arm.cost_micros;
    END LOOP;

    v_map_hash := pg_catalog.encode(
        pg_catalog.sha256(pg_catalog.convert_to(
            public.{CANONICAL_JSON_FUNCTION}(
                pg_catalog.jsonb_build_object('arms', v_generation_map)
            ),
            'UTF8'
        )),
        'hex'
    );
    v_arm_hash := pg_catalog.encode(
        pg_catalog.sha256(pg_catalog.convert_to(
            public.{CANONICAL_JSON_FUNCTION}(
                pg_catalog.jsonb_build_object(
                    'arm_ids', pg_catalog.to_jsonb(v_arm_ids)
                )
            ),
            'UTF8'
        )),
        'hex'
    );
    IF v_map_hash <> v_expected_map_sha THEN
        RAISE EXCEPTION 'billing generation map does not match relational evidence';
    END IF;

    SELECT * INTO v_authorization FROM public.provider_account_authorizations AS pa
     WHERE pa.provider_account_budget_id = v_ab.id
       AND pa.execution_backend = 'bedrock'
       AND pa.account_scope_sha256 = v_ab.account_scope_sha256
       AND pa.authorization_envelope_sha256 = v_epoch_hash
       AND pa.authorization_reference_sha256 = v_authorization_reference;
    IF NOT FOUND
       OR v_attempt_min < v_authorization.created_at
       OR v_attempt_max >= v_authorization.valid_until
       OR (v_authorization.revoked_at IS NOT NULL
           AND v_attempt_max >= v_authorization.revoked_at) THEN
        RAISE EXCEPTION 'billing credential epoch does not cover its requests';
    END IF;
    WITH RECURSIVE chain AS (
        SELECT pa.*, ARRAY[pa.id]::text[] AS path
          FROM public.provider_account_authorizations AS pa
         WHERE pa.id = v_authorization.id
        UNION ALL
        SELECT prior.*, chain.path || prior.id::text
          FROM chain
          JOIN public.provider_account_authorizations AS prior
            ON prior.id = chain.supersedes_authorization_id
         WHERE NOT prior.id::text = ANY(chain.path)
    )
    SELECT COALESCE(pg_catalog.bool_or(
               chain.authorization_envelope_sha256 =
               v_pb.account_authorization_envelope_sha256
           ), false),
           COALESCE(pg_catalog.bool_and(
               chain.provider_account_budget_id = v_ab.id
               AND chain.execution_backend = 'bedrock'
               AND chain.account_scope_sha256 = v_ab.account_scope_sha256
           ), false)
      INTO v_chain_reaches_root, v_chain_consistent
      FROM chain;
    IF NOT v_chain_reaches_root OR NOT v_chain_consistent THEN
        RAISE EXCEPTION 'billing credential chain does not reach the season authority';
    END IF;

    IF v_supersedes IS NULL THEN
        IF EXISTS (
            SELECT 1
            FROM public.bedrock_billing_crosscheck_arms AS m
            JOIN public.bedrock_billing_crosschecks AS c ON c.id = m.crosscheck_id
            WHERE m.arm_id = ANY(v_arm_ids)
              AND NOT EXISTS (
                  SELECT 1 FROM public.bedrock_billing_crosschecks AS successor
                  WHERE successor.supersedes_crosscheck_id = c.id
              )
        ) THEN
            RAISE EXCEPTION USING
                ERRCODE = 'FB001',
                MESSAGE = 'one or more billing arms already have active evidence';
        END IF;
    ELSE
        SELECT * INTO v_predecessor FROM public.bedrock_billing_crosschecks AS c
         WHERE c.id = v_supersedes
           AND c.season_id = p_season_id
           AND c.provider_account_budget_id = v_ab.id
         FOR UPDATE;
        IF NOT FOUND OR EXISTS (
            SELECT 1 FROM public.bedrock_billing_crosschecks AS successor
             WHERE successor.supersedes_crosscheck_id = v_predecessor.id
        ) THEN
            RAISE EXCEPTION USING
                ERRCODE = 'FB001',
                MESSAGE = 'billing correction predecessor is not the active record';
        END IF;
        SELECT pg_catalog.array_agg(m.arm_id::text ORDER BY m.arm_id::text)
          INTO v_prior_arm_ids
          FROM public.bedrock_billing_crosscheck_arms AS m
         WHERE m.crosscheck_id = v_predecessor.id;
        IF v_prior_arm_ids IS DISTINCT FROM v_arm_ids THEN
            RAISE EXCEPTION 'billing correction predecessor has another arm set';
        END IF;
        SELECT * INTO v_predecessor_event FROM public.cost_events AS ce
         WHERE ce.kind = 'bedrock_billing_adjustment'
           AND ce.accounting_json::jsonb ->> 'crosscheck_id' = v_predecessor.id;
        IF NOT FOUND OR v_predecessor_event.season_id <> p_season_id
           OR v_predecessor_event.amount_micros <>
              v_predecessor.ledger_delta_micros THEN
            RAISE EXCEPTION 'billing correction predecessor was not applied';
        END IF;
        v_prior_difference := v_predecessor.billing_difference_micros;
    END IF;

    v_difference := v_billed - v_rate;
    v_delta := v_difference - v_prior_difference;
    v_governed := GREATEST(0::bigint, v_delta);
    v_tolerance := GREATEST(10000::bigint, (v_rate + 49) / 50);
    v_status := CASE
        WHEN abs(v_difference) <= v_tolerance THEN 'accepted'
        ELSE 'discrepant'
    END;
    v_evidence := pg_catalog.jsonb_build_object(
        'schema_version', 'flavourbench-bedrock-billing-crosscheck-v1',
        'season_slug', v_season.slug,
        'account_scope_sha256', v_ab.account_scope_sha256,
        'source_kind', v_source_kind,
        'source_artifact_uri', v_source_artifact_uri,
        'source_artifact_sha256', v_source_artifact_sha,
        'statement_sha256', v_statement_sha,
        'coverage_start', p_request ->> 'coverage_start',
        'coverage_end', p_request ->> 'coverage_end',
        'arm_set_sha256', v_arm_hash,
        'generation_request_map_sha256', v_map_hash,
        'rate_card_estimated_micros', v_rate,
        'billed_usage_micros', v_billed,
        'billing_difference_micros', v_difference,
        'crosscheck_status', v_status,
        'supersedes_crosscheck_id', v_supersedes,
        'tolerance_micros', v_tolerance,
        'credits_policy', v_credits_policy,
        'authorization_reference_sha256', v_authorization_reference,
        'ledger_delta_micros', v_delta,
        'governed_budget_delta_micros', v_governed
    );
    v_evidence_hash := pg_catalog.encode(
        pg_catalog.sha256(pg_catalog.convert_to(
            public.{CANONICAL_JSON_FUNCTION}(v_evidence), 'UTF8'
        )),
        'hex'
    );
    v_crosscheck_id := pg_catalog.gen_random_uuid()::text;
    INSERT INTO public.bedrock_billing_crosschecks(
        id, season_id, provider_account_budget_id, status,
        supersedes_crosscheck_id, source_kind, source_artifact_uri,
        source_artifact_sha256, statement_sha256, coverage_start, coverage_end,
        arm_set_sha256, generation_request_map_sha256,
        rate_card_estimated_micros, billed_usage_micros,
        billing_difference_micros, ledger_delta_micros, tolerance_micros,
        credits_policy, authorization_reference_sha256,
        evidence_json, evidence_sha256, created_at
    ) VALUES (
        v_crosscheck_id, p_season_id, v_ab.id, v_status, v_supersedes,
        v_source_kind, v_source_artifact_uri, v_source_artifact_sha,
        v_statement_sha, v_coverage_start, v_coverage_end, v_arm_hash,
        v_map_hash, v_rate, v_billed, v_difference, v_delta, v_tolerance,
        v_credits_policy, v_authorization_reference,
        v_evidence, v_evidence_hash, CURRENT_TIMESTAMP
    );
    FOR v_membership IN
        SELECT value FROM pg_catalog.jsonb_array_elements(v_memberships)
    LOOP
        INSERT INTO public.bedrock_billing_crosscheck_arms(
            id, crosscheck_id, arm_id, generation_set_sha256, created_at
        ) VALUES (
            pg_catalog.gen_random_uuid()::text,
            v_crosscheck_id,
            v_membership ->> 'arm_id',
            v_membership ->> 'generation_set_sha256',
            CURRENT_TIMESTAMP
        );
    END LOOP;
    SELECT applied.ledger_delta_micros,
           applied.governed_delta_micros,
           applied.cost_halted,
           applied.idempotent
      INTO v_apply_delta, v_apply_governed, v_apply_halted, v_apply_idempotent
      FROM public.{BILLING_FUNCTION}(v_crosscheck_id) AS applied;
    IF v_apply_idempotent OR v_apply_delta <> v_delta
       OR v_apply_governed <> v_governed THEN
        RAISE EXCEPTION 'billing registration was not applied exactly once';
    END IF;
    crosscheck_id := v_crosscheck_id;
    evidence_sha256 := v_evidence_hash;
    arm_set_sha256 := v_arm_hash;
    generation_request_map_sha256 := v_map_hash;
    rate_card_estimated_micros := v_rate;
    billed_usage_micros := v_billed;
    billing_difference_micros := v_difference;
    ledger_delta_micros := v_delta;
    governed_delta_micros := v_governed;
    tolerance_micros := v_tolerance;
    crosscheck_status := v_status;
    cost_halted := v_apply_halted;
    RETURN NEXT;
END;
$$;
"""


POSTGRESQL_GRANTS = f"""
REVOKE ALL ON FUNCTION public.{RESERVE_FUNCTION}(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.{SETTLE_FUNCTION}(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.{BILLING_FUNCTION}(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.{REGISTER_BILLING_FUNCTION}(text, jsonb) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.{CANONICAL_JSON_FUNCTION}(jsonb) FROM PUBLIC;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'flavourbench_api') THEN
        GRANT EXECUTE ON FUNCTION public.{RESERVE_FUNCTION}(text) TO flavourbench_api;
        GRANT EXECUTE ON FUNCTION public.{SETTLE_FUNCTION}(text) TO flavourbench_api;
        GRANT EXECUTE ON FUNCTION public.{REGISTER_BILLING_FUNCTION}(text, jsonb)
            TO flavourbench_api;
        REVOKE INSERT ON public.bedrock_billing_crosschecks,
            public.bedrock_billing_crosscheck_arms FROM flavourbench_api;
        REVOKE UPDATE ON public.seasons, public.season_provider_budgets,
            public.provider_account_budgets, public.controlled_runs, public.battles
            FROM flavourbench_api;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'flavourbench_worker') THEN
        GRANT EXECUTE ON FUNCTION public.{SETTLE_FUNCTION}(text) TO flavourbench_worker;
        REVOKE UPDATE ON public.seasons, public.season_provider_budgets,
            public.provider_account_budgets, public.controlled_runs, public.battles
            FROM flavourbench_worker;
    END IF;
END;
$$;
DO $$
DECLARE
    v_role text;
    v_table text;
    v_columns text;
    v_excluded text[];
BEGIN
    FOR v_role, v_table, v_excluded IN
        SELECT * FROM (VALUES
            ('flavourbench_api', 'seasons',
             ARRAY['budget_used_micros', 'budget_reserved_micros']),
            ('flavourbench_api', 'provider_account_budgets',
             ARRAY['budget_used_micros', 'budget_reserved_micros']),
            ('flavourbench_api', 'controlled_runs',
             ARRAY['budget_used_micros', 'budget_reserved_micros']),
            ('flavourbench_api', 'battles',
             ARRAY['reserved_cost_micros', 'provider_reservations_json']),
            ('flavourbench_worker', 'battles',
             ARRAY['reserved_cost_micros', 'provider_reservations_json'])
        ) AS grants(role_name, table_name, excluded_columns)
    LOOP
        IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = v_role) THEN
            SELECT pg_catalog.string_agg(pg_catalog.quote_ident(a.attname), ', ')
              INTO v_columns
              FROM pg_catalog.pg_attribute AS a
              JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
              JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
             WHERE n.nspname = 'public' AND c.relname = v_table
               AND a.attnum > 0 AND NOT a.attisdropped
               AND NOT (a.attname = ANY(v_excluded));
            EXECUTE pg_catalog.format(
                'GRANT UPDATE (%s) ON TABLE public.%I TO %I',
                v_columns, v_table, v_role
            );
        END IF;
    END LOOP;
END;
$$;
"""


def _create_postgresql_authority() -> None:
    op.execute(POSTGRESQL_LEGACY_TRIGGER_REPAIRS)
    op.execute(POSTGRESQL_GUARDS)
    op.execute(POSTGRESQL_RESERVE_FUNCTION)
    op.execute(POSTGRESQL_SETTLE_FUNCTION)
    op.execute(POSTGRESQL_BILLING_FUNCTION)
    op.execute(POSTGRESQL_REGISTER_BILLING_FUNCTION)
    op.execute(POSTGRESQL_GRANTS)


def upgrade() -> None:
    bind = op.get_bind()
    _preflight(bind)
    _create_partial_indexes(bind)
    if bind.dialect.name == "postgresql":
        _create_postgresql_checks()
        _create_postgresql_authority()


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            f"DROP FUNCTION IF EXISTS public.{REGISTER_BILLING_FUNCTION}(text, jsonb)"
        )
        op.execute(
            f"DROP FUNCTION IF EXISTS public.{CANONICAL_JSON_FUNCTION}(jsonb)"
        )
        op.execute(f"DROP FUNCTION IF EXISTS public.{BILLING_FUNCTION}(text)")
        op.execute(f"DROP FUNCTION IF EXISTS public.{SETTLE_FUNCTION}(text)")
        op.execute(f"DROP FUNCTION IF EXISTS public.{RESERVE_FUNCTION}(text)")
        op.execute("DROP TRIGGER IF EXISTS trg_cost_event_authority ON public.cost_events")
        op.execute(
            "DROP TRIGGER IF EXISTS trg_bedrock_membership_seal "
            "ON public.bedrock_billing_crosscheck_arms"
        )
        op.execute(
            "DROP TRIGGER IF EXISTS trg_generation_attempt_arm_authority "
            "ON public.generation_attempts"
        )
        op.execute("DROP TRIGGER IF EXISTS trg_battle_reservation_authority ON public.battles")
        for table in (
            "seasons",
            "season_provider_budgets",
            "provider_account_budgets",
            "controlled_runs",
        ):
            op.execute(
                f"DROP TRIGGER IF EXISTS trg_budget_counter_authority_{table} "
                f"ON public.{table}"
            )
        op.execute(f"DROP FUNCTION IF EXISTS public.{COST_EVENT_GUARD_FUNCTION}()")
        op.execute(
            f"DROP FUNCTION IF EXISTS public.{BEDROCK_MEMBERSHIP_GUARD_FUNCTION}()"
        )
        op.execute(f"DROP FUNCTION IF EXISTS public.{ATTEMPT_ARM_GUARD_FUNCTION}()")
        op.execute(f"DROP FUNCTION IF EXISTS public.{BATTLE_GUARD_FUNCTION}()")
        op.execute(f"DROP FUNCTION IF EXISTS public.{COUNTER_GUARD_FUNCTION}()")
        op.execute("DROP INDEX IF EXISTS uq_cost_events_bedrock_crosscheck_adjustment")
        op.drop_constraint(
            "ck_bedrock_billing_crosschecks_not_self_superseding",
            "bedrock_billing_crosschecks",
            type_="check",
        )
        op.drop_constraint(
            "ck_cost_events_nonpositive_release",
            "cost_events",
            type_="check",
        )
        op.drop_constraint(
            "ck_cost_events_nonnegative_charge",
            "cost_events",
            type_="check",
        )
        op.drop_constraint(
            "ck_response_arms_nonnegative_cost",
            "response_arms",
            type_="check",
        )
    for name in reversed(COMMON_PARTIAL_INDEXES):
        op.execute(f"DROP INDEX IF EXISTS {name}")
