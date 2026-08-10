"""Separate public, official-research, and commercial retention authority.

Revision ID: 0026_retention_basis
Revises: 0025_task_evidence_database_guard
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import sqlalchemy as sa

from alembic import op

revision = "0026_retention_basis"
down_revision = "0025_task_evidence_database_guard"
branch_labels = None
depends_on = None

RETENTION_TRIGGER = "trg_battle_retention_authorization"
RETENTION_FUNCTION = "flavourbench_battle_retention_authorization"
SCOPE_FUNCTION = "flavourbench_battle_retention_basis_scope"
SCOPE_INSERT_TRIGGER = "trg_battle_retention_basis_insert"
SCOPE_UPDATE_TRIGGER = "trg_battle_retention_basis_update"
CHILD_RETENTION_FUNCTION = "flavourbench_child_retention_authorization"
CHILD_RETENTION_TRIGGERS = {
    "response_arms": "trg_response_arm_retention_authorization",
    "tool_calls": "trg_tool_call_retention_authorization",
    "validator_results": "trg_validator_retention_authorization",
    "run_events": "trg_run_event_retention_authorization",
    "incidents": "trg_incident_retention_authorization",
    "jobs": "trg_job_retention_authorization",
}
REDACTABLE_BASES_SQL = (
    "'public_nonconsented', 'commercial_private', "
    "'controlled_development', 'development_research'"
)

VALID_SCOPE_SQL = """
(
    (retention_basis = 'public_nonconsented'
     AND data_stratum = 'public_freeform' AND research_consent = false)
 OR (retention_basis = 'public_consented'
     AND data_stratum = 'public_freeform' AND research_consent = true)
 OR (retention_basis IN (
         'official_research', 'commercial_private', 'controlled_development'
     ) AND data_stratum = 'controlled' AND controlled_run_id IS NOT NULL
       AND research_consent = false)
 OR (retention_basis = 'development_research'
     AND data_stratum = 'development' AND research_consent = false)
 OR (retention_basis = 'legacy_operational' AND data_stratum = 'legacy')
)
"""


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _preflight_and_reconcile_commercial_terms(bind: sa.Connection) -> None:
    metadata = sa.MetaData()
    battles = sa.Table("battles", metadata, autoload_with=bind)
    runs = sa.Table("controlled_runs", metadata, autoload_with=bind)
    seasons = sa.Table("seasons", metadata, autoload_with=bind)
    organizations = sa.Table("organizations", metadata, autoload_with=bind)
    rows = bind.execute(
        sa.select(
            battles.c.id,
            battles.c.run_class,
            battles.c.data_stratum,
            battles.c.manifest_sha256,
            battles.c.protocol_bundle_sha256,
            battles.c.research_consent,
            battles.c.prompt_redacted,
            battles.c.created_at,
            battles.c.retention_until,
            runs.c.evaluation_order_id,
            seasons.c.official.label("season_official"),
            organizations.c.retention_policy_json,
        )
        .select_from(
            battles.join(seasons, seasons.c.id == battles.c.season_id)
            .outerjoin(runs, runs.c.id == battles.c.controlled_run_id)
            .outerjoin(organizations, organizations.c.id == runs.c.organization_id)
        )
    ).mappings()
    now = datetime.now(UTC)
    for row in rows:
        official_controlled = bool(
            row["data_stratum"] == "controlled"
            and row["evaluation_order_id"] is None
            and row["run_class"] == "official"
            and row["season_official"] is True
            and row["manifest_sha256"] not in {"", "unfrozen", "unresolved"}
            and row["protocol_bundle_sha256"]
            not in {"", "unfrozen", "unresolved"}
        )
        if row["prompt_redacted"] and (
            row["research_consent"] or official_controlled
        ):
            raise RuntimeError(
                "0026 preflight found content already redacted under a protected basis"
            )
        if row["evaluation_order_id"] is None:
            continue
        policy = row["retention_policy_json"]
        private_days = policy.get("privateEvidenceDays") if isinstance(policy, dict) else None
        if (
            isinstance(private_days, bool)
            or not isinstance(private_days, int)
            or not 1 <= private_days <= 3650
        ):
            raise RuntimeError(
                "0026 preflight cannot resolve a commercial privateEvidenceDays term"
            )
        expected_deadline = _aware(row["created_at"]) + timedelta(days=private_days)
        if row["prompt_redacted"] and expected_deadline > now:
            raise RuntimeError(
                "0026 preflight found commercial evidence redacted before its sealed term"
            )
        if not row["prompt_redacted"]:
            bind.execute(
                battles.update()
                .where(battles.c.id == row["id"])
                .values(retention_until=expected_deadline)
            )


def _backfill() -> None:
    op.execute(
        """
        UPDATE battles
           SET retention_basis = CASE
               WHEN data_stratum = 'controlled'
                    AND EXISTS (
                        SELECT 1 FROM controlled_runs AS run
                         WHERE run.id = battles.controlled_run_id
                           AND run.evaluation_order_id IS NOT NULL
                    ) THEN 'commercial_private'
               WHEN data_stratum = 'controlled'
                    AND run_class = 'official'
                    AND manifest_sha256 NOT IN ('', 'unfrozen', 'unresolved')
                    AND protocol_bundle_sha256 NOT IN ('', 'unfrozen', 'unresolved')
                    AND EXISTS (
                        SELECT 1 FROM seasons AS season
                         WHERE season.id = battles.season_id
                           AND season.official = true
                    ) THEN 'official_research'
               WHEN data_stratum = 'controlled' THEN 'controlled_development'
               WHEN data_stratum = 'development' THEN 'development_research'
               WHEN data_stratum = 'legacy' THEN 'legacy_operational'
               WHEN research_consent THEN 'public_consented'
               ELSE 'public_nonconsented'
           END
        """
    )


def _postgresql_scope_guards() -> None:
    op.create_check_constraint(
        "ck_battles_retention_basis_scope",
        "battles",
        VALID_SCOPE_SQL,
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION public.{SCOPE_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY INVOKER
        SET search_path = pg_catalog, public
        AS $$
        DECLARE
            commercial boolean := false;
            official_eligible boolean := false;
        BEGIN
            IF TG_OP = 'UPDATE'
               AND OLD.retention_basis IS DISTINCT FROM NEW.retention_basis THEN
                RAISE EXCEPTION 'battle retention basis is immutable';
            END IF;
            IF NEW.data_stratum = 'controlled' THEN
                SELECT run.evaluation_order_id IS NOT NULL
                  INTO commercial
                  FROM public.controlled_runs AS run
                 WHERE run.id = NEW.controlled_run_id;
                SELECT EXISTS(
                    SELECT 1 FROM public.seasons AS season
                    WHERE season.id = NEW.season_id
                      AND season.official IS TRUE
                      AND NEW.run_class = 'official'
                      AND NEW.manifest_sha256 NOT IN ('', 'unfrozen', 'unresolved')
                      AND NEW.protocol_bundle_sha256 NOT IN ('', 'unfrozen', 'unresolved')
                ) INTO official_eligible;
                IF (NEW.retention_basis = 'commercial_private') IS DISTINCT FROM commercial
                   OR (
                       NEW.retention_basis = 'official_research'
                       AND (commercial OR NOT official_eligible)
                   )
                   OR (
                       NEW.retention_basis = 'controlled_development'
                       AND (commercial OR official_eligible)
                   ) THEN
                    RAISE EXCEPTION
                        'controlled battle retention basis contradicts its sealed run';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$;

        DROP TRIGGER IF EXISTS {SCOPE_INSERT_TRIGGER} ON public.battles;
        CREATE TRIGGER {SCOPE_INSERT_TRIGGER}
        BEFORE INSERT ON public.battles
        FOR EACH ROW EXECUTE FUNCTION public.{SCOPE_FUNCTION}();
        DROP TRIGGER IF EXISTS {SCOPE_UPDATE_TRIGGER} ON public.battles;
        CREATE TRIGGER {SCOPE_UPDATE_TRIGGER}
        BEFORE UPDATE OF retention_basis, data_stratum, research_consent,
                         controlled_run_id, run_class, manifest_sha256,
                         protocol_bundle_sha256
        ON public.battles
        FOR EACH ROW EXECUTE FUNCTION public.{SCOPE_FUNCTION}();
        """
    )


def _postgresql_retention_guards() -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION public.{RETENTION_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY INVOKER
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
            IF OLD.prompt IS NOT NULL
               AND NEW.prompt IS NULL
               AND OLD.prompt_redacted IS FALSE
               AND NEW.prompt_redacted IS TRUE
               AND (
                    OLD.research_consent IS TRUE
                    OR OLD.retention_basis NOT IN ({REDACTABLE_BASES_SQL})
                    OR OLD.retention_until > CURRENT_TIMESTAMP
               ) THEN
                RAISE EXCEPTION
                    'battle retention redaction requires an expired redactable basis';
            END IF;
            RETURN NEW;
        END;
        $$;

        DROP TRIGGER IF EXISTS {RETENTION_TRIGGER} ON public.battles;
        CREATE TRIGGER {RETENTION_TRIGGER}
        BEFORE UPDATE OF prompt, prompt_redacted ON public.battles
        FOR EACH ROW EXECUTE FUNCTION public.{RETENTION_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION public.{CHILD_RETENTION_FUNCTION}()
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
                SELECT arm.battle_id INTO parent_battle_id
                  FROM public.response_arms AS arm
                 WHERE arm.id = row_json ->> 'arm_id';
            ELSIF TG_TABLE_NAME = 'run_events' THEN
                IF row_json ->> 'entity_type' = 'battle' THEN
                    parent_battle_id := row_json ->> 'entity_id';
                ELSIF row_json ->> 'entity_type' = 'response_arm' THEN
                    SELECT arm.battle_id INTO parent_battle_id
                      FROM public.response_arms AS arm
                     WHERE arm.id = row_json ->> 'entity_id';
                END IF;
            ELSIF TG_TABLE_NAME IN ('incidents', 'jobs') THEN
                parent_battle_id := row_json ->> 'battle_id';
            END IF;
            IF parent_battle_id IS NOT NULL THEN
                SELECT EXISTS(
                    SELECT 1 FROM public.battles AS battle
                    WHERE battle.id = parent_battle_id
                      AND battle.prompt IS NULL
                      AND battle.prompt_redacted IS TRUE
                      AND battle.research_consent IS FALSE
                      AND battle.retention_basis IN ({REDACTABLE_BASES_SQL})
                      AND battle.retention_until <= CURRENT_TIMESTAMP
                ) INTO authorized;
            END IF;
            IF NOT authorized THEN
                RAISE EXCEPTION 'retention redaction lacks an expired redactable basis';
            END IF;
            RETURN NEW;
        END;
        $$;
        """
    )
    trigger_specs = {
        "response_arms": (
            "answer_markdown, output_json, error_detail",
            "(OLD.answer_markdown IS DISTINCT FROM NEW.answer_markdown AND "
            "NEW.answer_markdown IS NULL) OR "
            "(OLD.output_json::jsonb IS DISTINCT FROM NEW.output_json::jsonb AND "
            "NEW.output_json::jsonb = jsonb_build_object('redacted', true)) OR "
            "(OLD.error_detail IS DISTINCT FROM NEW.error_detail AND "
            "NEW.error_detail IS NULL)",
        ),
        "tool_calls": (
            "arguments_json, result_text, structured_content_json",
            "NEW.arguments_json::jsonb = jsonb_build_object('redacted', true) AND "
            "NEW.result_text = '[REDACTED AFTER OPERATIONAL RETENTION]' AND "
            "NEW.structured_content_json::jsonb = jsonb_build_object('redacted', true)",
        ),
        "validator_results": (
            "detail_json",
            "NEW.detail_json::jsonb = jsonb_build_object('redacted', true)",
        ),
        "run_events": (
            "payload_json",
            "NEW.payload_json::jsonb = jsonb_build_object('redacted', true)",
        ),
        "incidents": (
            "detail",
            "NEW.detail = '[REDACTED AFTER OPERATIONAL RETENTION]'",
        ),
        "jobs": (
            "last_error",
            "OLD.last_error IS DISTINCT FROM NEW.last_error AND "
            "NEW.last_error = '[REDACTED AFTER OPERATIONAL RETENTION]'",
        ),
    }
    for table, (columns, condition) in trigger_specs.items():
        trigger = CHILD_RETENTION_TRIGGERS[table]
        op.execute(
            f"""
            DROP TRIGGER IF EXISTS {trigger} ON public.{table};
            CREATE TRIGGER {trigger}
            BEFORE UPDATE OF {columns} ON public.{table}
            FOR EACH ROW WHEN ({condition})
            EXECUTE FUNCTION public.{CHILD_RETENTION_FUNCTION}();
            """
        )


def _sqlite_guards() -> None:
    official_eligible = """
        EXISTS (
            SELECT 1 FROM seasons AS season
             WHERE season.id = NEW.season_id
               AND season.official = 1
               AND NEW.run_class = 'official'
               AND NEW.manifest_sha256 NOT IN ('', 'unfrozen', 'unresolved')
               AND NEW.protocol_bundle_sha256 NOT IN ('', 'unfrozen', 'unresolved')
        )
    """
    commercial = """
        EXISTS (
            SELECT 1 FROM controlled_runs AS run
             WHERE run.id = NEW.controlled_run_id
               AND run.evaluation_order_id IS NOT NULL
        )
    """
    noncommercial = """
        EXISTS (
            SELECT 1 FROM controlled_runs AS run
             WHERE run.id = NEW.controlled_run_id
               AND run.evaluation_order_id IS NULL
        )
    """
    scope = f"""
        (NEW.retention_basis = 'public_nonconsented'
         AND NEW.data_stratum = 'public_freeform' AND NEW.research_consent = 0)
     OR (NEW.retention_basis = 'public_consented'
         AND NEW.data_stratum = 'public_freeform' AND NEW.research_consent = 1)
     OR (NEW.retention_basis = 'commercial_private'
         AND NEW.data_stratum = 'controlled' AND NEW.research_consent = 0
         AND {commercial})
     OR (NEW.retention_basis = 'official_research'
         AND NEW.data_stratum = 'controlled' AND NEW.research_consent = 0
         AND {noncommercial} AND {official_eligible})
     OR (NEW.retention_basis = 'controlled_development'
         AND NEW.data_stratum = 'controlled' AND NEW.research_consent = 0
         AND {noncommercial} AND NOT {official_eligible})
     OR (NEW.retention_basis = 'development_research'
         AND NEW.data_stratum = 'development' AND NEW.research_consent = 0)
     OR (NEW.retention_basis = 'legacy_operational' AND NEW.data_stratum = 'legacy')
    """
    op.execute(f"DROP TRIGGER IF EXISTS {SCOPE_INSERT_TRIGGER}")
    op.execute(f"DROP TRIGGER IF EXISTS {SCOPE_UPDATE_TRIGGER}")
    op.execute(
        f"""
        CREATE TRIGGER {SCOPE_INSERT_TRIGGER}
        BEFORE INSERT ON battles
        FOR EACH ROW WHEN NOT ({scope})
        BEGIN
            SELECT RAISE(ABORT, 'battle retention basis contradicts collection scope');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {SCOPE_UPDATE_TRIGGER}
        BEFORE UPDATE OF retention_basis, data_stratum, research_consent,
                         controlled_run_id, run_class, manifest_sha256,
                         protocol_bundle_sha256
        ON battles
        FOR EACH ROW WHEN OLD.retention_basis IS NOT NEW.retention_basis OR NOT ({scope})
        BEGIN
            SELECT RAISE(ABORT, 'battle retention basis is immutable or invalid');
        END;
        """
    )
    op.execute(f"DROP TRIGGER IF EXISTS {RETENTION_TRIGGER}")
    op.execute(
        f"""
        CREATE TRIGGER {RETENTION_TRIGGER}
        BEFORE UPDATE OF prompt, prompt_redacted ON battles
        FOR EACH ROW
        WHEN OLD.prompt IS NOT NULL
          AND NEW.prompt IS NULL
          AND OLD.prompt_redacted = 0
          AND NEW.prompt_redacted = 1
          AND (
              OLD.research_consent = 1
              OR OLD.retention_basis NOT IN ({REDACTABLE_BASES_SQL})
              OR julianday(OLD.retention_until) > julianday('now')
          )
        BEGIN
            SELECT RAISE(
                ABORT,
                'battle retention redaction requires an expired redactable basis'
            );
        END;
        """
    )
    eligibility = (
        "b.prompt IS NULL AND b.prompt_redacted = 1 "
        "AND b.research_consent = 0 "
        f"AND b.retention_basis IN ({REDACTABLE_BASES_SQL}) "
        "AND julianday(b.retention_until) <= julianday('now')"
    )
    sqlite_specs = {
        "response_arms": (
            "answer_markdown, output_json, error_detail",
            "((OLD.answer_markdown IS NOT NEW.answer_markdown AND NEW.answer_markdown IS NULL) "
            "OR (OLD.output_json IS NOT NEW.output_json "
            "AND json(NEW.output_json) = json_object('redacted', json('true'))) "
            "OR (OLD.error_detail IS NOT NEW.error_detail AND NEW.error_detail IS NULL)) "
            "AND NOT EXISTS (SELECT 1 FROM battles AS b "
            f"WHERE b.id = NEW.battle_id AND {eligibility})",
        ),
        "tool_calls": (
            "arguments_json, result_text, structured_content_json",
            "json(NEW.arguments_json) = json_object('redacted', json('true')) "
            "AND NEW.result_text = '[REDACTED AFTER OPERATIONAL RETENTION]' "
            "AND json(NEW.structured_content_json) = json_object('redacted', json('true')) "
            "AND NOT EXISTS (SELECT 1 FROM response_arms AS arm JOIN battles AS b "
            f"ON b.id = arm.battle_id WHERE arm.id = NEW.arm_id AND {eligibility})",
        ),
        "validator_results": (
            "detail_json",
            "json(NEW.detail_json) = json_object('redacted', json('true')) "
            "AND NOT EXISTS (SELECT 1 FROM response_arms AS arm JOIN battles AS b "
            f"ON b.id = arm.battle_id WHERE arm.id = NEW.arm_id AND {eligibility})",
        ),
        "run_events": (
            "payload_json",
            "json(NEW.payload_json) = json_object('redacted', json('true')) AND NOT ("
            "(NEW.entity_type = 'battle' AND EXISTS (SELECT 1 FROM battles AS b "
            f"WHERE b.id = NEW.entity_id AND {eligibility})) OR "
            "(NEW.entity_type = 'response_arm' AND EXISTS (SELECT 1 FROM response_arms AS arm "
            "JOIN battles AS b ON b.id = arm.battle_id "
            f"WHERE arm.id = NEW.entity_id AND {eligibility})))",
        ),
        "incidents": (
            "detail",
            "NEW.detail = '[REDACTED AFTER OPERATIONAL RETENTION]' "
            "AND NOT EXISTS (SELECT 1 FROM battles AS b "
            f"WHERE b.id = NEW.battle_id AND {eligibility})",
        ),
        "jobs": (
            "last_error",
            "OLD.last_error IS NOT NEW.last_error "
            "AND NEW.last_error = '[REDACTED AFTER OPERATIONAL RETENTION]' "
            "AND NOT EXISTS (SELECT 1 FROM battles AS b "
            f"WHERE b.id = NEW.battle_id AND {eligibility})",
        ),
    }
    for table, (columns, condition) in sqlite_specs.items():
        trigger = CHILD_RETENTION_TRIGGERS[table]
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        op.execute(
            f"""
            CREATE TRIGGER {trigger}
            BEFORE UPDATE OF {columns} ON {table}
            FOR EACH ROW WHEN {condition}
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'retention redaction lacks an expired redactable basis'
                );
            END;
            """
        )


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("battles")}
    if "retention_basis" in columns:
        raise RuntimeError("retention_basis exists before revision 0026")
    _preflight_and_reconcile_commercial_terms(bind)
    op.add_column(
        "battles",
        sa.Column(
            "retention_basis",
            sa.String(length=32),
            nullable=False,
            server_default="legacy_operational",
        ),
    )
    _backfill()
    op.create_index("ix_battles_retention_basis", "battles", ["retention_basis"])
    if bind.dialect.name == "postgresql":
        op.alter_column("battles", "retention_basis", server_default=None)
        _postgresql_scope_guards()
        _postgresql_retention_guards()
    elif bind.dialect.name == "sqlite":
        _sqlite_guards()
    else:
        raise RuntimeError(f"unsupported database dialect for 0026: {bind.dialect.name}")


def downgrade() -> None:
    raise RuntimeError("downgrade across retention-basis policy is prohibited")
