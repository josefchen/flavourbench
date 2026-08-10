"""Authorize retention redaction and make governed spend monotonic.

Revision ID: 0015_release_integrity_controls
Revises: 0014_commercial_evidence_invariants
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0015_release_integrity_controls"
down_revision = "0014_commercial_evidence_invariants"
branch_labels = None
depends_on = None


RETENTION_TRIGGER = "trg_battle_retention_authorization"
RETENTION_FUNCTION = "flavourbench_battle_retention_authorization"
BUDGET_TRIGGER_PREFIX = "trg_governed_spend_monotonic"
BUDGET_FUNCTION = "flavourbench_governed_spend_monotonic"
CHILD_RETENTION_FUNCTION = "flavourbench_child_retention_authorization"
CHILD_RETENTION_TRIGGERS = {
    "response_arms": "trg_response_arm_retention_authorization",
    "tool_calls": "trg_tool_call_retention_authorization",
    "validator_results": "trg_validator_retention_authorization",
    "run_events": "trg_run_event_retention_authorization",
    "incidents": "trg_incident_retention_authorization",
    "jobs": "trg_job_retention_authorization",
}
BUDGET_TABLES = (
    "seasons",
    "season_provider_budgets",
    "provider_account_budgets",
    "controlled_runs",
)


def _preflight(bind: sa.Connection) -> None:
    inspector = sa.inspect(bind)
    required = {
        "battles": {"prompt", "prompt_redacted", "research_consent", "retention_until"},
        **{
            table: {"budget_used_micros", "budget_reserved_micros"}
            for table in BUDGET_TABLES
        },
    }
    for table, columns in required.items():
        observed = {column["name"] for column in inspector.get_columns(table)}
        missing = columns - observed
        if missing:
            raise RuntimeError(
                f"0015 preflight: {table} is missing required columns: {sorted(missing)}"
            )

    if bind.dialect.name == "postgresql":
        unauthorized = bind.execute(
            sa.text(
                """
                SELECT COUNT(*) FROM battles
                WHERE prompt_redacted IS TRUE
                  AND (research_consent IS TRUE OR retention_until > CURRENT_TIMESTAMP)
                """
            )
        ).scalar_one()
    elif bind.dialect.name == "sqlite":
        unauthorized = bind.execute(
            sa.text(
                """
                SELECT COUNT(*) FROM battles
                WHERE prompt_redacted = 1
                  AND (research_consent = 1 OR julianday(retention_until) > julianday('now'))
                """
            )
        ).scalar_one()
    else:
        raise RuntimeError(f"unsupported database dialect for 0015: {bind.dialect.name}")
    if int(unauthorized):
        raise RuntimeError(
            "0015 preflight: database contains redactions without an expired non-consent basis"
        )
    for table in BUDGET_TABLES:
        negative = bind.execute(
            sa.text(
                f"SELECT COUNT(*) FROM {table} "
                "WHERE budget_used_micros < 0 OR budget_reserved_micros < 0"
            )
        ).scalar_one()
        if int(negative):
            raise RuntimeError(f"0015 preflight: {table} contains negative budget counters")


def _create_postgresql_guards() -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {RETENTION_FUNCTION}()
        RETURNS trigger AS $$
        BEGIN
            IF OLD.prompt IS NOT NULL
               AND NEW.prompt IS NULL
               AND OLD.prompt_redacted IS FALSE
               AND NEW.prompt_redacted IS TRUE
               AND (
                    OLD.research_consent IS TRUE
                    OR OLD.retention_until > CURRENT_TIMESTAMP
               ) THEN
                RAISE EXCEPTION
                    'battle retention redaction requires expired non-consented content';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS {RETENTION_TRIGGER} ON battles;
        CREATE TRIGGER {RETENTION_TRIGGER}
        BEFORE UPDATE OF prompt, prompt_redacted ON battles
        FOR EACH ROW EXECUTE FUNCTION {RETENTION_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {BUDGET_FUNCTION}()
        RETURNS trigger AS $$
        BEGIN
            IF NEW.budget_used_micros < 0 OR NEW.budget_reserved_micros < 0 THEN
                RAISE EXCEPTION 'governed spend and reservations must be nonnegative';
            END IF;
            IF TG_OP = 'UPDATE'
               AND NEW.budget_used_micros < OLD.budget_used_micros THEN
                RAISE EXCEPTION 'governed spend cannot move backward';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table in BUDGET_TABLES:
        trigger = f"{BUDGET_TRIGGER_PREFIX}_{table}"
        op.execute(
            f"""
            DROP TRIGGER IF EXISTS {trigger} ON {table};
            CREATE TRIGGER {trigger}
            BEFORE INSERT OR UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION {BUDGET_FUNCTION}();
            """
        )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {CHILD_RETENTION_FUNCTION}()
        RETURNS trigger AS $$
        DECLARE
            authorized boolean := false;
        BEGIN
            IF TG_TABLE_NAME = 'response_arms' THEN
                SELECT EXISTS(
                    SELECT 1 FROM battles AS b
                    WHERE b.id = NEW.battle_id
                      AND b.prompt IS NULL
                      AND b.prompt_redacted IS TRUE
                      AND b.research_consent IS FALSE
                      AND b.retention_until <= CURRENT_TIMESTAMP
                ) INTO authorized;
            ELSIF TG_TABLE_NAME IN ('tool_calls', 'validator_results') THEN
                SELECT EXISTS(
                    SELECT 1
                    FROM response_arms AS a
                    JOIN battles AS b ON b.id = a.battle_id
                    WHERE a.id = NEW.arm_id
                      AND b.prompt IS NULL
                      AND b.prompt_redacted IS TRUE
                      AND b.research_consent IS FALSE
                      AND b.retention_until <= CURRENT_TIMESTAMP
                ) INTO authorized;
            ELSIF TG_TABLE_NAME = 'run_events' THEN
                IF NEW.entity_type = 'battle' THEN
                    SELECT EXISTS(
                        SELECT 1 FROM battles AS b
                        WHERE b.id = NEW.entity_id
                          AND b.prompt IS NULL
                          AND b.prompt_redacted IS TRUE
                          AND b.research_consent IS FALSE
                          AND b.retention_until <= CURRENT_TIMESTAMP
                    ) INTO authorized;
                ELSIF NEW.entity_type = 'response_arm' THEN
                    SELECT EXISTS(
                        SELECT 1
                        FROM response_arms AS a
                        JOIN battles AS b ON b.id = a.battle_id
                        WHERE a.id = NEW.entity_id
                          AND b.prompt IS NULL
                          AND b.prompt_redacted IS TRUE
                          AND b.research_consent IS FALSE
                          AND b.retention_until <= CURRENT_TIMESTAMP
                    ) INTO authorized;
                END IF;
            ELSIF TG_TABLE_NAME = 'incidents' THEN
                SELECT EXISTS(
                    SELECT 1 FROM battles AS b
                    WHERE b.id = NEW.battle_id
                      AND b.prompt IS NULL
                      AND b.prompt_redacted IS TRUE
                      AND b.research_consent IS FALSE
                      AND b.retention_until <= CURRENT_TIMESTAMP
                ) INTO authorized;
            ELSIF TG_TABLE_NAME = 'jobs' THEN
                SELECT EXISTS(
                    SELECT 1 FROM battles AS b
                    WHERE b.id = NEW.battle_id
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
        $$ LANGUAGE plpgsql;
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
            DROP TRIGGER IF EXISTS {trigger} ON {table};
            CREATE TRIGGER {trigger}
            BEFORE UPDATE OF {columns} ON {table}
            FOR EACH ROW WHEN ({condition})
            EXECUTE FUNCTION {CHILD_RETENTION_FUNCTION}();
            """
        )


def _create_sqlite_guards() -> None:
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
              OR julianday(OLD.retention_until) > julianday('now')
          )
        BEGIN
            SELECT RAISE(
                ABORT,
                'battle retention redaction requires expired non-consented content'
            );
        END;
        """
    )
    for table in BUDGET_TABLES:
        trigger = f"{BUDGET_TRIGGER_PREFIX}_{table}"
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}_insert")
        op.execute(
            f"""
            CREATE TRIGGER {trigger}
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            WHEN NEW.budget_used_micros < 0
              OR NEW.budget_reserved_micros < 0
              OR NEW.budget_used_micros < OLD.budget_used_micros
            BEGIN
                SELECT RAISE(ABORT, 'governed spend cannot move backward');
            END;
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {trigger}_insert
            BEFORE INSERT ON {table}
            FOR EACH ROW
            WHEN NEW.budget_used_micros < 0 OR NEW.budget_reserved_micros < 0
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'governed spend and reservations must be nonnegative'
                );
            END;
            """
        )
    eligibility = (
        "b.prompt IS NULL AND b.prompt_redacted = 1 "
        "AND b.research_consent = 0 "
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
            "AND NOT EXISTS (SELECT 1 FROM response_arms AS a JOIN battles AS b "
            f"ON b.id = a.battle_id WHERE a.id = NEW.arm_id AND {eligibility})",
        ),
        "validator_results": (
            "detail_json",
            "json(NEW.detail_json) = json_object('redacted', json('true')) "
            "AND NOT EXISTS (SELECT 1 FROM response_arms AS a JOIN battles AS b "
            f"ON b.id = a.battle_id WHERE a.id = NEW.arm_id AND {eligibility})",
        ),
        "run_events": (
            "payload_json",
            "json(NEW.payload_json) = json_object('redacted', json('true')) AND NOT ("
            "(NEW.entity_type = 'battle' AND EXISTS (SELECT 1 FROM battles AS b "
            f"WHERE b.id = NEW.entity_id AND {eligibility})) OR "
            "(NEW.entity_type = 'response_arm' AND EXISTS (SELECT 1 FROM response_arms AS a "
            "JOIN battles AS b ON b.id = a.battle_id "
            f"WHERE a.id = NEW.entity_id AND {eligibility})))",
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
                    'retention redaction lacks an expired non-consent basis'
                );
            END;
            """
        )


def upgrade() -> None:
    bind = op.get_bind()
    _preflight(bind)
    if bind.dialect.name == "postgresql":
        _create_postgresql_guards()
    else:
        _create_sqlite_guards()


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(f"DROP TRIGGER IF EXISTS {RETENTION_TRIGGER} ON battles")
        for table, trigger in CHILD_RETENTION_TRIGGERS.items():
            op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        for table in BUDGET_TABLES:
            trigger = f"{BUDGET_TRIGGER_PREFIX}_{table}"
            op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        op.execute(f"DROP FUNCTION IF EXISTS {RETENTION_FUNCTION}()")
        op.execute(f"DROP FUNCTION IF EXISTS {CHILD_RETENTION_FUNCTION}()")
        op.execute(f"DROP FUNCTION IF EXISTS {BUDGET_FUNCTION}()")
    elif bind.dialect.name == "sqlite":
        op.execute(f"DROP TRIGGER IF EXISTS {RETENTION_TRIGGER}")
        for trigger in CHILD_RETENTION_TRIGGERS.values():
            op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        for table in BUDGET_TABLES:
            trigger = f"{BUDGET_TRIGGER_PREFIX}_{table}"
            op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            op.execute(f"DROP TRIGGER IF EXISTS {trigger}_insert")
    else:
        raise RuntimeError(f"unsupported database dialect for 0015: {bind.dialect.name}")
