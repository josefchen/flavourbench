"""Seal snapshot evidence, normalized outputs, and append-only records.

Revision ID: 0013_snapshot_evidence_seal
Revises: 0012_snapshot_integrity
"""

from __future__ import annotations

import hashlib
import json

import sqlalchemy as sa

from alembic import op

revision = "0013_snapshot_evidence_seal"
down_revision = "0012_snapshot_integrity"
branch_labels = None
depends_on = None


SNAPSHOT_GUARD_TRIGGER = "trg_leaderboard_snapshot_integrity_guard"
SNAPSHOT_GUARD_FUNCTION = "flavourbench_leaderboard_snapshot_integrity_guard"
ARM_DIGEST_TRIGGER = "trg_response_arm_output_digest_write_once"
ARM_DIGEST_FUNCTION = "flavourbench_response_arm_output_digest_write_once"
TOOL_CALL_GUARD_TRIGGER = "trg_tool_call_trace_seal"
TOOL_CALL_GUARD_FUNCTION = "flavourbench_tool_call_trace_seal"
TOOL_CALL_REDACTION_SENTINEL = "[REDACTED AFTER OPERATIONAL RETENTION]"
VALIDATOR_GUARD_TRIGGER = "trg_validator_result_seal"
VALIDATOR_GUARD_FUNCTION = "flavourbench_validator_result_seal"
VALIDATOR_UNIQUE_INDEX = "uq_validator_results_arm_name_version"


def _json_sha256(value: object) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _backfill_output_digests(bind: sa.Connection) -> None:
    arms = sa.table(
        "response_arms",
        sa.column("id", sa.String()),
        sa.column("status", sa.String()),
        sa.column("answer_markdown", sa.Text()),
        sa.column("answer_markdown_sha256", sa.String()),
        sa.column("output_json", sa.JSON()),
        sa.column("output_json_sha256", sa.String()),
    )
    rows = bind.execute(
        sa.select(arms.c.id, arms.c.status, arms.c.answer_markdown, arms.c.output_json)
    ).mappings()
    for row in rows:
        answer = row["answer_markdown"]
        values = {
            "answer_markdown_sha256": (
                hashlib.sha256(answer.encode()).hexdigest() if isinstance(answer, str) else None
            ),
            "output_json_sha256": (
                _json_sha256(row["output_json"] or {})
                if row["status"] in {"complete", "failed", "uncertain"} or bool(row["output_json"])
                else None
            ),
        }
        bind.execute(sa.update(arms).where(arms.c.id == row["id"]).values(**values))


def _backfill_snapshot_cutoff(bind: sa.Connection) -> None:
    snapshots = sa.table(
        "leaderboard_snapshots",
        sa.column("id", sa.String()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("evidence_cutoff_at", sa.DateTime(timezone=True)),
    )
    bind.execute(
        sa.update(snapshots)
        .where(snapshots.c.evidence_cutoff_at.is_(None))
        .values(evidence_cutoff_at=snapshots.c.created_at)
    )


def _backfill_tool_call_digests(bind: sa.Connection) -> None:
    calls = sa.table(
        "tool_calls",
        sa.column("id", sa.String()),
        sa.column("arguments_json", sa.JSON()),
        sa.column("arguments_sha256", sa.String()),
        sa.column("structured_content_json", sa.JSON()),
        sa.column("structured_content_sha256", sa.String()),
    )
    rows = bind.execute(
        sa.select(
            calls.c.id,
            calls.c.arguments_json,
            calls.c.structured_content_json,
        )
    ).mappings()
    for row in rows:
        bind.execute(
            sa.update(calls)
            .where(calls.c.id == row["id"])
            .values(
                arguments_sha256=_json_sha256(
                    {"arguments": row["arguments_json"]}
                ),
                structured_content_sha256=_json_sha256(
                    {"structured": row["structured_content_json"]}
                ),
            )
        )


def _backfill_validator_digests(bind: sa.Connection) -> None:
    results = sa.table(
        "validator_results",
        sa.column("id", sa.String()),
        sa.column("detail_json", sa.JSON()),
        sa.column("detail_sha256", sa.String()),
    )
    rows = bind.execute(
        sa.select(results.c.id, results.c.detail_json)
    ).mappings()
    for row in rows:
        bind.execute(
            sa.update(results)
            .where(results.c.id == row["id"])
            .values(detail_sha256=_json_sha256({"detail": row["detail_json"]}))
        )


def _create_postgresql_guards(
    *,
    include_arm_digests: bool,
    include_tool_calls: bool,
    include_validators: bool,
) -> None:
    op.execute(
        f"""
        CREATE FUNCTION {SNAPSHOT_GUARD_FUNCTION}() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.publication_status <> 'draft'
                   OR NEW.publication_reference_sha256 IS NOT NULL
                   OR NEW.published_at IS NOT NULL
                   OR NEW.evidence_cutoff_at IS NULL
                   OR NEW.input_sha256 !~ '^[0-9a-f]{{64}}$'
                   OR NEW.input_evidence_sha256 IS NULL
                   OR NEW.input_evidence_sha256 !~ '^[0-9a-f]{{64}}$'
                   OR NEW.input_evidence_json IS NULL
                   OR NEW.payload_sha256 IS NULL
                   OR NEW.payload_sha256 !~ '^[0-9a-f]{{64}}$' THEN
                    RAISE EXCEPTION 'leaderboard snapshots must be inserted as sealed drafts';
                END IF;
                RETURN NEW;
            ELSIF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'leaderboard snapshots are append-only';
            END IF;

            IF OLD.season_id IS DISTINCT FROM NEW.season_id
               OR OLD.track IS DISTINCT FROM NEW.track
               OR OLD.cohort IS DISTINCT FROM NEW.cohort
               OR OLD.category IS DISTINCT FROM NEW.category
               OR OLD.data_stratum IS DISTINCT FROM NEW.data_stratum
               OR OLD.controlled_run_id IS DISTINCT FROM NEW.controlled_run_id
               OR OLD.input_sha256 IS DISTINCT FROM NEW.input_sha256
               OR OLD.input_evidence_sha256 IS DISTINCT FROM NEW.input_evidence_sha256
               OR OLD.input_evidence_json::text IS DISTINCT FROM NEW.input_evidence_json::text
               OR OLD.payload_sha256 IS DISTINCT FROM NEW.payload_sha256
               OR OLD.payload_json::text IS DISTINCT FROM NEW.payload_json::text
               OR OLD.evidence_cutoff_at IS DISTINCT FROM NEW.evidence_cutoff_at
               OR OLD.supersedes_snapshot_id IS DISTINCT FROM NEW.supersedes_snapshot_id
               OR OLD.created_at IS DISTINCT FROM NEW.created_at THEN
                RAISE EXCEPTION 'leaderboard snapshot content is immutable';
            END IF;

            IF OLD.publication_status = 'draft' AND NEW.publication_status = 'published' THEN
                IF NEW.publication_reference_sha256 IS NULL
                   OR NEW.publication_reference_sha256 !~ '^[0-9a-f]{{64}}$'
                   OR NEW.published_at IS NULL
                   OR NEW.input_sha256 !~ '^[0-9a-f]{{64}}$'
                   OR NEW.input_evidence_sha256 IS NULL
                   OR NEW.input_evidence_sha256 !~ '^[0-9a-f]{{64}}$'
                   OR NEW.input_evidence_json IS NULL
                   OR NEW.payload_sha256 IS NULL
                   OR NEW.payload_sha256 !~ '^[0-9a-f]{{64}}$' THEN
                    RAISE EXCEPTION 'published snapshots require sealed evidence and metadata';
                END IF;
            ELSIF OLD.publication_reference_sha256 IS DISTINCT FROM NEW.publication_reference_sha256
               OR OLD.published_at IS DISTINCT FROM NEW.published_at THEN
                RAISE EXCEPTION 'publication metadata can change only while publishing';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {SNAPSHOT_GUARD_TRIGGER}
        BEFORE INSERT OR UPDATE OR DELETE ON leaderboard_snapshots
        FOR EACH ROW EXECUTE FUNCTION {SNAPSHOT_GUARD_FUNCTION}()
        """
    )
    if include_tool_calls:
        op.execute(
            f"""
            CREATE FUNCTION {TOOL_CALL_GUARD_FUNCTION}() RETURNS trigger AS $$
            BEGIN
                IF TG_OP = 'DELETE' THEN
                    RAISE EXCEPTION 'tool-call trace records cannot be deleted';
                END IF;
                IF OLD.id IS DISTINCT FROM NEW.id
                   OR OLD.arm_id IS DISTINCT FROM NEW.arm_id
                   OR OLD.round_index IS DISTINCT FROM NEW.round_index
                   OR OLD.call_index IS DISTINCT FROM NEW.call_index
                   OR OLD.tool_call_id IS DISTINCT FROM NEW.tool_call_id
                   OR OLD.tool_name IS DISTINCT FROM NEW.tool_name
                   OR OLD.arguments_sha256 IS DISTINCT FROM NEW.arguments_sha256
                   OR OLD.result_sha256 IS DISTINCT FROM NEW.result_sha256
                   OR OLD.structured_content_sha256 IS DISTINCT FROM NEW.structured_content_sha256
                   OR OLD.latency_ms IS DISTINCT FROM NEW.latency_ms
                   OR OLD.is_error IS DISTINCT FROM NEW.is_error
                   OR OLD.created_at IS DISTINCT FROM NEW.created_at
                   OR NEW.result_text IS DISTINCT FROM '{TOOL_CALL_REDACTION_SENTINEL}'
                   OR NEW.arguments_json::jsonb IS DISTINCT FROM '{{"redacted": true}}'::jsonb
                   OR NEW.structured_content_json::jsonb
                        IS DISTINCT FROM '{{"redacted": true}}'::jsonb THEN
                    RAISE EXCEPTION 'tool-call traces are immutable except for retention redaction';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {TOOL_CALL_GUARD_TRIGGER}
            BEFORE UPDATE OR DELETE ON tool_calls
            FOR EACH ROW EXECUTE FUNCTION {TOOL_CALL_GUARD_FUNCTION}()
            """
        )
    if include_validators:
        op.execute(
            f"""
            CREATE FUNCTION {VALIDATOR_GUARD_FUNCTION}() RETURNS trigger AS $$
            BEGIN
                IF TG_OP = 'DELETE' THEN
                    RAISE EXCEPTION 'validator result records cannot be deleted';
                END IF;
                IF OLD.id IS DISTINCT FROM NEW.id
                   OR OLD.arm_id IS DISTINCT FROM NEW.arm_id
                   OR OLD.validator_name IS DISTINCT FROM NEW.validator_name
                   OR OLD.validator_version IS DISTINCT FROM NEW.validator_version
                   OR OLD.status IS DISTINCT FROM NEW.status
                   OR OLD.score_milli IS DISTINCT FROM NEW.score_milli
                   OR OLD.detail_sha256 IS DISTINCT FROM NEW.detail_sha256
                   OR OLD.created_at IS DISTINCT FROM NEW.created_at
                   OR NEW.detail_json::jsonb IS DISTINCT FROM '{{"redacted": true}}'::jsonb THEN
                    RAISE EXCEPTION 'invalid validator result mutation';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {VALIDATOR_GUARD_TRIGGER}
            BEFORE UPDATE OR DELETE ON validator_results
            FOR EACH ROW EXECUTE FUNCTION {VALIDATOR_GUARD_FUNCTION}()
            """
        )
    if not include_arm_digests:
        return
    op.execute(
        f"""
        CREATE FUNCTION {ARM_DIGEST_FUNCTION}() RETURNS trigger AS $$
        BEGIN
            IF OLD.answer_markdown_sha256 IS NOT NULL
               AND OLD.answer_markdown_sha256 IS DISTINCT FROM NEW.answer_markdown_sha256 THEN
                RAISE EXCEPTION 'response arm answer digest is write-once';
            END IF;
            IF OLD.output_json_sha256 IS NOT NULL
               AND OLD.output_json_sha256 IS DISTINCT FROM NEW.output_json_sha256 THEN
                RAISE EXCEPTION 'response arm output digest is write-once';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ARM_DIGEST_TRIGGER}
        BEFORE UPDATE OF answer_markdown_sha256, output_json_sha256 ON response_arms
        FOR EACH ROW EXECUTE FUNCTION {ARM_DIGEST_FUNCTION}()
        """
    )


def _create_sqlite_guards(
    *,
    include_arm_digests: bool,
    include_tool_calls: bool,
    include_validators: bool,
) -> None:
    op.execute(
        f"""
        CREATE TRIGGER {SNAPSHOT_GUARD_TRIGGER}_insert
        BEFORE INSERT ON leaderboard_snapshots
        FOR EACH ROW
        WHEN NEW.publication_status <> 'draft'
          OR NEW.publication_reference_sha256 IS NOT NULL
          OR NEW.published_at IS NOT NULL
          OR NEW.evidence_cutoff_at IS NULL
          OR length(NEW.input_sha256) <> 64
          OR NEW.input_sha256 GLOB '*[^0-9a-f]*'
          OR NEW.input_evidence_sha256 IS NULL
          OR length(NEW.input_evidence_sha256) <> 64
          OR NEW.input_evidence_sha256 GLOB '*[^0-9a-f]*'
          OR NEW.input_evidence_json IS NULL
          OR NEW.payload_sha256 IS NULL
          OR length(NEW.payload_sha256) <> 64
          OR NEW.payload_sha256 GLOB '*[^0-9a-f]*'
        BEGIN
            SELECT RAISE(ABORT, 'leaderboard snapshots must be inserted as sealed drafts');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {SNAPSHOT_GUARD_TRIGGER}_content
        BEFORE UPDATE ON leaderboard_snapshots
        FOR EACH ROW
        WHEN OLD.season_id IS NOT NEW.season_id
          OR OLD.track IS NOT NEW.track
          OR OLD.cohort IS NOT NEW.cohort
          OR OLD.category IS NOT NEW.category
          OR OLD.data_stratum IS NOT NEW.data_stratum
          OR OLD.controlled_run_id IS NOT NEW.controlled_run_id
          OR OLD.input_sha256 IS NOT NEW.input_sha256
          OR OLD.input_evidence_sha256 IS NOT NEW.input_evidence_sha256
          OR OLD.input_evidence_json IS NOT NEW.input_evidence_json
          OR OLD.payload_sha256 IS NOT NEW.payload_sha256
          OR OLD.payload_json IS NOT NEW.payload_json
          OR OLD.evidence_cutoff_at IS NOT NEW.evidence_cutoff_at
          OR OLD.supersedes_snapshot_id IS NOT NEW.supersedes_snapshot_id
          OR OLD.created_at IS NOT NEW.created_at
        BEGIN
            SELECT RAISE(ABORT, 'leaderboard snapshot content is immutable');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {SNAPSHOT_GUARD_TRIGGER}_publication
        BEFORE UPDATE ON leaderboard_snapshots
        FOR EACH ROW
        WHEN (
            OLD.publication_status = 'draft'
            AND NEW.publication_status = 'published'
            AND (
                NEW.publication_reference_sha256 IS NULL
                OR NEW.published_at IS NULL
                OR length(NEW.publication_reference_sha256) <> 64
                OR NEW.publication_reference_sha256 GLOB '*[^0-9a-f]*'
                OR length(NEW.input_sha256) <> 64
                OR NEW.input_sha256 GLOB '*[^0-9a-f]*'
                OR NEW.input_evidence_sha256 IS NULL
                OR length(NEW.input_evidence_sha256) <> 64
                OR NEW.input_evidence_sha256 GLOB '*[^0-9a-f]*'
                OR NEW.input_evidence_json IS NULL
                OR NEW.payload_sha256 IS NULL
                OR length(NEW.payload_sha256) <> 64
                OR NEW.payload_sha256 GLOB '*[^0-9a-f]*'
            )
        ) OR (
            NOT (
                OLD.publication_status = 'draft'
                AND NEW.publication_status = 'published'
            )
            AND (
                OLD.publication_reference_sha256 IS NOT NEW.publication_reference_sha256
                OR OLD.published_at IS NOT NEW.published_at
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid leaderboard publication metadata transition');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {SNAPSHOT_GUARD_TRIGGER}_delete
        BEFORE DELETE ON leaderboard_snapshots
        FOR EACH ROW
        BEGIN
            SELECT RAISE(ABORT, 'leaderboard snapshots are append-only');
        END;
        """
    )
    if include_tool_calls:
        op.execute(
            f"""
            CREATE TRIGGER {TOOL_CALL_GUARD_TRIGGER}_update
            BEFORE UPDATE ON tool_calls
            FOR EACH ROW
            WHEN OLD.id IS NOT NEW.id
              OR OLD.arm_id IS NOT NEW.arm_id
              OR OLD.round_index IS NOT NEW.round_index
              OR OLD.call_index IS NOT NEW.call_index
              OR OLD.tool_call_id IS NOT NEW.tool_call_id
              OR OLD.tool_name IS NOT NEW.tool_name
              OR OLD.arguments_sha256 IS NOT NEW.arguments_sha256
              OR OLD.result_sha256 IS NOT NEW.result_sha256
              OR OLD.structured_content_sha256 IS NOT NEW.structured_content_sha256
              OR OLD.latency_ms IS NOT NEW.latency_ms
              OR OLD.is_error IS NOT NEW.is_error
              OR OLD.created_at IS NOT NEW.created_at
              OR NEW.result_text IS NOT '{TOOL_CALL_REDACTION_SENTINEL}'
              OR json_type(NEW.arguments_json, '$') IS NOT 'object'
              OR json_type(NEW.arguments_json, '$.redacted') IS NOT 'true'
              OR (SELECT COUNT(*) FROM json_each(NEW.arguments_json)) IS NOT 1
              OR json_type(NEW.structured_content_json, '$') IS NOT 'object'
              OR json_type(NEW.structured_content_json, '$.redacted') IS NOT 'true'
              OR (
                  SELECT COUNT(*) FROM json_each(NEW.structured_content_json)
              ) IS NOT 1
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'tool-call traces are immutable except for retention redaction'
                );
            END;
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {TOOL_CALL_GUARD_TRIGGER}_delete
            BEFORE DELETE ON tool_calls
            FOR EACH ROW
            BEGIN
                SELECT RAISE(ABORT, 'tool-call trace records cannot be deleted');
            END;
            """
        )
    if include_validators:
        op.execute(
            f"""
            CREATE TRIGGER {VALIDATOR_GUARD_TRIGGER}_update
            BEFORE UPDATE ON validator_results
            FOR EACH ROW
            WHEN OLD.id IS NOT NEW.id
              OR OLD.arm_id IS NOT NEW.arm_id
              OR OLD.validator_name IS NOT NEW.validator_name
              OR OLD.validator_version IS NOT NEW.validator_version
              OR OLD.status IS NOT NEW.status
              OR OLD.score_milli IS NOT NEW.score_milli
              OR OLD.detail_sha256 IS NOT NEW.detail_sha256
              OR OLD.created_at IS NOT NEW.created_at
              OR json_type(NEW.detail_json, '$') IS NOT 'object'
              OR json_type(NEW.detail_json, '$.redacted') IS NOT 'true'
              OR (SELECT COUNT(*) FROM json_each(NEW.detail_json)) IS NOT 1
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'validator results are immutable except for retention redaction'
                );
            END;
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {VALIDATOR_GUARD_TRIGGER}_delete
            BEFORE DELETE ON validator_results
            FOR EACH ROW
            BEGIN
                SELECT RAISE(ABORT, 'validator result records cannot be deleted');
            END;
            """
        )
    if not include_arm_digests:
        return
    op.execute(
        f"""
        CREATE TRIGGER {ARM_DIGEST_TRIGGER}
        BEFORE UPDATE OF answer_markdown_sha256, output_json_sha256 ON response_arms
        FOR EACH ROW
        WHEN (
            OLD.answer_markdown_sha256 IS NOT NULL
            AND OLD.answer_markdown_sha256 IS NOT NEW.answer_markdown_sha256
        ) OR (
            OLD.output_json_sha256 IS NOT NULL
            AND OLD.output_json_sha256 IS NOT NEW.output_json_sha256
        )
        BEGIN
            SELECT RAISE(ABORT, 'response arm output digests are write-once');
        END;
        """
    )


def upgrade() -> None:
    bind = op.get_bind()
    table_names = sa.inspect(bind).get_table_names()
    has_response_arms = "response_arms" in table_names
    has_tool_calls = "tool_calls" in table_names
    has_validators = "validator_results" in table_names
    if has_response_arms:
        op.add_column(
            "response_arms",
            sa.Column("answer_markdown_sha256", sa.String(length=64), nullable=True),
        )
        op.add_column(
            "response_arms",
            sa.Column("output_json_sha256", sa.String(length=64), nullable=True),
        )
    if has_tool_calls:
        op.add_column(
            "tool_calls",
            sa.Column("arguments_sha256", sa.String(length=64), nullable=True),
        )
        op.add_column(
            "tool_calls",
            sa.Column("structured_content_sha256", sa.String(length=64), nullable=True),
        )
    if has_validators:
        op.add_column(
            "validator_results",
            sa.Column("detail_sha256", sa.String(length=64), nullable=True),
        )
    op.add_column(
        "leaderboard_snapshots",
        sa.Column("evidence_cutoff_at", sa.DateTime(timezone=True), nullable=True),
    )
    if has_response_arms:
        _backfill_output_digests(bind)
    if has_tool_calls:
        _backfill_tool_call_digests(bind)
    if has_validators:
        _backfill_validator_digests(bind)
        op.create_index(
            VALIDATOR_UNIQUE_INDEX,
            "validator_results",
            ["arm_id", "validator_name", "validator_version"],
            unique=True,
        )
    _backfill_snapshot_cutoff(bind)
    if bind.dialect.name == "postgresql":
        op.alter_column("leaderboard_snapshots", "evidence_cutoff_at", nullable=False)
        if has_tool_calls:
            op.alter_column("tool_calls", "arguments_sha256", nullable=False)
            op.alter_column("tool_calls", "structured_content_sha256", nullable=False)
        if has_validators:
            op.alter_column("validator_results", "detail_sha256", nullable=False)
        _create_postgresql_guards(
            include_arm_digests=has_response_arms,
            include_tool_calls=has_tool_calls,
            include_validators=has_validators,
        )
    elif bind.dialect.name == "sqlite":
        _create_sqlite_guards(
            include_arm_digests=has_response_arms,
            include_tool_calls=has_tool_calls,
            include_validators=has_validators,
        )


def downgrade() -> None:
    bind = op.get_bind()
    table_names = sa.inspect(bind).get_table_names()
    has_response_arms = "response_arms" in table_names
    has_tool_calls = "tool_calls" in table_names
    has_validators = "validator_results" in table_names
    if bind.dialect.name == "postgresql":
        if has_validators:
            op.execute(
                f"DROP TRIGGER IF EXISTS {VALIDATOR_GUARD_TRIGGER} ON validator_results"
            )
            op.execute(f"DROP FUNCTION IF EXISTS {VALIDATOR_GUARD_FUNCTION}()")
        if has_tool_calls:
            op.execute(f"DROP TRIGGER IF EXISTS {TOOL_CALL_GUARD_TRIGGER} ON tool_calls")
            op.execute(f"DROP FUNCTION IF EXISTS {TOOL_CALL_GUARD_FUNCTION}()")
        if has_response_arms:
            op.execute(f"DROP TRIGGER IF EXISTS {ARM_DIGEST_TRIGGER} ON response_arms")
            op.execute(f"DROP FUNCTION IF EXISTS {ARM_DIGEST_FUNCTION}()")
        op.execute(f"DROP TRIGGER IF EXISTS {SNAPSHOT_GUARD_TRIGGER} ON leaderboard_snapshots")
        op.execute(f"DROP FUNCTION IF EXISTS {SNAPSHOT_GUARD_FUNCTION}()")
    elif bind.dialect.name == "sqlite":
        if has_validators:
            op.execute(f"DROP TRIGGER IF EXISTS {VALIDATOR_GUARD_TRIGGER}_update")
            op.execute(f"DROP TRIGGER IF EXISTS {VALIDATOR_GUARD_TRIGGER}_delete")
        if has_tool_calls:
            op.execute(f"DROP TRIGGER IF EXISTS {TOOL_CALL_GUARD_TRIGGER}_update")
            op.execute(f"DROP TRIGGER IF EXISTS {TOOL_CALL_GUARD_TRIGGER}_delete")
        if has_response_arms:
            op.execute(f"DROP TRIGGER IF EXISTS {ARM_DIGEST_TRIGGER}")
        for suffix in ("insert", "content", "publication", "delete"):
            op.execute(f"DROP TRIGGER IF EXISTS {SNAPSHOT_GUARD_TRIGGER}_{suffix}")
    op.drop_column("leaderboard_snapshots", "evidence_cutoff_at")
    if has_validators:
        op.drop_index(VALIDATOR_UNIQUE_INDEX, table_name="validator_results")
        op.drop_column("validator_results", "detail_sha256")
    if has_response_arms:
        op.drop_column("response_arms", "output_json_sha256")
        op.drop_column("response_arms", "answer_markdown_sha256")
    if has_tool_calls:
        op.drop_column("tool_calls", "structured_content_sha256")
        op.drop_column("tool_calls", "arguments_sha256")
