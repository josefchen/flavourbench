"""Freeze commercial-run schedules, evidence snapshots, and complete MCP traces.

Revision ID: 0009_commercial_run_integrity
Revises: 0008_private_controlled_runs
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0009_commercial_run_integrity"
down_revision = "0008_private_controlled_runs"
branch_labels = None
depends_on = None


def _columns(bind: sa.Connection, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns(table)}


def _indexes(bind: sa.Connection, table: str) -> set[str]:
    return {index["name"] for index in sa.inspect(bind).get_indexes(table)}


def _unique_constraints(bind: sa.Connection, table: str) -> set[str | None]:
    return {constraint.get("name") for constraint in sa.inspect(bind).get_unique_constraints(table)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    json_array_default = (
        sa.text("'[]'::json") if bind.dialect.name == "postgresql" else sa.text("'[]'")
    )
    json_object_default = (
        sa.text("'{}'::json") if bind.dialect.name == "postgresql" else sa.text("'{}'")
    )

    if "seasons" in tables:
        season_columns = _columns(bind, "seasons")
        season_additions = {
            "analysis_plan_sha256": sa.Column(
                "analysis_plan_sha256",
                sa.String(length=64),
                nullable=False,
                server_default="unfrozen",
            ),
            "protocol_bundle_json": sa.Column(
                "protocol_bundle_json",
                sa.JSON(),
                nullable=False,
                server_default=json_object_default,
            ),
            "protocol_bundle_sha256": sa.Column(
                "protocol_bundle_sha256",
                sa.String(length=64),
                nullable=False,
                server_default="unfrozen",
            ),
        }
        for name, column in season_additions.items():
            if name not in season_columns:
                op.add_column("seasons", column)

    if "battles" in tables:
        battle_columns = _columns(bind, "battles")
        if "protocol_bundle_sha256" not in battle_columns:
            op.add_column(
                "battles",
                sa.Column(
                    "protocol_bundle_sha256",
                    sa.String(length=64),
                    nullable=False,
                    server_default="unfrozen",
                ),
            )
        if "ix_battles_protocol_bundle_sha256" not in _indexes(bind, "battles"):
            op.create_index(
                "ix_battles_protocol_bundle_sha256",
                "battles",
                ["protocol_bundle_sha256"],
            )

    if "response_arms" in tables:
        arm_columns = _columns(bind, "response_arms")
        arm_additions = {
            "observed_decoding_json": sa.Column(
                "observed_decoding_json",
                sa.JSON(),
                nullable=False,
                server_default=json_object_default,
            ),
            "protocol_bundle_sha256": sa.Column(
                "protocol_bundle_sha256",
                sa.String(length=64),
                nullable=False,
                server_default="unfrozen",
            ),
            "epicure_attestation_json": sa.Column(
                "epicure_attestation_json",
                sa.JSON(),
                nullable=False,
                server_default=json_object_default,
            ),
            "epicure_attestation_sha256": sa.Column(
                "epicure_attestation_sha256",
                sa.String(length=64),
                nullable=True,
            ),
        }
        for name, column in arm_additions.items():
            if name not in arm_columns:
                op.add_column("response_arms", column)
        if "ix_response_arms_protocol_bundle_sha256" not in _indexes(
            bind, "response_arms"
        ):
            op.create_index(
                "ix_response_arms_protocol_bundle_sha256",
                "response_arms",
                ["protocol_bundle_sha256"],
            )

    if "catalog_models" in tables:
        catalog_columns = _columns(bind, "catalog_models")
        if "catalog_source" not in catalog_columns:
            op.add_column(
                "catalog_models",
                sa.Column(
                    "catalog_source",
                    sa.String(length=48),
                    nullable=False,
                    server_default="openrouter",
                ),
            )
            op.create_index(
                "ix_catalog_models_catalog_source",
                "catalog_models",
                ["catalog_source"],
            )
        if "open_weight_evidence_json" not in catalog_columns:
            op.add_column(
                "catalog_models",
                sa.Column(
                    "open_weight_evidence_json",
                    sa.JSON(),
                    nullable=False,
                    server_default=json_object_default,
                ),
            )

    controlled_columns = _columns(bind, "controlled_runs")
    controlled_additions = {
        "submitted_endpoint_model_id": sa.Column(
            "submitted_endpoint_model_id",
            sa.String(length=200),
            nullable=False,
            server_default="unbound",
        ),
        "submitted_model_card_sha256": sa.Column(
            "submitted_model_card_sha256",
            sa.String(length=64),
            nullable=False,
            server_default="0" * 64,
        ),
        "data_policy_sha256": sa.Column(
            "data_policy_sha256",
            sa.String(length=64),
            nullable=False,
            server_default="0" * 64,
        ),
        "model_roster_json": sa.Column(
            "model_roster_json",
            sa.JSON(),
            nullable=False,
            server_default=json_array_default,
        ),
        "model_roster_sha256": sa.Column(
            "model_roster_sha256",
            sa.String(length=64),
            nullable=False,
            server_default="0" * 64,
        ),
        "task_schedule_sha256": sa.Column(
            "task_schedule_sha256",
            sa.String(length=64),
            nullable=False,
            server_default="0" * 64,
        ),
        "token_version": sa.Column(
            "token_version", sa.Integer(), nullable=False, server_default="1"
        ),
        "collection_completed_at": sa.Column(
            "collection_completed_at", sa.DateTime(timezone=True), nullable=True
        ),
        "closed_at": sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        "revoked_at": sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    }
    for name, column in controlled_additions.items():
        if name not in controlled_columns:
            op.add_column("controlled_runs", column)

    if "controlled_run_assignments" not in tables:
        op.create_table(
            "controlled_run_assignments",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("controlled_run_id", sa.String(length=36), nullable=False),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("task_id", sa.String(length=36), nullable=False),
            sa.Column("task_public_id", sa.String(length=80), nullable=False),
            sa.Column("task_revision", sa.Integer(), nullable=False),
            sa.Column("task_prompt_sha256", sa.String(length=64), nullable=False),
            sa.Column("task_family", sa.String(length=64), nullable=False),
            sa.Column("track", sa.String(length=32), nullable=False),
            sa.Column("model_ids_json", sa.JSON(), nullable=False),
            sa.Column("repetition_index", sa.Integer(), nullable=False),
            sa.Column("assignment_sha256", sa.String(length=64), nullable=False),
            sa.Column("assignment_seed", sa.String(length=64), nullable=False),
            sa.Column(
                "status",
                sa.String(length=24),
                nullable=False,
                server_default="pending",
            ),
            sa.Column("battle_id", sa.String(length=36), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.CheckConstraint(
                "track IN ('model_arena', 'epicure_uplift')",
                name="ck_controlled_run_assignments_track",
            ),
            sa.CheckConstraint(
                "status IN ('pending', 'queued', 'cancelled')",
                name="ck_controlled_run_assignments_status",
            ),
            sa.CheckConstraint("ordinal >= 0", name="ck_controlled_run_assignments_ordinal"),
            sa.CheckConstraint(
                "repetition_index >= 1",
                name="ck_controlled_run_assignments_repetition_index",
            ),
            sa.ForeignKeyConstraint(["controlled_run_id"], ["controlled_runs.id"]),
            sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
            sa.ForeignKeyConstraint(["battle_id"], ["battles.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "controlled_run_id",
                "ordinal",
                name="uq_controlled_assignments_run_ordinal",
            ),
            sa.UniqueConstraint(
                "controlled_run_id",
                "assignment_sha256",
                name="uq_controlled_assignments_run_sha256",
            ),
            sa.UniqueConstraint("battle_id", name="uq_controlled_assignments_battle"),
        )
    assignment_indexes = _indexes(bind, "controlled_run_assignments")
    for name, columns in {
        "ix_controlled_run_assignments_controlled_run_id": ["controlled_run_id"],
        "ix_controlled_run_assignments_task_id": ["task_id"],
        "ix_controlled_run_assignments_track": ["track"],
        "ix_controlled_run_assignments_assignment_sha256": ["assignment_sha256"],
        "ix_controlled_run_assignments_status": ["status"],
        "ix_controlled_run_assignments_battle_id": ["battle_id"],
    }.items():
        if name not in assignment_indexes:
            op.create_index(name, "controlled_run_assignments", columns)

    if "season_models" in tables:
        season_model_columns = _columns(bind, "season_models")
        if "rate_card_json" not in season_model_columns:
            op.add_column(
                "season_models",
                sa.Column(
                    "rate_card_json",
                    sa.JSON(),
                    nullable=False,
                    server_default=json_object_default,
                ),
            )
        if "rate_card_sha256" not in season_model_columns:
            op.add_column(
                "season_models",
                sa.Column(
                    "rate_card_sha256",
                    sa.String(length=64),
                    nullable=False,
                    server_default="unfrozen",
                ),
            )

    tool_additions = {
        "call_index": sa.Column("call_index", sa.Integer(), nullable=False, server_default="0"),
        "tool_call_id": sa.Column("tool_call_id", sa.String(length=200), nullable=True),
        "structured_content_json": sa.Column(
            "structured_content_json",
            sa.JSON(),
            nullable=False,
            server_default=json_object_default,
        ),
    }
    if "tool_calls" in tables:
        tool_columns = _columns(bind, "tool_calls")
        for name, column in tool_additions.items():
            if name not in tool_columns:
                op.add_column("tool_calls", column)
        old_tool_constraints = [
            constraint
            for constraint in sa.inspect(bind).get_unique_constraints("tool_calls")
            if constraint.get("column_names") == ["arm_id", "round_index"]
            and constraint.get("name")
        ]
        for constraint in old_tool_constraints:
            with op.batch_alter_table("tool_calls") as batch:
                batch.drop_constraint(constraint["name"], type_="unique")
        if "uq_tool_calls_arm_round_call" not in _unique_constraints(
            bind, "tool_calls"
        ):
            with op.batch_alter_table("tool_calls") as batch:
                batch.create_unique_constraint(
                    "uq_tool_calls_arm_round_call",
                    ["arm_id", "round_index", "call_index"],
                )

    snapshot_columns = _columns(bind, "leaderboard_snapshots")
    snapshot_additions = {
        "input_evidence_sha256": sa.Column(
            "input_evidence_sha256", sa.String(length=64), nullable=True
        ),
        "input_evidence_json": sa.Column("input_evidence_json", sa.JSON(), nullable=True),
        "payload_sha256": sa.Column("payload_sha256", sa.String(length=64), nullable=True),
        "supersedes_snapshot_id": sa.Column(
            "supersedes_snapshot_id", sa.String(length=36), nullable=True
        ),
    }
    for name, column in snapshot_additions.items():
        if name not in snapshot_columns:
            op.add_column("leaderboard_snapshots", column)
    snapshot_fks = sa.inspect(bind).get_foreign_keys("leaderboard_snapshots")
    if not any(
        fk.get("constrained_columns") == ["supersedes_snapshot_id"]
        and fk.get("referred_table") == "leaderboard_snapshots"
        for fk in snapshot_fks
    ):
        with op.batch_alter_table("leaderboard_snapshots") as batch:
            batch.create_foreign_key(
                "fk_leaderboard_snapshots_supersedes_snapshot_id",
                "leaderboard_snapshots",
                ["supersedes_snapshot_id"],
                ["id"],
            )
    snapshot_indexes = _indexes(bind, "leaderboard_snapshots")
    if "ix_leaderboard_snapshots_supersedes_snapshot_id" not in snapshot_indexes:
        op.create_index(
            "ix_leaderboard_snapshots_supersedes_snapshot_id",
            "leaderboard_snapshots",
            ["supersedes_snapshot_id"],
        )

    if "cost_events" in tables and (
        "uq_cost_events_battle_reconcile" not in _indexes(bind, "cost_events")
    ):
        op.create_index(
            "uq_cost_events_battle_reconcile",
            "cost_events",
            ["battle_id"],
            unique=True,
            postgresql_where=sa.text("kind = 'reconcile'"),
            sqlite_where=sa.text("kind = 'reconcile'"),
        )

    if bind.dialect.name == "postgresql" and "season_models" in tables:
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_battle_provenance_update()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.season_id IS DISTINCT FROM NEW.season_id OR
                   OLD.run_class IS DISTINCT FROM NEW.run_class OR
                   OLD.rank_eligible IS DISTINCT FROM NEW.rank_eligible OR
                   OLD.data_stratum IS DISTINCT FROM NEW.data_stratum OR
                   OLD.task_id IS DISTINCT FROM NEW.task_id OR
                   OLD.task_revision IS DISTINCT FROM NEW.task_revision OR
                   OLD.controlled_run_id IS DISTINCT FROM NEW.controlled_run_id OR
                   OLD.manifest_sha256 IS DISTINCT FROM NEW.manifest_sha256 OR
                   OLD.protocol_bundle_sha256 IS DISTINCT FROM NEW.protocol_bundle_sha256 OR
                   OLD.scheduler_version IS DISTINCT FROM NEW.scheduler_version OR
                   OLD.assignment_seed IS DISTINCT FROM NEW.assignment_seed OR
                   OLD.track_assignment_probability IS DISTINCT FROM
                       NEW.track_assignment_probability OR
                   OLD.model_assignment_probability IS DISTINCT FROM
                       NEW.model_assignment_probability OR
                   OLD.side_assignment_probability IS DISTINCT FROM
                       NEW.side_assignment_probability OR
                   OLD.track IS DISTINCT FROM NEW.track OR
                   OLD.category IS DISTINCT FROM NEW.category OR
                   OLD.prompt_sha256 IS DISTINCT FROM NEW.prompt_sha256 OR
                   OLD.client_nonce_sha256 IS DISTINCT FROM NEW.client_nonce_sha256 THEN
                    RAISE EXCEPTION
                        'battle scientific provenance is immutable; insert a superseding battle';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_response_arm_contract_update()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.battle_id IS DISTINCT FROM NEW.battle_id OR
                   OLD.side IS DISTINCT FROM NEW.side OR
                   OLD.condition IS DISTINCT FROM NEW.condition OR
                   OLD.model_id IS DISTINCT FROM NEW.model_id OR
                   OLD.provider_slug IS DISTINCT FROM NEW.provider_slug OR
                   OLD.prompt_sha256 IS DISTINCT FROM NEW.prompt_sha256 OR
                   OLD.system_prompt_sha256 IS DISTINCT FROM NEW.system_prompt_sha256 OR
                   OLD.schema_sha256 IS DISTINCT FROM NEW.schema_sha256 OR
                   OLD.tool_schema_sha256 IS DISTINCT FROM NEW.tool_schema_sha256 OR
                   OLD.decoding_json IS DISTINCT FROM NEW.decoding_json OR
                   OLD.protocol_bundle_sha256 IS DISTINCT FROM NEW.protocol_bundle_sha256 OR
                   OLD.epicure_release_id IS DISTINCT FROM NEW.epicure_release_id OR
                   OLD.epicure_bundle_sha256 IS DISTINCT FROM NEW.epicure_bundle_sha256 OR
                   OLD.epicure_application_sha256 IS DISTINCT FROM
                       NEW.epicure_application_sha256 THEN
                    RAISE EXCEPTION
                        'response-arm execution contract is immutable; insert a superseding arm';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_response_arm_contract_immutable ON response_arms;
            CREATE TRIGGER trg_response_arm_contract_immutable
            BEFORE UPDATE ON response_arms
            FOR EACH ROW EXECUTE FUNCTION flavourbench_prevent_response_arm_contract_update();
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_endpoint_contract_update()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.manifest_sha256 NOT IN ('', 'unfrozen', 'unresolved')
                   AND (
                       OLD.model_id IS DISTINCT FROM NEW.model_id OR
                       OLD.slot_role IS DISTINCT FROM NEW.slot_role OR
                       OLD.provider_slug IS DISTINCT FROM NEW.provider_slug OR
                       OLD.expected_actual_model_id IS DISTINCT FROM NEW.expected_actual_model_id OR
                       OLD.expected_actual_provider_slug IS DISTINCT FROM
                           NEW.expected_actual_provider_slug OR
                       OLD.supported_parameters_json IS DISTINCT FROM
                           NEW.supported_parameters_json OR
                       OLD.decoding_json IS DISTINCT FROM NEW.decoding_json OR
                       OLD.endpoint_max_completion_tokens IS DISTINCT FROM
                           NEW.endpoint_max_completion_tokens OR
                       OLD.endpoint_document_sha256 IS DISTINCT FROM NEW.endpoint_document_sha256 OR
                       OLD.endpoint_contract_sha256 IS DISTINCT FROM NEW.endpoint_contract_sha256 OR
                       OLD.rate_card_json IS DISTINCT FROM NEW.rate_card_json OR
                       OLD.rate_card_sha256 IS DISTINCT FROM NEW.rate_card_sha256 OR
                       OLD.worst_case_cost_micros IS DISTINCT FROM NEW.worst_case_cost_micros OR
                       OLD.manifest_sha256 IS DISTINCT FROM NEW.manifest_sha256 OR
                       OLD.eligible IS DISTINCT FROM NEW.eligible
                   ) THEN
                    RAISE EXCEPTION
                        'frozen season endpoint contract is immutable; create a new manifest';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_controlled_run_contract_update()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.season_id IS DISTINCT FROM NEW.season_id OR
                   OLD.organization_reference_sha256 IS DISTINCT FROM
                       NEW.organization_reference_sha256 OR
                   OLD.protocol_version IS DISTINCT FROM NEW.protocol_version OR
                   OLD.rater_plan_sha256 IS DISTINCT FROM NEW.rater_plan_sha256 OR
                   OLD.analysis_plan_sha256 IS DISTINCT FROM NEW.analysis_plan_sha256 OR
                   OLD.submitted_endpoint_model_id IS DISTINCT FROM
                       NEW.submitted_endpoint_model_id OR
                   OLD.submitted_model_card_sha256 IS DISTINCT FROM
                       NEW.submitted_model_card_sha256 OR
                   OLD.data_policy_sha256 IS DISTINCT FROM NEW.data_policy_sha256 OR
                   OLD.model_roster_json IS DISTINCT FROM NEW.model_roster_json OR
                   OLD.model_roster_sha256 IS DISTINCT FROM NEW.model_roster_sha256 OR
                   OLD.task_schedule_sha256 IS DISTINCT FROM NEW.task_schedule_sha256 OR
                   OLD.budget_cap_micros IS DISTINCT FROM NEW.budget_cap_micros OR
                   OLD.run_card_json IS DISTINCT FROM NEW.run_card_json OR
                   OLD.run_card_sha256 IS DISTINCT FROM NEW.run_card_sha256 OR
                   OLD.run_card_signature IS DISTINCT FROM NEW.run_card_signature THEN
                    RAISE EXCEPTION 'controlled-run contract is immutable; create a new run';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_controlled_run_contract_immutable ON controlled_runs;
            CREATE TRIGGER trg_controlled_run_contract_immutable
            BEFORE UPDATE ON controlled_runs
            FOR EACH ROW EXECUTE FUNCTION flavourbench_prevent_controlled_run_contract_update();
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_controlled_assignment_mutation()
            RETURNS trigger AS $$
            BEGIN
                IF TG_OP = 'DELETE' THEN
                    RAISE EXCEPTION 'controlled-run assignments are append-only';
                END IF;
                IF OLD.controlled_run_id IS DISTINCT FROM NEW.controlled_run_id OR
                   OLD.ordinal IS DISTINCT FROM NEW.ordinal OR
                   OLD.task_id IS DISTINCT FROM NEW.task_id OR
                   OLD.task_public_id IS DISTINCT FROM NEW.task_public_id OR
                   OLD.task_revision IS DISTINCT FROM NEW.task_revision OR
                   OLD.task_prompt_sha256 IS DISTINCT FROM NEW.task_prompt_sha256 OR
                   OLD.task_family IS DISTINCT FROM NEW.task_family OR
                   OLD.track IS DISTINCT FROM NEW.track OR
                   OLD.model_ids_json IS DISTINCT FROM NEW.model_ids_json OR
                   OLD.repetition_index IS DISTINCT FROM NEW.repetition_index OR
                   OLD.assignment_sha256 IS DISTINCT FROM NEW.assignment_sha256 OR
                   OLD.assignment_seed IS DISTINCT FROM NEW.assignment_seed THEN
                    RAISE EXCEPTION 'controlled-run assignment content is immutable';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_controlled_assignment_immutable
                ON controlled_run_assignments;
            CREATE TRIGGER trg_controlled_assignment_immutable
            BEFORE UPDATE OR DELETE ON controlled_run_assignments
            FOR EACH ROW EXECUTE FUNCTION flavourbench_prevent_controlled_assignment_mutation();
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_snapshot_content_update()
            RETURNS trigger AS $$
            BEGIN
                IF TG_OP = 'DELETE' THEN
                    RAISE EXCEPTION 'leaderboard snapshots are append-only; withdraw or supersede';
                END IF;
                IF OLD.season_id IS DISTINCT FROM NEW.season_id OR
                   OLD.track IS DISTINCT FROM NEW.track OR
                   OLD.cohort IS DISTINCT FROM NEW.cohort OR
                   OLD.category IS DISTINCT FROM NEW.category OR
                   OLD.data_stratum IS DISTINCT FROM NEW.data_stratum OR
                   OLD.controlled_run_id IS DISTINCT FROM NEW.controlled_run_id OR
                   OLD.input_sha256 IS DISTINCT FROM NEW.input_sha256 OR
                   OLD.input_evidence_sha256 IS DISTINCT FROM NEW.input_evidence_sha256 OR
                   OLD.input_evidence_json IS DISTINCT FROM NEW.input_evidence_json OR
                   OLD.payload_sha256 IS DISTINCT FROM NEW.payload_sha256 OR
                   OLD.supersedes_snapshot_id IS DISTINCT FROM NEW.supersedes_snapshot_id OR
                   OLD.payload_json IS DISTINCT FROM NEW.payload_json THEN
                    RAISE EXCEPTION 'leaderboard snapshot content is immutable';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_leaderboard_snapshot_content_immutable
                ON leaderboard_snapshots;
            CREATE TRIGGER trg_leaderboard_snapshot_content_immutable
            BEFORE UPDATE OR DELETE ON leaderboard_snapshots
            FOR EACH ROW EXECUTE FUNCTION flavourbench_prevent_snapshot_content_update();
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_frozen_task_mutation()
            RETURNS trigger AS $$
            DECLARE season_frozen_at timestamptz;
            BEGIN
                IF TG_OP = 'INSERT' THEN
                    SELECT frozen_at INTO season_frozen_at
                    FROM seasons WHERE id = NEW.season_id;
                    IF season_frozen_at IS NOT NULL THEN
                        RAISE EXCEPTION 'cannot insert into a frozen task registry';
                    END IF;
                    RETURN NEW;
                END IF;
                IF OLD.review_status = 'frozen' THEN
                    RAISE EXCEPTION 'frozen task content is immutable';
                END IF;
                IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_frozen_task_immutable ON tasks;
            CREATE TRIGGER trg_frozen_task_immutable
            BEFORE INSERT OR UPDATE OR DELETE ON tasks
            FOR EACH ROW EXECUTE FUNCTION flavourbench_prevent_frozen_task_mutation();
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_frozen_season_mutation()
            RETURNS trigger AS $$
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
                    OLD.protocol_bundle_json IS DISTINCT FROM NEW.protocol_bundle_json OR
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
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_frozen_season_immutable ON seasons;
            CREATE TRIGGER trg_frozen_season_immutable
            BEFORE UPDATE ON seasons
            FOR EACH ROW EXECUTE FUNCTION flavourbench_prevent_frozen_season_mutation();
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_frozen_roster_insert_delete()
            RETURNS trigger AS $$
            DECLARE target_season_id text;
            DECLARE season_frozen_at timestamptz;
            BEGIN
                target_season_id := CASE
                    WHEN TG_OP = 'DELETE' THEN OLD.season_id
                    ELSE NEW.season_id
                END;
                SELECT frozen_at INTO season_frozen_at
                FROM seasons WHERE id = target_season_id;
                IF season_frozen_at IS NOT NULL THEN
                    RAISE EXCEPTION 'frozen season roster cannot add or delete endpoints';
                END IF;
                IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS trg_frozen_roster_insert_delete ON season_models;
            CREATE TRIGGER trg_frozen_roster_insert_delete
            BEFORE INSERT OR DELETE ON season_models
            FOR EACH ROW EXECUTE FUNCTION flavourbench_prevent_frozen_roster_insert_delete();
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_reject_evidence_delete()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'FlavourBench evidence records cannot be deleted';
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        for table in (
            "battles",
            "response_arms",
            "tool_calls",
            "validator_results",
            "run_events",
            "incidents",
            "controlled_runs",
        ):
            trigger = f"trg_{table}_reject_delete"
            op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
            op.execute(
                f"""
                CREATE TRIGGER {trigger}
                BEFORE DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION flavourbench_reject_evidence_delete();
                """
            )
        for table in ("votes", "cost_events"):
            trigger = f"trg_{table}_reject_mutation"
            op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
            op.execute(
                f"""
                CREATE TRIGGER {trigger}
                BEFORE UPDATE OR DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION flavourbench_reject_evidence_delete();
                """
            )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for table in (
            "battles",
            "response_arms",
            "tool_calls",
            "votes",
            "validator_results",
            "cost_events",
            "run_events",
            "incidents",
            "controlled_runs",
        ):
            op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_reject_delete ON {table}")
        for table in ("votes", "cost_events"):
            op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_reject_mutation ON {table}")
        for trigger, table in (
            ("trg_response_arm_contract_immutable", "response_arms"),
            ("trg_frozen_roster_insert_delete", "season_models"),
            ("trg_frozen_season_immutable", "seasons"),
            ("trg_frozen_task_immutable", "tasks"),
            ("trg_leaderboard_snapshot_content_immutable", "leaderboard_snapshots"),
            ("trg_controlled_assignment_immutable", "controlled_run_assignments"),
            ("trg_controlled_run_contract_immutable", "controlled_runs"),
        ):
            op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        for function in (
            "flavourbench_prevent_response_arm_contract_update",
            "flavourbench_prevent_frozen_roster_insert_delete",
            "flavourbench_prevent_frozen_season_mutation",
            "flavourbench_prevent_frozen_task_mutation",
            "flavourbench_prevent_snapshot_content_update",
            "flavourbench_prevent_controlled_assignment_mutation",
            "flavourbench_prevent_controlled_run_contract_update",
        ):
            op.execute(f"DROP FUNCTION IF EXISTS {function}()")
        op.execute("DROP FUNCTION IF EXISTS flavourbench_reject_evidence_delete()")
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_battle_provenance_update()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.season_id IS DISTINCT FROM NEW.season_id OR
                   OLD.run_class IS DISTINCT FROM NEW.run_class OR
                   OLD.rank_eligible IS DISTINCT FROM NEW.rank_eligible OR
                   OLD.data_stratum IS DISTINCT FROM NEW.data_stratum OR
                   OLD.task_id IS DISTINCT FROM NEW.task_id OR
                   OLD.task_revision IS DISTINCT FROM NEW.task_revision OR
                   OLD.controlled_run_id IS DISTINCT FROM NEW.controlled_run_id OR
                   OLD.manifest_sha256 IS DISTINCT FROM NEW.manifest_sha256 OR
                   OLD.scheduler_version IS DISTINCT FROM NEW.scheduler_version OR
                   OLD.assignment_seed IS DISTINCT FROM NEW.assignment_seed OR
                   OLD.track_assignment_probability IS DISTINCT FROM
                       NEW.track_assignment_probability OR
                   OLD.model_assignment_probability IS DISTINCT FROM
                       NEW.model_assignment_probability OR
                   OLD.side_assignment_probability IS DISTINCT FROM
                       NEW.side_assignment_probability OR
                   OLD.track IS DISTINCT FROM NEW.track OR
                   OLD.category IS DISTINCT FROM NEW.category OR
                   OLD.prompt_sha256 IS DISTINCT FROM NEW.prompt_sha256 OR
                   OLD.client_nonce_sha256 IS DISTINCT FROM NEW.client_nonce_sha256 THEN
                    RAISE EXCEPTION
                        'battle scientific provenance is immutable; insert a superseding battle';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION flavourbench_prevent_endpoint_contract_update()
            RETURNS trigger AS $$
            BEGIN
                IF OLD.manifest_sha256 NOT IN ('', 'unfrozen', 'unresolved')
                   AND (
                       OLD.model_id IS DISTINCT FROM NEW.model_id OR
                       OLD.slot_role IS DISTINCT FROM NEW.slot_role OR
                       OLD.provider_slug IS DISTINCT FROM NEW.provider_slug OR
                       OLD.expected_actual_model_id IS DISTINCT FROM NEW.expected_actual_model_id OR
                       OLD.expected_actual_provider_slug IS DISTINCT FROM
                           NEW.expected_actual_provider_slug OR
                       OLD.supported_parameters_json IS DISTINCT FROM
                           NEW.supported_parameters_json OR
                       OLD.decoding_json IS DISTINCT FROM NEW.decoding_json OR
                       OLD.endpoint_max_completion_tokens IS DISTINCT FROM
                           NEW.endpoint_max_completion_tokens OR
                       OLD.endpoint_document_sha256 IS DISTINCT FROM NEW.endpoint_document_sha256 OR
                       OLD.endpoint_contract_sha256 IS DISTINCT FROM NEW.endpoint_contract_sha256 OR
                       OLD.worst_case_cost_micros IS DISTINCT FROM NEW.worst_case_cost_micros OR
                       OLD.manifest_sha256 IS DISTINCT FROM NEW.manifest_sha256
                   ) THEN
                    RAISE EXCEPTION
                        'frozen season endpoint contract is immutable; create a new manifest';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )

    if "uq_cost_events_battle_reconcile" in _indexes(bind, "cost_events"):
        op.drop_index("uq_cost_events_battle_reconcile", table_name="cost_events")

    if "response_arms" in sa.inspect(bind).get_table_names():
        arm_indexes = _indexes(bind, "response_arms")
        if "ix_response_arms_protocol_bundle_sha256" in arm_indexes:
            op.drop_index(
                "ix_response_arms_protocol_bundle_sha256",
                table_name="response_arms",
            )
        with op.batch_alter_table("response_arms") as batch:
            for name in (
                "epicure_attestation_sha256",
                "epicure_attestation_json",
                "protocol_bundle_sha256",
                "observed_decoding_json",
            ):
                if name in _columns(bind, "response_arms"):
                    batch.drop_column(name)

    if "battles" in sa.inspect(bind).get_table_names():
        battle_indexes = _indexes(bind, "battles")
        if "ix_battles_protocol_bundle_sha256" in battle_indexes:
            op.drop_index(
                "ix_battles_protocol_bundle_sha256",
                table_name="battles",
            )
        if "protocol_bundle_sha256" in _columns(bind, "battles"):
            with op.batch_alter_table("battles") as batch:
                batch.drop_column("protocol_bundle_sha256")

    if "seasons" in sa.inspect(bind).get_table_names():
        with op.batch_alter_table("seasons") as batch:
            for name in (
                "protocol_bundle_sha256",
                "protocol_bundle_json",
                "analysis_plan_sha256",
            ):
                if name in _columns(bind, "seasons"):
                    batch.drop_column(name)

    snapshot_indexes = _indexes(bind, "leaderboard_snapshots")
    if "ix_leaderboard_snapshots_supersedes_snapshot_id" in snapshot_indexes:
        op.drop_index(
            "ix_leaderboard_snapshots_supersedes_snapshot_id",
            table_name="leaderboard_snapshots",
        )
    snapshot_fks = sa.inspect(bind).get_foreign_keys("leaderboard_snapshots")
    with op.batch_alter_table("leaderboard_snapshots") as batch:
        for fk in snapshot_fks:
            if (
                fk.get("constrained_columns") == ["supersedes_snapshot_id"]
                and fk.get("name")
            ):
                batch.drop_constraint(str(fk["name"]), type_="foreignkey")
        for name in (
            "supersedes_snapshot_id",
            "payload_sha256",
            "input_evidence_json",
            "input_evidence_sha256",
        ):
            if name in _columns(bind, "leaderboard_snapshots"):
                batch.drop_column(name)

    with op.batch_alter_table("tool_calls") as batch:
        if "uq_tool_calls_arm_round_call" in _unique_constraints(bind, "tool_calls"):
            batch.drop_constraint("uq_tool_calls_arm_round_call", type_="unique")
        for name in ("structured_content_json", "tool_call_id", "call_index"):
            if name in _columns(bind, "tool_calls"):
                batch.drop_column(name)

    with op.batch_alter_table("season_models") as batch:
        for name in ("rate_card_sha256", "rate_card_json"):
            if name in _columns(bind, "season_models"):
                batch.drop_column(name)

    if "controlled_run_assignments" in sa.inspect(bind).get_table_names():
        op.drop_table("controlled_run_assignments")

    with op.batch_alter_table("controlled_runs") as batch:
        for name in (
            "revoked_at",
            "closed_at",
            "collection_completed_at",
            "token_version",
            "task_schedule_sha256",
            "model_roster_sha256",
            "model_roster_json",
            "data_policy_sha256",
            "submitted_model_card_sha256",
            "submitted_endpoint_model_id",
        ):
            if name in _columns(bind, "controlled_runs"):
                batch.drop_column(name)

    if "catalog_models" in sa.inspect(bind).get_table_names():
        catalog_indexes = _indexes(bind, "catalog_models")
        if "ix_catalog_models_catalog_source" in catalog_indexes:
            op.drop_index(
                "ix_catalog_models_catalog_source",
                table_name="catalog_models",
            )
        with op.batch_alter_table("catalog_models") as batch:
            for name in ("open_weight_evidence_json", "catalog_source"):
                if name in _columns(bind, "catalog_models"):
                    batch.drop_column(name)
