"""Bind battle arms and seal terminal commercial evidence.

Revision ID: 0014_commercial_evidence_invariants
Revises: 0013_snapshot_evidence_seal
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0014_commercial_evidence_invariants"
down_revision = "0013_snapshot_evidence_seal"
branch_labels = None
depends_on = None


BATTLE_LINK_TRIGGER = "trg_battle_arm_link_guard"
BATTLE_LINK_FUNCTION = "flavourbench_battle_arm_link_guard"
ARM_GUARD_TRIGGER = "trg_response_arm_evidence_guard"
ARM_GUARD_FUNCTION = "flavourbench_response_arm_evidence_guard"
ASSIGNMENT_GUARD_TRIGGER = "trg_controlled_assignment_lifecycle_guard"
ASSIGNMENT_GUARD_FUNCTION = "flavourbench_controlled_assignment_lifecycle_guard"
CONTROLLED_RUN_GUARD_TRIGGER = "trg_controlled_run_lifecycle_guard"
CONTROLLED_RUN_GUARD_FUNCTION = "flavourbench_controlled_run_lifecycle_guard"
VOTE_GUARD_TRIGGER = "trg_vote_evidence_guard"
VOTE_GUARD_FUNCTION = "flavourbench_vote_evidence_guard"
COST_EVENT_GUARD_TRIGGER = "trg_cost_event_append_only_guard"
COST_EVENT_GUARD_FUNCTION = "flavourbench_cost_event_append_only_guard"
JOB_GUARD_TRIGGER = "trg_job_terminal_evidence_guard"
JOB_GUARD_FUNCTION = "flavourbench_job_terminal_evidence_guard"
RUN_EVENT_GUARD_TRIGGER = "trg_run_event_evidence_guard"
RUN_EVENT_GUARD_FUNCTION = "flavourbench_run_event_evidence_guard"
INCIDENT_GUARD_TRIGGER = "trg_incident_evidence_guard"
INCIDENT_GUARD_FUNCTION = "flavourbench_incident_evidence_guard"
ARM_DIGEST_TRIGGER = "trg_response_arm_output_digest_write_once"
APPEND_ONLY_EVIDENCE_FUNCTION = "flavourbench_append_only_evidence_guard"
APPEND_ONLY_EVIDENCE_TABLES = (
    "generation_attempts",
    "admission_events",
    "bedrock_billing_crosschecks",
    "bedrock_billing_crosscheck_arms",
)


def _scalar_count(bind: sa.Connection, query: str) -> int:
    return int(bind.execute(sa.text(query)).scalar_one())


def _preflight(bind: sa.Connection) -> None:
    required_columns = {
        "battles": {
            "id",
            "season_id",
            "run_class",
            "rank_eligible",
            "status",
            "left_arm_id",
            "right_arm_id",
            "controlled_run_id",
            "data_stratum",
            "task_id",
            "task_revision",
            "prompt",
            "prompt_sha256",
            "prompt_redacted",
            "category",
            "track",
            "assignment_seed",
            "scheduler_version",
            "track_assignment_probability",
            "model_assignment_probability",
            "side_assignment_probability",
            "completed_at",
            "manifest_sha256",
            "protocol_bundle_sha256",
            "provider_reservations_json",
            "client_nonce_sha256",
            "research_consent",
            "requester_pseudonym",
            "reserved_cost_micros",
            "created_at",
            "retention_until",
        },
        "response_arms": {
            "id",
            "battle_id",
            "side",
            "condition",
            "model_id",
            "execution_backend",
            "provider_slug",
            "actual_provider_slug",
            "actual_model_id",
            "generation_id",
            "provider_generation_ids_json",
            "status",
            "answer_markdown",
            "answer_markdown_sha256",
            "output_json",
            "output_json_sha256",
            "prompt_sha256",
            "system_prompt_sha256",
            "schema_sha256",
            "tool_schema_sha256",
            "decoding_json",
            "observed_decoding_json",
            "protocol_bundle_sha256",
            "epicure_release_id",
            "epicure_bundle_sha256",
            "epicure_application_sha256",
            "epicure_attestation_json",
            "epicure_attestation_sha256",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cost_micros",
            "cost_reconciled",
            "cost_accounting_basis",
            "billing_reconciliation_status",
            "backend_response_schema_sha256",
            "backend_tool_schema_sha256",
            "latency_ms",
            "retries",
            "finish_reason",
            "error_code",
            "error_detail",
            "created_at",
            "completed_at",
        },
        "controlled_runs": {
            "id",
            "season_id",
            "organization_reference_sha256",
            "status",
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
            "budget_reserved_micros",
            "run_card_json",
            "run_card_sha256",
            "run_card_signature",
            "collection_completed_at",
            "closed_at",
            "revoked_at",
            "created_at",
        },
        "controlled_run_assignments": {
            "id",
            "controlled_run_id",
            "ordinal",
            "battle_id",
            "status",
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
        },
        "votes": {"id", "battle_id", "choice", "cohort", "created_at"},
        "jobs": {
            "id",
            "battle_id",
            "kind",
            "payload_json",
            "status",
            "created_at",
            "completed_at",
        },
        "cost_events": {
            "id",
            "battle_id",
            "arm_id",
            "kind",
            "amount_micros",
            "generation_id",
            "accounting_json",
            "created_at",
        },
        "generation_attempts": {
            "id",
            "attempt_id",
            "arm_id",
            "event_type",
            "payload_sha256",
            "created_at",
        },
        "admission_events": {
            "id",
            "pseudonym",
            "action",
            "admitted",
            "reason",
            "created_at",
        },
        "bedrock_billing_crosschecks": {
            "id",
            "season_id",
            "provider_account_budget_id",
            "supersedes_crosscheck_id",
            "evidence_sha256",
            "created_at",
        },
        "bedrock_billing_crosscheck_arms": {
            "id",
            "crosscheck_id",
            "arm_id",
            "generation_set_sha256",
            "created_at",
        },
        "run_events": {
            "id",
            "entity_type",
            "entity_id",
            "event_type",
            "payload_json",
            "created_at",
        },
        "incidents": {
            "id",
            "severity",
            "code",
            "detail",
            "battle_id",
            "created_at",
        },
    }
    inspector = sa.inspect(bind)
    available_tables = set(inspector.get_table_names())
    for table, expected in required_columns.items():
        if table not in available_tables:
            raise RuntimeError(f"0014 integrity preflight requires table {table}")
        available = {column["name"] for column in inspector.get_columns(table)}
        missing = sorted(expected - available)
        if missing:
            raise RuntimeError(
                f"0014 integrity preflight requires {table} columns: {', '.join(missing)}"
            )
    checks = {
        "battles contain malformed prompt-retention state": """
            SELECT COUNT(*) FROM battles
            WHERE prompt_redacted IS NULL
               OR (prompt_redacted IS TRUE AND prompt IS NOT NULL)
               OR (prompt_redacted IS NOT TRUE AND prompt IS NULL)
        """,
        "battles contain duplicate arm links": """
            SELECT COUNT(*) FROM battles
            WHERE left_arm_id IS NOT NULL
              AND right_arm_id IS NOT NULL
              AND left_arm_id = right_arm_id
        """,
        "battle arm links do not match arm ownership and side": """
            SELECT COUNT(*) FROM battles AS b
            WHERE (
                b.left_arm_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = b.left_arm_id
                      AND a.battle_id = b.id
                      AND a.side = 'left'
                )
            ) OR (
                b.right_arm_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = b.right_arm_id
                      AND a.battle_id = b.id
                      AND a.side = 'right'
                )
            )
        """,
        "running or terminal battles do not have two owned arm links": """
            SELECT COUNT(*) FROM battles AS b
            WHERE b.status IN ('running', 'complete', 'failed') AND (
                b.left_arm_id IS NULL
                OR b.right_arm_id IS NULL
                OR b.left_arm_id = b.right_arm_id
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = b.left_arm_id
                      AND a.battle_id = b.id
                      AND a.side = 'left'
                )
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = b.right_arm_id
                      AND a.battle_id = b.id
                      AND a.side = 'right'
                )
                OR (b.status IN ('complete', 'failed') AND b.completed_at IS NULL)
            )
        """,
        "response arms contain invalid sides": """
            SELECT COUNT(*) FROM response_arms WHERE side NOT IN ('left', 'right')
        """,
        "response arms contain invalid statuses": """
            SELECT COUNT(*) FROM response_arms
            WHERE status NOT IN ('queued', 'running', 'complete', 'failed', 'uncertain')
        """,
        "response arms contain malformed lifecycle or terminal evidence": """
            SELECT COUNT(*) FROM response_arms
            WHERE (status IN ('queued', 'running') AND completed_at IS NOT NULL)
               OR (status IN ('complete', 'failed', 'uncertain') AND (
                    completed_at IS NULL
                    OR completed_at < created_at
                    OR output_json_sha256 IS NULL
               ))
               OR (status = 'complete' AND (
                    (
                        answer_markdown IS NULL
                        AND NOT EXISTS (
                            SELECT 1 FROM battles AS b
                            WHERE b.id = response_arms.battle_id
                              AND b.prompt_redacted IS TRUE
                              AND b.prompt IS NULL
                        )
                    )
                    OR answer_markdown_sha256 IS NULL
                    OR actual_provider_slug IS NULL
                    OR actual_model_id IS NULL
                    OR generation_id IS NULL
                    OR cost_reconciled IS NOT TRUE
                    OR COALESCE(cost_accounting_basis, 'unrecorded') = 'unrecorded'
                    OR COALESCE(billing_reconciliation_status, 'unrecorded') =
                        'unrecorded'
               ))
               OR (status = 'failed' AND (
                    cost_reconciled IS NOT TRUE
                    OR COALESCE(cost_accounting_basis, 'unrecorded') = 'unrecorded'
                    OR COALESCE(billing_reconciliation_status, 'unrecorded') =
                        'unrecorded'
               ))
               OR (status = 'uncertain' AND cost_reconciled IS TRUE)
        """,
        "terminal battles do not match their terminal arm evidence": """
            SELECT COUNT(*) FROM battles AS b
            WHERE b.status IN ('complete', 'failed') AND (
                NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = b.left_arm_id
                      AND a.battle_id = b.id
                      AND a.side = 'left'
                      AND a.completed_at IS NOT NULL
                      AND a.completed_at <= b.completed_at
                      AND (
                          (b.status = 'complete' AND a.status = 'complete')
                          OR (b.status = 'failed' AND a.status IN (
                              'complete', 'failed', 'uncertain'
                          ))
                      )
                )
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = b.right_arm_id
                      AND a.battle_id = b.id
                      AND a.side = 'right'
                      AND a.completed_at IS NOT NULL
                      AND a.completed_at <= b.completed_at
                      AND (
                          (b.status = 'complete' AND a.status = 'complete')
                          OR (b.status = 'failed' AND a.status IN (
                              'complete', 'failed', 'uncertain'
                          ))
                      )
                )
            )
        """,
        "votes contain choices outside the frozen preference domain": """
            SELECT COUNT(*) FROM votes WHERE choice NOT IN ('left', 'right', 'tie', 'both_bad')
        """,
        "votes contain cohorts outside the governed rater domain": """
            SELECT COUNT(*) FROM votes
            WHERE cohort NOT IN (
                'public',
                'expert_independent',
                'expert_product_affiliated',
                'expert_provider_affiliated'
            )
        """,
        "jobs contain malformed lifecycle timestamps": """
            SELECT COUNT(*) FROM jobs
            WHERE status NOT IN ('queued', 'running', 'complete', 'failed', 'uncertain')
               OR (status IN ('queued', 'running') AND completed_at IS NOT NULL)
               OR (status IN ('complete', 'failed', 'uncertain') AND (
                    completed_at IS NULL OR completed_at < created_at
               ))
        """,
        "votes do not follow a valid completed anonymous battle": """
            SELECT COUNT(*) FROM votes AS v
            WHERE NOT EXISTS (
                SELECT 1 FROM battles AS b
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
                WHERE b.id = v.battle_id
                  AND b.status = 'complete'
                  AND b.completed_at IS NOT NULL
                  AND v.created_at >= b.completed_at
                  AND b.left_arm_id IS NOT NULL
                  AND b.right_arm_id IS NOT NULL
                  AND b.left_arm_id <> b.right_arm_id
            )
        """,
        "controlled assignments have invalid binding state": """
            SELECT COUNT(*) FROM controlled_run_assignments
            WHERE (status = 'pending' AND battle_id IS NOT NULL)
               OR (status = 'queued' AND battle_id IS NULL)
        """,
        "controlled runs have inconsistent lifecycle timestamps": """
            SELECT COUNT(*) FROM controlled_runs
            WHERE (status = 'active' AND (
                    collection_completed_at IS NOT NULL
                    OR closed_at IS NOT NULL
                    OR revoked_at IS NOT NULL
                  ))
               OR (status = 'collection_complete' AND (
                    collection_completed_at IS NULL
                    OR closed_at IS NOT NULL
                    OR revoked_at IS NOT NULL
                  ))
               OR (status = 'closed' AND (
                    collection_completed_at IS NULL
                    OR closed_at IS NULL
                    OR revoked_at IS NOT NULL
                  ))
               OR (status = 'revoked' AND revoked_at IS NULL)
        """,
        "queued controlled assignments do not match their battles": """
            SELECT COUNT(*) FROM controlled_run_assignments AS a
            WHERE a.status = 'queued' AND NOT EXISTS (
                SELECT 1 FROM battles AS b
                WHERE b.id = a.battle_id
                  AND b.controlled_run_id = a.controlled_run_id
                  AND b.data_stratum = 'controlled'
                  AND b.task_id = a.task_id
                  AND b.task_revision = a.task_revision
                  AND b.prompt_sha256 = a.task_prompt_sha256
                  AND b.category = a.task_family
                  AND b.track = a.track
                  AND b.assignment_seed = a.assignment_seed
                  AND b.scheduler_version = 'controlled-frozen-schedule-v1'
                  AND b.track_assignment_probability = '1/1'
                  AND b.model_assignment_probability = '1/1'
                  AND b.side_assignment_probability = '1/2'
            )
        """,
        "sealed controlled runs do not have an exact terminal assignment-battle bijection": """
            SELECT COUNT(*) FROM controlled_runs AS r
            WHERE r.status IN ('collection_complete', 'closed') AND (
                NOT EXISTS (
                    SELECT 1 FROM controlled_run_assignments AS a
                    WHERE a.controlled_run_id = r.id
                )
                OR EXISTS (
                    SELECT 1 FROM controlled_run_assignments AS a
                    LEFT JOIN battles AS b ON b.id = a.battle_id
                    LEFT JOIN response_arms AS left_arm
                      ON left_arm.id = b.left_arm_id
                     AND left_arm.battle_id = b.id
                     AND left_arm.side = 'left'
                    LEFT JOIN response_arms AS right_arm
                      ON right_arm.id = b.right_arm_id
                     AND right_arm.battle_id = b.id
                     AND right_arm.side = 'right'
                    WHERE a.controlled_run_id = r.id AND (
                        a.status <> 'queued'
                        OR a.battle_id IS NULL
                        OR b.id IS NULL
                        OR b.controlled_run_id <> r.id
                        OR b.status NOT IN ('complete', 'failed')
                        OR left_arm.id IS NULL
                        OR right_arm.id IS NULL
                        OR left_arm.status NOT IN ('complete', 'failed')
                        OR right_arm.status NOT IN ('complete', 'failed')
                    )
                )
                OR EXISTS (
                    SELECT 1 FROM battles AS b
                    LEFT JOIN controlled_run_assignments AS a
                      ON a.battle_id = b.id AND a.controlled_run_id = r.id
                    WHERE b.controlled_run_id = r.id AND a.id IS NULL
                )
            )
        """,
        "closed controlled runs contain unreconciled cost or job evidence": """
            SELECT COUNT(*) FROM controlled_runs AS r
            WHERE r.status = 'closed' AND (
                r.budget_reserved_micros <> 0
                OR EXISTS (
                    SELECT 1 FROM battles AS b
                    WHERE b.controlled_run_id = r.id
                      AND b.reserved_cost_micros <> 0
                )
                OR EXISTS (
                    SELECT 1 FROM battles AS b
                    JOIN response_arms AS a ON a.battle_id = b.id
                    WHERE b.controlled_run_id = r.id AND (
                        a.cost_reconciled IS NOT TRUE
                        OR COALESCE(a.cost_accounting_basis, 'unrecorded') = 'unrecorded'
                        OR COALESCE(a.billing_reconciliation_status, 'unrecorded') =
                            'unrecorded'
                        OR NOT EXISTS (
                            SELECT 1 FROM cost_events AS ce
                            WHERE ce.arm_id = a.id
                              AND ce.battle_id = b.id
                              AND ce.kind IN ('actual', 'actual_settlement')
                              AND ce.amount_micros = a.cost_micros
                              AND (
                                  ce.generation_id = a.generation_id
                                  OR (ce.generation_id IS NULL AND a.generation_id IS NULL)
                              )
                        )
                    )
                )
                OR EXISTS (
                    SELECT 1 FROM battles AS b
                    WHERE b.controlled_run_id = r.id
                      AND NOT EXISTS (
                          SELECT 1 FROM cost_events AS ce
                          WHERE ce.battle_id = b.id
                            AND ce.kind = 'reconcile'
                            AND ce.amount_micros = (
                                SELECT COALESCE(SUM(a.cost_micros), 0)
                                FROM response_arms AS a
                                WHERE a.battle_id = b.id
                            )
                      )
                )
                OR EXISTS (
                    SELECT 1 FROM battles AS b
                    WHERE b.controlled_run_id = r.id AND (
                        (SELECT COUNT(*) FROM jobs AS j WHERE j.battle_id = b.id) <> 1
                        OR EXISTS (
                            SELECT 1 FROM jobs AS j
                            WHERE j.battle_id = b.id
                              AND (j.status NOT IN ('complete', 'failed')
                                   OR j.completed_at IS NULL)
                        )
                    )
                )
            )
        """,
    }
    if bind.dialect.name == "postgresql":
        checks["complete response arms lack provider generation identity"] = """
            SELECT COUNT(*) FROM response_arms
            WHERE status = 'complete' AND (
                jsonb_typeof(provider_generation_ids_json::jsonb) <> 'array'
                OR jsonb_array_length(provider_generation_ids_json::jsonb) = 0
            )
        """
    elif bind.dialect.name == "sqlite":
        checks["complete response arms lack provider generation identity"] = """
            SELECT COUNT(*) FROM response_arms
            WHERE status = 'complete' AND (
                json_type(provider_generation_ids_json, '$') IS NOT 'array'
                OR json_array_length(provider_generation_ids_json) = 0
            )
        """
    failures = [message for message, query in checks.items() if _scalar_count(bind, query)]
    if failures:
        raise RuntimeError("0014 integrity preflight failed: " + "; ".join(failures))


def _create_checks(bind: sa.Connection) -> None:
    constraints = {
        "battles": (
            (
                "ck_battles_distinct_arm_links",
                "left_arm_id IS NULL OR right_arm_id IS NULL OR left_arm_id <> right_arm_id",
            ),
        ),
        "response_arms": (
            ("ck_response_arms_side", "side IN ('left', 'right')"),
            (
                "ck_response_arms_status",
                "status IN ('queued', 'running', 'complete', 'failed', 'uncertain')",
            ),
        ),
        "votes": (
            ("ck_votes_choice", "choice IN ('left', 'right', 'tie', 'both_bad')"),
            (
                "ck_votes_cohort",
                "cohort IN ('public', 'expert_independent', "
                "'expert_product_affiliated', 'expert_provider_affiliated')",
            ),
        ),
    }
    if bind.dialect.name == "sqlite":
        for table, table_constraints in constraints.items():
            with op.batch_alter_table(table, recreate="always") as batch:
                for name, condition in table_constraints:
                    batch.create_check_constraint(name, condition)
        return
    for table, table_constraints in constraints.items():
        for name, condition in table_constraints:
            op.create_check_constraint(name, table, condition)


def _create_postgresql_guards() -> None:
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {BATTLE_LINK_FUNCTION}() RETURNS trigger AS $$
        DECLARE
            parent_status text;
            lifecycle_changed boolean;
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.controlled_run_id IS NOT NULL THEN
                    SELECT status INTO parent_status
                    FROM controlled_runs
                    WHERE id = NEW.controlled_run_id
                    FOR UPDATE;
                    IF parent_status IS DISTINCT FROM 'active' THEN
                        RAISE EXCEPTION 'controlled battle requires an active parent run';
                    END IF;
                END IF;
                IF NEW.status <> 'queued'
                   OR NEW.left_arm_id IS NOT NULL
                   OR NEW.right_arm_id IS NOT NULL
                   OR NEW.completed_at IS NOT NULL
                   OR NEW.prompt IS NULL
                   OR NEW.prompt_redacted IS DISTINCT FROM FALSE THEN
                    RAISE EXCEPTION
                        'battles must be inserted queued, unredacted, and unlinked';
                END IF;
                RETURN NEW;
            END IF;
            IF OLD.id IS DISTINCT FROM NEW.id
               OR OLD.season_id IS DISTINCT FROM NEW.season_id
               OR OLD.run_class IS DISTINCT FROM NEW.run_class
               OR OLD.rank_eligible IS DISTINCT FROM NEW.rank_eligible
               OR OLD.data_stratum IS DISTINCT FROM NEW.data_stratum
               OR OLD.task_id IS DISTINCT FROM NEW.task_id
               OR OLD.task_revision IS DISTINCT FROM NEW.task_revision
               OR OLD.controlled_run_id IS DISTINCT FROM NEW.controlled_run_id
               OR OLD.manifest_sha256 IS DISTINCT FROM NEW.manifest_sha256
               OR OLD.protocol_bundle_sha256 IS DISTINCT FROM NEW.protocol_bundle_sha256
               OR OLD.scheduler_version IS DISTINCT FROM NEW.scheduler_version
               OR OLD.assignment_seed IS DISTINCT FROM NEW.assignment_seed
               OR OLD.track_assignment_probability IS DISTINCT FROM
                    NEW.track_assignment_probability
               OR OLD.model_assignment_probability IS DISTINCT FROM
                    NEW.model_assignment_probability
               OR OLD.side_assignment_probability IS DISTINCT FROM
                    NEW.side_assignment_probability
               OR OLD.provider_reservations_json::jsonb IS DISTINCT FROM
                    NEW.provider_reservations_json::jsonb
               OR OLD.track IS DISTINCT FROM NEW.track
               OR OLD.category IS DISTINCT FROM NEW.category
               OR OLD.prompt_sha256 IS DISTINCT FROM NEW.prompt_sha256
               OR OLD.client_nonce_sha256 IS DISTINCT FROM NEW.client_nonce_sha256
               OR OLD.research_consent IS DISTINCT FROM NEW.research_consent
               OR OLD.requester_pseudonym IS DISTINCT FROM NEW.requester_pseudonym
               OR OLD.created_at IS DISTINCT FROM NEW.created_at
               OR OLD.retention_until IS DISTINCT FROM NEW.retention_until THEN
                RAISE EXCEPTION 'battle scientific provenance is immutable';
            END IF;
            IF (
                OLD.prompt IS DISTINCT FROM NEW.prompt
                OR OLD.prompt_redacted IS DISTINCT FROM NEW.prompt_redacted
            ) AND NOT (
                OLD.prompt IS NOT NULL
                AND NEW.prompt IS NULL
                AND OLD.prompt_redacted IS FALSE
                AND NEW.prompt_redacted IS TRUE
            ) THEN
                RAISE EXCEPTION 'battle prompt permits only one-way retention redaction';
            END IF;
            lifecycle_changed := OLD.status IS DISTINCT FROM NEW.status
                OR OLD.left_arm_id IS DISTINCT FROM NEW.left_arm_id
                OR OLD.right_arm_id IS DISTINCT FROM NEW.right_arm_id
                OR OLD.completed_at IS DISTINCT FROM NEW.completed_at;
            IF lifecycle_changed AND NEW.controlled_run_id IS NOT NULL THEN
                SELECT status INTO parent_status
                FROM controlled_runs
                WHERE id = NEW.controlled_run_id
                FOR UPDATE;
                IF parent_status IS DISTINCT FROM 'active' THEN
                    RAISE EXCEPTION 'controlled battle lifecycle requires an active parent run';
                END IF;
            END IF;
            IF OLD.status IS DISTINCT FROM NEW.status AND NOT (
                (OLD.status = 'queued' AND NEW.status IN ('running', 'complete', 'failed'))
                OR (OLD.status = 'running' AND NEW.status IN ('complete', 'failed'))
            ) THEN
                RAISE EXCEPTION 'battle lifecycle transition is invalid';
            END IF;
            IF OLD.completed_at IS DISTINCT FROM NEW.completed_at AND NOT (
                OLD.status IN ('queued', 'running')
                AND NEW.status IN ('complete', 'failed')
                AND OLD.completed_at IS NULL
                AND NEW.completed_at IS NOT NULL
            ) THEN
                RAISE EXCEPTION 'battle completion timestamp is write-once';
            END IF;
            IF NEW.status IN ('queued', 'running') AND NEW.completed_at IS NOT NULL THEN
                RAISE EXCEPTION 'nonterminal battle cannot carry a completion timestamp';
            END IF;
            IF OLD.left_arm_id IS DISTINCT FROM NEW.left_arm_id THEN
                IF OLD.left_arm_id IS NOT NULL OR NEW.left_arm_id IS NULL OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.left_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'left'
                ) THEN
                    RAISE EXCEPTION 'battle left arm link is invalid or has been rebound';
                END IF;
            END IF;
            IF OLD.right_arm_id IS DISTINCT FROM NEW.right_arm_id THEN
                IF OLD.right_arm_id IS NOT NULL OR NEW.right_arm_id IS NULL OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.right_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'right'
                ) THEN
                    RAISE EXCEPTION 'battle right arm link is invalid or has been rebound';
                END IF;
            END IF;
            IF NEW.left_arm_id IS NOT NULL AND NEW.right_arm_id IS NOT NULL
               AND NEW.left_arm_id = NEW.right_arm_id THEN
                RAISE EXCEPTION 'battle arm links must be distinct';
            END IF;
            IF NEW.status IN ('running', 'complete', 'failed') AND (
                NEW.left_arm_id IS NULL
                OR NEW.right_arm_id IS NULL
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.left_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'left'
                )
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.right_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'right'
                )
            ) THEN
                RAISE EXCEPTION 'running or terminal battles require two owned arm links';
            END IF;
            IF NEW.status IN ('complete', 'failed') AND (
                NEW.completed_at IS NULL
                OR NEW.completed_at < NEW.created_at
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.left_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'left'
                      AND a.completed_at IS NOT NULL
                      AND NEW.completed_at >= a.completed_at
                      AND (
                          (NEW.status = 'complete' AND a.status = 'complete')
                          OR (NEW.status = 'failed' AND a.status IN (
                              'complete', 'failed', 'uncertain'
                          ))
                      )
                )
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.right_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'right'
                      AND a.completed_at IS NOT NULL
                      AND NEW.completed_at >= a.completed_at
                      AND (
                          (NEW.status = 'complete' AND a.status = 'complete')
                          OR (NEW.status = 'failed' AND a.status IN (
                              'complete', 'failed', 'uncertain'
                          ))
                      )
                )
            ) THEN
                RAISE EXCEPTION 'terminal battle record is incomplete';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {BATTLE_LINK_TRIGGER} ON battles;
        CREATE TRIGGER {BATTLE_LINK_TRIGGER}
        BEFORE INSERT OR UPDATE ON battles
        FOR EACH ROW EXECUTE FUNCTION {BATTLE_LINK_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {ARM_GUARD_FUNCTION}() RETURNS trigger AS $$
        DECLARE
            settlement boolean;
            settlement_core_changed boolean;
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.status <> 'queued'
                   OR NEW.completed_at IS NOT NULL
                   OR NEW.actual_provider_slug IS NOT NULL
                   OR NEW.actual_model_id IS NOT NULL
                   OR NEW.generation_id IS NOT NULL
                   OR NEW.answer_markdown IS NOT NULL
                   OR NEW.answer_markdown_sha256 IS NOT NULL
                   OR NEW.output_json_sha256 IS NOT NULL
                   OR NEW.cost_micros <> 0
                   OR NEW.cost_reconciled IS TRUE THEN
                    RAISE EXCEPTION 'response arms must be inserted as empty queued records';
                END IF;
                RETURN NEW;
            ELSIF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'response arms are append-only';
            END IF;
            IF OLD.id IS DISTINCT FROM NEW.id
               OR OLD.battle_id IS DISTINCT FROM NEW.battle_id
               OR OLD.side IS DISTINCT FROM NEW.side
               OR OLD.condition IS DISTINCT FROM NEW.condition
               OR OLD.model_id IS DISTINCT FROM NEW.model_id
               OR OLD.execution_backend IS DISTINCT FROM NEW.execution_backend
               OR OLD.provider_slug IS DISTINCT FROM NEW.provider_slug
               OR OLD.prompt_sha256 IS DISTINCT FROM NEW.prompt_sha256
               OR OLD.system_prompt_sha256 IS DISTINCT FROM NEW.system_prompt_sha256
               OR OLD.schema_sha256 IS DISTINCT FROM NEW.schema_sha256
               OR OLD.tool_schema_sha256 IS DISTINCT FROM NEW.tool_schema_sha256
               OR OLD.decoding_json::jsonb IS DISTINCT FROM NEW.decoding_json::jsonb
               OR OLD.protocol_bundle_sha256 IS DISTINCT FROM NEW.protocol_bundle_sha256
               OR OLD.epicure_release_id IS DISTINCT FROM NEW.epicure_release_id
               OR OLD.epicure_bundle_sha256 IS DISTINCT FROM NEW.epicure_bundle_sha256
               OR OLD.epicure_application_sha256 IS DISTINCT FROM
                    NEW.epicure_application_sha256 THEN
                RAISE EXCEPTION 'response-arm execution contract is immutable';
            END IF;
            IF OLD.status NOT IN ('complete', 'failed', 'uncertain')
               AND OLD.status IS DISTINCT FROM NEW.status
               AND NOT (
                    (OLD.status = 'queued' AND NEW.status IN (
                        'running', 'complete', 'failed', 'uncertain'
                    )) OR
                    (OLD.status = 'running' AND NEW.status IN (
                        'complete', 'failed', 'uncertain'
                    ))
               ) THEN
                RAISE EXCEPTION 'response-arm lifecycle transition is invalid';
            END IF;
            IF OLD.status IS DISTINCT FROM NEW.status
               AND NEW.status IN ('complete', 'failed', 'uncertain')
               AND (
                    NEW.completed_at IS NULL
                    OR NEW.completed_at < NEW.created_at
                    OR NEW.output_json_sha256 IS NULL
                    OR (
                        NEW.status = 'complete' AND (
                            NEW.answer_markdown IS NULL
                            OR NEW.answer_markdown_sha256 IS NULL
                            OR NEW.actual_provider_slug IS NULL
                            OR NEW.actual_model_id IS NULL
                            OR NEW.generation_id IS NULL
                            OR jsonb_typeof(NEW.provider_generation_ids_json::jsonb) <>
                                'array'
                            OR jsonb_array_length(NEW.provider_generation_ids_json::jsonb) = 0
                            OR NEW.cost_reconciled IS NOT TRUE
                            OR COALESCE(NEW.cost_accounting_basis, 'unrecorded') =
                                'unrecorded'
                            OR COALESCE(
                                NEW.billing_reconciliation_status,
                                'unrecorded'
                            ) = 'unrecorded'
                        )
                    )
                    OR (
                        NEW.status = 'failed' AND (
                            NEW.cost_reconciled IS NOT TRUE
                            OR COALESCE(NEW.cost_accounting_basis, 'unrecorded') =
                                'unrecorded'
                            OR COALESCE(
                                NEW.billing_reconciliation_status,
                                'unrecorded'
                            ) = 'unrecorded'
                        )
                    )
                    OR (NEW.status = 'uncertain' AND NEW.cost_reconciled IS TRUE)
               ) THEN
                RAISE EXCEPTION 'terminal response-arm record is incomplete';
            END IF;
            IF NEW.status IN ('queued', 'running') AND NEW.completed_at IS NOT NULL THEN
                RAISE EXCEPTION 'nonterminal response arm cannot carry a completion timestamp';
            END IF;
            IF OLD.status NOT IN ('complete', 'failed', 'uncertain') THEN
                RETURN NEW;
            END IF;

            IF OLD.completed_at IS DISTINCT FROM NEW.completed_at THEN
                RAISE EXCEPTION 'terminal response-arm completion timestamp is immutable';
            END IF;

            IF OLD.actual_provider_slug IS DISTINCT FROM NEW.actual_provider_slug
               OR OLD.actual_model_id IS DISTINCT FROM NEW.actual_model_id
               OR OLD.generation_id IS DISTINCT FROM NEW.generation_id
               OR OLD.provider_generation_ids_json::jsonb IS DISTINCT FROM
                    NEW.provider_generation_ids_json::jsonb
               OR OLD.observed_decoding_json::jsonb IS DISTINCT FROM
                    NEW.observed_decoding_json::jsonb
               OR OLD.epicure_attestation_json::jsonb IS DISTINCT FROM
                    NEW.epicure_attestation_json::jsonb
               OR OLD.epicure_attestation_sha256 IS DISTINCT FROM
                    NEW.epicure_attestation_sha256
               OR OLD.prompt_tokens IS DISTINCT FROM NEW.prompt_tokens
               OR OLD.completion_tokens IS DISTINCT FROM NEW.completion_tokens
               OR OLD.reasoning_tokens IS DISTINCT FROM NEW.reasoning_tokens
               OR OLD.backend_response_schema_sha256 IS DISTINCT FROM
                    NEW.backend_response_schema_sha256
               OR OLD.backend_tool_schema_sha256 IS DISTINCT FROM
                    NEW.backend_tool_schema_sha256
               OR OLD.latency_ms IS DISTINCT FROM NEW.latency_ms
               OR OLD.retries IS DISTINCT FROM NEW.retries
               OR OLD.finish_reason IS DISTINCT FROM NEW.finish_reason
               OR OLD.answer_markdown_sha256 IS DISTINCT FROM NEW.answer_markdown_sha256
               OR OLD.output_json_sha256 IS DISTINCT FROM NEW.output_json_sha256
               OR OLD.created_at IS DISTINCT FROM NEW.created_at THEN
                RAISE EXCEPTION 'terminal response-arm evidence is immutable';
            END IF;

            settlement := COALESCE((OLD.status = 'uncertain'
                AND NEW.status = 'failed'
                AND NEW.cost_reconciled IS TRUE
                AND NEW.cost_micros >= 0
                AND NEW.cost_accounting_basis = 'manual_authorized_settlement'
                AND NEW.billing_reconciliation_status = 'manual_authorized_settlement'
                AND NEW.error_code = 'CostExposureSettled'
                AND NEW.error_detail =
                    'Provider cost exposure was settled by an authorized record.'
                AND NEW.completed_at IS NOT DISTINCT FROM OLD.completed_at), FALSE);
            settlement_core_changed :=
                OLD.status IS DISTINCT FROM NEW.status
                OR OLD.cost_micros IS DISTINCT FROM NEW.cost_micros
                OR OLD.cost_reconciled IS DISTINCT FROM NEW.cost_reconciled
                OR OLD.cost_accounting_basis IS DISTINCT FROM NEW.cost_accounting_basis
                OR OLD.billing_reconciliation_status IS DISTINCT FROM
                    NEW.billing_reconciliation_status
                OR OLD.error_code IS DISTINCT FROM NEW.error_code
                OR OLD.completed_at IS DISTINCT FROM NEW.completed_at;
            IF settlement_core_changed AND settlement IS NOT TRUE THEN
                RAISE EXCEPTION 'terminal response-arm result is immutable';
            END IF;
            IF (
                OLD.answer_markdown IS DISTINCT FROM NEW.answer_markdown
                OR OLD.output_json::jsonb IS DISTINCT FROM NEW.output_json::jsonb
                OR (
                    OLD.error_detail IS DISTINCT FROM NEW.error_detail
                    AND settlement IS NOT TRUE
                )
            ) AND NOT EXISTS (
                SELECT 1 FROM battles AS b
                WHERE b.id = NEW.battle_id
                  AND b.prompt_redacted IS TRUE
                  AND b.prompt IS NULL
            ) THEN
                RAISE EXCEPTION
                    'terminal response content may be redacted only with its battle prompt';
            END IF;
            IF OLD.answer_markdown IS DISTINCT FROM NEW.answer_markdown
               AND NEW.answer_markdown IS NOT NULL THEN
                RAISE EXCEPTION 'terminal answer permits only privacy redaction';
            END IF;
            IF OLD.output_json::jsonb IS DISTINCT FROM NEW.output_json::jsonb
               AND NEW.output_json::jsonb IS DISTINCT FROM '{{"redacted": true}}'::jsonb THEN
                RAISE EXCEPTION 'terminal structured output permits only privacy redaction';
            END IF;
            IF OLD.error_detail IS DISTINCT FROM NEW.error_detail
               AND NEW.error_detail IS NOT NULL
               AND settlement IS NOT TRUE THEN
                RAISE EXCEPTION 'terminal error detail permits only settlement or redaction';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {ARM_GUARD_TRIGGER} ON response_arms;
        CREATE TRIGGER {ARM_GUARD_TRIGGER}
        BEFORE INSERT OR UPDATE OR DELETE ON response_arms
        FOR EACH ROW EXECUTE FUNCTION {ARM_GUARD_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {ASSIGNMENT_GUARD_FUNCTION}() RETURNS trigger AS $$
        DECLARE
            binding boolean;
            cancellation boolean;
            bound_cancellation boolean;
            parent_status text;
        BEGIN
            SELECT status INTO parent_status
            FROM controlled_runs
            WHERE id = NEW.controlled_run_id
            FOR UPDATE;
            IF parent_status IS DISTINCT FROM 'active' THEN
                RAISE EXCEPTION 'controlled assignment mutation requires an active parent run';
            END IF;
            IF TG_OP = 'INSERT' THEN
                IF NEW.status <> 'pending' OR NEW.battle_id IS NOT NULL THEN
                    RAISE EXCEPTION 'controlled assignments must be inserted pending and unbound';
                END IF;
                RETURN NEW;
            ELSIF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'controlled assignments are append-only';
            END IF;
            IF OLD.id IS DISTINCT FROM NEW.id
               OR OLD.controlled_run_id IS DISTINCT FROM NEW.controlled_run_id
               OR OLD.ordinal IS DISTINCT FROM NEW.ordinal
               OR OLD.task_id IS DISTINCT FROM NEW.task_id
               OR OLD.task_public_id IS DISTINCT FROM NEW.task_public_id
               OR OLD.task_revision IS DISTINCT FROM NEW.task_revision
               OR OLD.task_prompt_sha256 IS DISTINCT FROM NEW.task_prompt_sha256
               OR OLD.task_family IS DISTINCT FROM NEW.task_family
               OR OLD.track IS DISTINCT FROM NEW.track
               OR OLD.model_ids_json::jsonb IS DISTINCT FROM NEW.model_ids_json::jsonb
               OR OLD.repetition_index IS DISTINCT FROM NEW.repetition_index
               OR OLD.assignment_sha256 IS DISTINCT FROM NEW.assignment_sha256
               OR OLD.assignment_seed IS DISTINCT FROM NEW.assignment_seed
               OR OLD.created_at IS DISTINCT FROM NEW.created_at THEN
                RAISE EXCEPTION 'controlled assignment content is immutable';
            END IF;
            IF OLD.status IS NOT DISTINCT FROM NEW.status
               AND OLD.battle_id IS NOT DISTINCT FROM NEW.battle_id THEN
                RETURN NEW;
            END IF;
            binding := OLD.status = 'pending' AND NEW.status = 'queued'
                AND OLD.battle_id IS NULL AND NEW.battle_id IS NOT NULL;
            cancellation := OLD.status = 'pending' AND NEW.status = 'cancelled'
                AND OLD.battle_id IS NULL AND NEW.battle_id IS NULL;
            bound_cancellation := OLD.status = 'queued' AND NEW.status = 'cancelled'
                AND OLD.battle_id IS NOT NULL AND NEW.battle_id = OLD.battle_id;
            IF NOT (binding OR cancellation OR bound_cancellation) THEN
                RAISE EXCEPTION 'controlled assignment lifecycle is write-once';
            END IF;
            IF binding AND NOT EXISTS (
                SELECT 1 FROM battles AS b
                WHERE b.id = NEW.battle_id
                  AND b.controlled_run_id = NEW.controlled_run_id
                  AND b.data_stratum = 'controlled'
                  AND b.task_id = NEW.task_id
                  AND b.task_revision = NEW.task_revision
                  AND b.prompt_sha256 = NEW.task_prompt_sha256
                  AND b.category = NEW.task_family
                  AND b.track = NEW.track
                  AND b.assignment_seed = NEW.assignment_seed
                  AND b.scheduler_version = 'controlled-frozen-schedule-v1'
                  AND b.track_assignment_probability = '1/1'
                  AND b.model_assignment_probability = '1/1'
                  AND b.side_assignment_probability = '1/2'
            ) THEN
                RAISE EXCEPTION 'controlled assignment does not match its battle';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {ASSIGNMENT_GUARD_TRIGGER}
            ON controlled_run_assignments;
        CREATE TRIGGER {ASSIGNMENT_GUARD_TRIGGER}
        BEFORE INSERT OR UPDATE OR DELETE ON controlled_run_assignments
        FOR EACH ROW EXECUTE FUNCTION {ASSIGNMENT_GUARD_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {CONTROLLED_RUN_GUARD_FUNCTION}() RETURNS trigger AS $$
        DECLARE
            completing boolean;
            closing boolean;
            revoking boolean;
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.status <> 'active'
                   OR NEW.collection_completed_at IS NOT NULL
                   OR NEW.closed_at IS NOT NULL
                   OR NEW.revoked_at IS NOT NULL THEN
                    RAISE EXCEPTION 'controlled runs must be inserted active';
                END IF;
                RETURN NEW;
            ELSIF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'controlled runs are append-only';
            END IF;
            IF OLD.id IS DISTINCT FROM NEW.id THEN
                RAISE EXCEPTION 'controlled-run identity is immutable';
            END IF;
            IF OLD.season_id IS DISTINCT FROM NEW.season_id
               OR OLD.organization_reference_sha256 IS DISTINCT FROM
                    NEW.organization_reference_sha256
               OR OLD.protocol_version IS DISTINCT FROM NEW.protocol_version
               OR OLD.rater_plan_sha256 IS DISTINCT FROM NEW.rater_plan_sha256
               OR OLD.analysis_plan_sha256 IS DISTINCT FROM NEW.analysis_plan_sha256
               OR OLD.submitted_endpoint_model_id IS DISTINCT FROM
                    NEW.submitted_endpoint_model_id
               OR OLD.submitted_model_card_sha256 IS DISTINCT FROM
                    NEW.submitted_model_card_sha256
               OR OLD.data_policy_sha256 IS DISTINCT FROM NEW.data_policy_sha256
               OR OLD.model_roster_json::jsonb IS DISTINCT FROM NEW.model_roster_json::jsonb
               OR OLD.model_roster_sha256 IS DISTINCT FROM NEW.model_roster_sha256
               OR OLD.task_schedule_sha256 IS DISTINCT FROM NEW.task_schedule_sha256
               OR OLD.budget_cap_micros IS DISTINCT FROM NEW.budget_cap_micros
               OR OLD.run_card_json::jsonb IS DISTINCT FROM NEW.run_card_json::jsonb
               OR OLD.run_card_sha256 IS DISTINCT FROM NEW.run_card_sha256
               OR OLD.run_card_signature IS DISTINCT FROM NEW.run_card_signature
               OR OLD.created_at IS DISTINCT FROM NEW.created_at THEN
                RAISE EXCEPTION 'controlled-run signed contract is immutable';
            END IF;
            IF OLD.status IS NOT DISTINCT FROM NEW.status THEN
                IF OLD.collection_completed_at IS DISTINCT FROM NEW.collection_completed_at
                   OR OLD.closed_at IS DISTINCT FROM NEW.closed_at
                   OR OLD.revoked_at IS DISTINCT FROM NEW.revoked_at THEN
                    RAISE EXCEPTION 'controlled-run lifecycle timestamps are immutable';
                END IF;
                RETURN NEW;
            END IF;
            completing := OLD.status = 'active' AND NEW.status = 'collection_complete'
                AND OLD.collection_completed_at IS NULL
                AND NEW.collection_completed_at IS NOT NULL
                AND OLD.closed_at IS NULL AND NEW.closed_at IS NULL
                AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NULL;
            closing := OLD.status = 'collection_complete' AND NEW.status = 'closed'
                AND OLD.collection_completed_at IS NOT NULL
                AND NEW.collection_completed_at = OLD.collection_completed_at
                AND OLD.closed_at IS NULL AND NEW.closed_at IS NOT NULL
                AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NULL;
            revoking := OLD.status <> 'revoked' AND NEW.status = 'revoked'
                AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NOT NULL
                AND OLD.collection_completed_at IS NOT DISTINCT FROM
                    NEW.collection_completed_at
                AND OLD.closed_at IS NOT DISTINCT FROM NEW.closed_at;
            IF NOT (completing OR closing OR revoking) THEN
                RAISE EXCEPTION 'controlled-run lifecycle transition is invalid';
            END IF;
            IF completing OR closing THEN
                IF NOT EXISTS (
                    SELECT 1 FROM controlled_run_assignments AS a
                    WHERE a.controlled_run_id = NEW.id
                ) OR EXISTS (
                    SELECT 1 FROM controlled_run_assignments AS a
                    WHERE a.controlled_run_id = NEW.id
                      AND (a.status <> 'queued' OR a.battle_id IS NULL)
                ) OR EXISTS (
                    SELECT 1 FROM controlled_run_assignments AS a
                    LEFT JOIN battles AS b ON b.id = a.battle_id
                    WHERE a.controlled_run_id = NEW.id AND (
                        b.id IS NULL
                        OR b.controlled_run_id IS DISTINCT FROM a.controlled_run_id
                        OR b.data_stratum <> 'controlled'
                        OR b.task_id IS DISTINCT FROM a.task_id
                        OR b.task_revision IS DISTINCT FROM a.task_revision
                        OR b.prompt_sha256 IS DISTINCT FROM a.task_prompt_sha256
                        OR b.category IS DISTINCT FROM a.task_family
                        OR b.track IS DISTINCT FROM a.track
                        OR b.assignment_seed IS DISTINCT FROM a.assignment_seed
                        OR b.scheduler_version <> 'controlled-frozen-schedule-v1'
                        OR b.track_assignment_probability <> '1/1'
                        OR b.model_assignment_probability <> '1/1'
                        OR b.side_assignment_probability <> '1/2'
                    )
                ) OR EXISTS (
                    SELECT 1 FROM battles AS b
                    LEFT JOIN controlled_run_assignments AS a
                      ON a.battle_id = b.id AND a.controlled_run_id = NEW.id
                    WHERE b.controlled_run_id = NEW.id AND a.id IS NULL
                ) OR EXISTS (
                    SELECT 1 FROM battles AS b
                    LEFT JOIN response_arms AS left_arm
                      ON left_arm.id = b.left_arm_id
                     AND left_arm.battle_id = b.id
                     AND left_arm.side = 'left'
                    LEFT JOIN response_arms AS right_arm
                      ON right_arm.id = b.right_arm_id
                     AND right_arm.battle_id = b.id
                     AND right_arm.side = 'right'
                    WHERE b.controlled_run_id = NEW.id AND (
                        b.status NOT IN ('complete', 'failed')
                        OR left_arm.id IS NULL
                        OR right_arm.id IS NULL
                        OR left_arm.status NOT IN ('complete', 'failed')
                        OR right_arm.status NOT IN ('complete', 'failed')
                    )
                ) THEN
                    RAISE EXCEPTION 'controlled-run terminal bijection is invalid';
                END IF;
                IF closing AND (
                    NEW.budget_reserved_micros <> 0
                    OR EXISTS (
                        SELECT 1 FROM battles AS b
                        WHERE b.controlled_run_id = NEW.id
                          AND b.reserved_cost_micros <> 0
                    )
                    OR EXISTS (
                        SELECT 1 FROM battles AS b
                        JOIN response_arms AS a ON a.battle_id = b.id
                        WHERE b.controlled_run_id = NEW.id AND (
                            a.cost_reconciled IS NOT TRUE
                            OR COALESCE(a.cost_accounting_basis, 'unrecorded') =
                                'unrecorded'
                            OR COALESCE(a.billing_reconciliation_status, 'unrecorded') =
                                'unrecorded'
                            OR NOT EXISTS (
                                SELECT 1 FROM cost_events AS ce
                                WHERE ce.arm_id = a.id
                                  AND ce.battle_id = b.id
                                  AND ce.kind IN ('actual', 'actual_settlement')
                                  AND ce.amount_micros = a.cost_micros
                                  AND ce.generation_id IS NOT DISTINCT FROM a.generation_id
                            )
                        )
                    )
                    OR EXISTS (
                        SELECT 1 FROM battles AS b
                        WHERE b.controlled_run_id = NEW.id
                          AND NOT EXISTS (
                              SELECT 1 FROM cost_events AS ce
                              WHERE ce.battle_id = b.id
                                AND ce.kind = 'reconcile'
                                AND ce.amount_micros = (
                                    SELECT COALESCE(SUM(a.cost_micros), 0)
                                    FROM response_arms AS a
                                    WHERE a.battle_id = b.id
                                )
                          )
                    )
                    OR EXISTS (
                        SELECT 1 FROM battles AS b
                        WHERE b.controlled_run_id = NEW.id AND (
                            (SELECT COUNT(*) FROM jobs AS j WHERE j.battle_id = b.id) <> 1
                            OR EXISTS (
                                SELECT 1 FROM jobs AS j
                                WHERE j.battle_id = b.id
                                  AND (j.status NOT IN ('complete', 'failed')
                                       OR j.completed_at IS NULL)
                            )
                        )
                    )
                ) THEN
                    RAISE EXCEPTION 'controlled-run close requires reconciled cost evidence';
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {CONTROLLED_RUN_GUARD_TRIGGER} ON controlled_runs;
        CREATE TRIGGER {CONTROLLED_RUN_GUARD_TRIGGER}
        BEFORE INSERT OR UPDATE OR DELETE ON controlled_runs
        FOR EACH ROW EXECUTE FUNCTION {CONTROLLED_RUN_GUARD_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {VOTE_GUARD_FUNCTION}() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.choice NOT IN ('left', 'right', 'tie', 'both_bad')
                   OR NEW.cohort NOT IN (
                        'public',
                        'expert_independent',
                        'expert_product_affiliated',
                        'expert_provider_affiliated'
                   ) THEN
                    RAISE EXCEPTION 'vote is outside the frozen evidence domain';
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM battles AS b
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
                    WHERE b.id = NEW.battle_id
                      AND b.status = 'complete'
                      AND b.completed_at IS NOT NULL
                      AND NEW.created_at >= b.completed_at
                      AND b.left_arm_id IS NOT NULL
                      AND b.right_arm_id IS NOT NULL
                      AND b.left_arm_id <> b.right_arm_id
                ) THEN
                    RAISE EXCEPTION 'vote does not follow a completed anonymous battle';
                END IF;
                RETURN NEW;
            END IF;
            RAISE EXCEPTION 'votes are append-only';
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {VOTE_GUARD_TRIGGER} ON votes;
        CREATE TRIGGER {VOTE_GUARD_TRIGGER}
        BEFORE INSERT OR UPDATE OR DELETE ON votes
        FOR EACH ROW EXECUTE FUNCTION {VOTE_GUARD_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {COST_EVENT_GUARD_FUNCTION}() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                RETURN NEW;
            END IF;
            RAISE EXCEPTION 'cost events are append-only; record an adjustment event';
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {COST_EVENT_GUARD_TRIGGER} ON cost_events;
        CREATE TRIGGER {COST_EVENT_GUARD_TRIGGER}
        BEFORE INSERT OR UPDATE OR DELETE ON cost_events
        FOR EACH ROW EXECUTE FUNCTION {COST_EVENT_GUARD_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {APPEND_ONLY_EVIDENCE_FUNCTION}() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'append-only evidence must be superseded by a new linked record';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table in APPEND_ONLY_EVIDENCE_TABLES:
        trigger = f"trg_{table}_append_only_guard"
        op.execute(
            f"""
            DROP TRIGGER IF EXISTS {trigger} ON {table};
            CREATE TRIGGER {trigger}
            BEFORE UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION {APPEND_ONLY_EVIDENCE_FUNCTION}();
            """
        )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {RUN_EVENT_GUARD_FUNCTION}() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'run events are append-only';
            END IF;
            IF OLD.id IS DISTINCT FROM NEW.id
               OR OLD.entity_type IS DISTINCT FROM NEW.entity_type
               OR OLD.entity_id IS DISTINCT FROM NEW.entity_id
               OR OLD.event_type IS DISTINCT FROM NEW.event_type
               OR OLD.created_at IS DISTINCT FROM NEW.created_at
               OR OLD.payload_json::jsonb = jsonb_build_object('redacted', true)
               OR NEW.payload_json::jsonb <> jsonb_build_object('redacted', true) THEN
                RAISE EXCEPTION
                    'run-event evidence permits only one-way payload redaction';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {RUN_EVENT_GUARD_TRIGGER} ON run_events;
        CREATE TRIGGER {RUN_EVENT_GUARD_TRIGGER}
        BEFORE UPDATE OR DELETE ON run_events
        FOR EACH ROW EXECUTE FUNCTION {RUN_EVENT_GUARD_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {INCIDENT_GUARD_FUNCTION}() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'incidents are append-only';
            END IF;
            IF OLD.id IS DISTINCT FROM NEW.id
               OR OLD.severity IS DISTINCT FROM NEW.severity
               OR OLD.code IS DISTINCT FROM NEW.code
               OR OLD.battle_id IS DISTINCT FROM NEW.battle_id
               OR OLD.created_at IS DISTINCT FROM NEW.created_at
               OR OLD.detail = '[REDACTED AFTER OPERATIONAL RETENTION]'
               OR NEW.detail <> '[REDACTED AFTER OPERATIONAL RETENTION]' THEN
                RAISE EXCEPTION
                    'incident evidence permits only one-way detail redaction';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {INCIDENT_GUARD_TRIGGER} ON incidents;
        CREATE TRIGGER {INCIDENT_GUARD_TRIGGER}
        BEFORE UPDATE OR DELETE ON incidents
        FOR EACH ROW EXECUTE FUNCTION {INCIDENT_GUARD_FUNCTION}();
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {JOB_GUARD_FUNCTION}() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                IF NEW.status <> 'queued' OR NEW.completed_at IS NOT NULL THEN
                    RAISE EXCEPTION 'jobs must be inserted queued without completion time';
                END IF;
                RETURN NEW;
            ELSIF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'jobs are append-only';
            END IF;
            IF OLD.id IS DISTINCT FROM NEW.id
               OR OLD.battle_id IS DISTINCT FROM NEW.battle_id
               OR OLD.kind IS DISTINCT FROM NEW.kind
               OR OLD.payload_json::jsonb IS DISTINCT FROM NEW.payload_json::jsonb
               OR OLD.created_at IS DISTINCT FROM NEW.created_at THEN
                RAISE EXCEPTION 'job execution identity is immutable';
            END IF;
            IF OLD.status IS DISTINCT FROM NEW.status AND NOT (
                (OLD.status = 'queued' AND NEW.status IN ('running', 'failed'))
                OR (OLD.status = 'running' AND NEW.status IN (
                    'queued', 'complete', 'failed', 'uncertain'
                ))
                OR (OLD.status = 'uncertain' AND NEW.status = 'failed')
            ) THEN
                RAISE EXCEPTION 'job lifecycle transition is invalid';
            END IF;
            IF NEW.status IN ('queued', 'running') AND NEW.completed_at IS NOT NULL THEN
                RAISE EXCEPTION 'nonterminal job cannot carry a completion timestamp';
            END IF;
            IF NEW.status IN ('complete', 'failed', 'uncertain') AND (
                NEW.completed_at IS NULL OR NEW.completed_at < NEW.created_at
            ) THEN
                RAISE EXCEPTION 'terminal job evidence is incomplete';
            END IF;
            IF OLD.completed_at IS NOT NULL
               AND OLD.completed_at IS DISTINCT FROM NEW.completed_at THEN
                RAISE EXCEPTION 'job completion timestamp is write-once';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        DROP TRIGGER IF EXISTS {JOB_GUARD_TRIGGER} ON jobs;
        CREATE TRIGGER {JOB_GUARD_TRIGGER}
        BEFORE INSERT OR UPDATE OR DELETE ON jobs
        FOR EACH ROW EXECUTE FUNCTION {JOB_GUARD_FUNCTION}();
        """
    )


def _create_sqlite_guards() -> None:
    op.execute(
        f"""
        CREATE TRIGGER {BATTLE_LINK_TRIGGER}_insert
        BEFORE INSERT ON battles FOR EACH ROW
        WHEN NEW.status <> 'queued'
          OR NEW.left_arm_id IS NOT NULL
          OR NEW.right_arm_id IS NOT NULL
          OR NEW.completed_at IS NOT NULL
          OR NEW.prompt IS NULL
          OR NEW.prompt_redacted <> 0
          OR (
              NEW.controlled_run_id IS NOT NULL AND NOT EXISTS (
                  SELECT 1 FROM controlled_runs AS r
                  WHERE r.id = NEW.controlled_run_id AND r.status = 'active'
              )
          )
        BEGIN
            SELECT RAISE(ABORT, 'battles must be inserted queued with null arm links');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {BATTLE_LINK_TRIGGER}_update
        BEFORE UPDATE ON battles FOR EACH ROW
        WHEN OLD.id IS NOT NEW.id
          OR OLD.season_id IS NOT NEW.season_id
          OR OLD.run_class IS NOT NEW.run_class
          OR OLD.rank_eligible IS NOT NEW.rank_eligible
          OR OLD.data_stratum IS NOT NEW.data_stratum
          OR OLD.task_id IS NOT NEW.task_id
          OR OLD.task_revision IS NOT NEW.task_revision
          OR OLD.controlled_run_id IS NOT NEW.controlled_run_id
          OR OLD.manifest_sha256 IS NOT NEW.manifest_sha256
          OR OLD.protocol_bundle_sha256 IS NOT NEW.protocol_bundle_sha256
          OR OLD.scheduler_version IS NOT NEW.scheduler_version
          OR OLD.assignment_seed IS NOT NEW.assignment_seed
          OR OLD.track_assignment_probability IS NOT NEW.track_assignment_probability
          OR OLD.model_assignment_probability IS NOT NEW.model_assignment_probability
          OR OLD.side_assignment_probability IS NOT NEW.side_assignment_probability
          OR OLD.provider_reservations_json IS NOT NEW.provider_reservations_json
          OR OLD.track IS NOT NEW.track
          OR OLD.category IS NOT NEW.category
          OR OLD.prompt_sha256 IS NOT NEW.prompt_sha256
          OR OLD.client_nonce_sha256 IS NOT NEW.client_nonce_sha256
          OR OLD.research_consent IS NOT NEW.research_consent
          OR OLD.requester_pseudonym IS NOT NEW.requester_pseudonym
          OR OLD.created_at IS NOT NEW.created_at
          OR OLD.retention_until IS NOT NEW.retention_until
          OR (
              (OLD.prompt IS NOT NEW.prompt OR OLD.prompt_redacted IS NOT NEW.prompt_redacted)
              AND NOT (
                  OLD.prompt IS NOT NULL
                  AND NEW.prompt IS NULL
                  AND OLD.prompt_redacted = 0
                  AND NEW.prompt_redacted = 1
              )
          )
          OR (
            (
                OLD.status IS NOT NEW.status
                OR OLD.left_arm_id IS NOT NEW.left_arm_id
                OR OLD.right_arm_id IS NOT NEW.right_arm_id
                OR OLD.completed_at IS NOT NEW.completed_at
            )
            AND NEW.controlled_run_id IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM controlled_runs AS r
                WHERE r.id = NEW.controlled_run_id AND r.status = 'active'
            )
        ) OR (
            OLD.status IS NOT NEW.status AND NOT (
                (OLD.status = 'queued' AND NEW.status IN ('running', 'complete', 'failed'))
                OR (OLD.status = 'running' AND NEW.status IN ('complete', 'failed'))
            )
        ) OR (
            OLD.completed_at IS NOT NEW.completed_at AND NOT (
                OLD.status IN ('queued', 'running')
                AND NEW.status IN ('complete', 'failed')
                AND OLD.completed_at IS NULL
                AND NEW.completed_at IS NOT NULL
            )
        ) OR (
            NEW.status IN ('queued', 'running') AND NEW.completed_at IS NOT NULL
        ) OR (
            OLD.left_arm_id IS NOT NEW.left_arm_id AND (
                OLD.left_arm_id IS NOT NULL OR NEW.left_arm_id IS NULL OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.left_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'left'
                )
            )
        ) OR (
            OLD.right_arm_id IS NOT NEW.right_arm_id AND (
                OLD.right_arm_id IS NOT NULL OR NEW.right_arm_id IS NULL OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.right_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'right'
                )
            )
        ) OR (
            NEW.left_arm_id IS NOT NULL
            AND NEW.right_arm_id IS NOT NULL
            AND NEW.left_arm_id = NEW.right_arm_id
        ) OR (
            NEW.status IN ('running', 'complete', 'failed') AND (
                NEW.left_arm_id IS NULL
                OR NEW.right_arm_id IS NULL
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.left_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'left'
                )
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.right_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'right'
                )
            )
        ) OR (
            NEW.status IN ('complete', 'failed') AND (
                NEW.completed_at IS NULL
                OR NEW.completed_at < NEW.created_at
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.left_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'left'
                      AND a.completed_at IS NOT NULL
                      AND NEW.completed_at >= a.completed_at
                      AND (
                          (NEW.status = 'complete' AND a.status = 'complete')
                          OR (NEW.status = 'failed' AND a.status IN (
                              'complete', 'failed', 'uncertain'
                          ))
                      )
                )
                OR NOT EXISTS (
                    SELECT 1 FROM response_arms AS a
                    WHERE a.id = NEW.right_arm_id
                      AND a.battle_id = NEW.id
                      AND a.side = 'right'
                      AND a.completed_at IS NOT NULL
                      AND NEW.completed_at >= a.completed_at
                      AND (
                          (NEW.status = 'complete' AND a.status = 'complete')
                          OR (NEW.status = 'failed' AND a.status IN (
                              'complete', 'failed', 'uncertain'
                          ))
                      )
                )
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'battle lifecycle or arm-link mutation is invalid');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ARM_GUARD_TRIGGER}_insert
        BEFORE INSERT ON response_arms FOR EACH ROW
        WHEN NEW.status <> 'queued'
          OR NEW.completed_at IS NOT NULL
          OR NEW.actual_provider_slug IS NOT NULL
          OR NEW.actual_model_id IS NOT NULL
          OR NEW.generation_id IS NOT NULL
          OR NEW.answer_markdown IS NOT NULL
          OR NEW.answer_markdown_sha256 IS NOT NULL
          OR NEW.output_json_sha256 IS NOT NULL
          OR NEW.cost_micros <> 0
          OR NEW.cost_reconciled = 1
        BEGIN
            SELECT RAISE(ABORT, 'response arms must be inserted as empty queued records');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ARM_GUARD_TRIGGER}_contract
        BEFORE UPDATE ON response_arms FOR EACH ROW
        WHEN OLD.id IS NOT NEW.id
          OR OLD.battle_id IS NOT NEW.battle_id
          OR OLD.side IS NOT NEW.side
          OR OLD.condition IS NOT NEW.condition
          OR OLD.model_id IS NOT NEW.model_id
          OR OLD.execution_backend IS NOT NEW.execution_backend
          OR OLD.provider_slug IS NOT NEW.provider_slug
          OR OLD.prompt_sha256 IS NOT NEW.prompt_sha256
          OR OLD.system_prompt_sha256 IS NOT NEW.system_prompt_sha256
          OR OLD.schema_sha256 IS NOT NEW.schema_sha256
          OR OLD.tool_schema_sha256 IS NOT NEW.tool_schema_sha256
          OR OLD.decoding_json IS NOT NEW.decoding_json
          OR OLD.protocol_bundle_sha256 IS NOT NEW.protocol_bundle_sha256
          OR OLD.epicure_release_id IS NOT NEW.epicure_release_id
          OR OLD.epicure_bundle_sha256 IS NOT NEW.epicure_bundle_sha256
          OR OLD.epicure_application_sha256 IS NOT NEW.epicure_application_sha256
        BEGIN
            SELECT RAISE(ABORT, 'response-arm execution contract is immutable');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ARM_GUARD_TRIGGER}_lifecycle
        BEFORE UPDATE ON response_arms FOR EACH ROW
        WHEN (NEW.status IN ('queued', 'running') AND NEW.completed_at IS NOT NULL)
        OR (
            OLD.status NOT IN ('complete', 'failed', 'uncertain')
            AND OLD.status IS NOT NEW.status
            AND NOT (
                (OLD.status = 'queued' AND NEW.status IN (
                    'running', 'complete', 'failed', 'uncertain'
                )) OR
                (OLD.status = 'running' AND NEW.status IN (
                    'complete', 'failed', 'uncertain'
                ))
            )
        ) OR (
            OLD.status IS NOT NEW.status
            AND NEW.status IN ('complete', 'failed', 'uncertain')
            AND (
                NEW.completed_at IS NULL
                OR NEW.completed_at < NEW.created_at
                OR NEW.output_json_sha256 IS NULL
                OR (
                    NEW.status = 'complete' AND (
                        NEW.answer_markdown IS NULL
                        OR NEW.answer_markdown_sha256 IS NULL
                        OR NEW.actual_provider_slug IS NULL
                        OR NEW.actual_model_id IS NULL
                        OR NEW.generation_id IS NULL
                        OR json_type(NEW.provider_generation_ids_json, '$') IS NOT 'array'
                        OR json_array_length(NEW.provider_generation_ids_json) = 0
                        OR NEW.cost_reconciled <> 1
                        OR COALESCE(NEW.cost_accounting_basis, 'unrecorded') =
                            'unrecorded'
                        OR COALESCE(
                            NEW.billing_reconciliation_status,
                            'unrecorded'
                        ) = 'unrecorded'
                    )
                )
                OR (
                    NEW.status = 'failed' AND (
                        NEW.cost_reconciled <> 1
                        OR COALESCE(NEW.cost_accounting_basis, 'unrecorded') =
                            'unrecorded'
                        OR COALESCE(
                            NEW.billing_reconciliation_status,
                            'unrecorded'
                        ) = 'unrecorded'
                    )
                )
                OR (NEW.status = 'uncertain' AND NEW.cost_reconciled = 1)
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'response-arm lifecycle or terminal record is invalid');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ARM_GUARD_TRIGGER}_terminal
        BEFORE UPDATE ON response_arms FOR EACH ROW
        WHEN OLD.status IN ('complete', 'failed', 'uncertain') AND (
            OLD.actual_provider_slug IS NOT NEW.actual_provider_slug
            OR OLD.actual_model_id IS NOT NEW.actual_model_id
            OR OLD.generation_id IS NOT NEW.generation_id
            OR OLD.provider_generation_ids_json IS NOT NEW.provider_generation_ids_json
            OR OLD.observed_decoding_json IS NOT NEW.observed_decoding_json
            OR OLD.epicure_attestation_json IS NOT NEW.epicure_attestation_json
            OR OLD.epicure_attestation_sha256 IS NOT NEW.epicure_attestation_sha256
            OR OLD.prompt_tokens IS NOT NEW.prompt_tokens
            OR OLD.completion_tokens IS NOT NEW.completion_tokens
            OR OLD.reasoning_tokens IS NOT NEW.reasoning_tokens
            OR OLD.backend_response_schema_sha256 IS NOT NEW.backend_response_schema_sha256
            OR OLD.backend_tool_schema_sha256 IS NOT NEW.backend_tool_schema_sha256
            OR OLD.latency_ms IS NOT NEW.latency_ms
            OR OLD.retries IS NOT NEW.retries
            OR OLD.finish_reason IS NOT NEW.finish_reason
            OR OLD.answer_markdown_sha256 IS NOT NEW.answer_markdown_sha256
            OR OLD.output_json_sha256 IS NOT NEW.output_json_sha256
            OR OLD.created_at IS NOT NEW.created_at
        )
        BEGIN
            SELECT RAISE(ABORT, 'terminal response-arm evidence is immutable');
        END;
        """
    )
    settlement = """
        OLD.status = 'uncertain'
        AND NEW.status = 'failed'
        AND NEW.cost_reconciled = 1
        AND NEW.cost_micros >= 0
        AND NEW.cost_accounting_basis = 'manual_authorized_settlement'
        AND NEW.billing_reconciliation_status = 'manual_authorized_settlement'
        AND NEW.error_code = 'CostExposureSettled'
        AND NEW.error_detail =
            'Provider cost exposure was settled by an authorized record.'
        AND NEW.completed_at IS OLD.completed_at
    """
    op.execute(
        f"""
        CREATE TRIGGER {ARM_GUARD_TRIGGER}_result
        BEFORE UPDATE ON response_arms FOR EACH ROW
        WHEN OLD.status IN ('complete', 'failed', 'uncertain') AND (
            (
                (
                    OLD.status IS NOT NEW.status
                    OR OLD.cost_micros IS NOT NEW.cost_micros
                    OR OLD.cost_reconciled IS NOT NEW.cost_reconciled
                    OR OLD.cost_accounting_basis IS NOT NEW.cost_accounting_basis
                    OR OLD.billing_reconciliation_status IS NOT
                        NEW.billing_reconciliation_status
                    OR OLD.error_code IS NOT NEW.error_code
                    OR OLD.completed_at IS NOT NEW.completed_at
                )
                AND NOT COALESCE(({settlement}), 0)
            )
            OR (
                (
                    OLD.answer_markdown IS NOT NEW.answer_markdown
                    OR OLD.output_json IS NOT NEW.output_json
                    OR (
                        OLD.error_detail IS NOT NEW.error_detail
                        AND NOT COALESCE(({settlement}), 0)
                    )
                )
                AND NOT EXISTS (
                    SELECT 1 FROM battles AS b
                    WHERE b.id = NEW.battle_id
                      AND b.prompt_redacted = 1
                      AND b.prompt IS NULL
                )
            )
            OR (
                OLD.answer_markdown IS NOT NEW.answer_markdown
                AND NEW.answer_markdown IS NOT NULL
            )
            OR (
                OLD.output_json IS NOT NEW.output_json
                AND (
                    json_type(NEW.output_json, '$') IS NOT 'object'
                    OR json_type(NEW.output_json, '$.redacted') IS NOT 'true'
                    OR (SELECT COUNT(*) FROM json_each(NEW.output_json)) IS NOT 1
                )
            )
            OR (
                OLD.error_detail IS NOT NEW.error_detail
                AND NEW.error_detail IS NOT NULL
                AND NOT COALESCE(({settlement}), 0)
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'terminal response-arm result is immutable');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ARM_GUARD_TRIGGER}_delete
        BEFORE DELETE ON response_arms FOR EACH ROW
        BEGIN
            SELECT RAISE(ABORT, 'response arms are append-only');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ASSIGNMENT_GUARD_TRIGGER}_insert
        BEFORE INSERT ON controlled_run_assignments FOR EACH ROW
        WHEN NEW.status <> 'pending'
          OR NEW.battle_id IS NOT NULL
          OR NOT EXISTS (
              SELECT 1 FROM controlled_runs AS r
              WHERE r.id = NEW.controlled_run_id AND r.status = 'active'
          )
        BEGIN
            SELECT RAISE(ABORT, 'controlled assignments must be inserted pending and unbound');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ASSIGNMENT_GUARD_TRIGGER}_update
        BEFORE UPDATE ON controlled_run_assignments FOR EACH ROW
        WHEN NOT EXISTS (
              SELECT 1 FROM controlled_runs AS r
              WHERE r.id = NEW.controlled_run_id AND r.status = 'active'
          )
          OR OLD.id IS NOT NEW.id
          OR OLD.controlled_run_id IS NOT NEW.controlled_run_id
          OR OLD.ordinal IS NOT NEW.ordinal
          OR OLD.task_id IS NOT NEW.task_id
          OR OLD.task_public_id IS NOT NEW.task_public_id
          OR OLD.task_revision IS NOT NEW.task_revision
          OR OLD.task_prompt_sha256 IS NOT NEW.task_prompt_sha256
          OR OLD.task_family IS NOT NEW.task_family
          OR OLD.track IS NOT NEW.track
          OR OLD.model_ids_json IS NOT NEW.model_ids_json
          OR OLD.repetition_index IS NOT NEW.repetition_index
          OR OLD.assignment_sha256 IS NOT NEW.assignment_sha256
          OR OLD.assignment_seed IS NOT NEW.assignment_seed
          OR OLD.created_at IS NOT NEW.created_at
          OR (
              (OLD.status IS NOT NEW.status OR OLD.battle_id IS NOT NEW.battle_id)
              AND NOT (
                  (
                      OLD.status = 'pending' AND NEW.status = 'queued'
                      AND OLD.battle_id IS NULL AND NEW.battle_id IS NOT NULL
                  ) OR (
                      OLD.status = 'pending' AND NEW.status = 'cancelled'
                      AND OLD.battle_id IS NULL AND NEW.battle_id IS NULL
                  ) OR (
                      OLD.status = 'queued' AND NEW.status = 'cancelled'
                      AND OLD.battle_id IS NOT NULL AND NEW.battle_id = OLD.battle_id
                  )
              )
          )
          OR (
              OLD.status = 'pending' AND NEW.status = 'queued'
              AND OLD.battle_id IS NULL AND NEW.battle_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM battles AS b
                  WHERE b.id = NEW.battle_id
                    AND b.controlled_run_id = NEW.controlled_run_id
                    AND b.data_stratum = 'controlled'
                    AND b.task_id = NEW.task_id
                    AND b.task_revision = NEW.task_revision
                    AND b.prompt_sha256 = NEW.task_prompt_sha256
                    AND b.category = NEW.task_family
                    AND b.track = NEW.track
                    AND b.assignment_seed = NEW.assignment_seed
                    AND b.scheduler_version = 'controlled-frozen-schedule-v1'
                    AND b.track_assignment_probability = '1/1'
                    AND b.model_assignment_probability = '1/1'
                    AND b.side_assignment_probability = '1/2'
              )
          )
        BEGIN
            SELECT RAISE(ABORT, 'controlled assignment mutation is invalid');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {ASSIGNMENT_GUARD_TRIGGER}_delete
        BEFORE DELETE ON controlled_run_assignments FOR EACH ROW
        BEGIN
            SELECT RAISE(ABORT, 'controlled assignments are append-only');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {CONTROLLED_RUN_GUARD_TRIGGER}_insert
        BEFORE INSERT ON controlled_runs FOR EACH ROW
        WHEN NEW.status <> 'active'
          OR NEW.collection_completed_at IS NOT NULL
          OR NEW.closed_at IS NOT NULL
          OR NEW.revoked_at IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT, 'controlled runs must be inserted active');
        END;
        """
    )
    completing = """
        OLD.status = 'active' AND NEW.status = 'collection_complete'
        AND OLD.collection_completed_at IS NULL
        AND NEW.collection_completed_at IS NOT NULL
        AND OLD.closed_at IS NULL AND NEW.closed_at IS NULL
        AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NULL
    """
    closing = """
        OLD.status = 'collection_complete' AND NEW.status = 'closed'
        AND OLD.collection_completed_at IS NOT NULL
        AND NEW.collection_completed_at IS OLD.collection_completed_at
        AND OLD.closed_at IS NULL AND NEW.closed_at IS NOT NULL
        AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NULL
    """
    revoking = """
        OLD.status <> 'revoked' AND NEW.status = 'revoked'
        AND OLD.revoked_at IS NULL AND NEW.revoked_at IS NOT NULL
        AND OLD.collection_completed_at IS NEW.collection_completed_at
        AND OLD.closed_at IS NEW.closed_at
    """
    op.execute(
        f"""
        CREATE TRIGGER {CONTROLLED_RUN_GUARD_TRIGGER}_update
        BEFORE UPDATE ON controlled_runs FOR EACH ROW
        WHEN OLD.id IS NOT NEW.id
          OR OLD.season_id IS NOT NEW.season_id
          OR OLD.organization_reference_sha256 IS NOT NEW.organization_reference_sha256
          OR OLD.protocol_version IS NOT NEW.protocol_version
          OR OLD.rater_plan_sha256 IS NOT NEW.rater_plan_sha256
          OR OLD.analysis_plan_sha256 IS NOT NEW.analysis_plan_sha256
          OR OLD.submitted_endpoint_model_id IS NOT NEW.submitted_endpoint_model_id
          OR OLD.submitted_model_card_sha256 IS NOT NEW.submitted_model_card_sha256
          OR OLD.data_policy_sha256 IS NOT NEW.data_policy_sha256
          OR OLD.model_roster_json IS NOT NEW.model_roster_json
          OR OLD.model_roster_sha256 IS NOT NEW.model_roster_sha256
          OR OLD.task_schedule_sha256 IS NOT NEW.task_schedule_sha256
          OR OLD.budget_cap_micros IS NOT NEW.budget_cap_micros
          OR OLD.run_card_json IS NOT NEW.run_card_json
          OR OLD.run_card_sha256 IS NOT NEW.run_card_sha256
          OR OLD.run_card_signature IS NOT NEW.run_card_signature
          OR OLD.created_at IS NOT NEW.created_at
          OR (
              OLD.status IS NEW.status AND (
                  OLD.collection_completed_at IS NOT NEW.collection_completed_at
                  OR OLD.closed_at IS NOT NEW.closed_at
                  OR OLD.revoked_at IS NOT NEW.revoked_at
              )
          )
          OR (
              OLD.status IS NOT NEW.status
              AND NOT (({completing}) OR ({closing}) OR ({revoking}))
          )
          OR (
              (({completing}) OR ({closing})) AND (
                  NOT EXISTS (
                      SELECT 1 FROM controlled_run_assignments AS a
                      WHERE a.controlled_run_id = NEW.id
                  )
                  OR EXISTS (
                      SELECT 1 FROM controlled_run_assignments AS a
                      WHERE a.controlled_run_id = NEW.id
                        AND (a.status <> 'queued' OR a.battle_id IS NULL)
                  )
                  OR EXISTS (
                      SELECT 1 FROM controlled_run_assignments AS a
                      LEFT JOIN battles AS b ON b.id = a.battle_id
                      WHERE a.controlled_run_id = NEW.id AND (
                          b.id IS NULL
                          OR b.controlled_run_id IS NOT a.controlled_run_id
                          OR b.data_stratum <> 'controlled'
                          OR b.task_id IS NOT a.task_id
                          OR b.task_revision IS NOT a.task_revision
                          OR b.prompt_sha256 IS NOT a.task_prompt_sha256
                          OR b.category IS NOT a.task_family
                          OR b.track IS NOT a.track
                          OR b.assignment_seed IS NOT a.assignment_seed
                          OR b.scheduler_version <> 'controlled-frozen-schedule-v1'
                          OR b.track_assignment_probability <> '1/1'
                          OR b.model_assignment_probability <> '1/1'
                          OR b.side_assignment_probability <> '1/2'
                      )
                  )
                  OR EXISTS (
                      SELECT 1 FROM battles AS b
                      LEFT JOIN controlled_run_assignments AS a
                        ON a.battle_id = b.id AND a.controlled_run_id = NEW.id
                      WHERE b.controlled_run_id = NEW.id AND a.id IS NULL
                  )
                  OR EXISTS (
                      SELECT 1 FROM battles AS b
                      LEFT JOIN response_arms AS left_arm
                        ON left_arm.id = b.left_arm_id
                       AND left_arm.battle_id = b.id
                       AND left_arm.side = 'left'
                      LEFT JOIN response_arms AS right_arm
                        ON right_arm.id = b.right_arm_id
                       AND right_arm.battle_id = b.id
                       AND right_arm.side = 'right'
                      WHERE b.controlled_run_id = NEW.id AND (
                          b.status NOT IN ('complete', 'failed')
                          OR left_arm.id IS NULL
                          OR right_arm.id IS NULL
                          OR left_arm.status NOT IN ('complete', 'failed')
                          OR right_arm.status NOT IN ('complete', 'failed')
                      )
                  )
              )
          )
          OR (
              ({closing}) AND (
                  NEW.budget_reserved_micros <> 0
                  OR EXISTS (
                      SELECT 1 FROM battles AS b
                      WHERE b.controlled_run_id = NEW.id
                        AND b.reserved_cost_micros <> 0
                  )
                  OR EXISTS (
                      SELECT 1 FROM battles AS b
                      JOIN response_arms AS a ON a.battle_id = b.id
                      WHERE b.controlled_run_id = NEW.id AND (
                          a.cost_reconciled <> 1
                          OR COALESCE(a.cost_accounting_basis, 'unrecorded') =
                              'unrecorded'
                          OR COALESCE(a.billing_reconciliation_status, 'unrecorded') =
                              'unrecorded'
                          OR NOT EXISTS (
                              SELECT 1 FROM cost_events AS ce
                              WHERE ce.arm_id = a.id
                                AND ce.battle_id = b.id
                                AND ce.kind IN ('actual', 'actual_settlement')
                                AND ce.amount_micros = a.cost_micros
                                AND ce.generation_id IS a.generation_id
                          )
                      )
                  )
                  OR EXISTS (
                      SELECT 1 FROM battles AS b
                      WHERE b.controlled_run_id = NEW.id
                        AND NOT EXISTS (
                            SELECT 1 FROM cost_events AS ce
                            WHERE ce.battle_id = b.id
                              AND ce.kind = 'reconcile'
                              AND ce.amount_micros = (
                                  SELECT COALESCE(SUM(a.cost_micros), 0)
                                  FROM response_arms AS a
                                  WHERE a.battle_id = b.id
                              )
                        )
                  )
                  OR EXISTS (
                      SELECT 1 FROM battles AS b
                      WHERE b.controlled_run_id = NEW.id AND (
                          (SELECT COUNT(*) FROM jobs AS j WHERE j.battle_id = b.id) <> 1
                          OR EXISTS (
                              SELECT 1 FROM jobs AS j
                              WHERE j.battle_id = b.id
                                AND (j.status NOT IN ('complete', 'failed')
                                     OR j.completed_at IS NULL)
                          )
                      )
                  )
              )
          )
        BEGIN
            SELECT RAISE(ABORT, 'controlled-run lifecycle or terminal bijection is invalid');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {CONTROLLED_RUN_GUARD_TRIGGER}_delete
        BEFORE DELETE ON controlled_runs FOR EACH ROW
        BEGIN
            SELECT RAISE(ABORT, 'controlled runs are append-only');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {VOTE_GUARD_TRIGGER}_insert
        BEFORE INSERT ON votes FOR EACH ROW
        WHEN NEW.choice NOT IN ('left', 'right', 'tie', 'both_bad')
          OR NEW.cohort NOT IN (
              'public',
              'expert_independent',
              'expert_product_affiliated',
              'expert_provider_affiliated'
          )
          OR NOT EXISTS (
              SELECT 1 FROM battles AS b
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
              WHERE b.id = NEW.battle_id
                AND b.status = 'complete'
                AND b.completed_at IS NOT NULL
                AND NEW.created_at >= b.completed_at
                AND b.left_arm_id IS NOT NULL
                AND b.right_arm_id IS NOT NULL
                AND b.left_arm_id <> b.right_arm_id
          )
        BEGIN
            SELECT RAISE(ABORT, 'vote does not follow a valid completed battle');
        END;
        """
    )
    for operation in ("update", "delete"):
        op.execute(
            f"""
            CREATE TRIGGER {VOTE_GUARD_TRIGGER}_{operation}
            BEFORE {operation.upper()} ON votes FOR EACH ROW
            BEGIN
                SELECT RAISE(ABORT, 'votes are append-only');
            END;
            """
        )
    for operation in ("update", "delete"):
        op.execute(
            f"""
            CREATE TRIGGER {COST_EVENT_GUARD_TRIGGER}_{operation}
            BEFORE {operation.upper()} ON cost_events FOR EACH ROW
            BEGIN
                SELECT RAISE(ABORT, 'cost events are append-only');
            END;
            """
        )
    for table in APPEND_ONLY_EVIDENCE_TABLES:
        for operation in ("update", "delete"):
            op.execute(
                f"""
                CREATE TRIGGER trg_{table}_append_only_guard_{operation}
                BEFORE {operation.upper()} ON {table} FOR EACH ROW
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'append-only evidence must be superseded by a new linked record'
                    );
                END;
                """
            )
    op.execute(
        f"""
        CREATE TRIGGER {RUN_EVENT_GUARD_TRIGGER}_update
        BEFORE UPDATE ON run_events FOR EACH ROW
        WHEN OLD.id IS NOT NEW.id
          OR OLD.entity_type IS NOT NEW.entity_type
          OR OLD.entity_id IS NOT NEW.entity_id
          OR OLD.event_type IS NOT NEW.event_type
          OR OLD.created_at IS NOT NEW.created_at
          OR json(OLD.payload_json) = json_object('redacted', json('true'))
          OR json(NEW.payload_json) <> json_object('redacted', json('true'))
        BEGIN
            SELECT RAISE(
                ABORT,
                'run-event evidence permits only one-way payload redaction'
            );
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {RUN_EVENT_GUARD_TRIGGER}_delete
        BEFORE DELETE ON run_events FOR EACH ROW
        BEGIN
            SELECT RAISE(ABORT, 'run events are append-only');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {INCIDENT_GUARD_TRIGGER}_update
        BEFORE UPDATE ON incidents FOR EACH ROW
        WHEN OLD.id IS NOT NEW.id
          OR OLD.severity IS NOT NEW.severity
          OR OLD.code IS NOT NEW.code
          OR OLD.battle_id IS NOT NEW.battle_id
          OR OLD.created_at IS NOT NEW.created_at
          OR OLD.detail = '[REDACTED AFTER OPERATIONAL RETENTION]'
          OR NEW.detail <> '[REDACTED AFTER OPERATIONAL RETENTION]'
        BEGIN
            SELECT RAISE(
                ABORT,
                'incident evidence permits only one-way detail redaction'
            );
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {INCIDENT_GUARD_TRIGGER}_delete
        BEFORE DELETE ON incidents FOR EACH ROW
        BEGIN
            SELECT RAISE(ABORT, 'incidents are append-only');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {JOB_GUARD_TRIGGER}_insert
        BEFORE INSERT ON jobs FOR EACH ROW
        WHEN NEW.status <> 'queued' OR NEW.completed_at IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT, 'jobs must be inserted queued without completion time');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {JOB_GUARD_TRIGGER}_update
        BEFORE UPDATE ON jobs FOR EACH ROW
        WHEN OLD.id IS NOT NEW.id
          OR OLD.battle_id IS NOT NEW.battle_id
          OR OLD.kind IS NOT NEW.kind
          OR OLD.payload_json IS NOT NEW.payload_json
          OR OLD.created_at IS NOT NEW.created_at
          OR (
              OLD.status IS NOT NEW.status AND NOT (
                  (OLD.status = 'queued' AND NEW.status IN ('running', 'failed'))
                  OR (OLD.status = 'running' AND NEW.status IN (
                      'queued', 'complete', 'failed', 'uncertain'
                  ))
                  OR (OLD.status = 'uncertain' AND NEW.status = 'failed')
              )
          )
          OR (NEW.status IN ('queued', 'running') AND NEW.completed_at IS NOT NULL)
          OR (
              NEW.status IN ('complete', 'failed', 'uncertain') AND (
                  NEW.completed_at IS NULL OR NEW.completed_at < NEW.created_at
              )
          )
          OR (OLD.completed_at IS NOT NULL AND OLD.completed_at IS NOT NEW.completed_at)
        BEGIN
            SELECT RAISE(ABORT, 'job lifecycle or terminal evidence is invalid');
        END;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {JOB_GUARD_TRIGGER}_delete
        BEFORE DELETE ON jobs FOR EACH ROW
        BEGIN
            SELECT RAISE(ABORT, 'jobs are append-only');
        END;
        """
    )
    # Recreate the 0013 digest trigger, which SQLite drops while rebuilding
    # response_arms to add named checks.
    op.execute(f"DROP TRIGGER IF EXISTS {ARM_DIGEST_TRIGGER}")
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
    # Alembic creates ``alembic_version.version_num`` as VARCHAR(32).  This
    # revision identifier is longer, and PostgreSQL enforces that width when
    # Alembic records the successful upgrade.  Widen it inside the same
    # transaction before the version-table update occurs.
    if bind.dialect.name == "postgresql":
        op.execute(
            "ALTER TABLE alembic_version "
            "ALTER COLUMN version_num TYPE VARCHAR(128)"
        )
    _preflight(bind)
    _create_checks(bind)
    if bind.dialect.name == "postgresql":
        _create_postgresql_guards()
    elif bind.dialect.name == "sqlite":
        _create_sqlite_guards()
    else:
        raise RuntimeError(f"unsupported database dialect for 0014: {bind.dialect.name}")


def _drop_sqlite_guards() -> None:
    for trigger in (
        f"{BATTLE_LINK_TRIGGER}_insert",
        f"{BATTLE_LINK_TRIGGER}_update",
        f"{ARM_GUARD_TRIGGER}_insert",
        f"{ARM_GUARD_TRIGGER}_contract",
        f"{ARM_GUARD_TRIGGER}_lifecycle",
        f"{ARM_GUARD_TRIGGER}_terminal",
        f"{ARM_GUARD_TRIGGER}_result",
        f"{ARM_GUARD_TRIGGER}_delete",
        f"{ASSIGNMENT_GUARD_TRIGGER}_insert",
        f"{ASSIGNMENT_GUARD_TRIGGER}_update",
        f"{ASSIGNMENT_GUARD_TRIGGER}_delete",
        f"{CONTROLLED_RUN_GUARD_TRIGGER}_insert",
        f"{CONTROLLED_RUN_GUARD_TRIGGER}_update",
        f"{CONTROLLED_RUN_GUARD_TRIGGER}_delete",
        f"{VOTE_GUARD_TRIGGER}_insert",
        f"{VOTE_GUARD_TRIGGER}_update",
        f"{VOTE_GUARD_TRIGGER}_delete",
        f"{COST_EVENT_GUARD_TRIGGER}_update",
        f"{COST_EVENT_GUARD_TRIGGER}_delete",
        f"{RUN_EVENT_GUARD_TRIGGER}_update",
        f"{RUN_EVENT_GUARD_TRIGGER}_delete",
        f"{INCIDENT_GUARD_TRIGGER}_update",
        f"{INCIDENT_GUARD_TRIGGER}_delete",
        f"{JOB_GUARD_TRIGGER}_insert",
        f"{JOB_GUARD_TRIGGER}_update",
        f"{JOB_GUARD_TRIGGER}_delete",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    for table in APPEND_ONLY_EVIDENCE_TABLES:
        for operation in ("update", "delete"):
            op.execute(
                f"DROP TRIGGER IF EXISTS trg_{table}_append_only_guard_{operation}"
            )


def _drop_checks(bind: sa.Connection) -> None:
    constraints = {
        "battles": ("ck_battles_distinct_arm_links",),
        "response_arms": ("ck_response_arms_side", "ck_response_arms_status"),
        "votes": ("ck_votes_choice", "ck_votes_cohort"),
    }
    if bind.dialect.name == "sqlite":
        for table, names in constraints.items():
            with op.batch_alter_table(table, recreate="always") as batch:
                for name in names:
                    batch.drop_constraint(name, type_="check")
        return
    for table, names in constraints.items():
        for name in names:
            op.drop_constraint(name, table, type_="check")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for trigger, table in (
            (BATTLE_LINK_TRIGGER, "battles"),
            (ARM_GUARD_TRIGGER, "response_arms"),
            (ASSIGNMENT_GUARD_TRIGGER, "controlled_run_assignments"),
            (CONTROLLED_RUN_GUARD_TRIGGER, "controlled_runs"),
            (VOTE_GUARD_TRIGGER, "votes"),
            (COST_EVENT_GUARD_TRIGGER, "cost_events"),
            (RUN_EVENT_GUARD_TRIGGER, "run_events"),
            (INCIDENT_GUARD_TRIGGER, "incidents"),
            (JOB_GUARD_TRIGGER, "jobs"),
        ):
            op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        for table in APPEND_ONLY_EVIDENCE_TABLES:
            op.execute(
                f"DROP TRIGGER IF EXISTS trg_{table}_append_only_guard ON {table}"
            )
        for function in (
            BATTLE_LINK_FUNCTION,
            ARM_GUARD_FUNCTION,
            ASSIGNMENT_GUARD_FUNCTION,
            CONTROLLED_RUN_GUARD_FUNCTION,
            VOTE_GUARD_FUNCTION,
            COST_EVENT_GUARD_FUNCTION,
            RUN_EVENT_GUARD_FUNCTION,
            INCIDENT_GUARD_FUNCTION,
            JOB_GUARD_FUNCTION,
            APPEND_ONLY_EVIDENCE_FUNCTION,
        ):
            op.execute(f"DROP FUNCTION IF EXISTS {function}()")
    elif bind.dialect.name == "sqlite":
        _drop_sqlite_guards()
    _drop_checks(bind)
    if bind.dialect.name == "sqlite":
        # Restore the 0013 digest trigger after the downgrade table rebuild.
        op.execute(f"DROP TRIGGER IF EXISTS {ARM_DIGEST_TRIGGER}")
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
