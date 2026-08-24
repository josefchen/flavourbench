from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import MetaData, Table, create_engine, event, inspect, select, text
from sqlalchemy.exc import IntegrityError

from flavourbench.database import EXPECTED_SCHEMA_REVISION


def test_fresh_database_upgrades_to_head(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "migration-smoke.sqlite3"
    database_url = f"sqlite:///{database_path}"
    environment = {
        **os.environ,
        "FLAVOURBENCH_DATABASE_URL": database_url,
        "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
    }
    alembic = Path(sys.executable).with_name("alembic")
    result = subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    engine = create_engine(database_url)
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("leaderboard_snapshots")}
    indexes = {index["name"] for index in inspector.get_indexes("leaderboard_snapshots")}
    assert "data_stratum" in columns
    assert "controlled_run_id" in columns
    assert "publication_status" in columns
    assert "evidence_cutoff_at" in columns
    assert "ix_leaderboard_snapshots_data_stratum" in indexes
    assert "ix_leaderboard_snapshots_controlled_run_id" in indexes
    assert "ix_leaderboard_snapshots_publication_status" in indexes
    assert "uq_leaderboard_snapshots_one_published_public_scope" in indexes
    assert "uq_leaderboard_snapshots_one_published_controlled_scope" in indexes
    assert "controlled_runs" in inspector.get_table_names()
    assert "controlled_run_assignments" in inspector.get_table_names()
    assert "epicure_releases" in inspector.get_table_names()
    assert "season_provider_budgets" in inspector.get_table_names()
    assert "provider_account_budgets" in inspector.get_table_names()
    assert "provider_account_authorizations" in inspector.get_table_names()
    assert "bedrock_billing_crosschecks" in inspector.get_table_names()
    assert "bedrock_billing_crosscheck_arms" in inspector.get_table_names()
    assert "research_release_archives" in inspector.get_table_names()
    assert {
        "organizations",
        "organization_api_keys",
        "governance_acceptances",
        "model_submissions",
        "model_route_revisions",
        "route_contract_tests",
        "evaluation_orders",
        "evidence_bundles",
        "api_idempotency_keys",
    } <= set(inspector.get_table_names())
    assert {
        "input_evidence_sha256",
        "input_evidence_json",
        "payload_sha256",
        "supersedes_snapshot_id",
    } <= columns
    season_columns = {column["name"] for column in inspector.get_columns("seasons")}
    battle_columns = {column["name"] for column in inspector.get_columns("battles")}
    arm_columns = {column["name"] for column in inspector.get_columns("response_arms")}
    tool_call_columns = {column["name"] for column in inspector.get_columns("tool_calls")}
    validator_columns = {column["name"] for column in inspector.get_columns("validator_results")}
    validator_indexes = {index["name"] for index in inspector.get_indexes("validator_results")}
    assert {
        "analysis_plan_sha256",
        "protocol_bundle_json",
        "protocol_bundle_sha256",
    } <= season_columns
    assert "protocol_bundle_sha256" in battle_columns
    assert "provider_reservations_json" in battle_columns
    assert "retention_basis" in battle_columns
    assert {
        "observed_decoding_json",
        "protocol_bundle_sha256",
        "epicure_attestation_json",
        "epicure_attestation_sha256",
        "execution_backend",
        "cost_accounting_basis",
        "billing_reconciliation_status",
        "backend_response_schema_sha256",
        "backend_tool_schema_sha256",
        "answer_markdown_sha256",
        "output_json_sha256",
        "route_revision_id",
        "endpoint_descriptor_sha256",
    } <= arm_columns
    assert "organization_id" in {
        column["name"] for column in inspector.get_columns("controlled_runs")
    }
    assert {
        "arguments_sha256",
        "structured_content_sha256",
    } <= tool_call_columns
    assert "detail_sha256" in validator_columns
    assert "uq_validator_results_arm_name_version" in validator_indexes
    assert "account_authorization_envelope_sha256" in {
        column["name"] for column in inspector.get_columns("season_provider_budgets")
    }
    assert {
        "status",
        "opening_used_micros",
        "opening_reserved_micros",
        "opening_balance_json",
        "opening_balance_sha256",
        "credential_binding_json",
        "credential_binding_sha256",
        "authorization_hmac_sha256",
        "revoked_at",
    } <= {column["name"] for column in inspector.get_columns("provider_account_budgets")}
    assert {
        "execution_backend",
        "backend_contract_json",
        "backend_contract_sha256",
    } <= {column["name"] for column in inspector.get_columns("season_models")}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar() == (
            EXPECTED_SCHEMA_REVISION
        )
        trigger_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            ).scalars()
        )
        assert {
            "trg_response_arm_normal_finish_guard_insert",
            "trg_response_arm_normal_finish_guard_update",
            "trg_vote_normal_finish_guard_insert",
            "trg_battle_retention_basis_insert",
            "trg_battle_retention_basis_update",
            "trg_research_release_archives_no_update",
            "trg_research_release_archives_no_delete",
        } <= trigger_names


def test_postgresql_finish_guard_never_schema_qualifies_coalesce() -> None:
    project_root = Path(__file__).resolve().parents[1]
    migration_directory = project_root / "alembic" / "versions"
    corrective_migration = (
        migration_directory / "0022_postgresql_finish_guard_coalesce.py"
    ).read_text(encoding="utf-8")
    release_fence = (
        migration_directory / "0023_postgresql_finish_guard_release_fence.py"
    ).read_text(encoding="utf-8")
    assert hashlib.sha256(corrective_migration.encode()).hexdigest() == (
        "dfe7e5b02e3c053c58952e29a71fcdbaa14466a02171e572c16c3af872959b21"
    )
    assert "pg_catalog.coalesce" not in corrective_migration.lower()
    assert corrective_migration.count("COALESCE(") == 3
    assert "pg_catalog.coalesce(" not in release_fence.lower()
    assert release_fence.count("COALESCE(") == 3
    assert release_fence.count("CREATE OR REPLACE FUNCTION") == 2
    assert release_fence.count("CREATE TRIGGER") == 2
    assert "PostgreSQL downgrade across 0023 is prohibited" in release_fence


def test_commercial_evidence_invariants_are_database_enforced(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'commercial-evidence-invariants.sqlite3'}"
    environment = {
        **os.environ,
        "FLAVOURBENCH_DATABASE_URL": database_url,
        "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
    }
    alembic = Path(sys.executable).with_name("alembic")
    result = subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    engine = create_engine(database_url)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    metadata = MetaData()
    seasons = Table("seasons", metadata, autoload_with=engine)
    models = Table("catalog_models", metadata, autoload_with=engine)
    tasks = Table("tasks", metadata, autoload_with=engine)
    runs = Table("controlled_runs", metadata, autoload_with=engine)
    assignments = Table("controlled_run_assignments", metadata, autoload_with=engine)
    battles = Table("battles", metadata, autoload_with=engine)
    arms = Table("response_arms", metadata, autoload_with=engine)
    tool_calls = Table("tool_calls", metadata, autoload_with=engine)
    validator_results = Table("validator_results", metadata, autoload_with=engine)
    votes = Table("votes", metadata, autoload_with=engine)
    generation_attempts = Table("generation_attempts", metadata, autoload_with=engine)
    admission_events = Table("admission_events", metadata, autoload_with=engine)
    provider_account_budgets = Table("provider_account_budgets", metadata, autoload_with=engine)
    season_provider_budgets = Table("season_provider_budgets", metadata, autoload_with=engine)
    billing_crosschecks = Table("bedrock_billing_crosschecks", metadata, autoload_with=engine)
    billing_crosscheck_arms = Table(
        "bedrock_billing_crosscheck_arms", metadata, autoload_with=engine
    )
    run_events = Table("run_events", metadata, autoload_with=engine)
    incidents = Table("incidents", metadata, autoload_with=engine)
    now = datetime.now(UTC) - timedelta(days=40)
    arm_completed_at = now + timedelta(milliseconds=100)
    battle_completed_at = now + timedelta(milliseconds=200)
    later = now + timedelta(seconds=1)
    prompt_sha256 = "1" * 64

    with engine.begin() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        connection.execute(
            seasons.insert().values(
                id="invariant-season",
                slug="invariant-season",
                name="Invariant test season",
                status="active",
                official=True,
                manifest_sha256="2" * 64,
                prompt_registry_sha256="3" * 64,
                tool_registry_sha256="4" * 64,
                epicure_release_id="invariant-release",
                epicure_bundle_sha256="5" * 64,
                epicure_application_sha256="6" * 64,
                budget_cap_micros=1000,
                budget_used_micros=20,
                budget_reserved_micros=0,
                created_at=now,
            )
        )
        for model_id in ("invariant/model-a", "invariant/model-b"):
            connection.execute(
                models.insert().values(
                    model_id=model_id,
                    canonical_slug=model_id,
                    name=model_id,
                    family="invariant",
                    open_weight=False,
                    status="smoke_passed",
                    supports_tools=True,
                    supports_structured_outputs=True,
                    pricing_json={},
                    endpoint_json={},
                    discovered_at=now,
                    last_seen_at=now,
                )
            )
        connection.execute(
            tasks.insert().values(
                id="invariant-task",
                public_id="invariant-task",
                season_id="invariant-season",
                family="composition",
                prompt="Compose a test dish.",
                prompt_sha256=prompt_sha256,
                revision=1,
                split="test",
                review_status="frozen",
                provenance_json={},
                created_at=now,
            )
        )
        connection.execute(
            runs.insert().values(
                id="invariant-run",
                season_id="invariant-season",
                organization_reference_sha256="7" * 64,
                access_token_sha256="8" * 64,
                status="active",
                protocol_version="invariant-v1",
                rater_plan_sha256="9" * 64,
                analysis_plan_sha256="a" * 64,
                run_card_json={},
                run_card_sha256="b" * 64,
                run_card_signature="c" * 64,
                budget_cap_micros=1000,
                budget_used_micros=20,
                created_at=now,
            )
        )
        connection.execute(
            battles.insert().values(
                id="invariant-battle",
                season_id="invariant-season",
                run_class="official",
                rank_eligible=True,
                data_stratum="controlled",
                task_id="invariant-task",
                task_revision=1,
                controlled_run_id="invariant-run",
                manifest_sha256="2" * 64,
                protocol_bundle_sha256="d" * 64,
                scheduler_version="controlled-frozen-schedule-v1",
                assignment_seed="0" * 64,
                track_assignment_probability="1/1",
                model_assignment_probability="1/1",
                side_assignment_probability="1/2",
                track="model_arena",
                category="composition",
                prompt="Compose a test dish.",
                prompt_sha256=prompt_sha256,
                client_nonce_sha256="9" * 64,
                prompt_redacted=False,
                research_consent=False,
                retention_basis="official_research",
                release_review_status="not_requested",
                requester_pseudonym="8" * 64,
                status="queued",
                reserved_cost_micros=0,
                created_at=now,
                retention_until=now + timedelta(days=30),
            )
        )
        connection.execute(
            battles.insert().values(
                id="future-retention-battle",
                season_id="invariant-season",
                run_class="exploratory",
                rank_eligible=False,
                data_stratum="public_freeform",
                manifest_sha256="2" * 64,
                protocol_bundle_sha256="d" * 64,
                scheduler_version="coverage-balanced-server-random-v1",
                assignment_seed="1" * 64,
                track_assignment_probability="1/1",
                model_assignment_probability="1/1",
                side_assignment_probability="1/2",
                track="model_arena",
                category="composition",
                prompt="Future-retained prompt.",
                prompt_sha256="a" * 64,
                client_nonce_sha256="b" * 64,
                prompt_redacted=False,
                research_consent=False,
                retention_basis="public_nonconsented",
                release_review_status="not_requested",
                requester_pseudonym="c" * 64,
                status="queued",
                reserved_cost_micros=0,
                created_at=datetime.now(UTC),
                retention_until=datetime.now(UTC) + timedelta(days=30),
            )
        )
        connection.execute(
            battles.insert().values(
                id="consented-retention-battle",
                season_id="invariant-season",
                run_class="exploratory",
                rank_eligible=False,
                data_stratum="public_freeform",
                manifest_sha256="2" * 64,
                protocol_bundle_sha256="d" * 64,
                scheduler_version="coverage-balanced-server-random-v1",
                assignment_seed="2" * 64,
                track_assignment_probability="1/1",
                model_assignment_probability="1/1",
                side_assignment_probability="1/2",
                track="model_arena",
                category="composition",
                prompt="Consented prompt.",
                prompt_sha256="d" * 64,
                client_nonce_sha256="e" * 64,
                prompt_redacted=False,
                research_consent=True,
                retention_basis="public_consented",
                release_review_status="pending",
                requester_pseudonym="f" * 64,
                status="queued",
                reserved_cost_micros=0,
                created_at=now,
                retention_until=now + timedelta(days=30),
            )
        )
        for side, model_id in (
            ("left", "invariant/model-a"),
            ("right", "invariant/model-b"),
        ):
            answer = f"answer-{side}"
            connection.execute(
                arms.insert().values(
                    id=f"invariant-{side}-arm",
                    battle_id="invariant-battle",
                    side=side,
                    condition="epicure_on",
                    model_id=model_id,
                    execution_backend="openrouter",
                    provider_slug="invariant-provider",
                    provider_generation_ids_json=[],
                    status="queued",
                    output_json={},
                    prompt_sha256=prompt_sha256,
                    system_prompt_sha256="5" * 64,
                    schema_sha256="6" * 64,
                    tool_schema_sha256="4" * 64,
                    decoding_json={},
                    observed_decoding_json={},
                    protocol_bundle_sha256="d" * 64,
                    epicure_release_id="invariant-release",
                    epicure_bundle_sha256="5" * 64,
                    epicure_application_sha256="6" * 64,
                    epicure_attestation_json={},
                    prompt_tokens=10,
                    completion_tokens=10,
                    reasoning_tokens=0,
                    cost_micros=0,
                    cost_reconciled=False,
                    latency_ms=0,
                    retries=0,
                    created_at=now,
                )
            )
            terminal_values = {
                "actual_provider_slug": "invariant-provider",
                "actual_model_id": model_id,
                "generation_id": f"generation-{side}",
                "provider_generation_ids_json": [f"generation-{side}"],
                "status": "complete",
                "answer_markdown": answer,
                "answer_markdown_sha256": ("1" if side == "left" else "2") * 64,
                "output_json": {"answer_markdown": answer},
                "output_json_sha256": ("3" if side == "left" else "4") * 64,
                "cost_micros": 10,
                "cost_reconciled": True,
                "cost_accounting_basis": "generation_metadata",
                "billing_reconciliation_status": "complete",
                "latency_ms": 100,
                "finish_reason": "stop",
                "completed_at": arm_completed_at,
            }
            if side == "left":
                with pytest.raises(
                    IntegrityError,
                    match="normal provider finish reason",
                ):
                    connection.execute(
                        arms.update()
                        .where(arms.c.id == f"invariant-{side}-arm")
                        .values(**{**terminal_values, "finish_reason": "length"})
                    )
            connection.execute(
                arms.update()
                .where(arms.c.id == f"invariant-{side}-arm")
                .values(**terminal_values)
            )
        connection.execute(
            tool_calls.insert().values(
                id="invariant-tool-call",
                arm_id="invariant-left-arm",
                round_index=0,
                call_index=0,
                tool_call_id="call-invariant",
                tool_name="ingredient_pairing",
                arguments_json={"ingredient": "test"},
                arguments_sha256="a" * 64,
                result_text="sensitive tool result",
                structured_content_json={"result": "sensitive"},
                structured_content_sha256="b" * 64,
                result_sha256="c" * 64,
                latency_ms=1,
                is_error=False,
                created_at=now,
            )
        )
        connection.execute(
            validator_results.insert().values(
                id="invariant-validator-result",
                arm_id="invariant-left-arm",
                validator_name="constraint_test",
                validator_version="v1",
                status="pass",
                score_milli=1000,
                detail_json={"detail": "sensitive"},
                detail_sha256="d" * 64,
                created_at=now,
            )
        )
        connection.execute(
            battles.update()
            .where(battles.c.id == "invariant-battle")
            .values(
                left_arm_id="invariant-left-arm",
                right_arm_id="invariant-right-arm",
            )
        )
        connection.execute(
            assignments.insert().values(
                id="invariant-assignment",
                controlled_run_id="invariant-run",
                ordinal=0,
                task_id="invariant-task",
                task_public_id="invariant-task",
                task_revision=1,
                task_prompt_sha256=prompt_sha256,
                task_family="composition",
                track="model_arena",
                model_ids_json=["invariant/model-a", "invariant/model-b"],
                repetition_index=1,
                assignment_sha256="7" * 64,
                assignment_seed="0" * 64,
                status="pending",
                created_at=now,
            )
        )
        connection.execute(
            assignments.update()
            .where(assignments.c.id == "invariant-assignment")
            .values(status="queued", battle_id="invariant-battle")
        )
        connection.execute(
            battles.update()
            .where(battles.c.id == "invariant-battle")
            .values(status="complete", completed_at=battle_completed_at)
        )
        connection.execute(
            votes.insert().values(
                id="invariant-vote",
                battle_id="invariant-battle",
                rater_pseudonym="8" * 64,
                cohort="public",
                choice="left",
                reason_tags_json=[],
                rubric_json={},
                idempotency_key="invariant-vote-key",
                created_at=later,
            )
        )
        connection.execute(
            generation_attempts.insert().values(
                id="invariant-attempt-event",
                attempt_id="invariant-attempt",
                arm_id="invariant-left-arm",
                request_key_sha256="a" * 64,
                phase="initial",
                attempt_index=0,
                event_type="request_started",
                payload_sha256="b" * 64,
                metadata_json={},
                created_at=later,
            )
        )
        connection.execute(
            admission_events.insert().values(
                id="invariant-admission",
                pseudonym="c" * 64,
                action="create_battle",
                admitted=True,
                reason="within_limit",
                created_at=later,
            )
        )
        connection.execute(
            run_events.insert().values(
                id="invariant-run-event",
                entity_type="battle",
                entity_id="invariant-battle",
                event_type="invariant_recorded",
                payload_json={"sensitive": "value"},
                created_at=later,
            )
        )
        connection.execute(
            incidents.insert().values(
                id="invariant-incident",
                severity="info",
                code="InvariantRecorded",
                detail="sensitive detail",
                battle_id="invariant-battle",
                created_at=later,
            )
        )
        connection.execute(
            provider_account_budgets.insert().values(
                id="invariant-provider-account",
                execution_backend="bedrock",
                currency="USD",
                status="active",
                budget_cap_micros=1000,
                budget_used_micros=20,
                budget_reserved_micros=0,
                opening_used_micros=0,
                opening_reserved_micros=0,
                account_scope_sha256="d" * 64,
                authorization_reference_sha256="e" * 64,
                opening_balance_json={},
                opening_balance_sha256="f" * 64,
                credential_binding_json={},
                credential_binding_sha256="0" * 64,
                authorization_envelope_json={},
                authorization_envelope_sha256="1" * 64,
                authorization_hmac_sha256="2" * 64,
                valid_until=later + timedelta(days=30),
                created_at=now,
            )
        )
        connection.execute(
            season_provider_budgets.insert().values(
                id="invariant-season-provider-budget",
                season_id="invariant-season",
                execution_backend="bedrock",
                currency="USD",
                budget_cap_micros=1000,
                budget_used_micros=20,
                budget_reserved_micros=0,
                account_scope_sha256="d" * 64,
                authorization_reference_sha256="e" * 64,
                account_authorization_envelope_sha256="3" * 64,
                authorization_envelope_json={},
                authorization_envelope_sha256="4" * 64,
                valid_until=datetime.now(UTC) + timedelta(days=30),
                created_at=now,
            )
        )
        connection.execute(
            billing_crosschecks.insert().values(
                id="invariant-crosscheck",
                season_id="invariant-season",
                provider_account_budget_id="invariant-provider-account",
                status="accepted",
                source_kind="aws_cost_explorer",
                source_artifact_uri="s3://example/invariant.json",
                source_artifact_sha256="3" * 64,
                statement_sha256="4" * 64,
                coverage_start=now,
                coverage_end=later,
                arm_set_sha256="5" * 64,
                generation_request_map_sha256="6" * 64,
                rate_card_estimated_micros=10,
                billed_usage_micros=10,
                billing_difference_micros=0,
                ledger_delta_micros=0,
                tolerance_micros=1,
                credits_policy="gross_usage_before_credits",
                authorization_reference_sha256="7" * 64,
                evidence_json={},
                evidence_sha256="8" * 64,
                created_at=later,
            )
        )
        connection.execute(
            billing_crosscheck_arms.insert().values(
                id="invariant-crosscheck-arm",
                crosscheck_id="invariant-crosscheck",
                arm_id="invariant-left-arm",
                generation_set_sha256="9" * 64,
                created_at=later,
            )
        )

    inspector = inspect(engine)
    assert {row["name"] for row in inspector.get_check_constraints("battles")} >= {
        "ck_battles_distinct_arm_links"
    }
    assert {row["name"] for row in inspector.get_check_constraints("response_arms")} >= {
        "ck_response_arms_side",
        "ck_response_arms_status",
    }
    assert {row["name"] for row in inspector.get_check_constraints("votes")} >= {
        "ck_votes_choice",
        "ck_votes_cohort",
    }

    for table, row_id in (
        (generation_attempts, "invariant-attempt-event"),
        (admission_events, "invariant-admission"),
        (billing_crosschecks, "invariant-crosscheck"),
        (billing_crosscheck_arms, "invariant-crosscheck-arm"),
        (run_events, "invariant-run-event"),
        (incidents, "invariant-incident"),
    ):
        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    table.update()
                    .where(table.c.id == row_id)
                    .values(created_at=later + timedelta(seconds=1))
                )
        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.execute(table.delete().where(table.c.id == row_id))

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                battles.update()
                .where(battles.c.id == "invariant-battle")
                .values(
                    left_arm_id="invariant-right-arm",
                    right_arm_id="invariant-left-arm",
                )
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                arms.update()
                .where(arms.c.id == "invariant-left-arm")
                .values(actual_model_id="rewritten/model")
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                arms.update()
                .where(arms.c.id == "invariant-left-arm")
                .values(answer_markdown=None, output_json={"redacted": True})
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                battles.update()
                .where(battles.c.id == "invariant-battle")
                .values(prompt="rewritten prompt")
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                battles.update()
                .where(battles.c.id == "invariant-battle")
                .values(retention_until=later + timedelta(days=30))
            )

    for battle_id in ("future-retention-battle", "consented-retention-battle"):
        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    battles.update()
                    .where(battles.c.id == battle_id)
                    .values(prompt=None, prompt_redacted=True)
                )

    for table, row_id in (
        (seasons, "invariant-season"),
        (season_provider_budgets, "invariant-season-provider-budget"),
        (provider_account_budgets, "invariant-provider-account"),
        (runs, "invariant-run"),
    ):
        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    table.update().where(table.c.id == row_id).values(budget_used_micros=19)
                )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                run_events.update()
                .where(run_events.c.id == "invariant-run-event")
                .values(payload_json={"redacted": True})
            )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                incidents.update()
                .where(incidents.c.id == "invariant-incident")
                .values(detail="[REDACTED AFTER OPERATIONAL RETENTION]")
            )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                tool_calls.update()
                .where(tool_calls.c.id == "invariant-tool-call")
                .values(
                    arguments_json={"redacted": True},
                    result_text="[REDACTED AFTER OPERATIONAL RETENTION]",
                    structured_content_json={"redacted": True},
                )
            )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                validator_results.update()
                .where(validator_results.c.id == "invariant-validator-result")
                .values(detail_json={"redacted": True})
            )

    with pytest.raises(IntegrityError, match="expired redactable basis"):
        with engine.begin() as connection:
            connection.execute(
                battles.update()
                .where(battles.c.id == "invariant-battle")
                .values(prompt=None, prompt_redacted=True)
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                battles.update()
                .where(battles.c.id == "invariant-battle")
                .values(prompt="restored prompt", prompt_redacted=False)
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                assignments.update()
                .where(assignments.c.id == "invariant-assignment")
                .values(battle_id=None)
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                votes.insert().values(
                    id="early-vote",
                    battle_id="invariant-battle",
                    rater_pseudonym="9" * 64,
                    cohort="public",
                    choice="right",
                    reason_tags_json=[],
                    rubric_json={},
                    idempotency_key="early-vote-key",
                    created_at=now,
                )
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                votes.update().where(votes.c.id == "invariant-vote").values(choice="right")
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(votes.delete().where(votes.c.id == "invariant-vote"))

    with engine.begin() as connection:
        connection.execute(
            assignments.update()
            .where(assignments.c.id == "invariant-assignment")
            .values(status="cancelled")
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                assignments.update()
                .where(assignments.c.id == "invariant-assignment")
                .values(status="queued")
            )

    engine.dispose()
    result = subprocess.run(
        [str(alembic), "downgrade", "0013_snapshot_evidence_seal"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "downgrade across reviewer identity, task-validation ballots" in (
        result.stdout + result.stderr
    )


def test_snapshot_publication_constraints_are_database_enforced(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'snapshot-publication-constraints.sqlite3'}"
    environment = {
        **os.environ,
        "FLAVOURBENCH_DATABASE_URL": database_url,
        "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
    }
    alembic = Path(sys.executable).with_name("alembic")
    result = subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    engine = create_engine(database_url)
    insert_snapshot = text(
        "INSERT INTO leaderboard_snapshots "
        "(id, season_id, track, cohort, category, data_stratum, "
        "publication_status, input_sha256, input_evidence_sha256, "
        "input_evidence_json, payload_sha256, payload_json, created_at, evidence_cutoff_at) "
        "VALUES (:id, 'season-publication-constraints', 'model_arena', 'public', "
        "'all', 'public_freeform', 'draft', :digest, :evidence_digest, "
        "'{}', :payload_digest, '{}', :created_at, :created_at)"
    )
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            insert_snapshot,
            {
                "id": "snapshot-a",
                "digest": "a" * 64,
                "evidence_digest": "1" * 64,
                "payload_digest": "a" * 64,
                "created_at": now,
            },
        )
        connection.execute(
            insert_snapshot,
            {
                "id": "snapshot-b",
                "digest": "b" * 64,
                "evidence_digest": "2" * 64,
                "payload_digest": "b" * 64,
                "created_at": now,
            },
        )
        connection.execute(
            text(
                "UPDATE leaderboard_snapshots SET publication_status = 'published', "
                "publication_reference_sha256 = :reference, published_at = :published_at "
                "WHERE id = 'snapshot-a'"
            ),
            {"reference": "c" * 64, "published_at": now},
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE leaderboard_snapshots SET publication_status = 'draft' "
                    "WHERE id = 'snapshot-a'"
                )
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO leaderboard_snapshots "
                    "(id, season_id, track, cohort, category, data_stratum, "
                    "publication_status, input_sha256, payload_json, created_at, "
                    "evidence_cutoff_at) VALUES "
                    "('snapshot-unsealed-draft', 'season-publication-constraints', "
                    "'model_arena', 'public', 'all', 'public_freeform', 'draft', "
                    ":digest, '{}', :created_at, :created_at)"
                ),
                {"digest": "f" * 64, "created_at": now},
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE leaderboard_snapshots SET publication_status = 'published', "
                    "publication_reference_sha256 = :reference, published_at = :published_at "
                    "WHERE id = 'snapshot-b'"
                ),
                {"reference": "d" * 64, "published_at": now},
            )

    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE leaderboard_snapshots SET publication_status = 'withdrawn' "
                "WHERE id = 'snapshot-a'"
            )
        )
        connection.execute(
            text(
                "UPDATE leaderboard_snapshots SET publication_status = 'published', "
                "publication_reference_sha256 = :reference, published_at = :published_at "
                "WHERE id = 'snapshot-b'"
            ),
            {"reference": "d" * 64, "published_at": now},
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE leaderboard_snapshots SET publication_status = 'published' "
                    "WHERE id = 'snapshot-a'"
                )
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE leaderboard_snapshots SET payload_json = :payload "
                    "WHERE id = 'snapshot-b'"
                ),
                {"payload": '{"rewritten":true}'},
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM leaderboard_snapshots WHERE id = 'snapshot-b'"))

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO leaderboard_snapshots "
                    "(id, season_id, track, cohort, category, data_stratum, "
                    "publication_status, input_sha256, payload_json, created_at, "
                    "evidence_cutoff_at) VALUES "
                    "('snapshot-direct-published', 'season-publication-constraints', "
                    "'model_arena', 'public', 'all', 'public_freeform', 'published', "
                    ":digest, '{}', :created_at, :created_at)"
                ),
                {"digest": "e" * 64, "created_at": now},
            )


def test_tool_call_trace_is_database_sealed_with_one_way_redaction(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'tool-call-trace-seal.sqlite3'}"
    environment = {
        **os.environ,
        "FLAVOURBENCH_DATABASE_URL": database_url,
        "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
    }
    alembic = Path(sys.executable).with_name("alembic")
    result = subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    engine = create_engine(database_url)
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO tool_calls "
                "(id, arm_id, round_index, call_index, tool_call_id, tool_name, "
                "arguments_json, arguments_sha256, result_text, result_sha256, "
                "structured_content_json, structured_content_sha256, latency_ms, "
                "is_error, created_at) VALUES "
                "('tool-seal', 'arm-seal', 0, 0, 'provider-call', 'find_pairings', "
                ":arguments, :arguments_sha256, 'original result', :result_sha256, "
                ":structured, :structured_sha256, 12, 0, :created_at)"
            ),
            {
                "arguments": '{"ingredient":"tomato"}',
                "arguments_sha256": "a" * 64,
                "result_sha256": "b" * 64,
                "structured": '{"pairs":[]}',
                "structured_sha256": "c" * 64,
                "created_at": now,
            },
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE tool_calls SET result_text = 'rewritten result' WHERE id = 'tool-seal'"
                )
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE tool_calls SET arguments_json = :redacted, "
                    "result_text = '[REDACTED AFTER OPERATIONAL RETENTION]', "
                    "structured_content_json = :redacted WHERE id = 'tool-seal'"
                ),
                {"redacted": '{"redacted":true}'},
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE tool_calls SET tool_name = 'rewritten' WHERE id = 'tool-seal'")
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM tool_calls WHERE id = 'tool-seal'"))


def test_validator_results_are_unique_and_database_sealed(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'validator-result-seal.sqlite3'}"
    environment = {
        **os.environ,
        "FLAVOURBENCH_DATABASE_URL": database_url,
        "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
    }
    alembic = Path(sys.executable).with_name("alembic")
    result = subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    engine = create_engine(database_url)
    now = datetime.now(UTC)
    insert_validator = text(
        "INSERT INTO validator_results "
        "(id, arm_id, validator_name, validator_version, status, score_milli, "
        "detail_json, detail_sha256, created_at) VALUES "
        "(:id, 'arm-validator', 'identity_blinding', 'validator-v1', 'pass', 1000, "
        "'{}', :detail_sha256, :created_at)"
    )
    with engine.begin() as connection:
        connection.execute(
            insert_validator,
            {"id": "validator-a", "detail_sha256": "a" * 64, "created_at": now},
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                insert_validator,
                {"id": "validator-b", "detail_sha256": "b" * 64, "created_at": now},
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE validator_results SET status = 'fail' WHERE id = 'validator-a'")
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE validator_results SET detail_json = :redacted WHERE id = 'validator-a'"
                ),
                {"redacted": '{"redacted":true}'},
            )

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM validator_results WHERE id = 'validator-a'"))


def test_upgrade_refuses_incomplete_claimed_0006_schema(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "migration-from-0006.sqlite3"
    database_url = f"sqlite:///{database_path}"
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE seasons (id VARCHAR(36) PRIMARY KEY, slug VARCHAR(80) NOT NULL)")
        )
        connection.execute(
            text(
                "CREATE TABLE battles ("
                "id VARCHAR(36) PRIMARY KEY, season_id VARCHAR(36) NOT NULL, "
                "track VARCHAR(32) NOT NULL, status VARCHAR(32) NOT NULL)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE leaderboard_snapshots ("
                "id VARCHAR(36) PRIMARY KEY, season_id VARCHAR(36) NOT NULL, "
                "track VARCHAR(32) NOT NULL, cohort VARCHAR(48) NOT NULL, "
                "category VARCHAR(64) NOT NULL, input_sha256 VARCHAR(64) NOT NULL, "
                "payload_json JSON NOT NULL, created_at DATETIME NOT NULL)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE expert_reviewers ("
                "id VARCHAR(36) PRIMARY KEY, reviewer_code VARCHAR(80) NOT NULL)"
            )
        )
        connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(64) PRIMARY KEY)")
        )
        connection.execute(
            text("INSERT INTO alembic_version(version_num) VALUES ('0006_widen_rater_cohorts')")
        )

    environment = {
        **os.environ,
        "FLAVOURBENCH_DATABASE_URL": database_url,
        "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
    }
    alembic = Path(sys.executable).with_name("alembic")
    result = subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "0014 integrity preflight requires battles columns" in (result.stdout + result.stderr)


def test_retention_policy_downgrade_is_prohibited(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'migration-round-trip.sqlite3'}"
    environment = {
        **os.environ,
        "FLAVOURBENCH_DATABASE_URL": database_url,
        "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
    }
    alembic = Path(sys.executable).with_name("alembic")
    result = subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [str(alembic), "downgrade", "0026_retention_basis"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "downgrade across reviewer identity, task-validation ballots" in (
        result.stdout + result.stderr
    )
    engine = create_engine(database_url)
    inspector = inspect(engine)
    assert "controlled_run_assignments" in inspector.get_table_names()
    assert "rate_card_sha256" in {
        column["name"] for column in inspector.get_columns("season_models")
    }
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar() == (
            EXPECTED_SCHEMA_REVISION
        )


def test_upgrade_from_populated_0010_preserves_zero_spend_authorization(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    database_url = f"sqlite:///{tmp_path / 'migration-from-populated-0010.sqlite3'}"
    environment = {
        **os.environ,
        "FLAVOURBENCH_DATABASE_URL": database_url,
        "FLAVOURBENCH_AUTO_CREATE_SCHEMA": "false",
    }
    alembic = Path(sys.executable).with_name("alembic")
    result = subprocess.run(
        [str(alembic), "upgrade", "0010_lineage_budgets"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    engine = create_engine(database_url)
    metadata = MetaData()
    seasons = Table("seasons", metadata, autoload_with=engine)
    account_budgets = Table("provider_account_budgets", metadata, autoload_with=engine)
    season_budgets = Table("season_provider_budgets", metadata, autoload_with=engine)
    now = datetime.now(UTC)
    legacy_envelope_sha256 = "a" * 64
    with engine.begin() as connection:
        connection.execute(
            seasons.insert().values(
                id="legacy-season",
                slug="legacy-frozen-season",
                name="Legacy frozen season",
                status="active",
                official=False,
                manifest_sha256="b" * 64,
                prompt_registry_sha256="c" * 64,
                tool_registry_sha256="d" * 64,
                epicure_release_id="legacy-release",
                epicure_bundle_sha256="e" * 64,
                epicure_application_sha256="f" * 64,
                analysis_plan_sha256="1" * 64,
                protocol_bundle_json={},
                protocol_bundle_sha256="2" * 64,
                budget_cap_micros=5_000_000_000,
                budget_used_micros=0,
                budget_reserved_micros=0,
                created_at=now,
                frozen_at=now,
            )
        )
        connection.execute(
            account_budgets.insert().values(
                id="legacy-account-authorization",
                execution_backend="bedrock",
                currency="USD",
                budget_cap_micros=5_000_000_000,
                budget_used_micros=0,
                budget_reserved_micros=0,
                account_scope_sha256="3" * 64,
                authorization_reference_sha256="4" * 64,
                authorization_envelope_json={"schema_version": "legacy-v1"},
                authorization_envelope_sha256=legacy_envelope_sha256,
                valid_until=now + timedelta(days=30),
                created_at=now,
            )
        )
        connection.execute(
            season_budgets.insert().values(
                id="legacy-season-authorization",
                season_id="legacy-season",
                execution_backend="bedrock",
                currency="USD",
                budget_cap_micros=5_000_000_000,
                budget_used_micros=0,
                budget_reserved_micros=0,
                account_scope_sha256="3" * 64,
                authorization_reference_sha256="4" * 64,
                authorization_envelope_json={"schema_version": "legacy-season-v1"},
                authorization_envelope_sha256="5" * 64,
                valid_until=now + timedelta(days=30),
                created_at=now,
            )
        )

    result = subprocess.run(
        [str(alembic), "upgrade", "head"],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    migrated_metadata = MetaData()
    migrated_accounts = Table("provider_account_budgets", migrated_metadata, autoload_with=engine)
    migrated_season_budgets = Table(
        "season_provider_budgets", migrated_metadata, autoload_with=engine
    )
    with engine.connect() as connection:
        account = (
            connection.execute(
                select(migrated_accounts).where(
                    migrated_accounts.c.id == "legacy-account-authorization"
                )
            )
            .mappings()
            .one()
        )
        season_authorization = (
            connection.execute(
                select(migrated_season_budgets).where(
                    migrated_season_budgets.c.id == "legacy-season-authorization"
                )
            )
            .mappings()
            .one()
        )
        assert account["status"] == "pending_verification"
        assert account["budget_used_micros"] == 0
        assert account["budget_reserved_micros"] == 0
        assert account["opening_used_micros"] == 0
        assert account["opening_reserved_micros"] == 0
        assert account["authorization_envelope_sha256"] == legacy_envelope_sha256
        assert account["opening_balance_json"]["migration_status"] == (
            "revoked_requires_reprovision"
        )
        assert len(account["opening_balance_sha256"]) == 64
        assert len(account["credential_binding_sha256"]) == 64
        assert season_authorization["account_authorization_envelope_sha256"] == "unresolved"
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar() == (
            EXPECTED_SCHEMA_REVISION
        )
