from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from flavourbench.database import init_database, session_scope
from flavourbench.engine import reconcile_battle_cost
from flavourbench.main import _canonical_sha256, _snapshot_evidence_manifest
from flavourbench.models import (
    Battle,
    CatalogModel,
    ControlledRun,
    ControlledRunAssignment,
    LeaderboardSnapshot,
    ResponseArm,
    Season,
    SeasonModel,
    Task,
    ToolCall,
    ValidatorResult,
    Vote,
)
from flavourbench.service_ranking import model_leaderboard, snapshot_hash
from flavourbench.validators import VALIDATOR_VERSION


def _identifier(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def test_snapshot_manifest_binds_ranking_contract_fields() -> None:
    init_database()
    completed_at = datetime.now(UTC)
    evidence_cutoff_at = completed_at + timedelta(minutes=1)
    season_id = _identifier("season")
    season_slug = _identifier("snapshot-manifest")
    model_id = _identifier("provider/model")
    right_model_id = _identifier("provider/right-model")
    battle_id = _identifier("battle")
    controlled_run_id = _identifier("controlled-run")
    task_id = _identifier("task")
    assignment_id = _identifier("assignment")
    left_id = _identifier("arm-left")
    right_id = _identifier("arm-right")
    manifest_sha256 = "a" * 64
    protocol_sha256 = "b" * 64

    with session_scope() as session:
        season = Season(
            id=season_id,
            slug=season_slug,
            name="Snapshot manifest binding test",
            status="active",
            official=True,
            manifest_sha256=manifest_sha256,
            prompt_registry_sha256="c" * 64,
            tool_registry_sha256="d" * 64,
            epicure_release_id="epicure-binding-test",
            epicure_bundle_sha256="e" * 64,
            epicure_application_sha256="f" * 64,
            analysis_plan_sha256="1" * 64,
            protocol_bundle_json={"test": True},
            protocol_bundle_sha256=protocol_sha256,
        )
        model = CatalogModel(
            model_id=model_id,
            canonical_slug=model_id,
            name="Binding model",
            family="binding",
            catalog_source="test",
            status="discovered",
        )
        right_model = CatalogModel(
            model_id=right_model_id,
            canonical_slug=right_model_id,
            name="Binding right model",
            family="binding-right",
            catalog_source="test",
            status="discovered",
        )
        slot = SeasonModel(
            season_id=season_id,
            model_id=model_id,
            slot_role="binding",
            execution_backend="openrouter",
            provider_slug="provider-a",
            expected_actual_model_id=model_id,
            expected_actual_provider_slug="provider-a",
            supported_parameters_json=["max_tokens"],
            decoding_json={"max_tokens": 64},
            endpoint_max_completion_tokens=64,
            endpoint_document_sha256="2" * 64,
            endpoint_contract_sha256="3" * 64,
            backend_contract_json={"route": "fixed"},
            backend_contract_sha256="4" * 64,
            rate_card_json={"prompt": "0.1"},
            rate_card_sha256="5" * 64,
            eligible=True,
            worst_case_cost_micros=100,
            manifest_sha256=manifest_sha256,
        )
        right_slot = SeasonModel(
            season_id=season_id,
            model_id=right_model_id,
            slot_role="binding-right",
            execution_backend="openrouter",
            provider_slug="provider-a",
            expected_actual_model_id=right_model_id,
            expected_actual_provider_slug="provider-a",
            supported_parameters_json=["max_tokens"],
            decoding_json={"max_tokens": 64},
            endpoint_max_completion_tokens=64,
            endpoint_document_sha256="6" * 64,
            endpoint_contract_sha256="7" * 64,
            backend_contract_json={"route": "fixed-right"},
            backend_contract_sha256="8" * 64,
            rate_card_json={"prompt": "0.1"},
            rate_card_sha256="9" * 64,
            eligible=True,
            worst_case_cost_micros=100,
            manifest_sha256=manifest_sha256,
        )
        task_prompt = "Explain the pairing evidence."
        task = Task(
            id=task_id,
            public_id=_identifier("task-public"),
            season_id=season_id,
            family="evidence",
            prompt=task_prompt,
            prompt_sha256=hashlib.sha256(task_prompt.encode()).hexdigest(),
            revision=1,
            split="test",
            review_status="approved",
            provenance_json={"source": "test"},
        )
        controlled_run = ControlledRun(
            id=controlled_run_id,
            season_id=season_id,
            organization_reference_sha256="6" * 64,
            access_token_sha256="7" * 64,
            status="active",
            protocol_version="snapshot-binding-v1",
            rater_plan_sha256="8" * 64,
            analysis_plan_sha256=season.analysis_plan_sha256,
            model_roster_json=[model_id, right_model_id],
            model_roster_sha256="9" * 64,
            task_schedule_sha256="a" * 64,
            budget_cap_micros=1000,
            run_card_json={"test": True},
            run_card_sha256="b" * 64,
            run_card_signature="c" * 64,
        )
        battle = Battle(
            id=battle_id,
            season_id=season_id,
            run_class="official",
            rank_eligible=True,
            data_stratum="controlled",
            task_id=task_id,
            task_revision=1,
            controlled_run_id=controlled_run_id,
            manifest_sha256=manifest_sha256,
            protocol_bundle_sha256=protocol_sha256,
            scheduler_version="controlled-frozen-schedule-v1",
            assignment_seed="0" * 64,
            track_assignment_probability="1/1",
            model_assignment_probability="1/1",
            side_assignment_probability="1/2",
            track="model_arena",
            category="evidence",
            prompt=task_prompt,
            prompt_sha256=task.prompt_sha256,
            client_nonce_sha256="7" * 64,
            requester_pseudonym="8" * 64,
            status="queued",
            retention_until=datetime.now(UTC) + timedelta(days=30),
        )
        arms = [
            ResponseArm(
                id=arm_id,
                battle_id=battle_id,
                side=side,
                condition="epicure_on",
                model_id=model_id if side == "left" else right_model_id,
                execution_backend="openrouter",
                provider_slug="provider-a",
                status="queued",
                prompt_sha256=battle.prompt_sha256,
                system_prompt_sha256="9" * 64,
                schema_sha256="a" * 64,
                tool_schema_sha256=season.tool_registry_sha256,
                decoding_json={"max_tokens": 64},
                observed_decoding_json={"max_tokens": 64},
                protocol_bundle_sha256=protocol_sha256,
                epicure_release_id=season.epicure_release_id,
                epicure_bundle_sha256=season.epicure_bundle_sha256,
                epicure_application_sha256=season.epicure_application_sha256,
                epicure_attestation_json={"release_id": season.epicure_release_id},
                epicure_attestation_sha256="b" * 64,
            )
            for arm_id, side in ((left_id, "left"), (right_id, "right"))
        ]
        assignment = ControlledRunAssignment(
            id=assignment_id,
            controlled_run_id=controlled_run_id,
            ordinal=0,
            task_id=task_id,
            task_public_id=task.public_id,
            task_revision=1,
            task_prompt_sha256=task.prompt_sha256,
            task_family=task.family,
            track="model_arena",
            model_ids_json=[model_id, right_model_id],
            repetition_index=1,
            assignment_sha256="d" * 64,
            assignment_seed="0" * 64,
            status="pending",
        )
        session.add_all([season, model, right_model])
        session.flush()
        session.add_all([slot, right_slot, task, controlled_run])
        session.flush()
        session.add(battle)
        session.flush()
        session.add_all([assignment, *arms])
        session.flush()
        terminal_at = datetime.now(UTC)
        for arm in arms:
            arm.actual_provider_slug = "provider-a"
            arm.actual_model_id = arm.model_id
            arm.generation_id = f"generation-{arm.side}"
            arm.provider_generation_ids_json = [arm.generation_id]
            arm.status = "complete"
            arm.finish_reason = "stop"
            arm.answer_markdown = f"Normalized answer from {arm.side}."
            arm.output_json = {"answer_markdown": arm.answer_markdown}
            arm.cost_micros = 10
            arm.cost_reconciled = True
            arm.cost_accounting_basis = "generation_metadata"
            arm.billing_reconciliation_status = "complete"
            arm.latency_ms = 20
            arm.completed_at = terminal_at
        session.flush(arms)
        battle.left_arm_id = left_id
        battle.right_arm_id = right_id
        battle.status = "complete"
        battle.completed_at = terminal_at
        assignment.status = "queued"
        assignment.battle_id = battle_id
        session.flush([battle, assignment])
        controlled_run.status = "collection_complete"
        controlled_run.collection_completed_at = terminal_at
        session.flush([controlled_run])
        controlled_run.status = "closed"
        controlled_run.closed_at = terminal_at
        session.flush([controlled_run])
        session.add(
            ValidatorResult(
                arm_id=left_id,
                validator_name="identity_blinding",
                validator_version=VALIDATOR_VERSION,
                status="pass",
                detail_json={"matched": False},
            )
        )
        session.add(
            ToolCall(
                arm_id=left_id,
                round_index=0,
                call_index=0,
                tool_call_id="tool-call-1",
                tool_name="find_pairings",
                arguments_json={"ingredients": ["tomato"]},
                result_text="pairing evidence",
                structured_content_json={"pairs": []},
                result_sha256=hashlib.sha256(b"pairing evidence").hexdigest(),
                is_error=False,
            )
        )

    def manifest_digest() -> tuple[str, dict]:
        with session_scope() as session:
            season = session.get(Season, season_id)
            assert season is not None
            manifest = _snapshot_evidence_manifest(
                session,
                season=season,
                track="model_arena",
                cohort="public",
                category="all",
                data_stratum="controlled",
                controlled_run_id=controlled_run_id,
                evidence_cutoff_at=evidence_cutoff_at,
            )
            return _canonical_sha256(manifest), manifest

    def payload_digest() -> str:
        with session_scope() as session:
            season = session.get(Season, season_id)
            assert season is not None
            payload = model_leaderboard(
                session,
                season,
                "public",
                "all",
                "controlled",
                controlled_run_id,
                evidence_cutoff_at,
            )
            return snapshot_hash(payload)

    first_digest, first_manifest = manifest_digest()
    first_payload_digest = payload_digest()
    assert first_manifest["schema_version"] == "flavourbench-snapshot-evidence-v7"
    assert first_manifest["arena_acceptance_source_sha256"]
    assert first_manifest["arena_acceptance_policy_file_sha256"]
    assert first_manifest["postcollection_item_audit_events"] == []
    assert first_manifest["arena_method_validation_events"] == []
    assert first_manifest["analysis_observations"] == {
        "eligible_judgment_ids": [],
        "preference_observation_ids": [],
        "preference_observation_sha256": _canonical_sha256({"vote_ids": []}),
    }
    assert first_manifest["battles"][0]["left_arm_id"] == left_id
    assert first_manifest["arms"][0]["execution_backend"] == "openrouter"
    assert first_manifest["arms"][0]["answer_markdown_sha256"]
    assert first_manifest["arms"][0]["output_json_sha256"]
    assert first_manifest["assignments"][0]["id"] == assignment_id
    assert "answer_markdown_sha256" not in first_manifest["assignments"][0]
    assert first_manifest["season_models"][0]["model_id"] == model_id
    assert first_manifest["tool_calls"][0]["result_sha256"] == hashlib.sha256(
        b"pairing evidence"
    ).hexdigest()

    with session_scope() as session:
        session.add(
            Vote(
                battle_id=battle_id,
                rater_pseudonym="f" * 64,
                cohort="public",
                choice="tie",
                idempotency_key=_identifier("post-cutoff-vote"),
                created_at=evidence_cutoff_at + timedelta(seconds=1),
            )
        )
        session.add(
            ValidatorResult(
                arm_id=right_id,
                validator_name="constraint_acknowledgement",
                validator_version=VALIDATOR_VERSION,
                status="pass",
                detail_json={"post_cutoff": True},
                created_at=evidence_cutoff_at + timedelta(seconds=1),
            )
        )
    post_append_digest, _ = manifest_digest()
    assert post_append_digest == first_digest
    assert payload_digest() == first_payload_digest

    with session_scope() as session:
        session.execute(
            update(ToolCall)
            .where(ToolCall.arm_id == left_id)
            .values(result_text="silently rewritten trace")
        )
    with pytest.raises(RuntimeError, match="tool-call trace differs"):
        manifest_digest()
    with session_scope() as session:
        session.execute(
            update(ToolCall)
            .where(ToolCall.arm_id == left_id)
            .values(result_text="pairing evidence")
        )

    with session_scope() as session:
        session.execute(
            update(ResponseArm)
            .where(ResponseArm.id == left_id)
            .values(observed_decoding_json={"max_tokens": 32})
        )
    second_digest, _ = manifest_digest()
    assert second_digest != first_digest

    with session_scope() as session:
        session.execute(
            update(SeasonModel)
            .where(SeasonModel.season_id == season_id, SeasonModel.model_id == model_id)
            .values(expected_actual_provider_slug="provider-b")
        )
    third_digest, _ = manifest_digest()
    assert third_digest != second_digest

    with session_scope() as session:
        session.execute(
            update(Battle)
            .where(Battle.id == battle_id)
            .values(left_arm_id=right_id, right_arm_id=left_id)
        )
    fourth_digest, _ = manifest_digest()
    assert fourth_digest != third_digest

    with session_scope() as session:
        session.execute(
            update(ResponseArm)
            .where(ResponseArm.id == left_id)
            .values(answer_markdown="Rewritten normalized answer.")
        )
    with pytest.raises(RuntimeError, match="normalized answer content differs"):
        manifest_digest()


def test_cost_halt_withdraws_every_published_snapshot() -> None:
    init_database()
    completed_at = datetime.now(UTC)
    season_id = _identifier("cost-halt-season")
    model_id = _identifier("cost-halt-model")
    battle_id = _identifier("cost-halt-battle")
    arm_id = _identifier("cost-halt-arm")
    right_arm_id = _identifier("cost-halt-arm-right")
    snapshot_id = _identifier("cost-halt-snapshot")
    payload = {"rows": [], "accounting": {"complete": True}}
    evidence = {"scope": "cost-halt"}

    with session_scope() as session:
        snapshot = LeaderboardSnapshot(
            id=snapshot_id,
            season_id=season_id,
            track="model_arena",
            cohort="public",
            category="all",
            data_stratum="public_freeform",
            publication_status="draft",
            input_sha256=snapshot_hash(payload),
            input_evidence_sha256=_canonical_sha256(evidence),
            input_evidence_json=evidence,
            payload_sha256=snapshot_hash(payload),
            payload_json=payload,
            evidence_cutoff_at=completed_at,
        )
        battle = Battle(
            id=battle_id,
            season_id=season_id,
            run_class="official",
            rank_eligible=True,
            data_stratum="public_freeform",
            manifest_sha256="c" * 64,
            protocol_bundle_sha256="3" * 64,
            scheduler_version="cost-halt-test-v1",
            assignment_seed="4" * 64,
            track="model_arena",
            category="evidence",
            prompt="Cost halt test",
            prompt_sha256=hashlib.sha256(b"Cost halt test").hexdigest(),
            client_nonce_sha256="5" * 64,
            requester_pseudonym="6" * 64,
            status="queued",
            reserved_cost_micros=1,
            retention_until=datetime.now(UTC) + timedelta(days=30),
        )
        arms = [
            ResponseArm(
                id=current_arm_id,
                battle_id=battle_id,
                side=side,
                condition="epicure_on",
                model_id=model_id,
                provider_slug="provider-a",
                status="queued",
                prompt_sha256=hashlib.sha256(b"Cost halt test").hexdigest(),
                schema_sha256="7" * 64,
                tool_schema_sha256="e" * 64,
                epicure_release_id="cost-halt-release",
                epicure_bundle_sha256="f" * 64,
            )
            for current_arm_id, side in ((arm_id, "left"), (right_arm_id, "right"))
        ]
        session.add_all(
            [
                Season(
                    id=season_id,
                    slug=_identifier("cost-halt"),
                    name="Cost halt snapshot test",
                    status="active",
                    official=True,
                    manifest_sha256="c" * 64,
                    prompt_registry_sha256="d" * 64,
                    tool_registry_sha256="e" * 64,
                    epicure_release_id="cost-halt-release",
                    epicure_bundle_sha256="f" * 64,
                    epicure_application_sha256="1" * 64,
                    analysis_plan_sha256="2" * 64,
                    protocol_bundle_json={"test": True},
                    protocol_bundle_sha256="3" * 64,
                    budget_cap_micros=100,
                    budget_used_micros=0,
                    budget_reserved_micros=1,
                ),
                CatalogModel(
                    model_id=model_id,
                    canonical_slug=model_id,
                    name="Cost halt model",
                    family="test",
                    catalog_source="test",
                ),
            ]
        )
        session.flush()
        session.add_all([battle, snapshot])
        session.flush()
        session.add_all(arms)
        session.flush()
        terminal_at = datetime.now(UTC)
        for current_arm in arms:
            current_arm.status = "failed"
            current_arm.error_code = "CostHaltFixture"
            current_arm.error_detail = "Fixture terminal failure."
            current_arm.cost_micros = 2 if current_arm.side == "left" else 0
            current_arm.cost_reconciled = True
            current_arm.cost_accounting_basis = "fixture_reconciled"
            current_arm.billing_reconciliation_status = "complete"
            current_arm.completed_at = terminal_at
        session.flush(arms)
        battle.left_arm_id = arm_id
        battle.right_arm_id = right_arm_id
        battle.status = "failed"
        battle.completed_at = terminal_at
        session.flush()
        snapshot.publication_status = "published"
        snapshot.publication_reference_sha256 = "8" * 64
        snapshot.published_at = completed_at

    with session_scope() as session:
        battle = session.get(Battle, battle_id)
        assert battle is not None
        reconcile_battle_cost(session, battle)

    with session_scope() as session:
        season = session.get(Season, season_id)
        snapshot = session.get(LeaderboardSnapshot, snapshot_id)
        assert season is not None and snapshot is not None
        assert season.status == "cost_halted"
        assert snapshot.publication_status == "withdrawn"
