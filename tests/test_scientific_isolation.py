from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import flavourbench.arena as arena
import flavourbench.execution_policy as execution_policy_module
import flavourbench.protocol_contract as protocol_contract_module
import flavourbench.seed as seed_module
from flavourbench.config import get_settings
from flavourbench.database import EXPECTED_SCHEMA_REVISION, init_database, session_scope
from flavourbench.endpoint_contract import endpoint_contract_sha256
from flavourbench.execution_policy import assert_legacy_paid_cli_allowed
from flavourbench.main import app
from flavourbench.models import (
    Battle,
    CatalogModel,
    ControlledRun,
    ControlledRunAssignment,
    ResponseArm,
    RunEvent,
    Season,
    SeasonModel,
    Task,
    ValidatorResult,
    Vote,
)
from flavourbench.protocol_contract import build_protocol_bundle
from flavourbench.provider import FINAL_SCHEMA_SHA256, system_prompt_sha256
from flavourbench.ranking import model_leaderboard, snapshot_hash, uplift_leaderboard
from flavourbench.schemas import ControlledBattleCreate
from flavourbench.seed import seed_database
from flavourbench.service_ranking import (
    model_leaderboard as service_model_leaderboard,
)
from flavourbench.validators import VALIDATOR_VERSION

HEADERS = {
    "X-FlavourBench-Service-Token": "test-service-token",
    "X-FlavourBench-Pseudonym": "9" * 64,
}


def _bind_protocol(season: Season) -> None:
    season.tool_registry_sha256 = "d" * 64
    season.analysis_plan_sha256 = "c" * 64
    bundle, digest = build_protocol_bundle(
        tool_registry_sha256=season.tool_registry_sha256,
        epicure_release_id=season.epicure_release_id,
        epicure_bundle_sha256=season.epicure_bundle_sha256,
        epicure_application_sha256=season.epicure_application_sha256,
        analysis_plan_sha256=season.analysis_plan_sha256,
    )
    season.protocol_bundle_json = bundle
    season.protocol_bundle_sha256 = digest


def test_protocol_bundle_binds_budget_schema_migration_and_release_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = {
        "tool_registry_sha256": "1" * 64,
        "epicure_release_id": "epicure-test-release",
        "epicure_bundle_sha256": "2" * 64,
        "epicure_application_sha256": "3" * 64,
        "analysis_plan_sha256": "4" * 64,
    }
    bundle, digest = build_protocol_bundle(**arguments)
    assert {
        "account_authority.py",
        "budget_policy.py",
        "config.py",
        "controlled_integrity.py",
        "database.py",
        "main.py",
        "models.py",
        "ranking.py",
        "schemas.py",
        "security.py",
        "worker.py",
    } <= set(bundle["implementation_sha256"])
    assert bundle["release_inputs"]["alembic_head"] == EXPECTED_SCHEMA_REVISION
    assert all(
        len(bundle["release_inputs"][name]) == 64
        for name in (
            "alembic_head_sha256",
            "alembic_chain_sha256",
            "pyproject_sha256",
            "dependency_lock_sha256",
            "dockerfile_sha256",
        )
    )
    revision_hashes = bundle["release_inputs"]["alembic_revisions_sha256"]
    assert list(revision_hashes) == sorted(revision_hashes)
    assert "0001_initial.py" in revision_hashes
    assert "0014_commercial_evidence_invariants.py" in revision_hashes
    assert f"{EXPECTED_SCHEMA_REVISION}.py" in revision_hashes
    implementation_names = set(bundle["implementation_sha256"])
    for filename in implementation_names:
        assert protocol_contract_module._local_module_imports(filename) <= (
            implementation_names
        )

    original = protocol_contract_module._source_sha256
    monkeypatch.setattr(
        protocol_contract_module,
        "_source_sha256",
        lambda name: "0" * 64 if name == "budget_policy.py" else original(name),
    )
    changed_bundle, changed_digest = build_protocol_bundle(**arguments)
    assert changed_bundle["implementation_sha256"]["budget_policy.py"] == "0" * 64
    assert changed_digest != digest


def test_production_startup_never_writes_development_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_database()
    slug = "production-empty-draft"
    with session_scope() as session:
        season = session.scalar(select(Season).where(Season.slug == slug))
        if season is None:
            season = Season(
                slug=slug,
                name="Production draft",
                status="draft",
                epicure_release_id="operator-provisioned-release",
            )
            session.add(season)
            session.flush()
        season_id = season.id

    monkeypatch.setattr(
        seed_module,
        "get_settings",
        lambda: SimpleNamespace(
            environment="production",
            default_season_slug=slug,
        ),
    )
    assert seed_database()["status"] == "development_fixtures_disabled"
    assert seed_database()["status"] == "development_fixtures_disabled"

    with session_scope() as session:
        assert session.scalar(select(Task.id).where(Task.season_id == season_id).limit(1)) is None
        assert (
            session.scalar(
                select(SeasonModel.id).where(SeasonModel.season_id == season_id).limit(1)
            )
            is None
        )


def test_legacy_paid_cli_plane_is_disabled_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution_policy_module,
        "get_settings",
        lambda: SimpleNamespace(environment="production"),
    )
    with pytest.raises(RuntimeError, match="PostgreSQL-governed API and worker"):
        assert_legacy_paid_cli_allowed("legacy-paid-runner")

    source_root = Path(__file__).parents[1] / "src" / "flavourbench"
    for filename in (
        "live_smoke.py",
        "frontier_contract_runner.py",
        "real_dataset_runner.py",
        "bedrock_contract_smoke.py",
        "bedrock_b2_runner.py",
        "task_curation.py",
        "season0_compatibility.py",
        "season0_openrouter_compatibility.py",
        "season0_cost_recovery.py",
        "season0_judge_manifest.py",
        "reconcile.py",
    ):
        source = (source_root / filename).read_text(encoding="utf-8")
        assert "assert_legacy_paid_cli_allowed(" in source

    legacy_wrapper = (source_root / "legacy_paid_cli.py").read_text(encoding="utf-8")
    for command in (
        "flavourbench-run-season0-collection",
        "flavourbench-run-season0-judging",
        "flavourbench-recover-season0-throttles",
    ):
        assert f'assert_legacy_paid_cli_allowed("{command}")' in legacy_wrapper


def test_assignment_uses_server_entropy_and_provenance_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = "f" * 64
    settings = get_settings()
    server_is_model_track = arena._seeded_index(seed, "track", 100) < settings.model_track_percent
    nonce = next(
        f"scientific-isolation-client-nonce-{index}"
        for index in range(1000)
        if (
            int(
                hashlib.sha256(f"scientific-isolation-client-nonce-{index}".encode()).hexdigest(),
                16,
            )
            % 100
            < settings.model_track_percent
        )
        != server_is_model_track
    )
    monkeypatch.setattr(arena.secrets, "token_hex", lambda _length: seed)

    with TestClient(app) as client:
        response = client.post(
            "/v1/battles",
            headers=HEADERS,
            json={
                "prompt": "Create a practical savoury cauliflower dish with a crisp garnish.",
                "category": "composition",
                "clientNonce": nonce,
            },
        )
        assert response.status_code == 202, response.text
        battle_id = response.json()["battleId"]

    with session_scope() as session:
        battle = session.get(Battle, battle_id)
        assert battle is not None
        assert battle.assignment_seed == seed
        assert (battle.track == "model_arena") == server_is_model_track
        assert battle.run_class == "mock"
        assert battle.rank_eligible is False

    with pytest.raises(ValueError, match="scientific provenance is immutable"):
        with session_scope() as session:
            battle = session.get(Battle, battle_id)
            assert battle is not None
            battle.rank_eligible = True


def test_controlled_battle_snapshots_task_identity_and_revision() -> None:
    seed_database()
    with session_scope() as session:
        task = session.scalar(select(Task).order_by(Task.public_id))
        assert task is not None and task.revision == 1
        season = session.get(Season, task.season_id)
        assert season is not None
        controlled_run = ControlledRun(
            season_id=season.id,
            organization_reference_sha256="1" * 64,
            access_token_sha256="2" * 64,
            status="active",
            protocol_version="controlled-test-v1",
            rater_plan_sha256="3" * 64,
            analysis_plan_sha256="4" * 64,
            budget_cap_micros=1_000_000,
            run_card_json={"test": True},
            run_card_sha256="5" * 64,
            run_card_signature="6" * 64,
        )
        session.add(controlled_run)
        session.flush()
        slot = session.scalar(
            select(SeasonModel)
            .where(SeasonModel.season_id == season.id, SeasonModel.eligible.is_(True))
            .order_by(SeasonModel.model_id)
        )
        assert slot is not None
        assignment = ControlledRunAssignment(
            controlled_run_id=controlled_run.id,
            ordinal=0,
            task_id=task.id,
            task_public_id=task.public_id,
            task_revision=task.revision,
            task_prompt_sha256=task.prompt_sha256,
            task_family=task.family,
            track="epicure_uplift",
            model_ids_json=[slot.model_id],
            repetition_index=1,
            assignment_sha256="8" * 64,
            assignment_seed="9" * 64,
        )
        session.add(assignment)
        session.flush()
        battle = arena.create_battle(
            session,
            ControlledBattleCreate(
                taskPublicId=task.public_id,
                expectedAssignmentOrdinal=0,
                clientNonce="scientific-isolation-controlled-task",
            ),
            pseudonym="7" * 64,
            task=task,
            controlled_run=controlled_run,
            season_row=season,
        )
        assert battle.data_stratum == "controlled"
        assert battle.task_id == task.id
        assert battle.task_revision == 1
        assert battle.controlled_run_id == controlled_run.id


def _catalog_model(session, model_id: str) -> None:
    if session.get(CatalogModel, model_id) is None:
        session.add(
            CatalogModel(
                model_id=model_id,
                canonical_slug=model_id,
                name=model_id,
                family="isolation-provider",
                status="smoke_passed",
                supports_tools=True,
                supports_structured_outputs=True,
            )
        )


def _add_judgment(
    session,
    *,
    season: Season,
    suffix: str,
    track: str,
    choice: str,
    left_model: str,
    right_model: str,
    manifest_sha256: str | None = None,
    run_class: str = "official",
    rank_eligible: bool = True,
    provider_slug: str = "isolation-provider",
    data_stratum: str = "public_freeform",
    controlled_run_id: str | None = None,
) -> Battle:
    supported = [
        "max_tokens",
        "response_format",
        "structured_outputs",
        "tool_choice",
        "tools",
    ]
    decoding = {"max_tokens": 512}
    endpoint_document = hashlib.sha256(f"{provider_slug}:scientific-isolation".encode()).hexdigest()
    expected_provider = "mock" if provider_slug == "mock" else "Isolation Provider"
    for model_id in {left_model, right_model}:
        slot = session.scalar(
            select(SeasonModel).where(
                SeasonModel.season_id == season.id,
                SeasonModel.model_id == model_id,
            )
        )
        if slot is None:
            slot = SeasonModel(
                season_id=season.id,
                model_id=model_id,
                slot_role="test",
                provider_slug=provider_slug,
                expected_actual_model_id=model_id,
                expected_actual_provider_slug=expected_provider,
                supported_parameters_json=supported,
                decoding_json=decoding,
                endpoint_max_completion_tokens=512,
                endpoint_document_sha256=endpoint_document,
                endpoint_contract_sha256=endpoint_contract_sha256(
                    model_id=model_id,
                    provider_slug=provider_slug,
                    expected_actual_model_id=model_id,
                    expected_actual_provider_slug=expected_provider,
                    supported_parameters=supported,
                    decoding=decoding,
                    endpoint_max_completion_tokens=512,
                    endpoint_document_sha256=endpoint_document,
                ),
                manifest_sha256=season.manifest_sha256,
            )
            session.add(slot)
    prompt_sha = hashlib.sha256(f"prompt:{suffix}".encode()).hexdigest()
    task_id = None
    task_revision = None
    assignment: ControlledRunAssignment | None = None
    assignment_seed = hashlib.sha256(f"seed:{suffix}".encode()).hexdigest()
    if data_stratum == "controlled":
        if controlled_run_id is None:
            raise ValueError("controlled test judgment requires a controlled run id")
        task = Task(
            public_id=f"isolation-task-{suffix}",
            season_id=season.id,
            family="composition",
            prompt=f"Prompt {suffix} with enough text for a benchmark battle.",
            prompt_sha256=prompt_sha,
            revision=1,
            split="test",
            review_status="frozen",
        )
        session.add(task)
        session.flush()
        task_id = task.id
        task_revision = task.revision
        assignment_seed = "0" * 64
    battle = Battle(
        season_id=season.id,
        run_class=run_class,
        rank_eligible=rank_eligible,
        data_stratum=data_stratum,
        task_id=task_id,
        task_revision=task_revision,
        controlled_run_id=controlled_run_id,
        manifest_sha256=manifest_sha256 or season.manifest_sha256,
        protocol_bundle_sha256=season.protocol_bundle_sha256,
        scheduler_version=(
            arena.CONTROLLED_SCHEDULER_VERSION
            if data_stratum == "controlled"
            else "isolation-test-v1"
        ),
        assignment_seed=assignment_seed,
        track_assignment_probability=(
            "1/1" if data_stratum == "controlled" else "1/2"
        ),
        model_assignment_probability="1/1",
        side_assignment_probability="1/2",
        track=track,
        category="composition",
        prompt=f"Prompt {suffix} with enough text for a benchmark battle.",
        prompt_sha256=prompt_sha,
        client_nonce_sha256=hashlib.sha256(f"nonce:{suffix}".encode()).hexdigest(),
        research_consent=False,
        requester_pseudonym=hashlib.sha256(f"rater:{suffix}".encode()).hexdigest(),
        status="queued",
        retention_until=datetime.now(UTC) + timedelta(days=30),
    )
    session.add(battle)
    session.flush()
    if data_stratum == "controlled":
        assignment = ControlledRunAssignment(
            controlled_run_id=str(controlled_run_id),
            ordinal=0,
            task_id=str(task_id),
            task_public_id=task.public_id,
            task_revision=int(task_revision or 1),
            task_prompt_sha256=prompt_sha,
            task_family="composition",
            track=track,
            model_ids_json=(
                [left_model, right_model] if track == "model_arena" else [left_model]
            ),
            repetition_index=1,
            assignment_sha256=hashlib.sha256(
                f"assignment:{suffix}".encode()
            ).hexdigest(),
            assignment_seed=assignment_seed,
            status="pending",
        )
        session.add(assignment)
        session.flush()

    conditions = (
        ("epicure_on", "epicure_on") if track == "model_arena" else ("epicure_on", "epicure_off")
    )
    arms = []
    for side, model_id, condition in zip(
        ("left", "right"),
        (left_model, right_model),
        conditions,
        strict=True,
    ):
        is_mock = model_id.startswith("flavourbench/mock-") or provider_slug == "mock"
        arm = ResponseArm(
            battle_id=battle.id,
            side=side,
            condition=condition,
            model_id=model_id,
            provider_slug=provider_slug,
            status="queued",
            prompt_sha256=prompt_sha,
            system_prompt_sha256=system_prompt_sha256(condition),
            schema_sha256=FINAL_SCHEMA_SHA256,
            tool_schema_sha256=season.tool_registry_sha256,
            decoding_json={
                "max_tokens": 512,
                "seed": "provider_fixed_unsupported",
                "structured_output": True,
                "temperature": "provider_fixed_unsupported",
                "top_p": "provider_fixed_unsupported",
                "max_tool_rounds": get_settings().max_tool_rounds,
            },
            observed_decoding_json={
                "max_tokens": 512,
                "seed": "provider_fixed_unsupported",
                "temperature": "provider_fixed_unsupported",
                "top_p": "provider_fixed_unsupported",
            },
            protocol_bundle_sha256=season.protocol_bundle_sha256,
            epicure_release_id=season.epicure_release_id,
            epicure_bundle_sha256=season.epicure_bundle_sha256,
            epicure_application_sha256=season.epicure_application_sha256,
        )
        if condition == "epicure_on" and not is_mock:
            attestation = {
                "release_id": season.epicure_release_id,
                "bundle_sha256": season.epicure_bundle_sha256,
                "application_sha256": season.epicure_application_sha256,
                "tool_schema_sha256": season.tool_registry_sha256,
                "ingredient_count": 1,
                "embedding_dimensions": 1,
                "tool_count": 1,
                "mcp_protocol_version": "2025-06-18",
            }
            arm.epicure_attestation_json = attestation
            arm.epicure_attestation_sha256 = hashlib.sha256(
                json.dumps(attestation, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        session.add(arm)
        arms.append(arm)
    session.flush()
    for arm in arms:
        is_mock = arm.model_id.startswith("flavourbench/mock-") or provider_slug == "mock"
        arm.actual_provider_slug = "mock" if is_mock else "Isolation Provider"
        arm.actual_model_id = arm.model_id
        arm.generation_id = f"generation-{suffix}-{arm.side}"
        arm.provider_generation_ids_json = [arm.generation_id]
        arm.status = "complete"
        arm.finish_reason = "stop"
        arm.answer_markdown = f"Answer {arm.side}"
        arm.output_json = {"answer_markdown": arm.answer_markdown}
        arm.cost_reconciled = True
        arm.cost_accounting_basis = "fixture_reconciled"
        arm.billing_reconciliation_status = "complete"
        arm.completed_at = datetime.now(UTC)
    session.flush(arms)
    for arm in arms:
        session.add_all(
            [
                ValidatorResult(
                    arm_id=arm.id,
                    validator_name=validator_name,
                    validator_version=VALIDATOR_VERSION,
                    status="pass",
                    score_milli=1000,
                    detail_json={"fixture": True},
                )
                for validator_name in ("identity_blinding", "semantic_completion")
            ]
        )
    battle.left_arm_id = arms[0].id
    battle.right_arm_id = arms[1].id
    battle.status = "complete"
    battle.completed_at = datetime.now(UTC)
    if assignment is not None:
        assignment.status = "queued"
        assignment.battle_id = battle.id
    session.flush()
    if data_stratum == "public_freeform":
        session.add(
            RunEvent(
                entity_type="battle",
                entity_id=battle.id,
                event_type="battle_general_track_scope_admitted",
                payload_json={
                    "general_track_eligible": True,
                    "scope_protocol_sha256": "f" * 64,
                },
            )
        )
        session.flush()
    session.add(
        Vote(
            battle_id=battle.id,
            rater_pseudonym=hashlib.sha256(f"vote-rater:{suffix}".encode()).hexdigest(),
            cohort="public",
            choice=choice,
            idempotency_key=f"scientific-isolation-{suffix}",
        )
    )
    session.flush()
    return battle


def _controlled_run(session, season: Season, suffix: str) -> ControlledRun:
    digest = hashlib.sha256(suffix.encode()).hexdigest()
    run = ControlledRun(
        season_id=season.id,
        organization_reference_sha256=hashlib.sha256(f"organization:{suffix}".encode()).hexdigest(),
        access_token_sha256=hashlib.sha256(f"token:{suffix}".encode()).hexdigest(),
        status="active",
        protocol_version="isolation-test-v1",
        rater_plan_sha256=hashlib.sha256(f"rater:{suffix}".encode()).hexdigest(),
        analysis_plan_sha256=hashlib.sha256(f"analysis:{suffix}".encode()).hexdigest(),
        budget_cap_micros=1_000_000,
        run_card_json={"suffix": suffix},
        run_card_sha256=digest,
        run_card_signature=hashlib.sha256(f"signature:{suffix}".encode()).hexdigest(),
    )
    session.add(run)
    session.flush()
    return run


def test_rankings_and_snapshots_exclude_nonofficial_wrong_manifest_and_mock_rows() -> None:
    seed_database()
    manifest = "1" * 64
    model_a = "isolation-provider/model-a"
    model_b = "isolation-provider/model-b"
    season_slug = "scientific-isolation-season"

    with session_scope() as session:
        _catalog_model(session, model_a)
        _catalog_model(session, model_b)
        season = session.scalar(select(Season).where(Season.slug == season_slug))
        if season is None:
            season = Season(
                slug=season_slug,
                name="Scientific isolation test",
                status="active",
                official=True,
                manifest_sha256=manifest,
                epicure_release_id="isolation-release",
                epicure_bundle_sha256="e" * 64,
                epicure_application_sha256="a" * 64,
            )
            session.add(season)
            session.flush()
            _bind_protocol(season)

        _add_judgment(
            session,
            season=season,
            suffix="arena-valid",
            track="model_arena",
            choice="left",
            left_model=model_a,
            right_model=model_b,
        )
        _add_judgment(
            session,
            season=season,
            suffix="arena-both-bad",
            track="model_arena",
            choice="both_bad",
            left_model=model_a,
            right_model=model_b,
        )
        _add_judgment(
            session,
            season=season,
            suffix="arena-wrong-manifest",
            track="model_arena",
            choice="right",
            left_model=model_a,
            right_model=model_b,
            manifest_sha256="2" * 64,
        )
        _add_judgment(
            session,
            season=season,
            suffix="arena-exploratory",
            track="model_arena",
            choice="right",
            left_model=model_a,
            right_model=model_b,
            run_class="exploratory",
        )
        _add_judgment(
            session,
            season=season,
            suffix="arena-smoke",
            track="model_arena",
            choice="right",
            left_model=model_a,
            right_model=model_b,
            run_class="smoke",
        )
        _add_judgment(
            session,
            season=season,
            suffix="arena-not-rank-eligible",
            track="model_arena",
            choice="right",
            left_model=model_a,
            right_model=model_b,
            rank_eligible=False,
        )
        _add_judgment(
            session,
            season=season,
            suffix="arena-legacy-stratum",
            track="model_arena",
            choice="right",
            left_model=model_a,
            right_model=model_b,
            data_stratum="legacy",
        )
        _add_judgment(
            session,
            season=season,
            suffix="arena-mock-models",
            track="model_arena",
            choice="right",
            left_model="flavourbench/mock-openai-flagship",
            right_model="flavourbench/mock-anthropic-flagship",
            provider_slug="mock",
        )

        arena_board = model_leaderboard(
            session,
            season,
            "public",
            "all",
            data_stratum="public_freeform",
        )
        assert {row["competitor_id"] for row in arena_board["rows"]} == {model_a, model_b}
        assert all(row["battles"] == 1 for row in arena_board["rows"])
        assert all(row["judgments"] == 2 for row in arena_board["rows"])
        assert all(row["both_bad"] == 1 for row in arena_board["rows"])
        assert all(row["both_bad_rate"] == 0.5 for row in arena_board["rows"])
        assert arena_board["manifest_sha256"] == manifest
        assert arena_board["eligibility_filter"]["run_class"] == "official"
        first_snapshot = snapshot_hash(arena_board)
        assert first_snapshot == snapshot_hash(arena_board)

        _add_judgment(
            session,
            season=season,
            suffix="uplift-valid",
            track="epicure_uplift",
            choice="left",
            left_model=model_a,
            right_model=model_a,
        )
        _add_judgment(
            session,
            season=season,
            suffix="uplift-both-bad",
            track="epicure_uplift",
            choice="both_bad",
            left_model=model_a,
            right_model=model_a,
        )
        _add_judgment(
            session,
            season=season,
            suffix="uplift-wrong-manifest",
            track="epicure_uplift",
            choice="right",
            left_model=model_a,
            right_model=model_a,
            manifest_sha256="3" * 64,
        )
        uplift_board = uplift_leaderboard(
            session,
            season,
            "public",
            "all",
            data_stratum="public_freeform",
        )
        assert len(uplift_board["rows"]) == 1
        uplift_row = uplift_board["rows"][0]
        assert uplift_row["competitor_id"] == model_a
        assert uplift_row["battles"] == 1
        assert uplift_row["judgments"] == 2
        assert uplift_row["both_bad"] == 1
        assert uplift_row["epicure_wins"] == 1


def test_service_rankings_isolate_public_and_each_controlled_run() -> None:
    seed_database()
    manifest = "7" * 64
    model_a = "tenant-isolation/model-a"
    model_b = "tenant-isolation/model-b"
    with session_scope() as session:
        _catalog_model(session, model_a)
        _catalog_model(session, model_b)
        season = Season(
            slug="tenant-isolation-season",
            name="Tenant isolation test",
            status="active",
            official=True,
            manifest_sha256=manifest,
            epicure_release_id="tenant-isolation-release",
            epicure_bundle_sha256="e" * 64,
            epicure_application_sha256="a" * 64,
        )
        session.add(season)
        session.flush()
        _bind_protocol(season)
        run_a = _controlled_run(session, season, "tenant-a")
        run_b = _controlled_run(session, season, "tenant-b")

        _add_judgment(
            session,
            season=season,
            suffix="tenant-public",
            track="model_arena",
            choice="left",
            left_model=model_a,
            right_model=model_b,
        )
        _add_judgment(
            session,
            season=season,
            suffix="tenant-controlled-a",
            track="model_arena",
            choice="left",
            left_model=model_a,
            right_model=model_b,
            data_stratum="controlled",
            controlled_run_id=run_a.id,
        )
        _add_judgment(
            session,
            season=season,
            suffix="tenant-controlled-b",
            track="model_arena",
            choice="right",
            left_model=model_a,
            right_model=model_b,
            data_stratum="controlled",
            controlled_run_id=run_b.id,
        )

        public_board = service_model_leaderboard(
            session,
            season,
            "public",
            "all",
            data_stratum="public_freeform",
        )
        run_a_board = service_model_leaderboard(
            session,
            season,
            "public",
            "all",
            data_stratum="controlled",
            controlled_run_id=run_a.id,
        )
        run_b_board = service_model_leaderboard(
            session,
            season,
            "public",
            "all",
            data_stratum="controlled",
            controlled_run_id=run_b.id,
        )

        assert public_board["rows"] == []
        assert public_board["ranking_status"] == "insufficient_comparisons"
        for board in (run_a_board, run_b_board):
            assert len(board["rows"]) == 2
            assert all(row["battles"] == 1 for row in board["rows"])
            assert all(row["response_arms"] == 1 for row in board["rows"])
        assert public_board["controlled_run_id"] is None
        assert run_a_board["controlled_run_id"] == run_a.id
        assert run_b_board["controlled_run_id"] == run_b.id
        run_a_ratings = {row["competitor_id"]: row["rating"] for row in run_a_board["rows"]}
        run_b_ratings = {row["competitor_id"]: row["rating"] for row in run_b_board["rows"]}
        assert run_a_ratings[model_a] > run_a_ratings[model_b]
        assert run_b_ratings[model_a] < run_b_ratings[model_b]

        with pytest.raises(ValueError, match="controlled leaderboards require"):
            service_model_leaderboard(
                session,
                season,
                "public",
                "all",
                data_stratum="controlled",
            )
