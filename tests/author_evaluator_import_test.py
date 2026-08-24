from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import Session

from flavourbench.author_evaluator_import import (
    AuthorEvaluatorImportError,
    _validate_candidate,
    import_bundle,
    load_bundle,
)
from flavourbench.expert_calibration import TASK_SCOPE_QUARANTINE
from flavourbench.models import Base, Battle, ResponseArm, Season, Task, ToolCall

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_SHA256 = "94e917b6c202eb49953f3a8c22f897301eaa7ffba47116b83c915d17a6850b69"


def _season_id(session: Session, candidate_sha256: str) -> str:
    season_id = session.scalar(
        select(Season.id).where(Season.manifest_sha256 == candidate_sha256)
    )
    assert season_id is not None
    return season_id


def test_real_author_evaluator_pool_import_is_verified_isolated_and_idempotent() -> None:
    bundle = load_bundle(
        candidate_path=(
            ROOT
            / "artifacts"
            / "expert-calibration"
            / "candidate-v11"
            / f"candidate-pack-{CANDIDATE_SHA256}.json"
        ),
        comparison_manifest_path=(
            ROOT
            / "artifacts"
            / "season0"
            / "comparisons"
            / (
                "season0-comparisons-"
                "c6e9052d19737b39b540dafbd0cea53d1dd0c54b1a04584fd3775ddfe9f35ca7.json"
            )
        ),
        model_manifest_path=(
            ROOT
            / "artifacts"
            / "season0"
            / "manifests"
            / (
                "season0-model-manifest-"
                "3919def66686b4bd939c94cdd89659f63ae2afbbf03288413129e2ea8d6b83d2.json"
            )
        ),
        arm_directory=ROOT / "artifacts" / "season0" / "scored-v1" / "arms",
    )
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        first = import_bundle(session, bundle)
        session.commit()
        assert first["battles"] == 32
        assert first["arms"] == 64
        assert first["toolCalls"] == bundle.candidate["observed"]["real_epicure_calls"]
        assert first["syntheticArms"] == 0
        assert first["rankEligibleBattles"] == 0
        assert first["idempotent"] is False
        assert session.scalar(select(func.count()).select_from(Battle)) == 32
        assert session.scalar(select(func.count()).select_from(ResponseArm)) == 64
        assert session.scalar(select(func.count()).select_from(ToolCall)) == first["toolCalls"]
        imported_task_ids = set(session.scalars(select(Task.public_id)).all())
        assert imported_task_ids.isdisjoint(TASK_SCOPE_QUARANTINE)
        assert (
            session.scalar(
                select(func.count())
                .select_from(Battle)
                .where(
                    Battle.manifest_sha256 == CANDIDATE_SHA256,
                    Battle.data_stratum == "development",
                    Battle.run_class == "pilot",
                    Battle.rank_eligible.is_(False),
                )
            )
            == 32
        )

        second = import_bundle(session, bundle)
        assert second["idempotent"] is True

        source_battle_id = session.scalar(
            select(Battle.id)
            .where(Battle.manifest_sha256 == CANDIDATE_SHA256)
            .limit(1)
        )
        assert source_battle_id is not None
        session.execute(
            text(
                "UPDATE battles SET run_class = 'official', rank_eligible = 1 "
                "WHERE id = :battle_id"
            ),
            {"battle_id": source_battle_id},
        )
        session.commit()
        with pytest.raises(AuthorEvaluatorImportError, match="battle projection has drifted"):
            import_bundle(session, bundle)
        session.execute(
            text(
                "UPDATE battles SET run_class = 'pilot', rank_eligible = 0 "
                "WHERE id = :battle_id"
            ),
            {"battle_id": source_battle_id},
        )
        session.commit()

        source_task_id = session.scalar(
            select(Task.id).where(Task.season_id == _season_id(session, CANDIDATE_SHA256)).limit(1)
        )
        assert source_task_id is not None
        session.execute(
            text("UPDATE tasks SET split = 'scored' WHERE id = :task_id"),
            {"task_id": source_task_id},
        )
        session.commit()
        with pytest.raises(AuthorEvaluatorImportError, match="task projection has drifted"):
            import_bundle(session, bundle)
        session.execute(
            text("UPDATE tasks SET split = 'calibration' WHERE id = :task_id"),
            {"task_id": source_task_id},
        )
        session.commit()

        source_arm = session.scalar(
            select(ResponseArm)
            .join(Battle, Battle.id == ResponseArm.battle_id)
            .where(Battle.manifest_sha256 == CANDIDATE_SHA256)
            .limit(1)
        )
        assert source_arm is not None
        source_arm_id = source_arm.id
        source_condition = source_arm.condition
        source_model_id = source_arm.model_id
        source_answer = source_arm.answer_markdown
        alternate_model_id = session.scalar(
            select(ResponseArm.model_id)
            .join(Battle, Battle.id == ResponseArm.battle_id)
            .where(
                Battle.manifest_sha256 == CANDIDATE_SHA256,
                ResponseArm.model_id != source_model_id,
            )
            .limit(1)
        )
        assert alternate_model_id is not None

        for column, changed, original in (
            ("condition", "tampered_condition", source_condition),
            ("model_id", alternate_model_id, source_model_id),
            ("answer_markdown", f"{source_answer}\nTAMPERED", source_answer),
        ):
            session.execute(
                text(f"UPDATE response_arms SET {column} = :value WHERE id = :arm_id"),
                {"value": changed, "arm_id": source_arm_id},
            )
            session.commit()
            with pytest.raises(
                AuthorEvaluatorImportError,
                match="response-arm projection has drifted",
            ):
                import_bundle(session, bundle)
            session.execute(
                text(f"UPDATE response_arms SET {column} = :value WHERE id = :arm_id"),
                {"value": original, "arm_id": source_arm_id},
            )
            session.commit()

        source_tool = session.scalar(
            select(ToolCall)
            .join(ResponseArm, ResponseArm.id == ToolCall.arm_id)
            .join(Battle, Battle.id == ResponseArm.battle_id)
            .where(Battle.manifest_sha256 == CANDIDATE_SHA256)
            .limit(1)
        )
        assert source_tool is not None
        original_structured = source_tool.structured_content_json
        session.execute(
            text(
                "UPDATE tool_calls SET structured_content_json = :value "
                "WHERE id = :tool_id"
            ),
            {"value": json.dumps({"tampered": True}), "tool_id": source_tool.id},
        )
        session.commit()
        with pytest.raises(AuthorEvaluatorImportError, match="tool-call projection has drifted"):
            import_bundle(session, bundle)
        session.execute(
            text(
                "UPDATE tool_calls SET structured_content_json = :value "
                "WHERE id = :tool_id"
            ),
            {"value": json.dumps(original_structured), "tool_id": source_tool.id},
        )
        session.commit()
        assert import_bundle(session, bundle)["idempotent"] is True

        replacement_sha256 = "f" * 64
        replacement_bundle = replace(
            bundle,
            candidate={**bundle.candidate, "artifact_sha256": replacement_sha256},
        )
        replacement_result = import_bundle(session, replacement_bundle)
        session.commit()
        assert replacement_result["idempotent"] is False
        assert session.scalar(select(func.count()).select_from(Battle)) == 64
        assert session.scalar(select(func.count()).select_from(ResponseArm)) == 128
        assert session.scalar(select(func.count()).select_from(ToolCall)) == 2 * first["toolCalls"]
        assert session.scalar(
            select(Season.slug).where(Season.manifest_sha256 == replacement_sha256)
        ) == f"season-0-author-evaluator-{replacement_sha256[:12]}"

        session.execute(
            text("UPDATE response_arms SET finish_reason = 'length' WHERE id = :arm_id"),
            {"arm_id": source_arm_id},
        )
        session.commit()
        with pytest.raises(AuthorEvaluatorImportError, match="has drifted"):
            import_bundle(session, bundle)

    mutated = {
        **bundle.candidate,
        "selection_policy": {
            **bundle.candidate["selection_policy"],
            "specialist_scope_review_sha256": "0" * 64,
        },
    }
    with pytest.raises(AuthorEvaluatorImportError, match="selection evidence"):
        _validate_candidate(mutated)

    quarantined_item = {
        **bundle.candidate["items"][0],
        "task_id": sorted(TASK_SCOPE_QUARANTINE)[0],
    }
    mutated = {
        **bundle.candidate,
        "items": [quarantined_item, *bundle.candidate["items"][1:]],
    }
    with pytest.raises(AuthorEvaluatorImportError, match="quarantined tasks"):
        _validate_candidate(mutated)
