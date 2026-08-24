from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import Session

from flavourbench.current_pilot_review_import import (
    CurrentPilotReviewImportError,
    build_review_pool,
    import_review_pool,
)
from flavourbench.expert_review import isolated_uplift_workload_cell_targets
from flavourbench.models import Base, Battle, ResponseArm, RunEvent, Season, Task, ToolCall
from flavourbench.schemas import AuthorEvaluatorAdmissionCreate

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "artifacts/season1/current-quality-run/pilot-v19"
SUMMARY = PILOT_ROOT / (
    "summaries/real-exploratory-summary-"
    "f0bfa024315c4efdbcc59f8251fdea6fc6fd3592875c9ff0aee65f56ced78e20.json"
)
FAMILY_COUNTS = {
    "substitution": 14,
    "composition": 8,
    "cookability": 12,
    "evidence": 13,
}


def _pool():
    return build_review_pool(
        summary_path=SUMMARY,
        source_dir=PILOT_ROOT / "source",
        response_dir=PILOT_ROOT / "responses",
    )


def test_current_frontier_review_pool_is_real_complete_and_balanced() -> None:
    pool = _pool()
    observed = pool.manifest["observed"]

    assert observed["candidate_pairs"] == 47
    assert observed["source_arms"] == 94
    assert observed["candidate_pairs_by_family"] == FAMILY_COUNTS
    assert observed["real_provider_calls"] == 148
    assert observed["real_epicure_calls"] == 67
    assert observed["successful_real_epicure_calls"] == 25
    assert observed["synthetic_arms"] == 0
    assert {observed["left_epicure_on"], observed["right_epicure_on"]} == {23, 24}
    assert pool.manifest["claim_boundary"]["quality_judgments"] == 0


def test_unbalanced_real_pool_has_an_explicit_isolated_review_contract() -> None:
    targets = isolated_uplift_workload_cell_targets(FAMILY_COUNTS)

    assert targets["primary"]["epicure_uplift"] == FAMILY_COUNTS
    assert targets["primary_judgments"] == 47
    assert targets["reliability_repeats"] == 5
    assert targets["total_presentations"] == 52
    request = AuthorEvaluatorAdmissionCreate.model_validate(
        {
            "qualification_reference": "governance/reviewers/josef-chen.json",
            "conflict_disclosure_reference": "governance/reviewers/josef-chen.json",
            "candidate_pack_sha256": "a" * 64,
            "primary_judgments": 47,
            "primaryByFamily": FAMILY_COUNTS,
            "admission_decision_reference": "governance/current-frontier-review.md",
            "independent_validation_claim": False,
        }
    )
    assert request.primary_by_family is not None
    assert sum(request.primary_by_family.values()) == 47

    with pytest.raises(ValueError, match="sum"):
        AuthorEvaluatorAdmissionCreate.model_validate(
            {
                "qualification_reference": "reviewer record",
                "conflict_disclosure_reference": "reviewer record",
                "candidate_pack_sha256": "a" * 64,
                "primary_judgments": 47,
                "primaryByFamily": {**FAMILY_COUNTS, "composition": 7},
                "admission_decision_reference": "review decision",
            }
        )


def test_current_frontier_review_pool_import_is_isolated_and_idempotent() -> None:
    pool = _pool()
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        first = import_review_pool(session, pool)
        session.commit()
        assert first == {
            "schemaVersion": "flavourbench-current-frontier-review-pool-v1",
            "reviewPoolSha256": pool.artifact_sha256,
            "tasks": 4,
            "battles": 47,
            "arms": 94,
            "toolCalls": 67,
            "syntheticArms": 0,
            "rankEligibleBattles": 0,
            "idempotent": False,
            "eventId": first["eventId"],
        }
        assert session.scalar(select(func.count()).select_from(Task)) == 4
        assert session.scalar(select(func.count()).select_from(Battle)) == 47
        assert session.scalar(select(func.count()).select_from(ResponseArm)) == 94
        assert session.scalar(select(func.count()).select_from(ToolCall)) == 67
        assert (
            session.scalar(
                select(func.count())
                .select_from(Battle)
                .where(
                    Battle.run_class == "pilot",
                    Battle.data_stratum == "development",
                    Battle.rank_eligible.is_(False),
                )
            )
            == 47
        )
        assert import_review_pool(session, pool)["idempotent"] is True

        arm = session.scalar(select(ResponseArm).order_by(ResponseArm.id).limit(1))
        assert arm is not None
        original_answer = arm.answer_markdown
        session.execute(
            text("UPDATE response_arms SET answer_markdown = :value WHERE id = :arm_id"),
            {"value": f"{original_answer}\nTAMPERED", "arm_id": arm.id},
        )
        session.commit()
        with pytest.raises(CurrentPilotReviewImportError, match="projection has drifted"):
            import_review_pool(session, pool)

        event_row = session.scalar(
            select(RunEvent).where(RunEvent.entity_id == pool.artifact_sha256)
        )
        season = session.scalar(
            select(Season).where(Season.manifest_sha256 == pool.artifact_sha256)
        )
        assert event_row is not None
        assert season is not None
        assert season.official is False
