from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from flavourbench.current_pilot_review_import import import_review_pool
from flavourbench.models import Base, Battle, ResponseArm, Season, Task, ToolCall
from flavourbench.required_pilot_review_import import (
    EXPECTED_FAMILY_COUNTS,
    EXPECTED_MODEL_COUNTS,
    SCHEMA_VERSION,
    build_required_review_pool,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "artifacts/season1/current-quality-run/pilot-v24-required-epicure"
SUMMARY = PILOT_ROOT / (
    "summaries/real-exploratory-summary-"
    "f1a38e30042b9614fa82f5c38b43b98c7c9a18916c4541f21bb12d3bcce8ba70.json"
)


def _pool():
    return build_required_review_pool(
        summary_path=SUMMARY,
        source_dir=PILOT_ROOT / "source",
        response_dir=PILOT_ROOT / "responses",
    )


def test_required_review_pool_is_real_complete_and_balanced() -> None:
    pool = _pool()
    observed = pool.manifest["observed"]

    assert pool.manifest["schema_version"] == SCHEMA_VERSION
    assert observed["candidate_pairs"] == 43
    assert observed["source_arms"] == 86
    assert observed["candidate_pairs_by_family"] == EXPECTED_FAMILY_COUNTS
    assert observed["candidate_pairs_by_model"] == EXPECTED_MODEL_COUNTS
    assert observed["real_provider_calls"] == 277
    assert observed["real_epicure_calls"] == 182
    assert observed["successful_real_epicure_calls"] == 86
    assert observed["synthetic_arms"] == 0
    assert {observed["left_epicure_on"], observed["right_epicure_on"]} == {21, 22}
    assert pool.manifest["selection_policy"]["raw_answers_edited"] is False
    assert pool.manifest["claim_boundary"]["quality_judgments"] == 0


def test_required_review_pool_import_is_isolated_and_idempotent() -> None:
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

        assert first["schemaVersion"] == SCHEMA_VERSION
        assert first["battles"] == 43
        assert first["arms"] == 86
        assert first["toolCalls"] == 182
        assert first["syntheticArms"] == 0
        assert first["rankEligibleBattles"] == 0
        assert first["idempotent"] is False
        assert session.scalar(select(func.count()).select_from(Task)) == 4
        assert session.scalar(select(func.count()).select_from(Battle)) == 43
        assert session.scalar(select(func.count()).select_from(ResponseArm)) == 86
        assert session.scalar(select(func.count()).select_from(ToolCall)) == 182
        season = session.scalar(select(Season))
        assert season is not None
        assert season.official is False

        second = import_review_pool(session, pool)
        assert second["schemaVersion"] == SCHEMA_VERSION
        assert second["idempotent"] is True
