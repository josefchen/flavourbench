from pathlib import Path

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from flavourbench.current_pilot_review_import import import_review_pool
from flavourbench.frontier_multirun_assets import RunInput
from flavourbench.frontier_multirun_review_pool import (
    build_multirun_review_pool,
    write_review_pool,
)
from flavourbench.models import Base, Battle, ResponseArm, ToolCall

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/season1/current-quality-run/pilot-v27-eight-pairs"
RUN = RunInput(
    summary=(
        PILOT
        / "summaries/real-exploratory-summary-"
        "d0876f6e7b70d9803468b766b4df91f983fcf684c463766bbe9be1b35cda7018.json"
    ),
    sources=PILOT / "source",
    responses=PILOT / "responses",
)


def test_multirun_review_pool_contains_only_real_complete_blinded_pairs(
    tmp_path: Path,
) -> None:
    pool = build_multirun_review_pool([RUN])

    assert len(pool.pairs) == 50
    assert pool.manifest["observed"]["source_candidate_pairs_before_task_quarantine"] == 65
    assert pool.manifest["observed"]["task_quarantined_candidate_pairs"] == 15
    assert pool.manifest["observed"]["candidate_pairs"] == 50
    assert pool.manifest["observed"]["synthetic_arms"] == 0
    assert pool.manifest["observed"]["candidate_pairs_by_model"]["moonshotai/kimi-k3"] == 4
    assert pool.manifest["observed"]["candidate_pairs_by_model"]["x-ai/grok-4.5"] == 0
    assert pool.manifest["model_contracts"]["moonshotai/kimi-k3"] == {
        "canonical_model_slug": "k3",
        "provider_tag": "kimi-code-direct",
        "execution_backend": "kimi_direct",
    }
    assert all(
        {arm.condition for arm in pair.arms} == {"epicure_off", "epicure_on"}
        and {arm.side for arm in pair.arms} == {"left", "right"}
        for pair in pool.pairs
    )
    assert pool.manifest["claim_boundary"]["quality_judgments"] == 0
    assert pool.manifest["claim_boundary"]["rank_eligible"] is False

    path = write_review_pool(pool, tmp_path)
    assert path.is_file()
    assert pool.artifact_sha256 in path.name


def test_multirun_review_pool_projects_to_blinded_database_rows() -> None:
    pool = build_multirun_review_pool([RUN])
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        result = import_review_pool(session, pool)
        session.commit()
        assert result["battles"] == 50
        assert result["arms"] == 100
        assert result["syntheticArms"] == 0
        assert result["rankEligibleBattles"] == 0
        assert session.scalar(select(func.count()).select_from(Battle)) == 50
        assert session.scalar(select(func.count()).select_from(ResponseArm)) == 100
        assert session.scalar(select(func.count()).select_from(ToolCall)) == 104
        assert import_review_pool(session, pool)["idempotent"] is True
