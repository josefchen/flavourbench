from pathlib import Path

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from flavourbench.current_pilot_review_import import import_review_pool
from flavourbench.frontier_model_arena_review_pool import (
    StratumInput,
    build_model_arena_review_pool,
    write_model_arena_review_pool,
)
from flavourbench.frontier_multirun_assets import RunInput
from flavourbench.models import Base, Battle, ResponseArm, Season, Task, ToolCall

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/season1/current-quality-run"


def _run(root: str, digest: str) -> RunInput:
    base = PILOT / root
    return RunInput(
        summary=base / "summaries" / f"real-exploratory-summary-{digest}.json",
        sources=base / "source",
        responses=base / "responses",
    )


STRICT = StratumInput(
    label="strict",
    runs=(
        _run(
            "pilot-v27-eight-pairs",
            "d0876f6e7b70d9803468b766b4df91f983fcf684c463766bbe9be1b35cda7018",
        ),
        _run(
            "pilot-v28-replenishment",
            "a3e6674274a270d51424e86d1726b3a52abade109ec27590f5adc4bde8fa5a05",
        ),
    ),
)
HIGH_RESOURCE = StratumInput(
    label="high-resource",
    runs=(
        _run(
            "pilot-v29-high-resource",
            "9710525c84feed31ed7ddfa6ae172cff915b36b8fdab8f7dedbeba2bdb0c8084",
        ),
        _run(
            "pilot-v30-floor-replenishment",
            "6fe0e3ff11572069900bb1a06b24bc7377ea6440c92d24e4567f5138db4553b6",
        ),
        _run(
            "pilot-v32-floor-replenishment",
            "26b0392db5c4e1ae3a4e8f7ce53b4981f0b9bf0ab9e9ea27d71fd5b9a17593fe",
        ),
        _run(
            "pilot-v33-mistral-floor",
            "93e134e2bacd766afb5bb18cd558d6d352991b5acb05eb162464a9ff47b3260e",
        ),
    ),
)


def test_model_arena_pool_is_real_blinded_connected_and_complete(tmp_path: Path) -> None:
    pool = build_model_arena_review_pool((STRICT, HIGH_RESOURCE))
    reversed_pool = build_model_arena_review_pool((HIGH_RESOURCE, STRICT))
    observed = pool.manifest["observed"]

    assert reversed_pool.artifact_sha256 == pool.artifact_sha256
    assert reversed_pool.manifest == pool.manifest
    assert pool.manifest["track"] == "model_arena"
    assert observed["source_candidate_comparisons_before_task_quarantine"] == 843
    assert observed["source_response_arms_before_task_quarantine"] == 196
    assert observed["task_quarantined_source_response_arms"] == 29
    assert observed["task_quarantined_candidate_comparisons"] == 118
    assert observed["candidate_comparisons"] == 725
    assert observed["candidate_comparisons_by_stratum"] == {
        "high-resource": 345,
        "strict": 380,
    }
    assert observed["unique_task_ids"] == 20
    assert observed["task_stratum_clusters"] == 20
    assert observed["models"] == 14
    assert observed["source_response_arms"] == 167
    assert observed["comparison_graph_component_sizes"] == [14]
    assert len(observed["candidate_comparisons_by_model_pair"]) == 91
    assert sum(
        row["candidate_comparisons"]
        for row in observed["candidate_comparisons_by_model_pair"]
    ) == 725
    assert observed["model_pair_family_cells"] == 364
    assert observed["missing_model_pair_family_cells"] == 79
    assert observed["evidence_units"]["raw_comparison_rows"] == 725
    assert observed["evidence_units"]["unique_response_arms"] == 167
    assert observed["evidence_units"]["maximum_comparisons_per_reused_response_arm"] == 12
    assert observed["synthetic_arms"] == 0
    assert pool.manifest["claim_boundary"]["quality_judgments"] == 0
    assert all(
        pair.arms[0].condition == pair.arms[1].condition == "epicure_on"
        and pair.arms[0].response["model"]["requested_model_id"]
        != pair.arms[1].response["model"]["requested_model_id"]
        and pair.arms[0].response["task"]["public_id"] == pair.arms[1].response["task"]["public_id"]
        for pair in pool.pairs
    )
    paths = write_model_arena_review_pool(pool, tmp_path)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    assert pool.artifact_sha256 in paths["pool"].name
    assert paths["coverage_figure"].suffix == ".pdf"
    macros = paths["macros"].read_text(encoding="utf-8")
    assert r"\newcommand{\FrontierArenaGrossCandidateComparisons}{843}" in macros
    assert r"\newcommand{\FrontierArenaQuarantinedComparisonCount}{118}" in macros
    assert r"\newcommand{\FrontierArenaMissingModelPairFamilyCells}{79}" in macros
    assert r"\newcommand{\FrontierArenaResponseReuseMaximum}{12}" in macros

    nonvisual_paths = write_model_arena_review_pool(
        pool,
        tmp_path / "nonvisual",
        render_figure=False,
    )
    assert "coverage_figure" not in nonvisual_paths
    assert "coverage_svg" not in nonvisual_paths
    assert all(path.is_file() and path.stat().st_size > 0 for path in nonvisual_paths.values())


def test_model_arena_pool_projects_to_real_blinded_review_rows() -> None:
    pool = build_model_arena_review_pool((STRICT, HIGH_RESOURCE))
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

        assert result["battles"] == 725
        assert result["arms"] == 1450
        assert result["toolCalls"] == 3373
        assert result["syntheticArms"] == 0
        assert result["rankEligibleBattles"] == 0
        assert session.scalar(select(func.count()).select_from(Task)) == 20
        assert session.scalar(select(func.count()).select_from(Battle)) == 725
        assert session.scalar(select(func.count()).select_from(ResponseArm)) == 1450
        assert session.scalar(select(func.count()).select_from(ToolCall)) == 3373
        assert session.scalar(select(func.count()).select_from(Season)) == 1
        assert (
            session.scalar(
                select(func.count()).select_from(Battle).where(
                    Battle.track == "model_arena",
                    Battle.data_stratum == "development",
                    Battle.rank_eligible.is_(False),
                )
            )
            == 725
        )
        assert import_review_pool(session, pool)["idempotent"] is True
