from __future__ import annotations

import json
from pathlib import Path

from flavourbench.current_frontier_task_quarantine import (
    build_quarantine_artifact,
    quarantine_task_ids,
    verify_quarantine_artifact,
    write_quarantine_artifact,
)
from flavourbench.frontier_coverage_repair import build_coverage_repair_schedule
from flavourbench.frontier_model_arena_review_pool import (
    StratumInput,
    build_model_arena_review_pool,
)
from flavourbench.frontier_multirun_assets import RunInput
from flavourbench.frontier_multirun_review_pool import build_multirun_review_pool

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/season1/current-quality-run"
TASK_VALIDITY = (
    ROOT
    / "artifacts/season1/task-validity/development-v2/"
    "development-task-validity-v2-"
    "86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json"
)


def _run(name: str, digest: str) -> RunInput:
    root = CURRENT / name
    return RunInput(
        summary=root / "summaries" / f"real-exploratory-summary-{digest}.json",
        sources=root / "source",
        responses=root / "responses",
    )


STRICT = (
    _run(
        "pilot-v27-eight-pairs",
        "d0876f6e7b70d9803468b766b4df91f983fcf684c463766bbe9be1b35cda7018",
    ),
    _run(
        "pilot-v28-replenishment",
        "a3e6674274a270d51424e86d1726b3a52abade109ec27590f5adc4bde8fa5a05",
    ),
)
HIGH_RESOURCE = (
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
    _run(
        "pilot-v42-cohere-direct",
        "b32df510da8125b91248bdc29f8f7c7cc6b9ab2abccabe762e89cfa00b9965b0",
    ),
    _run(
        "pilot-v43-cohere-direct",
        "814af7f7aaa5be5b76674292ef3a5f09a05a1303f969ab78ddbf47536cf68e2d",
    ),
    _run(
        "pilot-v44-cohere-direct",
        "b39a1de038f40f8d16a73597788ed9263d91dd4a54e556fcd29578b4284d8b30",
    ),
)


def _corrected_pools():
    arena = build_model_arena_review_pool(
        (
            StratumInput("strict", STRICT),
            StratumInput("high-resource", HIGH_RESOURCE),
        )
    )
    strict = build_multirun_review_pool(STRICT)
    high_resource = build_multirun_review_pool(HIGH_RESOURCE)
    return arena, strict, high_resource


def test_quarantine_is_content_addressed_and_non_destructive(tmp_path: Path) -> None:
    artifact = build_quarantine_artifact()
    verify_quarantine_artifact(artifact)
    path = write_quarantine_artifact(tmp_path)

    assert artifact["status"] == "quarantine_pending_qualified_adjudication"
    assert artifact["record_count"] == 4
    assert {row["task_id"] for row in artifact["records"]} == quarantine_task_ids()
    assert artifact["scope"]["raw_tasks_mutated"] is False
    assert artifact["scope"]["raw_response_arms_mutated"] is False
    assert artifact["claim_boundary"]["declares_tasks_definitively_invalid"] is False
    assert artifact["artifact_sha256"] in path.name


def test_corrected_real_pools_quarantine_exact_exposure_without_synthetic_data() -> None:
    arena, strict, high_resource = _corrected_pools()
    observed = arena.manifest["observed"]

    assert observed["source_candidate_comparisons_before_task_quarantine"] == 1_024
    assert observed["source_response_arms_before_task_quarantine"] == 218
    assert observed["task_quarantined_source_response_arms"] == 33
    assert observed["task_quarantined_candidate_comparisons"] == 148
    assert observed["candidate_comparisons"] == 876
    assert observed["source_response_arms"] == 185
    assert observed["unique_task_ids"] == 20
    assert observed["task_stratum_clusters"] == 20
    assert observed["model_pair_family_cells"] == 480
    assert observed["missing_model_pair_family_cells"] == 94
    assert observed["evidence_units"] == {
        "raw_comparison_rows": 876,
        "unique_task_ids": 20,
        "task_stratum_clusters": 20,
        "unique_response_arms": 185,
        "response_arm_presentations": 1_752,
        "minimum_comparisons_per_reused_response_arm": 2,
        "maximum_comparisons_per_reused_response_arm": 13,
        "median_comparisons_per_reused_response_arm": 9,
        "independence_unit_for_uncertainty": "task_cluster",
        "response_reuse_policy": (
            "shared response arms remain locked inside their task cluster during "
            "resampling; comparison rows are not treated as independent"
        ),
        "scalar_effective_sample_size_claimed": False,
    }
    assert strict.manifest["observed"]["source_candidate_pairs_before_task_quarantine"] == 101
    assert strict.manifest["observed"]["task_quarantined_candidate_pairs"] == 15
    assert strict.manifest["observed"]["candidate_pairs"] == 86
    assert high_resource.manifest["observed"][
        "source_candidate_pairs_before_task_quarantine"
    ] == 110
    assert high_resource.manifest["observed"]["task_quarantined_candidate_pairs"] == 17
    assert high_resource.manifest["observed"]["candidate_pairs"] == 93
    assert all(
        item["task_id"] not in quarantine_task_ids()
        for pool in (arena, strict, high_resource)
        for item in pool.manifest["items"]
    )
    assert sum(
        pool.manifest["observed"]["synthetic_arms"]
        for pool in (arena, strict, high_resource)
    ) == 0


def test_coverage_repair_freezes_exact_real_call_cells_and_closes_empty_families() -> None:
    arena, strict, high_resource = _corrected_pools()
    task_validity = json.loads(TASK_VALIDITY.read_text(encoding="utf-8"))
    schedule = build_coverage_repair_schedule(
        arena_pool=arena.manifest,
        uplift_pools=(strict.manifest, high_resource.manifest),
        task_validity=task_validity,
    )

    assert [row["family"] for row in schedule["anchors"]] == list(
        ("composition", "cookability", "evidence", "substitution")
    )
    assert {row["task_id"] for row in schedule["anchors"]}.isdisjoint(
        quarantine_task_ids()
    )
    assert schedule["counts"]["current_missing_model_pair_family_cells"] == 94
    assert schedule["counts"]["missing_endpoint_task_cells"] == 13
    assert schedule["counts"]["required_new_real_arms"] == 25
    assert schedule["counts"]["projected_missing_model_pair_family_cells_after_schedule"] == 0
    assert schedule["claim_boundary"]["zero_synthetic_arms"] is True
    assert schedule["claim_boundary"]["rank_eligible"] is False
    assert schedule["source"]["task_validity_sha256"] == task_validity["artifact_sha256"]
