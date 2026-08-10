from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from flavourbench.frontier_contract_runner import ArtifactExposure, IntegrityError
from flavourbench.frontier_coverage_postrun import (
    ARENA_SCHEMA_VERSION,
    BUNDLE_SCHEMA_VERSION,
    COVERAGE_SCHEMA_VERSION,
    UPLIFT_SCHEMA_VERSION,
    build_corrected_documents,
    load_base_inputs,
    load_historical_responses,
    materialize_postrun,
)
from flavourbench.frontier_coverage_repair_executor import (
    CoverageState,
    RunAccounting,
    build_materialization,
)
from flavourbench.real_dataset_runner import DatasetSource, ResponseArtifact
from flavourbench.real_task_bank import sha256_json

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/season1/current-quality-run"
SCHEDULE = CURRENT / "frontier-coverage-repair-v1" / (
    "frontier-coverage-repair-"
    "45ffc02f56b16b04f2fb4ce51c3561ddb99bd0cad55bf3a7c5162107b2085857.json"
)
ARENA = CURRENT / "frontier-model-arena-review-pool-quarantine-v1" / (
    "frontier-model-arena-review-pool-"
    "407e7fc6413e6d009c942eb51d9603d7cb958f0f282ffe90e1dc8ff28c3b6ac3.json"
)
STRICT = CURRENT / "frontier-strict-review-pool-quarantine-v1" / (
    "frontier-multirun-review-pool-"
    "0da4c58326a936daef3d9e6ac606cfb5abaff2e9d93784754c56a302c662f38c.json"
)
HIGH = CURRENT / "frontier-high-resource-review-pool-quarantine-v1" / (
    "frontier-multirun-review-pool-"
    "cd47055d12e6360a1ad0bfaa73fe4b2cef5bd1f5666150968bdfeeaf9eca024c.json"
)
TASKS = ROOT / "artifacts/season1/task-validity/development-v2" / (
    "development-task-validity-v2-"
    "86fd22c6e3fb331df3bfd18c2363572bb39036f1f1230d30a8bef085195d1119.json"
)
ROUTES = (
    CURRENT / "manifest-v29-high-resource" / (
        "flavourbench-routed-unranked-"
        "f87ee4f9d94c087e8c7486fa3e8bf8800b13d5dae34fa2f95af5848c3eb705de.json"
    ),
    CURRENT / "manifest-v42-high-resource-cohere-direct" / (
        "flavourbench-cohere-unranked-"
        "fd28d55f78056d4d668a8f610a8de63228f7aabdc05fdfb5bfa4389d837d8a22.json"
    ),
)
HISTORICAL_RESPONSE_DIRS = tuple(
    CURRENT / run / "responses"
    for run in (
        "pilot-v27-eight-pairs",
        "pilot-v28-replenishment",
        "pilot-v29-high-resource",
        "pilot-v30-floor-replenishment",
        "pilot-v32-floor-replenishment",
        "pilot-v33-mistral-floor",
        "pilot-v42-cohere-direct",
        "pilot-v43-cohere-direct",
        "pilot-v44-cohere-direct",
    )
)


@pytest.fixture(scope="module")
def upstream():
    materialization = build_materialization(
        schedule_path=SCHEDULE,
        arena_path=ARENA,
        task_validity_path=TASKS,
        route_manifest_paths=ROUTES,
    )
    strict, high, arena = load_base_inputs(
        strict_path=STRICT,
        high_path=HIGH,
        arena_path=ARENA,
        materialization=materialization,
    )
    historical = load_historical_responses(HISTORICAL_RESPONSE_DIRS)
    return materialization, strict, high, arena, historical


def _fake_completed_state(materialization, root: Path) -> CoverageState:  # type: ignore[no-untyped-def]
    responses: dict[tuple[str, str], ResponseArtifact] = {}
    sources: dict[str, DatasetSource] = {}
    reservations: dict[str, dict[str, object]] = {}
    finalizations: dict[str, dict[str, object]] = {}
    response_root = root / "responses"
    response_root.mkdir(parents=True)
    for cell in materialization.cells:
        work_item = cell.work_item
        source_digest = sha256_json(
            {"fixture": "real-source-placeholder", "work_item_id": work_item.work_item_id}
        )
        source_path = root / "source" / f"source-{source_digest}.json"
        exposure = ArtifactExposure(
            path=source_path,
            artifact_sha256=source_digest,
            status="complete",
            requested_model_id=work_item.candidate.model_id,
            requested_provider=work_item.candidate.provider_tag,
            candidate_manifest_sha256=work_item.manifest_sha256,
            actual_cost_usd=Decimal("0.000001"),
            forecast_usd=cell.forecast.forecast_usd,
            admitted_cap_usd=cell.forecast.forecast_usd,
            exposure_usd=Decimal("0.000001"),
            exposure_basis="fully_reconciled_actual",
            contract_passed=True,
        )
        sources[work_item.work_item_id] = DatasetSource(
            path=source_path,
            artifact_sha256=source_digest,
            work_item_id=work_item.work_item_id,
            artifact={"artifact_sha256": source_digest},
            exposure=exposure,
        )
        response_digests: list[str] = []
        for condition in cell.conditions:
            generation_id = f"coverage-{work_item.work_item_id[:16]}-{condition}"
            trace = []
            if condition == "epicure_on":
                result = json.dumps(
                    {"fixture": "verified real Epicure result", "cell": cell.schedule_cell_id}
                )
                trace = [
                    {
                        "arguments": {"ingredients": ["tomato", "miso"]},
                        "result": result,
                        "result_sha256": hashlib.sha256(result.encode()).hexdigest(),
                        "is_error": False,
                    }
                ]
            payload = {
                "schema_version": "flavourbench-real-exploratory-response-v1",
                "artifact_type": "model_response",
                "official": False,
                "rank_eligible": False,
                "research_result": False,
                "research_release_eligible": False,
                "condition": condition,
                "work_item_id": work_item.work_item_id,
                "execution_policy_sha256": materialization.policy.sha256,
                "model": {
                    "requested_model_id": work_item.candidate.model_id,
                    "canonical_model_slug": work_item.candidate.canonical_model_slug,
                    "actual_model_id": work_item.candidate.canonical_model_slug,
                    "actual_provider": work_item.candidate.provider_tag,
                    "provider_tag": work_item.candidate.provider_tag,
                    "execution_backend": work_item.candidate.execution_backend,
                },
                "task": {
                    "public_id": work_item.task.public_id,
                    "family": work_item.task.family,
                    "prompt_sha256": work_item.task.prompt_sha256,
                    "review_status": "candidate",
                },
                "response": {
                    "requested_model_id": work_item.candidate.model_id,
                    "actual_model_id": work_item.candidate.canonical_model_slug,
                    "actual_provider": work_item.candidate.provider_tag,
                    "answer_markdown": f"Verified fixture answer {generation_id}",
                    "finish_reason": "stop",
                    "generation_id": generation_id,
                    "generation_ids": [generation_id],
                    "cost_micros": 1,
                    "cost_reconciled": True,
                    "tool_trace": trace,
                    "latency_ms": 1,
                },
                "cost": {
                    "all_generation_usage_accounted": True,
                    "all_generation_costs_reconciled": True,
                },
                "provenance": {
                    "epicure_access": condition == "epicure_on",
                    "epicure": {
                        "release_id": materialization.epicure["release_id"],
                        "bundle_sha256": materialization.epicure["bundle_sha256"],
                        "application_sha256": materialization.epicure["application_sha256"],
                    },
                    "epicure_tool_schema_sha256": materialization.epicure[
                        "tool_schema_sha256"
                    ],
                },
                "source": {"artifact_sha256": source_digest},
            }
            digest = sha256_json(payload)
            document = {**payload, "artifact_sha256": digest}
            path = response_root / (
                f"response-{work_item.work_item_id}-{condition}-{digest}.json"
            )
            path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            response = ResponseArtifact(
                path=path,
                artifact_sha256=digest,
                work_item_id=work_item.work_item_id,
                condition=condition,
                task_id=work_item.task.public_id,
                task_family=work_item.task.family,
                model_id=work_item.candidate.model_id,
                provider_tag=work_item.candidate.provider_tag,
                source_artifact_sha256=source_digest,
                actual_cost_usd=Decimal("0.000001"),
                tool_used=condition == "epicure_on",
            )
            responses[(work_item.work_item_id, condition)] = response
            response_digests.append(digest)
        reservations[work_item.work_item_id] = {"entry_sha256": "a" * 64}
        finalizations[work_item.work_item_id] = {
            "source_artifact_sha256": source_digest,
            "response_artifact_sha256s": sorted(response_digests),
        }
    accounting = RunAccounting(
        source_count=len(sources),
        actual_cost_usd=Decimal("0.000013"),
        exposure_usd=Decimal("0.000013"),
        orphan_reservation_usd=Decimal(0),
        artifact_sha256s=frozenset(source.artifact_sha256 for source in sources.values()),
        sources=sources,
        reservations=reservations,
        finalizations=finalizations,
        blockers=(),
    )
    return CoverageState(accounting=accounting, responses=responses, ledger=())


def test_completed_repair_builds_exact_corrected_inputs(upstream, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    materialization, strict_base, high_base, arena_base, historical = upstream
    state = _fake_completed_state(materialization, tmp_path)
    documents = build_corrected_documents(
        materialization=materialization,
        coverage_state=state,
        strict_base=strict_base,
        high_base=high_base,
        arena_base=arena_base,
        historical_responses=historical,
    )
    assert documents.strict["schema_version"] == UPLIFT_SCHEMA_VERSION
    assert documents.high["schema_version"] == UPLIFT_SCHEMA_VERSION
    assert documents.arena["schema_version"] == ARENA_SCHEMA_VERSION
    assert documents.coverage["schema_version"] == COVERAGE_SCHEMA_VERSION
    assert documents.strict["observed"]["candidate_pairs"] == 86
    assert documents.high["observed"]["candidate_pairs"] == 106
    assert documents.coverage["uplift"]["combined_pairs_after"] == 192
    assert documents.arena["observed"]["candidate_comparisons"] == 1043
    assert documents.arena["observed"]["coverage_repair_candidate_comparisons_added"] == 167
    assert documents.arena["observed"]["source_response_arms"] == 197
    assert documents.arena["observed"]["missing_model_pair_family_cells"] == 0
    assert documents.arena["observed"]["comparison_graph_component_sizes"] == [16]
    assert documents.coverage["model_arena"]["missing_cells_before_by_family"] == {
        "composition": 17,
        "cookability": 27,
        "evidence": 27,
        "substitution": 23,
    }
    assert documents.coverage["model_arena"]["missing_cells_after"] == 0
    assert documents.coverage["interpretation"]["family_specific_ranking_supported"] is False
    source_records = documents.coverage["coverage_repair"]["source_records"]
    assert len(source_records) == 13
    assert len({row["source_artifact_sha256"] for row in source_records}) == 13
    assert sum(
        len(row["response_artifact_sha256s_by_condition"]) for row in source_records
    ) == 25
    assert documents.coverage["coverage_repair"][
        "source_records_commitment_sha256"
    ] == sha256_json(source_records)
    assert all(document["claim_boundary"]["official"] is False for document in (
        documents.strict,
        documents.high,
        documents.arena,
        documents.coverage,
    ))


def test_completed_materialization_is_content_addressed_and_deterministic(
    upstream, tmp_path: Path  # type: ignore[no-untyped-def]
) -> None:
    materialization, strict_base, high_base, arena_base, historical = upstream
    state = _fake_completed_state(materialization, tmp_path / "state")
    first = build_corrected_documents(
        materialization=materialization,
        coverage_state=state,
        strict_base=strict_base,
        high_base=high_base,
        arena_base=arena_base,
        historical_responses=historical,
    )
    second = build_corrected_documents(
        materialization=materialization,
        coverage_state=state,
        strict_base=strict_base,
        high_base=high_base,
        arena_base=arena_base,
        historical_responses=historical,
    )
    assert first == second
    for document in (first.strict, first.high, first.arena, first.coverage):
        payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
        assert sha256_json(payload) == document["artifact_sha256"]


def test_missing_coverage_arm_fails_closed(upstream, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    materialization, strict_base, high_base, arena_base, historical = upstream
    state = _fake_completed_state(materialization, tmp_path)
    responses = dict(state.responses)
    responses.pop(next(iter(responses)))
    incomplete = CoverageState(
        accounting=state.accounting,
        responses=responses,
        ledger=state.ledger,
    )
    with pytest.raises(IntegrityError, match="coverage repair is incomplete"):
        build_corrected_documents(
            materialization=materialization,
            coverage_state=incomplete,
            strict_base=strict_base,
            high_base=high_base,
            arena_base=arena_base,
            historical_responses=historical,
        )


def test_historical_response_index_must_cover_every_committed_arm(upstream) -> None:  # type: ignore[no-untyped-def]
    materialization, strict_base, high_base, arena_base, historical = upstream
    required = str(strict_base["items"][0]["left"]["response_artifact_sha256"])
    incomplete = dict(historical)
    incomplete.pop(required)
    state = CoverageState(
        accounting=RunAccounting(
            source_count=0,
            actual_cost_usd=Decimal(0),
            exposure_usd=Decimal(0),
            orphan_reservation_usd=Decimal(0),
            artifact_sha256s=frozenset(),
            sources={},
            reservations={},
            finalizations={},
            blockers=(),
        ),
        responses={},
        ledger=(),
    )
    with pytest.raises(IntegrityError, match="historical response index is incomplete"):
        build_corrected_documents(
            materialization=materialization,
            coverage_state=state,
            strict_base=strict_base,
            high_base=high_base,
            arena_base=arena_base,
            historical_responses=incomplete,
        )


def test_current_no_call_state_cannot_emit_paper_bundle(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match="coverage repair is incomplete"):
        materialize_postrun(
            schedule_path=SCHEDULE,
            arena_base_path=ARENA,
            strict_base_path=STRICT,
            high_base_path=HIGH,
            task_validity_path=TASKS,
            route_manifest_paths=ROUTES,
            historical_response_directories=HISTORICAL_RESPONSE_DIRS,
            coverage_source_directory=tmp_path / "source",
            coverage_corrections_directory=tmp_path / "corrections",
            coverage_response_directory=tmp_path / "responses",
            coverage_ledger_path=tmp_path / "ledger.jsonl",
            output_directory=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_successful_command_writes_bundle_without_mutating_history(
    upstream, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
) -> None:
    materialization, _strict, _high, _arena, _historical = upstream
    state = _fake_completed_state(materialization, tmp_path / "state")
    monkeypatch.setattr(
        "flavourbench.frontier_coverage_postrun._coverage_state",
        lambda *args, **kwargs: state,
    )
    sentinel = HISTORICAL_RESPONSE_DIRS[0] / next(
        path.name for path in sorted(HISTORICAL_RESPONSE_DIRS[0].glob("*.json"))
    )
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    bundle, paths = materialize_postrun(
        schedule_path=SCHEDULE,
        arena_base_path=ARENA,
        strict_base_path=STRICT,
        high_base_path=HIGH,
        task_validity_path=TASKS,
        route_manifest_paths=ROUTES,
        historical_response_directories=HISTORICAL_RESPONSE_DIRS,
        coverage_source_directory=tmp_path / "unused-source",
        coverage_corrections_directory=tmp_path / "unused-corrections",
        coverage_response_directory=tmp_path / "unused-responses",
        coverage_ledger_path=tmp_path / "unused-ledger.jsonl",
        output_directory=tmp_path / "output",
    )
    assert bundle["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert bundle["counts"] == {
        "coverage_source_records": 13,
        "coverage_new_real_arms": 25,
        "synthetic_arms": 0,
        "quality_judgments": 0,
    }
    assert set(paths) == {"strict", "high", "arena", "coverage", "bundle"}
    for path in paths.values():
        document = json.loads(path.read_text())
        digest = document.pop("artifact_sha256")
        assert digest in path.name
        assert sha256_json(document) == digest
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before


def test_bundle_schema_constant_is_frozen() -> None:
    assert BUNDLE_SCHEMA_VERSION == "flavourbench-frontier-corrected-paper-input-bundle-v1"
