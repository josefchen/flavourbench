from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from flavourbench.prospective_task_acquisition import canonical_sha256
from flavourbench.task_validation_automated_replay import (
    PINNED_REPLAY_PHYSICAL_SHA256,
    PINNED_REPLAY_SEMANTIC_SHA256,
)
from flavourbench.task_validation_contamination_replay_v2 import (
    ASSIGNED_PROMPT_COUNT,
    BENCHMARK_SCAN_IMPLEMENTATION_SHA256,
    BENCHMARK_SCAN_IMPLEMENTATION_VERSION,
    BENCHMARK_SNAPSHOT_SCHEMA,
    PINNED_BENCHMARK_SNAPSHOT_PHYSICAL_SHA256,
    PINNED_BENCHMARK_SNAPSHOT_SEMANTIC_SHA256,
    PINNED_REPLAY_V2_PHYSICAL_SHA256,
    PINNED_REPLAY_V2_SEMANTIC_SHA256,
    PINNED_WEB_SNAPSHOT_PHYSICAL_SHA256,
    PINNED_WEB_SNAPSHOT_SEMANTIC_SHA256,
    REPLAY_SCHEMA,
    WEB_SNAPSHOT_SCHEMA,
    ContaminationReplayV2Error,
    ReplayV2Paths,
    _build_benchmark_scan_index,
    _scan_benchmark_prompt,
    build_replay_v2,
    verify_benchmark_snapshot,
    verify_pinned_replay_v2,
    verify_replay_v2,
    verify_web_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIRECTORY = ROOT / "artifacts/season1/task-validation-campaign-v6/contamination-replay-v2"
BENCHMARK_PATH = (
    CAPTURE_DIRECTORY / f"benchmark-snapshot-{PINNED_BENCHMARK_SNAPSHOT_SEMANTIC_SHA256}.json"
)
WEB_PATH = CAPTURE_DIRECTORY / f"web-snapshot-{PINNED_WEB_SNAPSHOT_SEMANTIC_SHA256}.json"
REPLAY_PATH = CAPTURE_DIRECTORY / f"contamination-replay-v2-{PINNED_REPLAY_V2_SEMANTIC_SHA256}.json"
PATHS = ReplayV2Paths.from_root(
    ROOT,
    benchmark_snapshot=BENCHMARK_PATH,
    web_snapshot=WEB_PATH,
)

BENCHMARK_SCHEMA_PATH = (
    ROOT / "contracts/season1/task-validation-contamination-benchmark-snapshot-v2.schema.json"
)
WEB_SCHEMA_PATH = (
    ROOT / "contracts/season1/task-validation-contamination-web-snapshot-v2.schema.json"
)
REPLAY_SCHEMA_PATH = ROOT / "contracts/season1/task-validation-contamination-replay-v2.schema.json"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def replay() -> dict:
    summary = verify_pinned_replay_v2(REPLAY_PATH, PATHS)
    assert summary == {
        "artifact_sha256": PINNED_REPLAY_V2_SEMANTIC_SHA256,
        "physical_sha256": PINNED_REPLAY_V2_PHYSICAL_SHA256,
        "status": "no_go",
        "assigned_prompts": 180,
        "benchmark_records": 4671,
        "exact_fuzzy_ngram_semantic_coverage_percent": {
            "exact": 100,
            "fuzzy": 100,
            "ngram": 100,
            "semantic": 100,
        },
        "web_replay_performed": False,
        "calibration_cases_observed": 0,
        "rank_eligible": False,
    }
    return _json(REPLAY_PATH)


@pytest.mark.parametrize(
    ("schema_path", "artifact_path", "schema_version"),
    [
        (BENCHMARK_SCHEMA_PATH, BENCHMARK_PATH, BENCHMARK_SNAPSHOT_SCHEMA),
        (WEB_SCHEMA_PATH, WEB_PATH, WEB_SNAPSHOT_SCHEMA),
        (REPLAY_SCHEMA_PATH, REPLAY_PATH, REPLAY_SCHEMA),
    ],
)
def test_published_artifacts_are_draft_2020_12_schema_valid(
    schema_path: Path, artifact_path: Path, schema_version: str
) -> None:
    schema = _json(schema_path)
    artifact = _json(artifact_path)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(artifact)
    assert artifact["schema_version"] == schema_version


def test_published_artifacts_have_exact_semantic_and_physical_pins() -> None:
    for path, semantic, physical in (
        (
            BENCHMARK_PATH,
            PINNED_BENCHMARK_SNAPSHOT_SEMANTIC_SHA256,
            PINNED_BENCHMARK_SNAPSHOT_PHYSICAL_SHA256,
        ),
        (
            WEB_PATH,
            PINNED_WEB_SNAPSHOT_SEMANTIC_SHA256,
            PINNED_WEB_SNAPSHOT_PHYSICAL_SHA256,
        ),
        (REPLAY_PATH, PINNED_REPLAY_V2_SEMANTIC_SHA256, PINNED_REPLAY_V2_PHYSICAL_SHA256),
    ):
        document = _json(path)
        assert document["artifact_sha256"] == semantic
        assert hashlib.sha256(path.read_bytes()).hexdigest() == physical
        unsigned = {key: value for key, value in document.items() if key != "artifact_sha256"}
        assert canonical_sha256(unsigned) == semantic


def test_licensed_benchmark_snapshot_is_complete_for_its_declared_nonexhaustive_scope() -> None:
    document = verify_benchmark_snapshot(BENCHMARK_PATH)
    assert len(document["records"]) == 4671
    assert len({row["source_reference_sha256"] for row in document["records"]}) == 4671
    assert document["coverage"] == {
        "captured_dataset_slices": 4,
        "captured_records": 4671,
        "known_relevant_culinary_datasets_excluded": 2,
        "external_benchmark_universe_exhaustive": False,
        "all_captured_records_have_declared_license": True,
    }
    assert {row["license_id"] for row in document["records"]} == {
        "MIT",
        "Apache-2.0",
        "CC-BY-SA-4.0",
    }
    assert all(row["source_url"].startswith("https://") for row in document["records"])
    assert all(
        row["content_downloaded"] is False and row["scan_performed"] is False
        for row in document["excluded_culinary_dataset_receipts"]
    )


def test_web_snapshot_preserves_receipts_but_distributes_no_unknown_license_text() -> None:
    assignment_rows = json.loads(PATHS.v1_inputs.review_assignment.read_text())["assignment_rows"]
    document = verify_web_snapshot(WEB_PATH, assignment_rows)
    assert len(document["query_receipts"]) == ASSIGNED_PROMPT_COUNT
    assert len(document["result_records"]) == 1769
    assert document["collection_failures"] == []
    assert document["coverage"]["known_positive_source_urls_returned"] == 0
    assert document["claim_boundary"]["captured_response_replayable"] is False
    forbidden = {"text", "title", "description", "text_sha256"}
    assert all(not (forbidden & set(record)) for record in document["result_records"])
    assert all(
        receipt["raw_response_retained"] is False
        and receipt["raw_file"] is None
        and len(receipt["raw_response_sha256"]) == 64
        and receipt["raw_response_bytes"] > 0
        for receipt in document["query_receipts"]
    )
    assert not (CAPTURE_DIRECTORY / "raw-web").exists()
    assert list(CAPTURE_DIRECTORY.glob("*.xml")) == []


def test_replay_covers_four_methods_for_every_prompt_and_marks_web_unperformed(
    replay: dict,
) -> None:
    coverage = replay["coverage"]
    assert coverage["assigned_prompts"] == coverage["scan_records"] == 180
    for method in ("exact", "fuzzy", "ngram", "semantic"):
        assert coverage["method_coverage"][method] == {
            "performed": True,
            "assigned_prompts": 180,
            "prompts_scanned": 180,
            "coverage_percent": 100,
            "hit_count": replay["findings"]["method_hit_counts"][method],
        }
    assert coverage["method_coverage"]["web"] == {
        "performed": False,
        "assigned_prompts": 180,
        "prompts_scanned": 0,
        "coverage_percent": 0,
        "hit_count": 0,
    }
    records = replay["findings"]["records"]
    assert len(records) == len({row["candidate_id"] for row in records}) == 180
    assert [row["assignment_ordinal"] for row in records] == list(range(1, 181))
    assert all(
        [method["method"] for method in row["methods"]]
        == [
            "exact",
            "fuzzy",
            "ngram",
            "semantic",
            "web",
        ]
        for row in records
    )
    assert all(row["methods"][-1]["performed"] is False for row in records)
    assert all(row["human_disposition"] is None for row in records)


def test_no_go_is_precise_about_calibration_and_uncalibrated_semantic_behavior(
    replay: dict,
) -> None:
    assert replay["status"] == replay["decision"]["disposition"] == "no_go"
    assert replay["decision"]["full_campaign_contamination_method_requirement_satisfied"] is False
    assert replay["calibration"] == {
        "real_labeled_calibration_artifact_observed": False,
        "cases_observed": 0,
        "precision_threshold_verified": False,
        "recall_threshold_verified": False,
        "paraphrase_recall_threshold_verified": False,
        "test_fixtures_count_as_evidence": False,
    }
    diagnostics = replay["findings"]["detector_diagnostics"]
    assert diagnostics["benchmark_prompt_record_pairs"] == 180 * 4671
    assert diagnostics["semantic_report_hit_rate_million"] >= 50_000
    assert diagnostics["semantic_report_hit_rate_at_least_five_percent"] is True
    assert replay["findings"]["automated_hits_are_human_ground_truth"] is False
    assert replay["external_evidence_assessment"] == {
        "benchmark_corpus_search_performed": True,
        "benchmark_corpus_coverage_complete": False,
        "external_web_search_captured": True,
        "external_web_search_replay_performed": False,
        "external_web_provider_known_positive_validation_passed": False,
        "external_web_method_suitable_for_admission": False,
        "external_result_text_redistribution_rights_confirmed": False,
        "known_relevant_culinary_corpora_excluded": 2,
        "model_training_membership_tested": False,
    }
    assert replay["claim_boundary"] == {
        "contamination_limited": True,
        "contamination_free": False,
        "official_task_bank": False,
        "rank_eligible": False,
        "task_bank_import_authorized": False,
        "campaign_audit_passed": False,
        "human_contamination_decision_observed": False,
        "model_calls": 0,
        "epicure_calls": 0,
        "paid_provider_calls": 0,
        "synthetic_tasks": 0,
    }


def test_replay_rebuild_is_byte_stable_and_v1_is_unchanged(replay: dict) -> None:
    assert build_replay_v2(PATHS) == replay
    v1_path = (
        ROOT
        / "artifacts/season1/task-validation-campaign-v6"
        / f"automated-replay-{PINNED_REPLAY_SEMANTIC_SHA256}.json"
    )
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == PINNED_REPLAY_PHYSICAL_SHA256


def test_rehashed_go_claim_still_fails_deterministic_rebuild(replay: dict) -> None:
    forged = copy.deepcopy(replay)
    forged["status"] = "go"
    forged["decision"]["disposition"] = "go"
    forged["decision"]["full_campaign_contamination_method_requirement_satisfied"] = True
    forged["claim_boundary"]["rank_eligible"] = True
    forged.pop("artifact_sha256")
    forged["artifact_sha256"] = canonical_sha256(forged)
    with pytest.raises(ContaminationReplayV2Error, match="deterministic rebuild"):
        verify_replay_v2(forged, PATHS)


def test_rehashed_web_text_injection_fails_closed(tmp_path: Path) -> None:
    forged = _json(WEB_PATH)
    forged["result_records"][0]["description"] = "unlicensed injected snippet"
    forged.pop("artifact_sha256")
    forged["artifact_sha256"] = canonical_sha256(forged)
    path = tmp_path / "web.json"
    path.write_text(json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n")
    assignment_rows = json.loads(PATHS.v1_inputs.review_assignment.read_text())["assignment_rows"]
    with pytest.raises(ContaminationReplayV2Error, match="web result record integrity"):
        verify_web_snapshot(path, assignment_rows)


def test_raw_benchmark_byte_mutation_fails_closed(tmp_path: Path) -> None:
    document = _json(BENCHMARK_PATH)
    raw_source = CAPTURE_DIRECTORY / document["dataset_receipts"][0]["metadata_receipt"]["raw_file"]
    raw_directory = tmp_path / "raw-benchmark"
    raw_directory.mkdir()
    raw_destination = raw_directory / raw_source.name
    raw_destination.write_bytes(raw_source.read_bytes() + b"\n")
    # Preserve every other captured byte so only the selected mutation can fail.
    for source in (CAPTURE_DIRECTORY / "raw-benchmark").iterdir():
        if source.name != raw_source.name:
            shutil.copyfile(source, raw_directory / source.name)
    snapshot = tmp_path / BENCHMARK_PATH.name
    snapshot.write_bytes(BENCHMARK_PATH.read_bytes())
    with pytest.raises(ContaminationReplayV2Error, match="raw capture digest mismatch"):
        verify_benchmark_snapshot(snapshot)


def test_published_replay_cannot_be_substituted_by_symlink(tmp_path: Path) -> None:
    link = tmp_path / "replay.json"
    link.symlink_to(REPLAY_PATH)
    with pytest.raises(ContaminationReplayV2Error, match="symlinked"):
        verify_pinned_replay_v2(link, PATHS)


def test_small_fixture_exercises_all_four_licensed_corpus_methods() -> None:
    text = (
        "How should carrots be roasted with cumin, lemon, olive oil, and parsley "
        "without making their surfaces soggy?"
    )
    record = {
        "source_reference_sha256": hashlib.sha256(b"fixture-reference").hexdigest(),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_url": "https://example.org/fixture",
        "dataset": "fixture",
        "license_id": "CC0-1.0",
    }
    index = _build_benchmark_scan_index({"records": [record]})
    methods, hits = _scan_benchmark_prompt(
        text,
        index=index,
        completed_at="2026-08-08T00:00:00Z",
        corpus_snapshot_sha256="a" * 64,
    )
    assert [method["method"] for method in methods] == [
        "exact",
        "fuzzy",
        "ngram",
        "semantic",
        "web",
    ]
    assert all(method["performed"] for method in methods[:4])
    assert methods[-1]["performed"] is False
    assert {hit["method"] for hit in hits} == {"exact", "fuzzy", "ngram", "semantic"}
    assert all(hit["similarity_milli"] == 1000 for hit in hits)
    assert methods[0]["implementation_version"] == BENCHMARK_SCAN_IMPLEMENTATION_VERSION
    assert methods[0]["implementation_sha256"] == BENCHMARK_SCAN_IMPLEMENTATION_SHA256
