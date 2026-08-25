from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "paper/generated/complete-core"
FIGURE_DIRECTORY = ROOT / "paper/figures/complete-core"
ANALYSIS = DIRECTORY / "complete-core-external-substitution-validation.json"
PROTOCOL = ROOT / "paper/protocols/external_substitution_validation_v1.json"
BUILDER = ROOT / "paper/build_external_substitution_validation_assets.py"
CHECKPOINT_LOADER = ROOT / "paper/build_public_scorer_sensitivity_assets.py"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def test_external_substitution_validation_is_content_bound() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    payload = dict(analysis)
    recorded = payload.pop("artifact_sha256")
    assert recorded == hashlib.sha256(_canonical(payload)).hexdigest()
    assert analysis["schema_version"] == "flavourbench-external-substitution-validation-v1"
    assert analysis["status"] == "post_collection_label_independent_convergent_validation"
    assert (
        analysis["inputs"]["protocol_physical_sha256"]
        == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    )
    assert (
        analysis["inputs"]["builder_physical_sha256"]
        == hashlib.sha256(BUILDER.read_bytes()).hexdigest()
    )
    assert (
        analysis["inputs"]["checkpoint_loader_physical_sha256"]
        == hashlib.sha256(CHECKPOINT_LOADER.read_bytes()).hexdigest()
    )

    for record in analysis["companion_files"]:
        path = DIRECTORY / record["name"]
        data = path.read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
    for record in analysis["figure_files"]:
        path = FIGURE_DIRECTORY / record["name"]
        data = path.read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]


def test_external_substitution_mapping_and_raw_data_boundary() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    assert analysis["coverage"] == {
        "raw_train_records": 49_044,
        "raw_test_records": 10_747,
        "source_mapped_test_events": 5_674,
        "target_mapped_test_events": 5_408,
        "mapped_test_events": 3_282,
        "mapped_test_event_rate": 3_282 / 10_747,
        "unique_mapped_test_pairs": 1_469,
        "unique_source_ingredients": 357,
        "unique_target_ingredients": 429,
        "novel_unique_pairs": 594,
        "novel_unique_pair_rate": 594 / 1_469,
        "exact_manual_aliases": 0,
    }
    assert analysis["raw_external_rows_redistributed"] is False
    assert analysis["design"]["mapping"] == "exact canonical token equality"
    assert analysis["design"]["manual_aliases"] == 0
    assert analysis["design"]["deduplication_unit"] == "directed source-target pair"

    summary = list(
        csv.DictReader(
            io.StringIO(
                (DIRECTORY / "complete-core-external-substitution-validation.csv").read_text(
                    encoding="utf-8"
                )
            )
        )
    )
    assert len(summary) == 3
    assert "source" not in summary[0]
    assert "target" not in summary[0]


def test_external_substitution_primary_results_clear_the_fixed_null() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    assert analysis["all_three_primary_tests_reject_after_holm"] is True
    assert analysis["minimum_primary_equal_source_percentile"] > 0.75
    assert analysis["minimum_novel_pair_equal_source_percentile"] > 0.70
    assert [row["checkpoint"] for row in analysis["checkpoint_results"]] == [
        "cooc",
        "core",
        "chem",
    ]
    for row in analysis["checkpoint_results"]:
        primary = row["primary_all_unique_pairs"]
        novel = row["sensitivity_novel_unique_pairs"]
        retrieval = row["full_vocabulary_retrieval"]
        assert primary["unique_sources"] == 357
        assert primary["unique_pairs"] == 1_469
        assert primary["percentile_95_interval"][0] > 0.5
        assert primary["holm_adjusted_p"] < 0.001
        assert primary["holm_reject_at_familywise_0_05"] is True
        assert novel["unique_sources"] == 268
        assert novel["unique_pairs"] == 594
        assert novel["percentile_95_interval"][0] > 0.5
        assert retrieval["mean_reciprocal_rank"] > 10 * retrieval["analytic_chance_mrr"]
        assert retrieval["hit_at_10"] > 20 * retrieval["analytic_chance_hit_at_10"]

    boundary = analysis["claim_boundary"]
    assert "not corpus-independent validation" in boundary
    assert "does not validate the unrecovered primary runtime" in boundary
    assert "cooked-food outcomes" in boundary
