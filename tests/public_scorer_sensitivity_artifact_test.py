from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "paper/generated/complete-core"
ANALYSIS = DIRECTORY / "complete-core-public-scorer-sensitivity.json"
BUILDER = ROOT / "paper/build_public_scorer_sensitivity_assets.py"
INGREDIENT_TAGS_COMMIT = "14ddf04aba81a76b75efa6554041f6bff48992c6"
INGREDIENT_TAGS_SHA256 = "8f52e83a072069f436ab7d851ed0251e775da92afc46e0deaa61d49d91014772"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def test_public_scorer_sensitivity_release_is_self_verifying() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    payload = dict(analysis)
    recorded = payload.pop("artifact_sha256")
    assert recorded == hashlib.sha256(_canonical(payload)).hexdigest()
    assert analysis["schema_version"] == "flavourbench-public-epicure-scorer-sensitivity-v1"
    assert analysis["status"] == "post_collection_fixed_task_public_scorer_sensitivity"
    assert analysis["analysis_timing"] == {
        "prespecified_primary_analysis": False,
        "posthoc_sensitivity": True,
        "model_provider_calls_added": 0,
        "model_responses_changed": False,
        "task_prompts_or_candidate_sets_changed": False,
    }
    assert analysis["design"]["models"] == 27
    assert analysis["design"]["tasks"] == 534
    assert analysis["design"]["model_task_cells"] == 14_418
    assert analysis["design"]["public_checkpoint_count"] == 3
    assert analysis["design"]["missing_public_checkpoint_ingredients"] == 0
    assert (
        analysis["inputs"]["builder_physical_sha256"]
        == hashlib.sha256(BUILDER.read_bytes()).hexdigest()
    )
    assert analysis["inputs"]["ingredient_tags_sha256"] == INGREDIENT_TAGS_SHA256
    assert analysis["inputs"]["ingredient_tags_source_commit"] == INGREDIENT_TAGS_COMMIT
    assert analysis["inputs"]["ingredient_tags_source_url"] == (
        "https://raw.githubusercontent.com/KAIKAKU-AI/epicure-mcp/"
        f"{INGREDIENT_TAGS_COMMIT}/data/ingredient_tags.csv"
    )

    for record in analysis["companion_files"]:
        path = DIRECTORY / record["name"]
        data = path.read_bytes()
        assert len(data) == record["bytes"]
        assert data.count(b"\n") == record["newline_count"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        expected_newlines = (
            record["records"] if record["format"] == "jsonl" else record["records"] + 1
        )
        assert record["newline_count"] == expected_newlines


def test_public_score_maps_and_leaderboards_are_complete() -> None:
    score_maps = [
        json.loads(line)
        for line in (DIRECTORY / "complete-core-public-scorer-score-maps.jsonl")
        .read_bytes()
        .splitlines()
    ]
    assert len(score_maps) == 3 * 534
    assert len({(row["checkpoint"], row["task_id"]) for row in score_maps}) == len(score_maps)
    assert Counter(row["checkpoint"] for row in score_maps) == Counter(
        {"cooc": 534, "core": 534, "chem": 534}
    )
    for row in score_maps:
        scores = row["selection_scores_bps"]
        assert len(scores) == 56
        assert max(scores.values()) == 10_000
        assert all(isinstance(value, int) and 0 <= value <= 10_000 for value in scores.values())

    leaderboard = list(
        csv.DictReader(
            io.StringIO(
                (DIRECTORY / "complete-core-public-scorer-leaderboard.csv").read_text(
                    encoding="utf-8"
                )
            )
        )
    )
    assert len(leaderboard) == 3 * 27
    assert len({(row["checkpoint"], row["model_id"]) for row in leaderboard}) == len(leaderboard)
    for checkpoint in ("cooc", "core", "chem"):
        rows = [row for row in leaderboard if row["checkpoint"] == checkpoint]
        assert sorted(int(row["point_rank"]) for row in rows) == list(range(1, 28))
        assert next(row for row in rows if int(row["point_rank"]) == 1)["model_id"] == (
            "x-ai/grok-4.6"
        )
        assert all(math.isfinite(float(row["score"])) for row in rows)


def test_public_checkpoint_sensitivity_claims_match_the_records() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    results = analysis["checkpoint_results"]
    assert [row["checkpoint"] for row in results] == ["cooc", "core", "chem"]
    assert analysis["all_public_checkpoint_point_estimates_preserve_release_leader"] is True
    assert analysis["original_point_leader_model_id"] == "x-ai/grok-4.6"
    assert 0.90 <= analysis["minimum_model_rank_spearman"] <= 1.0
    assert 0.95 <= analysis["maximum_model_rank_spearman"] <= 1.0
    assert 0.86 <= analysis["minimum_pair_order_agreement"] <= 1.0
    assert 0.91 <= analysis["maximum_pair_order_agreement"] <= 1.0
    assert all(row["stratified_anchor_bootstrap"]["replicates"] == 20_000 for row in results)
    assert "conditional on the released 534 prompts" in analysis["claim_boundary"]
    assert (
        "does not validate the scorer against human culinary judgments"
        in analysis["claim_boundary"]
    )
