from __future__ import annotations

import csv
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path

import pytest

from paper.reproduce_powered_release import (
    COVERAGE_REPAIR_MODEL_IDS,
    PoweredReleaseError,
    verify_release,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic(document: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _fixture(tmp_path: Path, model_ids: list[str] | None = None) -> Path:
    model_ids = model_ids or [f"model-{index:02d}" for index in range(20)]
    models = [
        {
            "model_id": model_id,
            "model_name": model_id,
            "point_estimate_rank": index + 1,
            "flavourbench_score": 100 - index,
            "availability": {"scheduled": 640},
        }
        for index, model_id in enumerate(model_ids)
    ]
    pairs = [
        {"left_model_id": left, "right_model_id": right}
        for left, right in combinations(model_ids, 2)
    ]
    repeats = [{"model_id": model_id, "tasks": 64} for model_id in model_ids]

    leaderboard_buffer = io.StringIO(newline="")
    leaderboard_writer = csv.DictWriter(
        leaderboard_buffer, fieldnames=("model_id",), lineterminator="\n"
    )
    leaderboard_writer.writeheader()
    leaderboard_writer.writerows({"model_id": model_id} for model_id in model_ids)
    leaderboard = leaderboard_buffer.getvalue().encode()
    leaderboard_path = tmp_path / "leaderboard.csv"
    leaderboard_path.write_bytes(leaderboard)

    pairwise_buffer = io.StringIO(newline="")
    pairwise_writer = csv.DictWriter(
        pairwise_buffer,
        fieldnames=("left_model_id", "right_model_id"),
        lineterminator="\n",
    )
    pairwise_writer.writeheader()
    pairwise_writer.writerows(pairs)
    pairwise = pairwise_buffer.getvalue().encode()
    pairwise_path = tmp_path / "pairwise.csv"
    pairwise_path.write_bytes(pairwise)

    release: dict[str, object] = {
        "schema_version": "flavourbench-selection-powered-release-v1",
        "status": "final_complete",
        "inputs": {
            "primary_responses": {"count": 12_800},
            "repeat_responses": {"count": 1_280},
        },
        "tables": {
            "leaderboard": {
                "filename": leaderboard_path.name,
                "sha256": _sha256(leaderboard),
            },
            "pairwise": {"filename": pairwise_path.name, "sha256": _sha256(pairwise)},
        },
        "analysis": {
            "status": "final_complete",
            "models": models,
            "pairwise_comparisons": pairs,
            "repeatability": repeats,
            "definitive_top_model_id": None,
        },
    }
    release["artifact_sha256"] = _semantic(release)
    release_path = tmp_path / f"flavourbench-powered-release-{release['artifact_sha256']}.json"
    release_path.write_text(json.dumps(release, sort_keys=True) + "\n", encoding="utf-8")
    return release_path


def _fixture_v2(tmp_path: Path, model_ids: list[str] | None = None) -> Path:
    predecessor = _fixture(tmp_path, model_ids)
    release = json.loads(predecessor.read_text(encoding="utf-8"))
    release.pop("artifact_sha256")
    release["schema_version"] = "flavourbench-selection-powered-release-v2-anchor-free"
    analysis = release["analysis"]
    analysis["dnf_classification"] = False
    analysis["failure_handling"] = "excluded_from_quality_score_and_retained_in_coverage"
    for index, model in enumerate(analysis["models"]):
        model.pop("availability")
        model["score_status"] = "scored"
        model["coverage"] = {
            "scheduled": 640,
            "valid_scored": 640 - index,
        }
    for repeat in analysis["repeatability"]:
        repeat["scheduled"] = repeat.pop("tasks")
    release["artifact_sha256"] = _semantic(release)
    destination = tmp_path / f"flavourbench-powered-release-{release['artifact_sha256']}.json"
    destination.write_text(json.dumps(release, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _fixture_joint(tmp_path: Path) -> Path:
    model_ids = [
        *COVERAGE_REPAIR_MODEL_IDS,
        "qwen/qwen3.8-2.4t-a95b",
        "openai/gpt-5.6-luna-pro",
        "deepseek/deepseek-v4-flash-0731",
        *[f"model-{index:02d}" for index in range(8)],
    ]
    predecessor = _fixture_v2(tmp_path, model_ids)
    release = json.loads(predecessor.read_text(encoding="utf-8"))
    release.pop("artifact_sha256")
    release["schema_version"] = "flavourbench-selection-powered-joint-release-v1"
    for model in release["analysis"]["models"]:
        model["coverage"]["scheduled"] = 1_280
    for repeat in release["analysis"]["repeatability"]:
        repeat["scheduled"] = 128
    release["analysis"].update(
        {
            "design": {"unique_anchor_clusters": 1_178, "shared_anchor_clusters": 102},
            "inference": {
                "independence_unit": "anchor_ingredient",
                "independent_cluster_count": 1_178,
                "shared_anchor_tasks_move_together": True,
            },
            "panel_replication": {"status": "descriptive"},
        }
    )
    release["inputs"] = {
        "panel_1_primary": {"count": len(model_ids) * 640},
        "panel_2_primary": {"count": len(model_ids) * 640},
        "panel_1_repeat": {"count": len(model_ids) * 64},
        "panel_2_repeat": {"count": len(model_ids) * 64},
        "response_lineage": {
            "panel_1_base_plan_sha256": "1" * 64,
            "panel_1_qwen_replacement_plan_sha256": "2" * 64,
            "panel_1_superseded_qwen_responses_used": False,
            "panel_1_fable_replacement_plan_sha256": None,
            "panel_1_superseded_fable_replacement_plan_sha256": "3" * 64,
            "panel_1_coverage_repair_plan_sha256": "4" * 64,
            "panel_1_coverage_repair_model_ids": COVERAGE_REPAIR_MODEL_IDS,
            "panel_1_superseded_coverage_route_responses_used": False,
            "panel_1_deepseek_repair_plan_sha256": "8" * 64,
            "panel_1_deepseek_repair_model_ids": ["deepseek/deepseek-v4-pro-0813"],
            "panel_1_deepseek_repair_provider_tag": "gmicloud/fp8",
            "panel_1_superseded_deepseek_route_responses_used": False,
            "panel_2_plan_sha256": "9" * 64,
            "panel_2_base_plan_sha256": "5" * 64,
            "panel_2_replacement_plan_sha256": "6" * 64,
            "panel_2_replacement_model_ids": [
                "openai/gpt-5.6-luna-pro",
                "deepseek/deepseek-v4-flash-0731",
            ],
            "panel_2_superseded_route_responses_used": False,
            "panel_2_reuses_panel_1_responses": False,
            "panel_2_coverage_repair_plan_sha256": "7" * 64,
            "panel_2_coverage_repair_model_ids": COVERAGE_REPAIR_MODEL_IDS,
            "panel_2_superseded_coverage_route_responses_used": False,
            "panel_2_deepseek_repair_plan_sha256": "9" * 64,
            "panel_2_deepseek_repair_model_ids": ["deepseek/deepseek-v4-pro-0813"],
            "panel_2_deepseek_repair_provider_tag": "gmicloud/fp8",
            "panel_2_superseded_deepseek_route_responses_used": False,
            "deepseek_quality_scores_inspected_before_source_freeze": False,
        },
    }
    release["artifact_sha256"] = _semantic(release)
    destination = tmp_path / f"flavourbench-joint-release-{release['artifact_sha256']}.json"
    destination.write_text(json.dumps(release, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def test_verifies_compact_release(tmp_path: Path) -> None:
    summary = verify_release(_fixture(tmp_path))
    assert summary["status"] == "verified"
    assert summary["models"] == 20
    assert summary["pairwise_comparisons"] == 190


def test_verifies_anchor_free_success_only_release(tmp_path: Path) -> None:
    summary = verify_release(_fixture_v2(tmp_path))
    assert summary["status"] == "verified"
    assert summary["models"] == 20
    assert summary["leader_model_id"] == "model-00"


def test_verifies_joint_complete_coverage_repair_lineage(tmp_path: Path) -> None:
    summary = verify_release(_fixture_joint(tmp_path))
    assert summary["status"] == "verified"
    assert summary["independent_anchor_clusters"] == 1_178


def test_rejects_table_drift(tmp_path: Path) -> None:
    release = _fixture(tmp_path)
    (tmp_path / "leaderboard.csv").write_text("model_id\nmodel-00\n", encoding="utf-8")
    with pytest.raises(PoweredReleaseError, match="leaderboard table hash failed"):
        verify_release(release)
