from __future__ import annotations

import csv
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path

import pytest

from paper.reproduce_powered_release import PoweredReleaseError, verify_release


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


def _fixture(tmp_path: Path) -> Path:
    model_ids = [f"model-{index:02d}" for index in range(20)]
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


def test_verifies_compact_release(tmp_path: Path) -> None:
    summary = verify_release(_fixture(tmp_path))
    assert summary["status"] == "verified"
    assert summary["models"] == 20
    assert summary["pairwise_comparisons"] == 190


def test_rejects_table_drift(tmp_path: Path) -> None:
    release = _fixture(tmp_path)
    (tmp_path / "leaderboard.csv").write_text("model_id\nmodel-00\n", encoding="utf-8")
    with pytest.raises(PoweredReleaseError, match="leaderboard table hash failed"):
        verify_release(release)
