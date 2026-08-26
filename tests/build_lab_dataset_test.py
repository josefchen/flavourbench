from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "hf/dataset/build_lab_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_lab_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def _rows(payload: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in payload.splitlines()]


def test_lab_dataset_is_reproducible_and_anchor_disjoint() -> None:
    files, manifest = BUILDER.build_files(
        panel_1_path=BUILDER.DEFAULT_PANEL_1,
        panel_2_path=BUILDER.DEFAULT_PANEL_2,
        official_plan_path=BUILDER.DEFAULT_OFFICIAL_PLAN,
    )
    assert (
        manifest["artifact_sha256"]
        == "36b660e75cb3e209526ab6549f7d3358958f048627e7cf1ebd0a25c23294aba0"
    )
    assert manifest["counts"] == {
        "train_tasks": 270,
        "validation_tasks": 72,
        "evaluation_tasks": 84,
        "sft_train": 270,
        "sft_validation": 72,
        "sft_format_control_train": 270,
        "sft_format_control_validation": 72,
        "dpo_train": 1080,
        "dpo_validation": 288,
        "grpo_train": 270,
        "grpo_validation": 72,
        "supplemental_cultural_composition": 283,
    }
    train = _rows(files["train_tasks.jsonl"])
    validation = _rows(files["validation_tasks.jsonl"])
    evaluation = _rows(files["evaluation_tasks.jsonl"])
    train_anchors = {str(row["anchor_ingredient"]) for row in train}
    validation_anchors = {str(row["anchor_ingredient"]) for row in validation}
    evaluation_anchors = {str(row["anchor_ingredient"]) for row in evaluation}
    assert not train_anchors & validation_anchors
    assert not train_anchors & evaluation_anchors
    assert not validation_anchors & evaluation_anchors
    assert all(
        row["official_leaderboard_eligible"] is False for row in train + validation + evaluation
    )
    assert all(
        row["official_test_anchor_overlap"] is False for row in train + validation + evaluation
    )
    assert all(row["lab_split"] == "evaluation" for row in evaluation)
    dpo_rows = _rows(files["dpo_train.jsonl"])
    assert min(float(row["reward_margin"]) for row in dpo_rows) >= 0.05
    sft_rows = _rows(files["sft_train.jsonl"])
    assert all(int(row["optimal_margin_bps"]) > 0 for row in sft_rows)
    for file_row in manifest["files"]:
        payload = files[file_row["name"]]
        assert len(payload) == file_row["bytes"]
        assert payload.count(b"\n") == file_row["rows"]
        assert hashlib.sha256(payload).hexdigest() == file_row["sha256"]


def test_committed_lab_dataset_matches_the_builder() -> None:
    files, manifest = BUILDER.build_files(
        panel_1_path=BUILDER.DEFAULT_PANEL_1,
        panel_2_path=BUILDER.DEFAULT_PANEL_2,
        official_plan_path=BUILDER.DEFAULT_OFFICIAL_PLAN,
    )
    directory = REPOSITORY / "hf/dataset/data-lab"
    for name, payload in files.items():
        assert (directory / name).read_bytes() == payload
    committed_manifest = json.loads((directory / "DATA_MANIFEST.json").read_text())
    assert committed_manifest == manifest
