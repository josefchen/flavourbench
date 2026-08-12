from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "hf/dataset/build_powered_dataset.py"
    spec = importlib.util.spec_from_file_location("build_powered_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_powered_dataset_tables_and_manifest_are_deterministic() -> None:
    module = _module()
    model_ids = ("model/a", "model/b", "model/c", "model/d")
    release = {
        "artifact_sha256": "a" * 64,
        "inputs": {
            "model_response_sources": {
                "base_models": ["model/a"],
                "deepseek_model_id": "model/b",
                "cohere_model_ids": ["model/c", "model/d"],
            }
        },
        "analysis": {
            "models": [
                {
                    "model_id": model_id,
                    "point_estimate_rank": index + 1,
                    "flavourbench_score": 90.0 - index,
                }
                for index, model_id in enumerate(model_ids)
            ],
            "pairwise_comparisons": [
                {
                    "left_model_id": "model/a",
                    "right_model_id": "model/b",
                    "holm_significant": True,
                }
            ],
        },
    }
    plan = {
        "artifact_sha256": "b" * 64,
        "roster": {
            "models": [
                {
                    "model_id": model_id,
                    "canonical_model_slug": f"{model_id}-dated",
                    "execution_backend": "openrouter",
                    "provider_tag": "provider",
                    "provider_name": "Provider",
                    "endpoint_execution_sha256": str(index) * 64,
                }
                for index, model_id in enumerate(model_ids, start=1)
            ]
        },
    }
    taskset = {"artifact_sha256": "c" * 64, "tasks": [{"task_id": "task-1"}]}
    repeat_panel = {"artifact_sha256": "d" * 64, "tasks": [{"task_id": "repeat-1"}]}
    primary = [{"model_id": model_id, "task_id": "task-1"} for model_id in model_ids]
    repeat = [{"model_id": model_id, "task_id": "repeat-1"} for model_id in model_ids]
    files = module._expected_files(
        release=release,
        taskset=taskset,
        repeat_panel=repeat_panel,
        plan=plan,
        primary_documents=primary,
        repeat_documents=repeat,
    )
    assert set(files) == {
        "models.jsonl",
        "tasks.jsonl",
        "primary_observations.jsonl",
        "repeat_observations.jsonl",
        "leaderboard.jsonl",
        "pairwise_comparisons.jsonl",
        "DATA_MANIFEST.json",
    }
    manifest = json.loads(files["DATA_MANIFEST.json"])
    payload = dict(manifest)
    recorded = payload.pop("artifact_sha256")
    assert recorded == module.hashlib.sha256(module._canonical(payload)).hexdigest()
    assert {row["name"]: row["rows"] for row in manifest["files"]} == {
        "models.jsonl": 4,
        "tasks.jsonl": 1,
        "primary_observations.jsonl": 4,
        "repeat_observations.jsonl": 4,
        "leaderboard.jsonl": 4,
        "pairwise_comparisons.jsonl": 1,
    }
