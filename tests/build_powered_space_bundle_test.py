from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "hf/space/build_powered_space_bundle.py"
    spec = importlib.util.spec_from_file_location("build_powered_space_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_space_bundle_verifies_dataset_and_truncates_answers(tmp_path: Path) -> None:
    module = _module()
    release = {
        "schema_version": "flavourbench-selection-powered-release-v1",
        "status": "final_complete",
        "benchmark": "FlavourBench",
        "track": "test track",
        "analysis": {"models": [], "pairwise_comparisons": []},
        "claim_boundary": {},
    }
    release["artifact_sha256"] = module.hashlib.sha256(module._canonical(release)).hexdigest()
    tables = {
        "models.jsonl": [{"model_id": "model/a"}],
        "tasks.jsonl": [{"task_id": "task-1"}],
        "primary_observations.jsonl": [
            {
                "model_id": "model/a",
                "task_id": "task-1",
                "status": "completed",
                "scoring": {"score": 100.0},
                "generation": {
                    "answer_markdown": "x" * 1700,
                    "actual_model_id": "model/a-dated",
                    "actual_provider": "Provider",
                    "finish_reason": "stop",
                    "latency_ms": 10,
                    "cost_micros": 2,
                },
                "artifact_sha256": "a" * 64,
            }
        ],
        "pairwise_comparisons.jsonl": [],
    }
    records = []
    for name, rows in tables.items():
        payload = b"".join(module._canonical(row) + b"\n" for row in rows)
        (tmp_path / name).write_bytes(payload)
        records.append(
            {
                "name": name,
                "rows": len(rows),
                "sha256": module.hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    manifest = {
        "schema_version": "flavourbench-hf-powered-dataset-manifest-v1",
        "release_artifact_sha256": release["artifact_sha256"],
        "files": records,
    }
    manifest["artifact_sha256"] = module.hashlib.sha256(module._canonical(manifest)).hexdigest()
    (tmp_path / "DATA_MANIFEST.json").write_text(json.dumps(manifest))
    bundle = module.build_bundle(release=release, dataset_directory=tmp_path)
    observation = bundle["primary_observations"][0]
    assert len(observation["answer_excerpt"]) == 1600
    assert observation["answer_truncated"] is True
    assert bundle["release_artifact_sha256"] == release["artifact_sha256"]
    assert module._semantic_valid(bundle)
