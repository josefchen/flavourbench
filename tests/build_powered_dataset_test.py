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
        provider_attempt_documents=[{"event_sha256": "e" * 64}],
    )
    assert set(files) == {
        "models.jsonl",
        "tasks.jsonl",
        "primary_observations.jsonl",
        "repeat_observations.jsonl",
        "provider_attempt_events.jsonl",
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
        "provider_attempt_events.jsonl": 1,
        "leaderboard.jsonl": 4,
        "pairwise_comparisons.jsonl": 1,
    }


def test_v42_dataset_labels_each_response_lineage() -> None:
    module = _module()
    groups = {
        "base_model_ids": [f"base-{index:02d}" for index in range(16)],
        "cohere_model_ids": ["cohere-a", "cohere-r"],
        "frontier_model_ids": [f"frontier-{index:02d}" for index in range(6)],
        "deepseek_model_ids": ["deepseek-v4-pro"],
        "successor_model_ids": ["fable-5"],
    }
    model_ids = [model_id for values in groups.values() for model_id in values]
    release = {
        "inputs": {
            "model_response_sources": {
                "schema_version": "flavourbench-selection-composite-response-sources-v7",
                **groups,
            }
        },
        "analysis": {
            "models": [
                {
                    "model_id": model_id,
                    "point_estimate_rank": index + 1,
                    "flavourbench_score": 80.0 - index,
                }
                for index, model_id in enumerate(model_ids)
            ],
            "pairwise_comparisons": [],
        },
    }
    plan = {
        "roster": {
            "models": [
                {
                    "model_id": model_id,
                    "canonical_model_slug": f"{model_id}-dated",
                    "execution_backend": "openrouter",
                    "provider_tag": "provider",
                    "provider_name": "Provider",
                    "endpoint_execution_sha256": str(index % 10) * 64,
                }
                for index, model_id in enumerate(model_ids)
            ]
        }
    }
    tables = module._tables(
        release=release,
        taskset={"tasks": []},
        repeat_panel={"tasks": []},
        plan=plan,
        primary_documents=[],
        repeat_documents=[],
        provider_attempt_documents=[],
    )
    sources = {row["model_id"]: row["response_source"] for row in tables["models"]}
    assert {sources[model_id] for model_id in groups["base_model_ids"]} == {"powered-v31-base"}
    assert {sources[model_id] for model_id in groups["cohere_model_ids"]} == {
        "powered-v35-clean-cohere"
    }
    assert {sources[model_id] for model_id in groups["frontier_model_ids"]} == {
        "powered-v38-frontier-refresh"
    }
    assert sources["deepseek-v4-pro"] == "powered-v39-deepseek-repair"
    assert sources["fable-5"] == "powered-v42-fable-complete-block"


def test_provider_attempt_export_rejects_unbound_or_missing_events(tmp_path: Path) -> None:
    module = _module()
    run = tmp_path / "run"
    journal = run / "attempts/provider-attempts.jsonl"
    journal.parent.mkdir(parents=True)
    event = {
        "schema_version": "flavourbench-powered-attempt-event-v1",
        "plan_sha256": "a" * 64,
        "recorded_at": "2026-08-14T00:00:00Z",
        "event": {"arm_id": "arm-1", "event_type": "response_received"},
    }
    event["event_sha256"] = module.hashlib.sha256(module._canonical(event)).hexdigest()
    journal.write_bytes(module._canonical(event) + b"\n")
    response = {
        "model_id": "model/a",
        "arm_id": "arm-1",
        "plan_sha256": "a" * 64,
        "attempt_event_sha256s": [event["event_sha256"]],
    }
    assert module._provider_attempt_documents(
        response_documents=[response],
        source_directories={"model/a": run},
    ) == [event]

    response["attempt_event_sha256s"] = ["f" * 64]
    try:
        module._provider_attempt_documents(
            response_documents=[response],
            source_directories={"model/a": run},
        )
    except module.PoweredDatasetBuildError as exc:
        assert "lineage is incomplete" in str(exc)
    else:
        raise AssertionError("missing provider-attempt evidence was accepted")
