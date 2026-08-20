from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module():
    _load_module("build_powered_dataset", "hf/dataset/build_powered_dataset.py")
    return _load_module("build_joint_powered_dataset", "hf/dataset/build_joint_powered_dataset.py")


def test_dataset_export_uses_first_completed_overlay_response(tmp_path: Path) -> None:
    base = _load_module("build_powered_dataset", "hf/dataset/build_powered_dataset.py")
    plan = {"roster": {"models": [{"model_id": "model/a", "slot_id": "slot-a"}]}}

    def write(directory: Path, *, status: str, marker: str) -> dict[str, object]:
        document: dict[str, object] = {
            "schema_version": "flavourbench-powered-response-v1",
            "status": status,
            "model_id": "model/a",
            "task_id": "task-1",
            "cell_id": "c" * 64,
            "marker": marker,
        }
        payload = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
        document["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
        target = directory / "responses" / "primary" / "slot-a"
        target.mkdir(parents=True)
        path = target / f"response-{'c' * 64}-{document['artifact_sha256']}.json"
        path.write_text(json.dumps(document))
        return document

    original = tmp_path / "original"
    repair_1 = tmp_path / "repair-1"
    repair_2 = tmp_path / "repair-2"
    write(original, status="failed", marker="preserved-failure")
    selected = write(repair_1, status="completed", marker="first-normal-completion")
    write(repair_2, status="completed", marker="later-normal-completion")
    rows = base._response_documents(
        panel="primary",
        final_plan=plan,
        task_ids=["task-1"],
        source_directories={"model/a": (original, repair_1, repair_2)},
    )
    assert rows == [selected]


def test_joint_dataset_tables_and_manifest_preserve_both_panels() -> None:
    module = _module()
    model_ids = ("model/a", "model/b")
    release = {
        "artifact_sha256": "a" * 64,
        "analysis": {
            "models": [
                {
                    "model_id": model_id,
                    "point_estimate_rank": index + 1,
                    "flavourbench_score": 75.0 - index,
                }
                for index, model_id in enumerate(model_ids)
            ],
            "panel_replication": {
                "models": [
                    {
                        "model_id": model_id,
                        "panel_1_score": 75.0 - index,
                        "panel_2_score": 74.0 - index,
                    }
                    for index, model_id in enumerate(model_ids)
                ]
            },
            "pairwise_comparisons": [
                {
                    "left_model_id": "model/a",
                    "right_model_id": "model/b",
                    "holm_significant": True,
                }
            ],
        },
    }
    joint_plan = {
        "artifact_sha256": "b" * 64,
        "design": {
            "panel_count": 2,
            "scheduled_primary_tasks_per_model": 2,
            "scheduled_repeat_tasks_per_model": 1,
            "shared_anchor_clusters": 1,
            "unique_anchor_clusters": 11,
        },
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
    panel_1_plan = {
        "artifact_sha256": "c" * 64,
        "roster": {"models": [dict(row) for row in joint_plan["roster"]["models"]]},
    }
    panel_2_routes = [dict(row) for row in joint_plan["roster"]["models"]]
    panel_2_routes[0] = {**panel_2_routes[0], "provider_tag": "provider-panel-2"}
    panel_2_plan = {
        "artifact_sha256": "d" * 64,
        "roster": {"models": panel_2_routes},
    }
    panel_1_taskset = {
        "artifact_sha256": "e" * 64,
        "tasks": [{"task_id": "panel-1-task"}],
    }
    panel_2_taskset = {
        "artifact_sha256": "f" * 64,
        "tasks": [{"task_id": "panel-2-task"}],
    }
    panel_1_repeat = {"artifact_sha256": "1" * 64}
    panel_2_repeat = {"artifact_sha256": "2" * 64}
    primary = [
        {"model_id": model_id, "task_id": task_id}
        for model_id in model_ids
        for task_id in ("panel-1-task", "panel-2-task")
    ]
    repeat = [{"model_id": model_id, "task_id": "repeat"} for model_id in model_ids]
    files = module._expected_files(
        release=release,
        joint_plan=joint_plan,
        panel_1_plan=panel_1_plan,
        panel_2_plan=panel_2_plan,
        panel_1_taskset=panel_1_taskset,
        panel_2_taskset=panel_2_taskset,
        panel_1_repeat=panel_1_repeat,
        panel_2_repeat=panel_2_repeat,
        primary_documents=primary,
        repeat_documents=repeat,
        attempt_documents=[{"event_sha256": "3" * 64}],
    )

    tasks = [json.loads(line) for line in files["tasks.jsonl"].splitlines()]
    assert tasks == [
        {"panel_index": 1, "task_id": "panel-1-task"},
        {"panel_index": 2, "task_id": "panel-2-task"},
    ]
    manifest = json.loads(files["DATA_MANIFEST.json"])
    assert manifest["schema_version"] == (
        "flavourbench-hf-powered-dataset-manifest-v4-two-panel-routes"
    )
    assert manifest["unique_anchor_clusters"] == 11
    assert manifest["shared_anchor_clusters"] == 1
    assert manifest["scheduled_primary_tasks_per_model"] == 2
    model_rows = [json.loads(line) for line in files["models.jsonl"].splitlines()]
    assert model_rows[0]["panel_replication"]["panel_1_score"] == 75.0
    assert model_rows[0]["panel_1_route"]["provider_tag"] == "provider"
    assert model_rows[0]["panel_2_route"]["provider_tag"] == "provider-panel-2"
    assert manifest["panel_plan_artifact_sha256s"] == ["c" * 64, "d" * 64]
    assert {row["name"]: row["rows"] for row in manifest["files"]} == {
        "models.jsonl": 2,
        "tasks.jsonl": 2,
        "primary_observations.jsonl": 4,
        "repeat_observations.jsonl": 2,
        "provider_attempt_events.jsonl": 1,
        "leaderboard.jsonl": 2,
        "pairwise_comparisons.jsonl": 1,
    }


def test_joint_dataset_requires_the_release_response_commitment() -> None:
    module = _module()
    documents = [
        {"artifact_sha256": "a" * 64, "generation": {"cost_micros": 12}},
        {"artifact_sha256": "b" * 64, "generation": None},
    ]
    release = {
        "inputs": {
            "panel_1_primary": {
                "count": 2,
                "artifact_set_sha256": module.hashlib.sha256(
                    module._canonical(["a" * 64, "b" * 64])
                ).hexdigest(),
                "spend_micros": 12,
            }
        }
    }
    module._require_response_commitment(release, "panel_1_primary", documents)
    release["inputs"]["panel_1_primary"]["spend_micros"] = 13
    try:
        module._require_response_commitment(release, "panel_1_primary", documents)
    except module.PoweredDatasetBuildError:
        pass
    else:
        raise AssertionError("response commitment drift was accepted")
