from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from flavourbench.epicure_selection_common_core_analysis_v1 import (
    _source_items,
    load_complete_common_core,
)
from flavourbench.epicure_selection_complete_core_plan_v84 import (
    selected_task_ids,
    verify_plan,
)
from flavourbench.epicure_selection_complete_core_sources_v1 import source_graph
from flavourbench.epicure_selection_powered_analysis import _sha256_file, _verify_semantic
from flavourbench.selection_response_parser_v3 import score_answer_v3

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "data-complete-core"


class CompleteCoreDatasetBuildError(RuntimeError):
    """The public complete-common-core dataset failed a release invariant."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompleteCoreDatasetBuildError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompleteCoreDatasetBuildError(f"input is not a JSON object: {path}")
    return value


def _semantic_valid(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == hashlib.sha256(_canonical(payload)).hexdigest())


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(dict(row)) + b"\n" for row in rows)


def _task_order(plan: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    panel_1, panel_2 = selected_task_ids(plan)
    if len(panel_1) != 267 or len(panel_2) != 267:
        raise CompleteCoreDatasetBuildError("complete-core panel task count differs")
    return panel_1, panel_2


def _response_rows(
    *,
    repository: Path,
    release_panel: str,
    data: Any,
    taskset: Mapping[str, Any],
    model_sources: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected_artifacts = set(data.response_artifact_sha256s)
    tasks = {str(row["task_id"]): row for row in taskset["tasks"]}
    selected: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    observed_artifacts: set[str] = set()

    for model_id in data.model_ids:
        for directory, source_plan in _source_items(model_sources[model_id]):
            source_models = {str(row["model_id"]): row for row in source_plan["roster"]["models"]}
            source_model = source_models.get(model_id)
            if source_model is None:
                raise CompleteCoreDatasetBuildError(
                    f"{model_id} is absent from a response source plan"
                )
            response_directory = directory / "responses" / "primary" / str(source_model["slot_id"])
            for path in sorted(response_directory.glob("response-*.json")):
                document = _load(path)
                artifact = str(document.get("artifact_sha256") or "")
                if artifact not in selected_artifacts:
                    continue
                if not _verify_semantic(document):
                    raise CompleteCoreDatasetBuildError(f"response semantic hash failed: {path}")
                key = (str(document["model_id"]), str(document["task_id"]))
                prior = selected.get(key)
                if prior is not None and prior[1] != document:
                    raise CompleteCoreDatasetBuildError(
                        f"selected response cell has conflicting source bytes: {key}"
                    )
                selected[key] = (path, document)
                observed_artifacts.add(artifact)

    if observed_artifacts != selected_artifacts:
        raise CompleteCoreDatasetBuildError("selected response artifact set is incomplete")

    rows: list[dict[str, Any]] = []
    for model_index, model_id in enumerate(data.model_ids):
        for task_index, task_id in enumerate(data.task_ids):
            path, document = selected[(model_id, task_id)]
            generation = document.get("generation") or {}
            scoring = score_answer_v3(tasks[task_id], str(generation["answer_markdown"]))
            if not scoring["parseable"] or document["status"] != "completed":
                raise CompleteCoreDatasetBuildError("selected response is not release-valid")
            if abs(float(scoring["score"]) - float(data.scores[model_index, task_index])) > 1e-12:
                raise CompleteCoreDatasetBuildError("selected response score differs from release")
            if str(scoring["observed_selection"]) != str(data.selections[model_index][task_index]):
                raise CompleteCoreDatasetBuildError(
                    "selected response selection differs from release"
                )
            rows.append(
                {
                    "release_panel": release_panel,
                    "source_path": str(path.resolve().relative_to(repository)),
                    "response_artifact_sha256": document["artifact_sha256"],
                    "release_scoring": scoring,
                    "response": document,
                }
            )
    return rows


def build_files(
    *,
    repository: Path,
    release_path: Path,
    plan_path: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    repository = repository.resolve()
    release = _load(release_path)
    plan = _load(plan_path)
    if not _semantic_valid(release) or release.get("status") != "final_complete_common_core":
        raise CompleteCoreDatasetBuildError("release is not the final complete common core")
    if not verify_plan(plan):
        raise CompleteCoreDatasetBuildError("complete-core analysis plan failed verification")
    if release["inputs"]["complete_core_plan"] != {
        "semantic_sha256": plan["artifact_sha256"],
        "physical_sha256": _sha256_file(plan_path),
    }:
        raise CompleteCoreDatasetBuildError("release and complete-core plan differ")

    graph = source_graph(repository)
    panel_1_ids, panel_2_ids = _task_order(plan)
    common = {
        "panel": "primary",
        "analysis_score_function": score_answer_v3,
        "allowed_source_roster_differences": {
            "deepseek/deepseek-v4-pro-0813": frozenset({"endpoint_sha256"})
        },
    }
    panel_1 = load_complete_common_core(
        plan=graph.panel_1_plan,
        taskset=graph.panel_1_taskset,
        repeat_panel=graph.panel_1_repeat,
        task_ids=panel_1_ids,
        model_sources=graph.panel_1_model_sources,
        **common,
    )
    panel_2 = load_complete_common_core(
        plan=graph.panel_2_plan,
        taskset=graph.panel_2_taskset,
        repeat_panel=graph.panel_2_repeat,
        task_ids=panel_2_ids,
        model_sources=graph.panel_2_model_sources,
        **common,
    )
    expected_inputs = release["inputs"]
    for label, panel in (("panel_1_responses", panel_1), ("panel_2_responses", panel_2)):
        observed = {
            "count": len(panel.response_artifact_sha256s),
            "artifact_set_sha256": hashlib.sha256(
                _canonical(list(panel.response_artifact_sha256s))
            ).hexdigest(),
        }
        if observed != expected_inputs[label]:
            raise CompleteCoreDatasetBuildError(f"{label} differs from release")

    task_by_panel = (
        ("panel_1", graph.panel_1_taskset, panel_1_ids),
        ("panel_2", graph.panel_2_taskset, panel_2_ids),
    )
    task_rows: list[dict[str, Any]] = []
    for panel_label, taskset, task_ids in task_by_panel:
        task_index = {str(row["task_id"]): row for row in taskset["tasks"]}
        task_rows.extend(
            {"release_panel": panel_label, **dict(task_index[task_id])} for task_id in task_ids
        )

    response_rows = [
        *_response_rows(
            repository=repository,
            release_panel="panel_1",
            data=panel_1,
            taskset=graph.panel_1_taskset,
            model_sources=graph.panel_1_model_sources,
        ),
        *_response_rows(
            repository=repository,
            release_panel="panel_2",
            data=panel_2,
            taskset=graph.panel_2_taskset,
            model_sources=graph.panel_2_model_sources,
        ),
    ]
    if len(response_rows) != 14_418:
        raise CompleteCoreDatasetBuildError("primary response cardinality differs")

    analysis_by_model = {str(row["model_id"]): dict(row) for row in release["analysis"]["models"]}
    replication_by_model = {
        str(row["model_id"]): dict(row)
        for row in release["analysis"]["panel_replication"]["models"]
    }
    routes = {str(row["model_id"]): dict(row) for row in plan["roster"]["models"]}
    if set(analysis_by_model) != set(routes) or set(replication_by_model) != set(routes):
        raise CompleteCoreDatasetBuildError("release and route roster differ")
    model_rows = [
        {
            **analysis_by_model[model_id],
            "route": routes[model_id],
            "execution_backend": routes[model_id]["execution_backend"],
            "provider_name": routes[model_id]["provider_name"],
            "provider_tag": routes[model_id]["provider_tag"],
            "panel_replication": replication_by_model[model_id],
        }
        for model_id in panel_1.model_ids
    ]
    leaderboard_rows = sorted(
        model_rows,
        key=lambda row: (int(row["point_estimate_rank"]), str(row["model_id"])),
    )
    pairwise_rows = [dict(row) for row in release["analysis"]["pairwise_comparisons"]]

    files = {
        "release.json": release_path.read_bytes(),
        "analysis_plan.json": plan_path.read_bytes(),
        "models.jsonl": _jsonl(model_rows),
        "tasks.jsonl": _jsonl(task_rows),
        "primary_observations.jsonl": _jsonl(response_rows),
        "leaderboard.jsonl": _jsonl(leaderboard_rows),
        "pairwise_comparisons.jsonl": _jsonl(pairwise_rows),
    }
    manifest: dict[str, Any] = {
        "schema_version": "flavourbench-hf-complete-core-dataset-manifest-v1",
        "status": "final_complete_common_core",
        "release_artifact_sha256": release["artifact_sha256"],
        "release_physical_sha256": _sha256_file(release_path),
        "complete_core_plan_artifact_sha256": plan["artifact_sha256"],
        "complete_core_plan_physical_sha256": _sha256_file(plan_path),
        "model_count": 27,
        "task_count": 534,
        "primary_observation_count": 14_418,
        "pairwise_comparison_count": 351,
        "independence_unit": "anchor_ingredient",
        "unique_anchor_clusters": 534,
        "files": [
            {
                "name": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                **({"rows": payload.count(b"\n")} if name.endswith(".jsonl") else {}),
            }
            for name, payload in sorted(files.items())
        ],
    }
    manifest["artifact_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    return files, manifest


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final FlavourBench Hub dataset")
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    files, manifest = build_files(
        repository=args.repository,
        release_path=args.release,
        plan_path=args.plan,
    )
    expected = {**files, "DATA_MANIFEST.json": _json_bytes(manifest)}
    if args.check:
        observed_names = {
            path.name for path in args.output.iterdir() if path.is_file() and not path.is_symlink()
        }
        if observed_names != set(expected):
            raise CompleteCoreDatasetBuildError("dataset output inventory differs")
        for name, payload in expected.items():
            path = args.output / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise CompleteCoreDatasetBuildError(f"dataset output differs: {name}")
        print(f"OK: complete-core dataset {manifest['artifact_sha256']}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        _write_atomic(args.output / name, payload)
    print(
        f"Wrote {len(expected)} files, {manifest['primary_observation_count']} responses, "
        f"manifest {manifest['artifact_sha256']}"
    )


if __name__ == "__main__":
    main()
