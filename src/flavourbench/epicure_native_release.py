"""Build the public replay dataset for the Epicure-native leaderboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_native_taskset import parse_final_choice, verify_taskset
from .frontier_manifest import verify_manifest_content_address

SCHEMA_VERSION = "flavourbench-epicure-native-release-v1"
CONDITIONS = ("epicure_off", "epicure_on")


class ReleaseError(RuntimeError):
    """The public replay dataset is incomplete or internally inconsistent."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseError(f"input must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"expected a JSON object: {path}")
    return value


def _artifact_digest(path: Path, value: Mapping[str, Any]) -> str:
    digest = str(value.get("artifact_sha256") or "")
    unhashed = {key: item for key, item in value.items() if key != "artifact_sha256"}
    accepted = {
        _sha256(unhashed),
        hashlib.sha256(
            json.dumps(unhashed, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
    }
    if len(digest) != 64 or digest not in accepted or digest[:12] not in path.name:
        raise ReleaseError(f"artifact content address does not verify: {path}")
    return digest


def _tool_projection(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = response.get("response")
    if not isinstance(result, Mapping):
        return []
    trace = result.get("tool_trace")
    if not isinstance(trace, list):
        return []
    projected: list[dict[str, Any]] = []
    for event in trace:
        if not isinstance(event, Mapping):
            continue
        tool_result = event.get("result")
        projected.append(
            {
                "name": str(event.get("name") or ""),
                "arguments": event.get("arguments")
                if isinstance(event.get("arguments"), Mapping)
                else {},
                "is_error": event.get("is_error") is True,
                "latency_ms": event.get("latency_ms")
                if isinstance(event.get("latency_ms"), int)
                else None,
                "result_sha256": _sha256(tool_result),
            }
        )
    return projected


def build_release(
    *,
    manifest_path: Path,
    taskset_path: Path,
    leaderboard_path: Path,
    source_directory: Path,
    response_directory: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if not verify_manifest_content_address(manifest):
        raise ReleaseError("manifest content address does not verify")
    manifest_sha = str(manifest["content_address"]["digest"])
    taskset = _read_json(taskset_path)
    if not verify_taskset(taskset):
        raise ReleaseError("taskset content address does not verify")
    leaderboard = _read_json(leaderboard_path)
    leaderboard_sha = _artifact_digest(leaderboard_path, leaderboard)
    if (
        leaderboard.get("status") != "complete_automated_leaderboard"
        or leaderboard.get("official_track") is not True
        or leaderboard.get("progress", {}).get("attempted_pairs") != 640
        or leaderboard.get("progress", {}).get("complete_models") != 20
        or leaderboard.get("provenance", {}).get("manifest_sha256") != manifest_sha
        or leaderboard.get("provenance", {}).get("taskset_artifact_sha256")
        != taskset.get("artifact_sha256")
    ):
        raise ReleaseError("leaderboard is not the complete official automated track")

    model_entries: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("models") or []:
        if not isinstance(entry, Mapping):
            raise ReleaseError("manifest contains a malformed model")
        model = entry.get("model")
        slot = entry.get("slot")
        route = entry.get("execution_route")
        endpoint = entry.get("endpoint")
        if not all(isinstance(value, Mapping) for value in (model, slot, route, endpoint)):
            raise ReleaseError("manifest model is missing identity or route")
        model_id = str(model.get("id") or "")
        model_entries[model_id] = {
            "model_id": model_id,
            "display_name": str(model.get("name") or model_id),
            "canonical_model_slug": str(model.get("canonical_slug") or ""),
            "slot_id": str(slot.get("slot_id") or ""),
            "execution_backend": str(route.get("selected_backend") or ""),
            "provider_route": str(endpoint.get("tag") or endpoint.get("provider_name") or ""),
        }
    if len(model_entries) != 20:
        raise ReleaseError("release requires exactly 20 models")

    tasks: dict[str, dict[str, Any]] = {}
    public_tasks: list[dict[str, Any]] = []
    for task in taskset.get("tasks") or []:
        if not isinstance(task, Mapping):
            raise ReleaseError("taskset contains a malformed task")
        task_id = str(task.get("task_id") or "")
        tasks[task_id] = dict(task)
        public_tasks.append(
            {
                "task_id": task_id,
                "family": task.get("family"),
                "scoring_family": task.get("scoring_family"),
                "prompt": task.get("prompt"),
                "prompt_sha256": task.get("prompt_sha256"),
                "choices": task.get("choices"),
                "expected_choice": task.get("expected_choice"),
                "reference_tool_call": task.get("reference_tool_call"),
                "reference_tool_result": task.get("reference_tool_result"),
                "reference_tool_result_sha256": task.get("reference_tool_result_sha256"),
            }
        )
    if len(tasks) != 32:
        raise ReleaseError("release requires exactly 32 tasks")

    sources: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    for path in sorted(source_directory.glob("*.json")):
        source = _read_json(path)
        digest = _artifact_digest(path, source)
        if source.get("candidate_manifest_sha256") != manifest_sha:
            continue
        key = (
            str(source.get("requested_model_id") or ""),
            str(source.get("dataset_task_id") or ""),
        )
        if key[0] not in model_entries or key[1] not in tasks or key in sources:
            raise ReleaseError(f"source membership is malformed or duplicated: {path}")
        sources[key] = (source, digest)
    if len(sources) != 640:
        raise ReleaseError(f"release requires 640 source pairs, found {len(sources)}")

    responses: dict[tuple[str, str, str], tuple[dict[str, Any], str]] = {}
    for path in sorted(response_directory.glob("*.json")):
        response = _read_json(path)
        digest = _artifact_digest(path, response)
        if response.get("manifest_sha256") != manifest_sha:
            continue
        model = response.get("model")
        task = response.get("task")
        if not isinstance(model, Mapping) or not isinstance(task, Mapping):
            raise ReleaseError(f"response identity is malformed: {path}")
        key = (
            str(model.get("requested_model_id") or ""),
            str(task.get("public_id") or ""),
            str(response.get("condition") or ""),
        )
        if (
            key[0] not in model_entries
            or key[1] not in tasks
            or key[2] not in CONDITIONS
            or key in responses
        ):
            raise ReleaseError(f"response membership is malformed or duplicated: {path}")
        responses[key] = (response, digest)

    observations: list[dict[str, Any]] = []
    for model_id in model_entries:
        for task_id in sorted(tasks):
            source, source_sha = sources[(model_id, task_id)]
            for condition in CONDITIONS:
                response_pair = responses.get((model_id, task_id, condition))
                response = response_pair[0] if response_pair else {}
                response_sha = response_pair[1] if response_pair else None
                result = response.get("response")
                result = result if isinstance(result, Mapping) else {}
                answer = str(result.get("answer_markdown") or "")
                observed = parse_final_choice(answer)
                expected = str(tasks[task_id]["expected_choice"])
                normal = result.get("finish_reason") == "stop"
                observations.append(
                    {
                        "model_id": model_id,
                        "task_id": task_id,
                        "condition": condition,
                        "source_status": source.get("status"),
                        "source_artifact_sha256": source_sha,
                        "response_artifact_sha256": response_sha,
                        "actual_model_id": result.get("actual_model_id"),
                        "actual_provider": result.get("actual_provider"),
                        "finish_reason": result.get("finish_reason"),
                        "latency_ms": result.get("latency_ms"),
                        "cost_micros": result.get("cost_micros"),
                        "answer_markdown": answer,
                        "observed_choice": observed,
                        "expected_choice": expected,
                        "parseable_normal_completion": observed is not None and normal,
                        "correct": observed == expected and normal,
                        "tool_trace": _tool_projection(response),
                    }
                )
    if len(observations) != 1_280:
        raise ReleaseError("release observation grid is incomplete")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "FlavourBench",
        "track": "Epicure-native exact-choice",
        "release_status": "complete_public_automated_leaderboard",
        "human_judgments": 0,
        "counts": {
            "models": 20,
            "tasks": 32,
            "families": 4,
            "assigned_pairs": 640,
            "assigned_arms": 1_280,
            "observed_response_arms": len(responses),
        },
        "models": list(model_entries.values()),
        "tasks": public_tasks,
        "observations": observations,
        "leaderboard": leaderboard,
        "provenance": {
            "manifest_sha256": manifest_sha,
            "taskset_artifact_sha256": taskset["artifact_sha256"],
            "task_set_sha256": taskset["task_set_sha256"],
            "leaderboard_artifact_sha256": leaderboard_sha,
            "epicure": taskset["epicure_provenance"],
        },
    }
    payload["artifact_sha256"] = _sha256(payload)
    if not verify_release(payload):
        raise ReleaseError("constructed release failed its replay checks")
    return payload


def verify_release(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    models = document.get("models")
    tasks = document.get("tasks")
    observations = document.get("observations")
    leaderboard = document.get("leaderboard")
    if not (
        document.get("schema_version") == SCHEMA_VERSION
        and recorded == _sha256(payload)
        and isinstance(models, list)
        and len(models) == 20
        and isinstance(tasks, list)
        and len(tasks) == 32
        and isinstance(observations, list)
        and len(observations) == 1_280
        and isinstance(leaderboard, Mapping)
        and leaderboard.get("official_track") is True
    ):
        return False
    model_ids = {str(model.get("model_id") or "") for model in models if isinstance(model, Mapping)}
    task_ids = {str(task.get("task_id") or "") for task in tasks if isinstance(task, Mapping)}
    keys = {
        (
            str(row.get("model_id") or ""),
            str(row.get("task_id") or ""),
            str(row.get("condition") or ""),
        )
        for row in observations
        if isinstance(row, Mapping)
    }
    expected = {
        (model_id, task_id, condition)
        for model_id in model_ids
        for task_id in task_ids
        for condition in CONDITIONS
    }
    return len(model_ids) == 20 and len(task_ids) == 32 and keys == expected


def write_release(document: Mapping[str, Any], output_directory: Path) -> Path:
    if not verify_release(document):
        raise ReleaseError("refusing to write an invalid release")
    digest = str(document["artifact_sha256"])
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"epicure-native-release-{digest}.json"
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise ReleaseError("content-addressed release conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_directory, delete=False
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.link(temporary, destination)
    destination.chmod(0o644)
    temporary.unlink()
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--response-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_release(
        manifest_path=args.manifest,
        taskset_path=args.taskset,
        leaderboard_path=args.leaderboard,
        source_directory=args.source_directory,
        response_directory=args.response_directory,
    )
    path = write_release(document, args.output_directory)
    print(
        json.dumps(
            {
                "release": str(path),
                "artifact_sha256": document["artifact_sha256"],
                "observations": len(document["observations"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
