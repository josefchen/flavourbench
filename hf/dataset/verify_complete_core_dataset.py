from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


class CompleteCoreDatasetVerifyError(RuntimeError):
    """The published complete-core dataset failed verification."""


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
        raise CompleteCoreDatasetVerifyError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompleteCoreDatasetVerifyError(f"input is not a JSON object: {path}")
    return value


def _semantic_valid(document: dict[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == hashlib.sha256(_canonical(payload)).hexdigest())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise CompleteCoreDatasetVerifyError(f"JSONL row is not an object: {path.name}")
    return rows


def verify_dataset(directory: Path) -> dict[str, Any]:
    manifest = _load(directory / "DATA_MANIFEST.json")
    if (
        not _semantic_valid(manifest)
        or manifest.get("schema_version") != "flavourbench-hf-complete-core-dataset-manifest-v1"
        or manifest.get("status") != "final_complete_common_core"
    ):
        raise CompleteCoreDatasetVerifyError("dataset manifest failed verification")
    inventory = {str(row["name"]): row for row in manifest["files"]}
    expected = {*inventory, "DATA_MANIFEST.json"}
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != expected:
        raise CompleteCoreDatasetVerifyError("dataset file inventory differs")
    for name, record in inventory.items():
        path = directory / name
        payload = path.read_bytes()
        if path.is_symlink() or len(payload) != int(record["bytes"]):
            raise CompleteCoreDatasetVerifyError(f"dataset file size differs: {name}")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise CompleteCoreDatasetVerifyError(f"dataset file hash differs: {name}")
        if name.endswith(".jsonl") and payload.count(b"\n") != int(record["rows"]):
            raise CompleteCoreDatasetVerifyError(f"dataset row count differs: {name}")

    release = _load(directory / "release.json")
    if (
        not _semantic_valid(release)
        or release["artifact_sha256"] != manifest["release_artifact_sha256"]
        or release.get("status") != "final_complete_common_core"
    ):
        raise CompleteCoreDatasetVerifyError("release and manifest differ")
    models = _jsonl(directory / "models.jsonl")
    tasks = _jsonl(directory / "tasks.jsonl")
    responses = _jsonl(directory / "primary_observations.jsonl")
    leaderboard = _jsonl(directory / "leaderboard.jsonl")
    pairwise = _jsonl(directory / "pairwise_comparisons.jsonl")

    model_ids = [str(row["model_id"]) for row in models]
    task_ids = [str(row["task_id"]) for row in tasks]
    if len(model_ids) != 27 or len(set(model_ids)) != 27:
        raise CompleteCoreDatasetVerifyError("model roster differs")
    if len(task_ids) != 534 or len(set(task_ids)) != 534:
        raise CompleteCoreDatasetVerifyError("task roster differs")
    if len(leaderboard) != 27 or [row["point_estimate_rank"] for row in leaderboard] != list(
        range(1, 28)
    ):
        raise CompleteCoreDatasetVerifyError("leaderboard order differs")
    if (
        len(pairwise) != 351
        or len({(row["left_model_id"], row["right_model_id"]) for row in pairwise}) != 351
    ):
        raise CompleteCoreDatasetVerifyError("pairwise comparison family differs")

    cells: set[tuple[str, str]] = set()
    artifacts: dict[str, list[str]] = {"panel_1": [], "panel_2": []}
    panel_counts: Counter[str] = Counter()
    for row in responses:
        panel = str(row.get("release_panel") or "")
        response = row.get("response")
        scoring = row.get("release_scoring")
        if (
            panel not in artifacts
            or not isinstance(response, dict)
            or not isinstance(scoring, dict)
        ):
            raise CompleteCoreDatasetVerifyError("response wrapper differs")
        if not _semantic_valid(response):
            raise CompleteCoreDatasetVerifyError("response semantic hash differs")
        if response.get("status") != "completed" or scoring.get("parseable") is not True:
            raise CompleteCoreDatasetVerifyError("common-core response is not valid")
        model_id = str(response["model_id"])
        task_id = str(response["task_id"])
        if model_id not in model_ids or task_id not in task_ids:
            raise CompleteCoreDatasetVerifyError("response key is outside the release grid")
        key = (model_id, task_id)
        if key in cells:
            raise CompleteCoreDatasetVerifyError("response cell is duplicated")
        cells.add(key)
        artifacts[panel].append(str(response["artifact_sha256"]))
        panel_counts[panel] += 1
    if len(cells) != 14_418 or cells != {(model, task) for model in model_ids for task in task_ids}:
        raise CompleteCoreDatasetVerifyError("response matrix is incomplete")
    for panel, values in artifacts.items():
        observed_commitment = {
            "count": panel_counts[panel],
            "artifact_set_sha256": hashlib.sha256(_canonical(sorted(values))).hexdigest(),
        }
        if observed_commitment != release["inputs"][f"{panel}_responses"]:
            raise CompleteCoreDatasetVerifyError(f"{panel} response commitment differs")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the published FlavourBench dataset")
    parser.add_argument("--dataset-directory", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_dataset(args.dataset_directory)
    print(
        "OK: 27 models, 534 tasks, 14,418 complete responses, 351 pairs, "
        f"manifest {manifest['artifact_sha256']}"
    )


if __name__ == "__main__":
    main()
