from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE.parent / "dataset" / "data-complete-core"
DEFAULT_LAB_DATASET = HERE.parent / "dataset" / "data-lab"
DEFAULT_STABILITY = (
    HERE.parents[1] / "paper/generated/complete-core/complete-core-stability-analysis.json"
)
DEFAULT_OUTPUT = HERE / "data-complete-core" / "flavourbench-complete-core-space.json"


class CompleteCoreSpaceBuildError(RuntimeError):
    """The complete-common-core Space bundle failed verification."""


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
        raise CompleteCoreSpaceBuildError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompleteCoreSpaceBuildError(f"input is not a JSON object: {path}")
    return value


def _semantic_valid(document: dict[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == hashlib.sha256(_canonical(payload)).hexdigest())


def _jsonl(path: Path, *, expected_sha256: str, expected_rows: int) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise CompleteCoreSpaceBuildError(f"dataset table hash failed: {path.name}")
    rows = [json.loads(line) for line in payload.splitlines() if line]
    if len(rows) != expected_rows or any(not isinstance(row, dict) for row in rows):
        raise CompleteCoreSpaceBuildError(f"dataset table cardinality failed: {path.name}")
    return rows


def build_bundle(
    *,
    dataset_directory: Path,
    lab_dataset_directory: Path,
    stability_path: Path = DEFAULT_STABILITY,
) -> dict[str, Any]:
    manifest = _load(dataset_directory / "DATA_MANIFEST.json")
    if (
        not _semantic_valid(manifest)
        or manifest.get("schema_version") != "flavourbench-hf-complete-core-dataset-manifest-v1"
        or manifest.get("status") != "final_complete_common_core"
    ):
        raise CompleteCoreSpaceBuildError("dataset manifest failed verification")
    records = {str(row["name"]): row for row in manifest["files"]}

    def rows(name: str) -> list[dict[str, Any]]:
        record = records[name]
        return _jsonl(
            dataset_directory / name,
            expected_sha256=str(record["sha256"]),
            expected_rows=int(record["rows"]),
        )

    release_record = records["release.json"]
    release_path = dataset_directory / "release.json"
    release_bytes = release_path.read_bytes()
    if hashlib.sha256(release_bytes).hexdigest() != release_record["sha256"]:
        raise CompleteCoreSpaceBuildError("release physical hash failed")
    release = _load(release_path)
    if (
        not _semantic_valid(release)
        or release.get("status") != "final_complete_common_core"
        or release.get("artifact_sha256") != manifest["release_artifact_sha256"]
    ):
        raise CompleteCoreSpaceBuildError("release and dataset manifest differ")

    models = rows("models.jsonl")
    tasks = rows("tasks.jsonl")
    primary = rows("primary_observations.jsonl")
    pairwise = rows("pairwise_comparisons.jsonl")

    lab_manifest = _load(lab_dataset_directory / "DATA_MANIFEST.json")
    if (
        not _semantic_valid(lab_manifest)
        or lab_manifest.get("schema_version") != "flavourbench-lab-dataset-v2"
        or lab_manifest.get("status") != "preregistered_transfer_maps_not_official_leaderboard_test"
    ):
        raise CompleteCoreSpaceBuildError("lab dataset manifest failed verification")
    lab_records = {str(row["name"]): row for row in lab_manifest["files"]}

    def lab_rows(name: str) -> list[dict[str, Any]]:
        record = lab_records[name]
        return _jsonl(
            lab_dataset_directory / name,
            expected_sha256=str(record["sha256"]),
            expected_rows=int(record["rows"]),
        )

    lab_tasks = [*lab_rows("train_tasks.jsonl"), *lab_rows("validation_tasks.jsonl")]
    stability = _load(stability_path)
    if (
        not _semantic_valid(stability)
        or stability.get("schema_version") != "flavourbench-task-count-stability-v1"
        or stability.get("status") != "retrospective_precision_and_stability_analysis"
    ):
        raise CompleteCoreSpaceBuildError("task-count stability analysis failed verification")
    official_ids = {str(row["task_id"]) for row in tasks}
    official_anchors = {str(row["anchor_ingredient"]) for row in tasks}
    lab_ids = {str(row["task_id"]) for row in lab_tasks}
    lab_anchors = {str(row["anchor_ingredient"]) for row in lab_tasks}
    if (
        len(lab_tasks) != 342
        or len(lab_ids) != 342
        or len(lab_anchors) != 342
        or lab_ids & official_ids
        or lab_anchors & official_anchors
    ):
        raise CompleteCoreSpaceBuildError("lab and official reward maps are not disjoint")
    compact_observations: list[dict[str, Any]] = []
    for row in primary:
        response = row["response"]
        generation = response.get("generation") or {}
        answer = str(generation.get("answer_markdown") or "")
        compact_observations.append(
            {
                "release_panel": row["release_panel"],
                "model_id": response["model_id"],
                "task_id": response["task_id"],
                "status": response["status"],
                "scoring": row["release_scoring"],
                "answer_excerpt": answer[:1800],
                "answer_truncated": len(answer) > 1800,
                "actual_model_id": generation.get("actual_model_id"),
                "actual_provider": generation.get("actual_provider"),
                "finish_reason": generation.get("finish_reason"),
                "latency_ms": generation.get("latency_ms"),
                "cost_micros": generation.get("cost_micros"),
                "artifact_sha256": response["artifact_sha256"],
            }
        )

    bundle: dict[str, Any] = {
        "schema_version": "flavourbench-complete-core-space-bundle-v1",
        "release_artifact_sha256": release["artifact_sha256"],
        "dataset_manifest_sha256": manifest["artifact_sha256"],
        "status": release["status"],
        "benchmark": release["benchmark"],
        "track": release["track"],
        "design": release["design"],
        "analysis": release["analysis"],
        "claim_boundary": release["claim_boundary"],
        "models": models,
        "tasks": tasks,
        "lab": {
            "dataset_manifest_sha256": lab_manifest["artifact_sha256"],
            "status": lab_manifest["status"],
            "task_count": len(lab_tasks),
            "train_tasks": int(lab_manifest["counts"]["train_tasks"]),
            "validation_tasks": int(lab_manifest["counts"]["validation_tasks"]),
            "evaluation_tasks": int(lab_manifest["counts"]["evaluation_tasks"]),
            "official_anchor_overlap": 0,
        },
        "lab_tasks": lab_tasks,
        "stability_analysis": stability,
        "primary_observations": compact_observations,
        "pairwise_comparisons": pairwise,
    }
    if (
        len(models) != 27
        or len(tasks) != 534
        or len(compact_observations) != 14_418
        or len(pairwise) != 351
    ):
        raise CompleteCoreSpaceBuildError("Space bundle cardinality differs")
    bundle["artifact_sha256"] = hashlib.sha256(_canonical(bundle)).hexdigest()
    return bundle


def _bytes(bundle: dict[str, Any]) -> bytes:
    return (json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


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
    parser = argparse.ArgumentParser(description="Build the final FlavourBench Space bundle")
    parser.add_argument("--dataset-directory", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lab-dataset-directory", type=Path, default=DEFAULT_LAB_DATASET)
    parser.add_argument("--stability", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bundle = build_bundle(
        dataset_directory=args.dataset_directory,
        lab_dataset_directory=args.lab_dataset_directory,
        stability_path=args.stability,
    )
    payload = _bytes(bundle)
    if args.check:
        if args.output.is_symlink() or not args.output.is_file():
            raise CompleteCoreSpaceBuildError("Space bundle output is absent")
        if args.output.read_bytes() != payload:
            raise CompleteCoreSpaceBuildError("generated Space bundle differs")
        print(f"OK: complete-core Space bundle {bundle['artifact_sha256']}")
        return
    _write_atomic(args.output, payload)
    print(f"Wrote {args.output} ({len(payload)} bytes, {bundle['artifact_sha256']})")


if __name__ == "__main__":
    main()
