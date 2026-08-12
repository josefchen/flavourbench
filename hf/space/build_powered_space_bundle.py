from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE.parent / "dataset" / "data-powered"
DEFAULT_OUTPUT = HERE / "data-powered" / "flavourbench-powered-space.json"


class PoweredSpaceBuildError(RuntimeError):
    """The powered Space bundle failed verification."""


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
        raise PoweredSpaceBuildError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PoweredSpaceBuildError(f"input is not a JSON object: {path}")
    return value


def _semantic_valid(document: dict[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == hashlib.sha256(_canonical(payload)).hexdigest())


def _jsonl(path: Path, *, expected_sha256: str, expected_rows: int) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PoweredSpaceBuildError(f"dataset table hash failed: {path.name}")
    rows = [json.loads(line) for line in payload.splitlines() if line]
    if len(rows) != expected_rows or any(not isinstance(row, dict) for row in rows):
        raise PoweredSpaceBuildError(f"dataset table cardinality failed: {path.name}")
    return rows


def build_bundle(*, release: dict[str, Any], dataset_directory: Path) -> dict[str, Any]:
    if not _semantic_valid(release) or release.get("status") != "final_complete":
        raise PoweredSpaceBuildError("release is not the final powered release")
    manifest = _load(dataset_directory / "DATA_MANIFEST.json")
    if (
        not _semantic_valid(manifest)
        or manifest.get("release_artifact_sha256") != release["artifact_sha256"]
    ):
        raise PoweredSpaceBuildError("dataset manifest and release differ")
    records = {str(row["name"]): row for row in manifest["files"]}

    def rows(name: str) -> list[dict[str, Any]]:
        record = records[name]
        return _jsonl(
            dataset_directory / name,
            expected_sha256=str(record["sha256"]),
            expected_rows=int(record["rows"]),
        )

    models = rows("models.jsonl")
    tasks = rows("tasks.jsonl")
    primary = rows("primary_observations.jsonl")
    pairwise = rows("pairwise_comparisons.jsonl")
    compact_observations = []
    for row in primary:
        generation = row.get("generation") or {}
        answer = str(generation.get("answer_markdown") or "")
        compact_observations.append(
            {
                "model_id": row["model_id"],
                "task_id": row["task_id"],
                "status": row["status"],
                "scoring": row["scoring"],
                "answer_excerpt": answer[:1600],
                "answer_truncated": len(answer) > 1600,
                "actual_model_id": generation.get("actual_model_id"),
                "actual_provider": generation.get("actual_provider"),
                "finish_reason": generation.get("finish_reason"),
                "latency_ms": generation.get("latency_ms"),
                "cost_micros": generation.get("cost_micros"),
                "artifact_sha256": row["artifact_sha256"],
            }
        )
    bundle: dict[str, Any] = {
        "schema_version": "flavourbench-powered-space-bundle-v1",
        "release_artifact_sha256": release["artifact_sha256"],
        "dataset_manifest_sha256": manifest["artifact_sha256"],
        "status": release["status"],
        "benchmark": release["benchmark"],
        "track": release["track"],
        "analysis": release["analysis"],
        "claim_boundary": release["claim_boundary"],
        "models": models,
        "tasks": tasks,
        "primary_observations": compact_observations,
        "pairwise_comparisons": pairwise,
    }
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
    parser = argparse.ArgumentParser(description="Build the compact powered FlavourBench Space")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--dataset-directory", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    bundle = build_bundle(release=_load(args.release), dataset_directory=args.dataset_directory)
    payload = _bytes(bundle)
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != payload:
            raise PoweredSpaceBuildError("generated Space bundle differs")
        print(f"OK: Space bundle {bundle['artifact_sha256']}")
        return
    _write_atomic(args.output, payload)
    print(f"Wrote {args.output} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
