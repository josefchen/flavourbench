#!/usr/bin/env python3
"""Materialize the verified reward-transfer evidence for the public dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from flavourbench.reward_transfer import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_RESULTS,
    TRAINED_CONDITIONS,
    RewardTransferError,
    canonical_bytes,
    load_plan,
    load_verified_evaluation,
    semantic_sha256,
    verify_content_addressed,
    verify_evaluation_gate,
    verify_training_run,
)

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY / "hf/dataset/data-analysis"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _write_atomic(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise RewardTransferError(f"refusing to replace symlink: {path}")
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


def _load_analysis(path: Path, *, split: str, plan_hash: str, gate_hash: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RewardTransferError(f"reward-transfer analysis is unavailable: {path}")
    analysis = json.loads(path.read_text(encoding="utf-8"))
    verify_content_addressed(analysis, label=f"{split} reward-transfer analysis")
    expected_status = (
        "primary_analysis_complete_before_public_replication"
        if split == "primary"
        else "public_replication_analysis_complete"
    )
    expected = {
        "schema_version": "flavourbench-reward-transfer-analysis-v1",
        "status": expected_status,
        "split": split,
        "protocol_artifact_sha256": plan_hash,
        "evaluation_gate_artifact_sha256": gate_hash,
    }
    for key, value in expected.items():
        if analysis.get(key) != value:
            raise RewardTransferError(f"{split} reward-transfer analysis differs at {key}")
    return analysis


def _parse_time(value: object) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as error:
        raise RewardTransferError(f"invalid reward-transfer timestamp: {value}") from error


def build_release(
    *,
    results: Path = DEFAULT_RESULTS,
    checkpoints: Path = DEFAULT_CHECKPOINTS,
) -> dict[str, bytes]:
    plan = load_plan()
    gate_path = results / "evaluation-gate.json"
    gate = verify_evaluation_gate(gate_path, checkpoints=checkpoints)
    primary_master, primary_tasks, primary_runs = load_verified_evaluation(
        results / "primary",
        split="primary",
        gate_path=gate_path,
        checkpoints=checkpoints,
    )
    public_master, public_tasks, public_runs = load_verified_evaluation(
        results / "public",
        split="public",
        gate_path=gate_path,
        checkpoints=checkpoints,
    )
    primary_analysis = _load_analysis(
        results / "primary/analysis.json",
        split="primary",
        plan_hash=plan["artifact_sha256"],
        gate_hash=gate["artifact_sha256"],
    )
    public_analysis = _load_analysis(
        results / "public/analysis.json",
        split="public",
        plan_hash=plan["artifact_sha256"],
        gate_hash=gate["artifact_sha256"],
    )
    if (
        public_master.get("primary_analysis_artifact_sha256") != primary_analysis["artifact_sha256"]
        or public_analysis.get("primary_analysis_artifact_sha256")
        != primary_analysis["artifact_sha256"]
    ):
        raise RewardTransferError("public replication does not bind the sealed primary analysis")
    if not (
        _parse_time(gate["created_at_utc"])
        <= _parse_time(primary_master["completed_at_utc"])
        <= _parse_time(primary_analysis["completed_at_utc"])
        <= _parse_time(public_master["completed_at_utc"])
        <= _parse_time(public_analysis["completed_at_utc"])
    ):
        raise RewardTransferError("reward-transfer gate and analysis chronology differs")

    identities = [("pretrained_base", None)] + [
        (condition, int(seed)) for condition in TRAINED_CONDITIONS for seed in plan["seeds"]
    ]
    response_files: dict[str, bytes] = {}
    evaluation_wrappers: list[dict[str, Any]] = []
    for split, tasks, rows_by_run, master in (
        ("primary", primary_tasks, primary_runs, primary_master),
        ("public", public_tasks, public_runs, public_master),
    ):
        responses: list[dict[str, Any]] = []
        task_ids = [str(task["task_id"]) for task in tasks]
        for condition, seed in identities:
            by_id = {str(row["task_id"]): row for row in rows_by_run[(condition, seed)]}
            responses.extend({"evaluation_split": split, **by_id[task_id]} for task_id in task_ids)
        response_files[f"reward-transfer-{split}-responses.jsonl"] = _jsonl_bytes(responses)
        for record in master["runs"]:
            manifest_path = results / split / str(record["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verify_content_addressed(manifest, label=f"{split} evaluation run")
            evaluation_wrappers.append({"evaluation_split": split, "manifest": manifest})

    training_wrappers: list[dict[str, Any]] = []
    for condition in TRAINED_CONDITIONS:
        for seed in plan["seeds"]:
            manifest = verify_training_run(
                checkpoints / condition / f"seed-{seed}",
                plan=plan,
                condition=condition,
                seed=int(seed),
            )
            training_wrappers.append({"manifest": manifest})

    files = {
        "reward-transfer-evaluation-gate.json": _json_bytes(gate),
        "reward-transfer-primary-analysis.json": _json_bytes(primary_analysis),
        "reward-transfer-public-analysis.json": _json_bytes(public_analysis),
        "reward-transfer-training-manifests.jsonl": _jsonl_bytes(training_wrappers),
        "reward-transfer-evaluation-manifests.jsonl": _jsonl_bytes(evaluation_wrappers),
        **response_files,
    }
    release: dict[str, Any] = {
        "schema_version": "flavourbench-reward-transfer-release-v1",
        "status": "complete",
        "protocol_artifact_sha256": plan["artifact_sha256"],
        "evaluation_gate_artifact_sha256": gate["artifact_sha256"],
        "primary_evaluation_artifact_sha256": primary_master["artifact_sha256"],
        "primary_analysis_artifact_sha256": primary_analysis["artifact_sha256"],
        "public_evaluation_artifact_sha256": public_master["artifact_sha256"],
        "public_analysis_artifact_sha256": public_analysis["artifact_sha256"],
        "counts": {
            "training_runs": 6,
            "evaluation_runs_per_split": 7,
            "primary_tasks": len(primary_tasks),
            "primary_response_rows": len(primary_tasks) * len(identities),
            "public_tasks": len(public_tasks),
            "public_response_rows": len(public_tasks) * len(identities),
        },
        "adapter_distribution": (
            "Exact adapter hashes and training manifests are included; weight files are "
            "distributed "
            "separately from the anonymous paper supplement."
        ),
        "files": [
            {
                "name": name,
                "bytes": len(payload),
                "rows": payload.count(b"\n") if name.endswith(".jsonl") else None,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in sorted(files.items())
        ],
    }
    release["artifact_sha256"] = semantic_sha256(release)
    files["reward-transfer-release-manifest.json"] = _json_bytes(release)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = build_release(results=args.results, checkpoints=args.checkpoints)
    if args.check:
        for name, payload in files.items():
            path = args.output / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise RewardTransferError(f"released reward-transfer artifact differs: {path}")
        print("OK: reward-transfer release")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        _write_atomic(args.output / name, payload)
    print(f"built {len(files)} reward-transfer artifacts at {args.output}")


if __name__ == "__main__":
    main()
