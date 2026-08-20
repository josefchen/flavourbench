#!/usr/bin/env python3
"""Verify the compact FlavourBench complete-core statistical release offline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


class ReleaseVerificationError(RuntimeError):
    """The compact release is incomplete or internally inconsistent."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseVerificationError(f"release is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"release is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseVerificationError("release root must be an object")
    return value


def verify_release(path: Path) -> dict[str, Any]:
    """Verify semantic address, design invariants, statistics, and CSV commitments."""

    release = _read_json(path)
    payload = dict(release)
    recorded = str(payload.pop("artifact_sha256", ""))
    if not recorded or recorded != _sha256(_canonical(payload)):
        raise ReleaseVerificationError("release semantic SHA-256 differs")
    if path.name != f"flavourbench-complete-core-release-{recorded}.json":
        raise ReleaseVerificationError("release filename is not its semantic address")
    if (
        release.get("schema_version") != "flavourbench-complete-common-core-release-v1"
        or release.get("status") != "final_complete_common_core"
        or release.get("all_ranked_models_have_identical_task_count") is not True
        or release.get("failed_or_unparseable_cells_scored_as_zero") is not False
        or release.get("dnf_rows_emitted") is not False
    ):
        raise ReleaseVerificationError("release contract differs")

    design = release.get("design") or {}
    analysis = release.get("analysis") or {}
    models = analysis.get("models") or []
    pairs = analysis.get("pairwise_comparisons") or []
    if not (
        design.get("ranked_models") == 27
        and design.get("primary_tasks_per_model") == 534
        and len(models) == 27
        and len(pairs) == 351
        and analysis.get("resolved_pair_count")
        == sum(bool(row.get("holm_significant")) for row in pairs)
    ):
        raise ReleaseVerificationError("release cardinality or pair summary differs")
    model_ids = [str(row.get("model_id") or "") for row in models]
    if len(set(model_ids)) != 27 or not all(model_ids):
        raise ReleaseVerificationError("model roster is empty or duplicated")
    if any(
        (row.get("coverage") or {}).get("scheduled") != 534
        or (row.get("coverage") or {}).get("valid_scored") != 534
        or (row.get("coverage") or {}).get("valid_scored_rate") != 1.0
        for row in models
    ):
        raise ReleaseVerificationError("a ranked model is not complete on all 534 tasks")
    expected_pairs = {
        tuple(sorted((left, right)))
        for i, left in enumerate(model_ids)
        for right in model_ids[i + 1 :]
    }
    observed_pairs = {
        tuple(sorted((str(row.get("left_model_id")), str(row.get("right_model_id")))))
        for row in pairs
    }
    if observed_pairs != expected_pairs or any(
        row.get("shared_valid_tasks") != 534 for row in pairs
    ):
        raise ReleaseVerificationError("pairwise matrix is not the complete 27-model graph")

    tables = release.get("tables") or {}
    table_rows: dict[str, int] = {}
    for name, expected_rows in (("leaderboard", 27), ("pairwise", 351)):
        table = tables.get(name) or {}
        filename = str(table.get("filename") or "")
        table_path = path.parent / filename
        if table_path.is_symlink() or not table_path.is_file():
            raise ReleaseVerificationError(f"{name} CSV is missing")
        raw = table_path.read_bytes()
        if _sha256(raw) != table.get("sha256"):
            raise ReleaseVerificationError(f"{name} CSV SHA-256 differs")
        rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
        if len(rows) != expected_rows:
            raise ReleaseVerificationError(f"{name} CSV row count differs")
        table_rows[name] = len(rows)

    return {
        "release_semantic_sha256": recorded,
        "release_physical_sha256": _sha256(path.read_bytes()),
        "models": len(models),
        "tasks_per_model": design["primary_tasks_per_model"],
        "pairwise_rows": table_rows["pairwise"],
        "resolved_pairs": analysis["resolved_pair_count"],
        "status": "PASS",
    }


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_release(args.release), sort_keys=True))


if __name__ == "__main__":
    run()
