"""Verify the compact, provider-free FlavourBench powered release.

This replay intentionally needs only the checked-in release JSON and its two CSV
tables.  It verifies their content addresses and the complete common-task
statistical result contract without making provider or Epicure calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path
from typing import Any


class PoweredReleaseError(RuntimeError):
    """Raised when the public release fails closed."""


def _reject_constant(value: str) -> None:
    raise PoweredReleaseError(f"non-finite JSON constant: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise PoweredReleaseError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoweredReleaseError(f"cannot read release: {path}") from exc
    if not isinstance(document, dict):
        raise PoweredReleaseError("release root must be an object")
    return document


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PoweredReleaseError(message)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PoweredReleaseError(f"cannot read table: {path}") from exc
    return list(csv.DictReader(io.StringIO(text)))


def verify_release(path: Path) -> dict[str, Any]:
    """Verify one compact release and return a short reproducibility summary."""

    _require(path.is_file() and not path.is_symlink(), "release must be a regular file")
    release = _load_json(path)
    stated = str(release.get("artifact_sha256", ""))
    semantic_payload = dict(release)
    semantic_payload.pop("artifact_sha256", None)
    _require(len(stated) == 64, "release semantic hash is malformed")
    _require(_canonical_sha256(semantic_payload) == stated, "release semantic hash failed")
    _require(stated in path.name, "release filename is not content addressed")
    _require(release.get("status") == "final_complete", "release is not final_complete")
    _require(
        release.get("schema_version") == "flavourbench-selection-powered-release-v1",
        "unexpected release schema",
    )

    analysis = release.get("analysis")
    _require(isinstance(analysis, dict), "analysis is missing")
    _require(analysis.get("status") == "final_complete", "analysis is not final_complete")
    models = analysis.get("models")
    pairs = analysis.get("pairwise_comparisons")
    repeats = analysis.get("repeatability")
    _require(isinstance(models, list) and len(models) >= 2, "expected at least two models")
    model_count = len(models)
    pair_count = model_count * (model_count - 1) // 2
    _require(isinstance(pairs, list) and len(pairs) == pair_count, "pairwise row count failed")
    _require(isinstance(repeats, list) and len(repeats) == model_count, "repeat row count failed")

    model_ids = [str(row["model_id"]) for row in models]
    _require(len(set(model_ids)) == model_count, "model IDs are not unique")
    _require(
        all(int(row["availability"]["scheduled"]) == 640 for row in models),
        "every model must have 640 scheduled primary tasks",
    )
    _require(
        all(int(row["tasks"]) == 64 for row in repeats),
        "every model must have 64 scheduled repeat tasks",
    )
    _require({str(row["model_id"]) for row in repeats} == set(model_ids), "repeat roster drift")

    expected_pairs = {frozenset(pair) for pair in combinations(model_ids, 2)}
    observed_pairs = {
        frozenset((str(row["left_model_id"]), str(row["right_model_id"]))) for row in pairs
    }
    _require(observed_pairs == expected_pairs, "pairwise comparison grid is incomplete")

    inputs = release.get("inputs")
    _require(isinstance(inputs, dict), "release inputs are missing")
    _require(
        int(inputs["primary_responses"]["count"]) == model_count * 640,
        "primary response count differs from the model/task grid",
    )
    _require(
        int(inputs["repeat_responses"]["count"]) == model_count * 64,
        "repeat response count differs from the model/repeat grid",
    )

    tables = release.get("tables")
    _require(isinstance(tables, dict), "release table commitments are missing")
    table_rows: dict[str, list[dict[str, str]]] = {}
    for label, expected_rows in (("leaderboard", model_count), ("pairwise", pair_count)):
        table = tables.get(label)
        _require(isinstance(table, dict), f"{label} table commitment is missing")
        table_path = path.parent / str(table.get("filename", ""))
        _require(table_path.is_file() and not table_path.is_symlink(), f"{label} table is missing")
        _require(_file_sha256(table_path) == table.get("sha256"), f"{label} table hash failed")
        rows = _csv_rows(table_path)
        _require(len(rows) == expected_rows, f"{label} table row count failed")
        table_rows[label] = rows

    _require(
        {row["model_id"] for row in table_rows["leaderboard"]} == set(model_ids),
        "leaderboard roster differs from release",
    )
    table_pairs = {
        frozenset((row["left_model_id"], row["right_model_id"])) for row in table_rows["pairwise"]
    }
    _require(table_pairs == expected_pairs, "pairwise CSV grid differs from release")

    ranked = sorted(
        (row for row in models if row.get("point_estimate_rank") is not None),
        key=lambda row: (int(row["point_estimate_rank"]), str(row["model_id"])),
    )
    _require(ranked, "release has no rank-eligible model")
    _require(
        [int(row["point_estimate_rank"]) for row in ranked] == list(range(1, len(ranked) + 1)),
        "eligible point ranks are not contiguous",
    )
    return {
        "status": "verified",
        "release": path.name,
        "artifact_sha256": stated,
        "models": len(models),
        "tasks_per_model": 640,
        "primary_responses": model_count * 640,
        "repeat_responses": model_count * 64,
        "pairwise_comparisons": len(pairs),
        "leader_model_id": ranked[0]["model_id"],
        "leader_score": ranked[0]["flavourbench_score"],
        "definitive_top_model_id": analysis.get("definitive_top_model_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_release(args.release), sort_keys=True))


if __name__ == "__main__":
    main()
