from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RELEASE = ROOT / "paper" / "generated" / "epicure-native" / "epicure-native-release.json"
DEFAULT_OUTPUT = HERE / "data"
TABLE_ORDER = ("models", "tasks", "observations", "paired_outcomes", "leaderboard")


class DatasetBuildError(RuntimeError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def _verify_release(release: dict[str, Any]) -> None:
    payload = dict(release)
    stated = payload.pop("artifact_sha256", None)
    computed = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if stated != computed:
        raise DatasetBuildError(
            f"Release semantic hash mismatch: stated={stated!r}, computed={computed!r}"
        )


def _paired_outcome(off: dict[str, Any], on: dict[str, Any]) -> str:
    if not off["parseable_normal_completion"] or not on["parseable_normal_completion"]:
        return "incomplete"
    if off["correct"] and on["correct"]:
        return "both_correct"
    if off["correct"]:
        return "off_only"
    if on["correct"]:
        return "on_only"
    return "neither"


def _score_ranks(release: dict[str, Any]) -> dict[str, int]:
    ordered = sorted(
        release["leaderboard"]["models"],
        key=lambda row: (
            -float(row["epicure_benchmark_score"]),
            str(row["display_name"]).casefold(),
        ),
    )
    result: dict[str, int] = {}
    previous_score: float | None = None
    current_rank = 0
    for position, row in enumerate(ordered, start=1):
        score = float(row["epicure_benchmark_score"])
        if previous_score is None or score != previous_score:
            current_rank = position
            previous_score = score
        result[str(row["model_id"])] = current_rank
    return result


def _build_tables(release: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    leaderboard_by_model = {row["model_id"]: row for row in release["leaderboard"]["models"]}
    task_by_id = {task["task_id"]: task for task in release["tasks"]}
    score_ranks = _score_ranks(release)

    model_rows = []
    for model in release["models"]:
        leaderboard = leaderboard_by_model[model["model_id"]]
        model_rows.append(
            {
                **model,
                "rank": score_ranks[model["model_id"]],
                "release_order": leaderboard["rank"],
                "evaluation_status": leaderboard["evaluation_status"],
                "epicure_benchmark_score": leaderboard["epicure_benchmark_score"],
                "tool_on_accuracy_percent": leaderboard["conditions"]["epicure_on"][
                    "accuracy_percent"
                ],
                "uplift_percentage_points": leaderboard["uplift_percentage_points"],
            }
        )

    grouped: defaultdict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for observation in release["observations"]:
        grouped[(observation["model_id"], observation["task_id"])][observation["condition"]] = (
            observation
        )

    paired_rows = []
    for model in release["models"]:
        for task in release["tasks"]:
            key = (model["model_id"], task["task_id"])
            pair = grouped[key]
            if set(pair) != {"epicure_off", "epicure_on"}:
                raise DatasetBuildError(f"Incomplete assigned condition pair: {key!r}")
            off = pair["epicure_off"]
            on = pair["epicure_on"]
            paired_rows.append(
                {
                    "model_id": model["model_id"],
                    "display_name": model["display_name"],
                    "task_id": task["task_id"],
                    "family": task_by_id[task["task_id"]]["family"],
                    "expected_choice": task["expected_choice"],
                    "off_observed_choice": off["observed_choice"],
                    "on_observed_choice": on["observed_choice"],
                    "off_correct": off["correct"],
                    "on_correct": on["correct"],
                    "off_parseable_normal_completion": off["parseable_normal_completion"],
                    "on_parseable_normal_completion": on["parseable_normal_completion"],
                    "off_response_artifact_sha256": off["response_artifact_sha256"],
                    "on_response_artifact_sha256": on["response_artifact_sha256"],
                    "paired_outcome": _paired_outcome(off, on),
                    "epicure_tool_trace": on["tool_trace"],
                }
            )

    leaderboard_rows = []
    leaderboard_models = sorted(
        release["leaderboard"]["models"],
        key=lambda row: (
            -float(row["epicure_benchmark_score"]),
            str(row["display_name"]).casefold(),
        ),
    )
    for row in leaderboard_models:
        leaderboard_rows.append(
            {
                "rank": score_ranks[row["model_id"]],
                "release_order": row["rank"],
                "model_id": row["model_id"],
                "display_name": row["display_name"],
                "execution_backend": row["execution_backend"],
                "evaluation_status": row["evaluation_status"],
                "epicure_benchmark_score": row["epicure_benchmark_score"],
                "tool_off_accuracy_percent": row["conditions"]["epicure_off"]["accuracy_percent"],
                "tool_off_correct": row["conditions"]["epicure_off"]["correct"],
                "tool_off_parseable_answers": row["conditions"]["epicure_off"]["parseable_answers"],
                "tool_off_normal_completions": row["conditions"]["epicure_off"][
                    "normal_completions"
                ],
                "tool_on_accuracy_percent": row["conditions"]["epicure_on"]["accuracy_percent"],
                "uplift_percentage_points": row["uplift_percentage_points"],
                "tool_off_wilson_95": row["conditions"]["epicure_off"]["wilson_95"],
                "tool_on_wilson_95": row["conditions"]["epicure_on"]["wilson_95"],
                "paired_outcomes": row["paired_outcomes"],
                "normal_response_arms": (
                    row["conditions"]["epicure_off"]["normal_completions"]
                    + row["conditions"]["epicure_on"]["normal_completions"]
                ),
                "expected_response_arms": row["expected_pairs"] * 2,
                "total_observed_cost_usd": row["total_observed_cost_usd"],
            }
        )

    return {
        "models": model_rows,
        "tasks": release["tasks"],
        "observations": release["observations"],
        "paired_outcomes": paired_rows,
        "leaderboard": leaderboard_rows,
    }


def _expected_files(release: dict[str, Any]) -> dict[str, bytes]:
    tables = _build_tables(release)
    files = {f"{name}.jsonl": _jsonl(tables[name]) for name in TABLE_ORDER}
    manifest = {
        "schema_version": "flavourbench-hf-dataset-manifest-v1",
        "release_artifact_sha256": release["artifact_sha256"],
        "files": [
            {
                "name": name,
                "rows": len(tables[name.removesuffix(".jsonl")]),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in files.items()
        ],
    }
    files["DATA_MANIFEST.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    return files


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build(release_path: Path, output: Path, *, check: bool) -> None:
    release = json.loads(release_path.read_bytes())
    _verify_release(release)
    expected = _expected_files(release)
    if check:
        mismatches = [
            name
            for name, payload in expected.items()
            if not (output / name).is_file() or (output / name).read_bytes() != payload
        ]
        if mismatches:
            raise DatasetBuildError(f"Generated dataset mismatch: {', '.join(mismatches)}")
        print(f"OK: {len(expected) - 1} tables match release {release['artifact_sha256']}")
        return

    for name, payload in expected.items():
        _write_atomic(output / name, payload)
    print(f"Wrote {len(expected) - 1} tables to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public FlavourBench HF dataset tables")
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build(args.release, args.output, check=args.check)


if __name__ == "__main__":
    main()
