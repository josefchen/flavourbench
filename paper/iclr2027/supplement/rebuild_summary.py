#!/usr/bin/env python3
"""Recompute FlavourBench score summaries from the anonymous selected response matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


class ReconstructionError(RuntimeError):
    """The selected release cannot be reconstructed exactly."""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ReconstructionError(f"input is not a regular file: {path}")
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ReconstructionError(f"JSONL row is not an object: {path}")
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reconstruct(dataset: Path, output: Path) -> dict[str, object]:
    tasks = _jsonl(dataset / "tasks.jsonl")
    observations = _jsonl(dataset / "primary_observations.jsonl")
    published = _jsonl(dataset / "leaderboard.jsonl")

    family_by_task = {str(row["task_id"]): str(row["family"]) for row in tasks}
    if len(tasks) != 534 or len(family_by_task) != 534:
        raise ReconstructionError("task roster is not the 534-task release")

    scores: dict[str, list[float]] = defaultdict(list)
    family_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    cells: set[tuple[str, str]] = set()
    for wrapper in observations:
        response = wrapper.get("response")
        scoring = wrapper.get("release_scoring")
        if not isinstance(response, dict) or not isinstance(scoring, dict):
            raise ReconstructionError("response wrapper differs")
        model_id = str(response["model_id"])
        task_id = str(response["task_id"])
        key = (model_id, task_id)
        if key in cells or task_id not in family_by_task or scoring.get("parseable") is not True:
            raise ReconstructionError("response grid is duplicated, incomplete, or unparseable")
        cells.add(key)
        score = float(scoring["score_bps"]) / 100.0
        scores[model_id].append(score)
        family_scores[(model_id, family_by_task[task_id])].append(score)

    if (
        len(scores) != 27
        or len(cells) != 14_418
        or any(len(values) != 534 for values in scores.values())
    ):
        raise ReconstructionError("response matrix is not rectangular")

    published_by_id = {str(row["model_id"]): row for row in published}
    if set(published_by_id) != set(scores):
        raise ReconstructionError("published and reconstructed model rosters differ")

    rows: list[dict[str, object]] = []
    for model_id, values in scores.items():
        mean = statistics.fmean(values)
        expected = float(published_by_id[model_id]["flavourbench_score"])
        if not math.isclose(mean, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ReconstructionError(f"primary score differs for {model_id}")
        row: dict[str, object] = {
            "model_id": model_id,
            "model_name": published_by_id[model_id]["model_name"],
            "tasks": len(values),
            "flavourbench_score": f"{mean:.12f}",
        }
        for family in ("substitution", "pairing", "constraint"):
            observed = family_scores[(model_id, family)]
            if len(observed) != 178:
                raise ReconstructionError(f"family coverage differs for {model_id}/{family}")
            family_mean = statistics.fmean(observed)
            expected_family = float(published_by_id[model_id]["family_scores"][family])
            if not math.isclose(family_mean, expected_family, rel_tol=0.0, abs_tol=1e-12):
                raise ReconstructionError(f"family score differs for {model_id}/{family}")
            row[f"{family}_score"] = f"{family_mean:.12f}"
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["flavourbench_score"]), str(row["model_id"])))
    for rank, row in enumerate(rows, start=1):
        row["reconstructed_rank"] = rank
        if int(published_by_id[str(row["model_id"])]["point_estimate_rank"]) != rank:
            raise ReconstructionError(f"point rank differs for {row['model_id']}")
    _write_csv(
        output / "reconstructed_leaderboard.csv",
        [
            "reconstructed_rank",
            "model_id",
            "model_name",
            "tasks",
            "flavourbench_score",
            "substitution_score",
            "pairing_score",
            "constraint_score",
        ],
        rows,
    )

    task_rows: list[dict[str, object]] = []
    for family in ("substitution", "pairing", "constraint"):
        selected = [row for row in tasks if row["family"] == family]
        distinct = [len(set(row["selection_scores_bps"].values())) for row in selected]
        chance_mean = statistics.fmean(float(row["chance_score_bps"]) for row in selected) / 100
        median_gap = statistics.median(float(row["optimal_margin_bps"]) for row in selected) / 100
        task_rows.append(
            {
                "family": family,
                "tasks": len(selected),
                "exact_chance_mean": f"{chance_mean:.6f}",
                "median_top_gap": f"{median_gap:.6f}",
                "median_distinct_scores": f"{statistics.median(distinct):.1f}",
            }
        )
    _write_csv(
        output / "reconstructed_task_diagnostics.csv",
        ["family", "tasks", "exact_chance_mean", "median_top_gap", "median_distinct_scores"],
        task_rows,
    )
    return {
        "models": len(rows),
        "tasks": len(tasks),
        "observations": len(observations),
        "top_model_id": rows[0]["model_id"],
        "top_score": rows[0]["flavourbench_score"],
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.dataset_directory, args.output_directory), sort_keys=True))


if __name__ == "__main__":
    main()
