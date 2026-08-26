#!/usr/bin/env python3
"""Fail-closed audit for the prospective reward-transfer inputs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_LAB_DATA = REPOSITORY / "hf/dataset/data-lab"
DEFAULT_PUBLIC_TASKS = REPOSITORY / "hf/dataset/data-complete-core/tasks.jsonl"
PRIMARY_FAMILIES = ("substitution", "pairing", "constraint")
EXPECTED_SELECTIONS = {"".join(labels) for labels in itertools.combinations("ABCDEFGH", 3)}


class AuditError(RuntimeError):
    """An input violates the frozen study contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _selection(completion: object) -> str:
    return str(completion).removeprefix("FINAL_SELECTION: ").replace(",", "")


def audit(lab_data: Path, public_tasks_path: Path) -> dict[str, Any]:
    manifest_path = lab_data / "DATA_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload = dict(manifest)
    recorded_manifest_hash = str(manifest_payload.pop("artifact_sha256"))
    if hashlib.sha256(_canonical(manifest_payload)).hexdigest() != recorded_manifest_hash:
        raise AuditError("lab manifest semantic hash differs")
    for record in manifest["files"]:
        path = lab_data / str(record["name"])
        if not path.is_file() or path.is_symlink():
            raise AuditError(f"manifest member is unavailable or unsafe: {path}")
        payload = path.read_bytes()
        if len(payload) != int(record["bytes"]):
            raise AuditError(f"byte count differs: {path.name}")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise AuditError(f"physical hash differs: {path.name}")

    split_files = {
        "train": "train_tasks.jsonl",
        "validation": "validation_tasks.jsonl",
        "evaluation": "evaluation_tasks.jsonl",
    }
    splits = {name: _rows(lab_data / filename) for name, filename in split_files.items()}
    expected_counts = {"train": 270, "validation": 72, "evaluation": 84}
    expected_per_stratum = {"train": 45, "validation": 12, "evaluation": 14}
    for split, rows in splits.items():
        if len(rows) != expected_counts[split]:
            raise AuditError(f"{split} row count differs")
        strata = Counter((row["family"], row["source_panel"]) for row in rows)
        expected = Counter(
            {
                (family, panel): expected_per_stratum[split]
                for family in PRIMARY_FAMILIES
                for panel in ("panel_1", "panel_2")
            }
        )
        if strata != expected:
            raise AuditError(f"{split} family-by-panel balance differs")
        if len({str(row["task_id"]) for row in rows}) != len(rows):
            raise AuditError(f"{split} task IDs are not unique")
        if len({str(row["anchor_ingredient"]) for row in rows}) != len(rows):
            raise AuditError(f"{split} anchors are not unique")
        if len({str(row["prompt_sha256"]) for row in rows}) != len(rows):
            raise AuditError(f"{split} prompt hashes are not unique")
        for row in rows:
            scores = row.get("selection_scores_bps")
            if not isinstance(scores, dict) or set(scores) != EXPECTED_SELECTIONS:
                raise AuditError(f"{row['task_id']} does not contain all 56 actions")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10_000
                for value in scores.values()
            ):
                raise AuditError(f"{row['task_id']} has an invalid reward value")
            if scores.get(str(row["optimal_selection"])) != 10_000:
                raise AuditError(f"{row['task_id']} optimum does not score 10,000 bps")
            expected_chance = round(mean(int(value) for value in scores.values()))
            if int(row["chance_score_bps"]) != expected_chance:
                raise AuditError(f"{row['task_id']} chance score differs")

    overlap: dict[str, dict[str, int]] = {}
    for left, right in itertools.combinations(splits, 2):
        left_rows = splits[left]
        right_rows = splits[right]
        overlap[f"{left}__{right}"] = {
            "anchor": len(
                {str(row["anchor_ingredient"]) for row in left_rows}
                & {str(row["anchor_ingredient"]) for row in right_rows}
            ),
            "task_id": len(
                {str(row["task_id"]) for row in left_rows}
                & {str(row["task_id"]) for row in right_rows}
            ),
            "prompt_sha256": len(
                {str(row["prompt_sha256"]) for row in left_rows}
                & {str(row["prompt_sha256"]) for row in right_rows}
            ),
            "exact_candidate_set": len(
                {
                    tuple(sorted(str(value) for value in row["choices"].values()))
                    for row in left_rows
                }
                & {
                    tuple(sorted(str(value) for value in row["choices"].values()))
                    for row in right_rows
                }
            ),
        }
        if any(overlap[f"{left}__{right}"].values()):
            raise AuditError(f"exact split overlap detected: {left}/{right}")

    public_tasks = _rows(public_tasks_path)
    if len(public_tasks) != 534:
        raise AuditError("public replication task count differs")
    development_anchors = {
        str(row["anchor_ingredient"]) for rows in splits.values() for row in rows
    }
    public_anchors = {str(row["anchor_ingredient"]) for row in public_tasks}
    if development_anchors & public_anchors:
        raise AuditError("lab and public replication anchors overlap")

    reward_rows = {
        split: _rows(lab_data / f"sft_{split}.jsonl") for split in ("train", "validation")
    }
    control_rows = {
        split: _rows(lab_data / f"sft_format_control_{split}.jsonl")
        for split in ("train", "validation")
    }
    control_summary: dict[str, Any] = {}
    for split in ("train", "validation"):
        tasks = {str(row["task_id"]): row for row in splits[split]}
        reward = {str(row["task_id"]): row for row in reward_rows[split]}
        control = {str(row["task_id"]): row for row in control_rows[split]}
        if set(tasks) != set(reward) or set(tasks) != set(control):
            raise AuditError(f"{split} SFT views do not cover the task split exactly")
        for task_id, task in tasks.items():
            if (
                reward[task_id]["prompt"] != task["prompt"]
                or control[task_id]["prompt"] != task["prompt"]
            ):
                raise AuditError(f"{split} SFT prompt differs: {task_id}")
            reward_selection = _selection(reward[task_id]["completion"])
            control_selection = _selection(control[task_id]["completion"])
            if reward_selection != task["optimal_selection"]:
                raise AuditError(f"reward SFT target differs: {task_id}")
            if control_selection == task["optimal_selection"]:
                raise AuditError(f"control SFT preserves an optimum: {task_id}")
            if int(control[task_id]["control_reward_bps"]) != int(
                task["selection_scores_bps"][control_selection]
            ):
                raise AuditError(f"control reward metadata differs: {task_id}")
        for family in PRIMARY_FAMILIES:
            for panel in ("panel_1", "panel_2"):
                ids = {
                    task_id
                    for task_id, task in tasks.items()
                    if task["family"] == family and task["source_panel"] == panel
                }
                if Counter(_selection(reward[task_id]["completion"]) for task_id in ids) != Counter(
                    _selection(control[task_id]["completion"]) for task_id in ids
                ):
                    raise AuditError(f"control label marginals differ: {split}/{family}/{panel}")
        control_scores = [float(row["control_reward_bps"]) / 100 for row in control.values()]
        chance_scores = [float(task["chance_score_bps"]) / 100 for task in tasks.values()]
        control_summary[split] = {
            "rows": len(control),
            "accidental_optima": 0,
            "control_mean_score": mean(control_scores),
            "exact_chance_mean_score": mean(chance_scores),
            "control_minus_chance": mean(control_scores) - mean(chance_scores),
        }

    vocabulary = {
        split: {str(value) for row in rows for value in row["choices"].values()}
        for split, rows in splits.items()
    }
    report: dict[str, Any] = {
        "schema_version": "flavourbench-reward-transfer-data-audit-v1",
        "status": "pass",
        "lab_dataset_artifact_sha256": recorded_manifest_hash,
        "public_tasks_sha256": hashlib.sha256(public_tasks_path.read_bytes()).hexdigest(),
        "counts": {split: len(rows) for split, rows in splits.items()},
        "strata": {
            split: {
                f"{family}/{panel}": count
                for (family, panel), count in sorted(
                    Counter((row["family"], row["source_panel"]) for row in rows).items()
                )
            }
            for split, rows in splits.items()
        },
        "exact_overlap": overlap,
        "public_anchor_overlap": 0,
        "control": control_summary,
        "candidate_vocabulary": {
            "train": len(vocabulary["train"]),
            "validation": len(vocabulary["validation"]),
            "evaluation": len(vocabulary["evaluation"]),
            "train_evaluation_shared": len(vocabulary["train"] & vocabulary["evaluation"]),
        },
        "interpretation": (
            "No exact task, anchor, prompt, or candidate-set leakage was found. Shared individual "
            "candidate ingredients are expected and make the estimand transfer to new anchors, not "
            "generalization to an unseen vocabulary."
        ),
    }
    report["artifact_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-data", type=Path, default=DEFAULT_LAB_DATA)
    parser.add_argument("--public-tasks", type=Path, default=DEFAULT_PUBLIC_TASKS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.lab_data, args.public_tasks)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
