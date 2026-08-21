"""Small, dependency-free scoring surface for the Hugging Face Space."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

REPORT_SCHEMA_VERSION = "flavourbench-lab-report-v1"
PRIMARY_FAMILIES = ("substitution", "pairing", "constraint")
_MARKER = re.compile(r"FINAL_SELECTION\s*:\s*", flags=re.IGNORECASE)
_LABEL_TRIPLE = re.compile(
    r"^\s*([A-H])\s*,\s*([A-H])\s*,\s*([A-H])"
    r"\s*(?:[\x60*_]+\s*)?(?:<\|close\|>response\s*)?$",
    flags=re.IGNORECASE,
)


class SpaceLabError(ValueError):
    """The uploaded lab artifact is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _normal_name(value: str) -> str:
    return " ".join(value.replace("_", " ").casefold().split())


def _parse(task: Mapping[str, Any], completion: str) -> str | None:
    if len(completion.encode()) > 1024 * 1024:
        raise SpaceLabError("completion exceeds the 1 MiB endpoint limit")
    choices = task.get("choices") or {}
    if set(choices) != set("ABCDEFGH"):
        return None
    names = {_normal_name(str(name)): str(label) for label, name in choices.items()}
    if len(names) != len(choices):
        return None
    matches = tuple(_MARKER.finditer(completion))
    candidates: set[str] = set()
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(completion)
        segment = completion[match.end() : stop].splitlines()[0].strip()
        label_match = _LABEL_TRIPLE.fullmatch(segment)
        if label_match:
            labels = tuple(label.upper() for label in label_match.groups())
            if len(set(labels)) == 3:
                candidates.add("".join(sorted(labels)))
            continue
        rendered = segment.strip().strip("*_").strip(chr(96)).strip()
        ingredients = tuple(_normal_name(value) for value in rendered.split(","))
        if len(ingredients) == 3 and all(value in names for value in ingredients):
            labels = tuple(names[value] for value in ingredients)
            if len(set(labels)) == 3:
                candidates.add("".join(sorted(labels)))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _extract(record: Mapping[str, Any]) -> str | None:
    if str(record.get("status") or "completed") not in {"completed", "success", "ok"}:
        return None
    for key in ("response", "completion", "answer", "answer_markdown"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return None


def _records(payload: str) -> list[dict[str, Any]]:
    if len(payload.encode()) > 16 * 1024 * 1024:
        raise SpaceLabError("artifact exceeds the 16 MiB Space limit")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        output = []
        for number, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SpaceLabError(f"invalid JSON on line {number}") from error
            if not isinstance(row, dict):
                raise SpaceLabError(f"line {number} is not an object") from None
            output.append(dict(row))
        return output
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return [dict(row) for row in value]
    if isinstance(value, dict) and isinstance(value.get("responses"), list):
        rows = value["responses"]
        if all(isinstance(row, dict) for row in rows):
            return [dict(row) for row in rows]
    if isinstance(value, dict):
        return [dict(value)]
    raise SpaceLabError("artifact must be JSON Lines, an array, or an object with responses")


def score_completion(
    tasks_by_id: Mapping[str, Mapping[str, Any]], task_id: str, completion: str
) -> dict[str, Any]:
    task = tasks_by_id.get(task_id)
    if task is None:
        raise SpaceLabError(f"unknown task_id: {task_id}")
    selection = _parse(task, completion)
    score_bps = int(task["selection_scores_bps"].get(selection, 0)) if selection else 0
    return {
        "task_id": task_id,
        "observed_selection": selection,
        "parseable": selection is not None,
        "score_bps": score_bps,
        "score": score_bps / 100,
        "reward": score_bps / 10_000,
        "optimal": score_bps == 10_000,
        "optimal_selection": task["optimal_selection"],
    }


def score_payload(
    tasks: Sequence[Mapping[str, Any]], payload: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    responses = _records(payload)
    task_by_id = {str(task["task_id"]): task for task in tasks}
    response_by_id: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(responses, start=1):
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise SpaceLabError(f"response {number} has no task_id")
        if task_id not in task_by_id:
            raise SpaceLabError(f"unknown task_id: {task_id}")
        if task_id in response_by_id:
            raise SpaceLabError(f"duplicate task_id: {task_id}")
        response_by_id[task_id] = row

    per_task = []
    for task in tasks:
        task_id = str(task["task_id"])
        response = response_by_id.get(task_id)
        completion = _extract(response) if response is not None else None
        scoring = (
            score_completion(task_by_id, task_id, completion)
            if completion is not None
            else {
                "task_id": task_id,
                "observed_selection": None,
                "parseable": False,
                "score_bps": 0,
                "score": 0.0,
                "reward": 0.0,
                "optimal": False,
                "optimal_selection": task["optimal_selection"],
            }
        )
        per_task.append(
            {
                "family": task["family"],
                "anchor_ingredient": task.get("anchor_ingredient"),
                "status": (
                    "valid"
                    if scoring["parseable"]
                    else "missing"
                    if response is None
                    else "invalid"
                ),
                **scoring,
            }
        )

    valid = sum(row["parseable"] is True for row in per_task)
    complete = valid == len(tasks)
    family_rows = []
    for family in PRIMARY_FAMILIES:
        rows = [row for row in per_task if row["family"] == family]
        accepted = [row for row in rows if row["parseable"]]
        family_rows.append(
            {
                "family": family,
                "tasks": len(rows),
                "valid": len(accepted),
                "coverage": len(accepted) / len(rows),
                "score": (
                    sum(float(row["score"]) for row in accepted) / len(accepted)
                    if rows and len(accepted) == len(rows)
                    else None
                ),
                "diagnostic_valid_score": (
                    sum(float(row["score"]) for row in accepted) / len(accepted)
                    if accepted
                    else None
                ),
            }
        )
    comparable_score = (
        sum(float(row["score"]) for row in family_rows) / len(PRIMARY_FAMILIES)
        if complete
        else None
    )
    diagnostic_families = [
        float(row["diagnostic_valid_score"])
        for row in family_rows
        if row["diagnostic_valid_score"] is not None
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "comparable": complete,
        "flavourbench_score": comparable_score,
        "diagnostic_valid_score": (
            sum(diagnostic_families) / len(diagnostic_families) if diagnostic_families else None
        ),
        "coverage": {
            "tasks": len(tasks),
            "submitted": len(response_by_id),
            "valid": valid,
            "missing": sum(row["status"] == "missing" for row in per_task),
            "invalid": sum(row["status"] == "invalid" for row in per_task),
            "fraction_valid": valid / len(tasks),
        },
        "families": family_rows,
        "task_set_semantic_sha256": hashlib.sha256(
            _canonical(sorted((dict(task) for task in tasks), key=lambda row: str(row["task_id"])))
        ).hexdigest(),
        "response_set_semantic_sha256": hashlib.sha256(
            _canonical(sorted(responses, key=lambda row: str(row.get("task_id"))))
        ).hexdigest(),
        "parser": "flavourbench-selection-set-parser-v3",
        "scoring": "exact-selection-lookup-bps-v1; equal-family macro mean",
        "inference": None,
        "per_task": per_task,
    }
    report["artifact_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report, per_task


def score_payload_json(tasks: Sequence[Mapping[str, Any]], payload: str) -> dict[str, Any]:
    """Convenience wrapper used by the named Gradio batch endpoint."""

    report, _ = score_payload(tasks, payload)
    return report
