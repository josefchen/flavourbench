"""Public, provider-neutral FlavourBench evaluation and reward SDK.

The lab API deliberately depends on released task records rather than private Epicure
infrastructure.  A task record carries the complete, precomputed reward map needed to score a
completion locally.  This makes evaluation deterministic and keeps provider credentials out of
FlavourBench artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .selection_response_parser_v3 import parse_final_selection_v3, score_answer_v3

LAB_REPORT_SCHEMA_VERSION = "flavourbench-lab-report-v1"
LAB_RESPONSE_SCHEMA_VERSION = "flavourbench-lab-response-v1"
PRIMARY_FAMILIES = ("substitution", "pairing", "constraint")
EXPECTED_SELECTIONS = frozenset("".join(labels) for labels in combinations("ABCDEFGH", 3))
DEFAULT_DATASET_REPO = "josefchen/flavourbench"
DEFAULT_EVALUATION_FILE = "data-complete-core/tasks.jsonl"


class LabValidationError(ValueError):
    """A task set or response artifact violates the public lab contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: object) -> str:
    """Return the SHA-256 of canonical JSON bytes."""

    return hashlib.sha256(_canonical(value)).hexdigest()


def verify_report(report: Mapping[str, Any]) -> str:
    """Verify a content-addressed lab report and its internal coverage invariants."""

    payload = dict(report)
    recorded = str(payload.pop("artifact_sha256", ""))
    if not recorded or recorded != semantic_sha256(payload):
        raise LabValidationError("report semantic digest failed")
    if payload.get("schema_version") != LAB_REPORT_SCHEMA_VERSION:
        raise LabValidationError("report schema version differs")
    coverage = payload.get("coverage")
    per_task = payload.get("per_task")
    if not isinstance(coverage, Mapping) or not isinstance(per_task, list):
        raise LabValidationError("report coverage or per-task table is absent")
    tasks = int(coverage.get("tasks", -1))
    valid = sum(isinstance(row, Mapping) and row.get("parseable") is True for row in per_task)
    missing = sum(isinstance(row, Mapping) and row.get("status") == "missing" for row in per_task)
    invalid = sum(isinstance(row, Mapping) and row.get("status") == "invalid" for row in per_task)
    if (
        tasks != len(per_task)
        or int(coverage.get("submitted", -1)) != valid + invalid
        or int(coverage.get("valid", -1)) != valid
        or int(coverage.get("missing", -1)) != missing
        or int(coverage.get("invalid", -1)) != invalid
        or valid + missing + invalid != tasks
    ):
        raise LabValidationError("report coverage does not match its per-task table")
    fraction = valid / tasks if tasks else 0.0
    if abs(float(coverage.get("fraction_valid", -1.0)) - fraction) > 1e-12:
        raise LabValidationError("report coverage fraction differs")
    comparable = valid == tasks
    if bool(payload.get("comparable")) != comparable or (
        (payload.get("flavourbench_score") is not None) != comparable
    ):
        raise LabValidationError("report score eligibility and coverage differ")
    families = payload.get("families")
    if not isinstance(families, list) or {
        str(row.get("family")) for row in families if isinstance(row, Mapping)
    } != set(PRIMARY_FAMILIES):
        raise LabValidationError("report family table differs")
    family_rows = {str(row["family"]): row for row in families}

    def same_optional_number(left: Any, right: float | None) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return abs(float(left) - right) <= 1e-12

    for family in PRIMARY_FAMILIES:
        task_rows = [row for row in per_task if row.get("family") == family]
        if not task_rows:
            raise LabValidationError("report contains an empty primary family")
        accepted = [row for row in task_rows if row.get("parseable") is True]
        reported = family_rows[family]
        if (
            int(reported.get("tasks", -1)) != len(task_rows)
            or int(reported.get("valid", -1)) != len(accepted)
            or abs(float(reported.get("coverage", -1.0)) - len(accepted) / len(task_rows)) > 1e-12
        ):
            raise LabValidationError("report family coverage differs")
        expected_score = (
            float(np.mean([float(row["score"]) for row in accepted])) if accepted else None
        )
        if not same_optional_number(reported.get("diagnostic_valid_score"), expected_score):
            raise LabValidationError("report family diagnostic score differs")
        complete_family_score = expected_score if len(accepted) == len(task_rows) else None
        if not same_optional_number(reported.get("score"), complete_family_score):
            raise LabValidationError("report family score differs")
    expected_score = (
        float(np.mean([float(family_rows[family]["score"]) for family in PRIMARY_FAMILIES]))
        if comparable
        else None
    )
    if not same_optional_number(payload.get("flavourbench_score"), expected_score):
        raise LabValidationError("report FlavourBench Score differs")
    return recorded


def read_json_records(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSON array/object or JSON Lines file into object records."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise LabValidationError(f"not a regular input file: {source}")
    if source.stat().st_size > 128 * 1024 * 1024:
        raise LabValidationError("input exceeds the 128 MiB lab-artifact limit")
    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise LabValidationError(f"invalid JSON on line {line_number}") from error
            if not isinstance(row, dict):
                raise LabValidationError(f"line {line_number} is not a JSON object") from None
            records.append(dict(row))
        return records
    if isinstance(value, list):
        if not all(isinstance(row, dict) for row in value):
            raise LabValidationError("JSON array contains a non-object row")
        return [dict(row) for row in value]
    if isinstance(value, dict):
        nested = value.get("responses") or value.get("tasks") or value.get("records")
        if isinstance(nested, list) and all(isinstance(row, dict) for row in nested):
            return [dict(row) for row in nested]
        return [dict(value)]
    raise LabValidationError("JSON input must be an object or array")


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write canonical JSON Lines in caller-provided order."""

    destination = Path(path)
    if destination.is_symlink():
        raise LabValidationError(f"refusing to replace symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical(dict(row)) + b"\n" for row in rows)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_hub_tasks(
    *,
    repo_id: str = DEFAULT_DATASET_REPO,
    filename: str = DEFAULT_EVALUATION_FILE,
    revision: str | None = None,
    token: str | bool | None = None,
) -> list[dict[str, Any]]:
    """Download and validate a task table from the Hugging Face dataset repository."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - installation guard
        raise LabValidationError(
            "Hub loading requires `pip install huggingface-hub` or a local --tasks file"
        ) from error
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    tasks = read_json_records(path)
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed unless every task contains a valid 56-action reward surface."""

    if not tasks:
        raise LabValidationError("task set is empty")
    seen_ids: set[str] = set()
    for index, task in enumerate(tasks):
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id in seen_ids:
            raise LabValidationError(f"task {index} has a missing or duplicate task_id")
        seen_ids.add(task_id)
        choices = task.get("choices")
        if not isinstance(choices, Mapping) or set(choices) != set("ABCDEFGH"):
            raise LabValidationError(f"{task_id} does not contain choices A through H")
        scores = task.get("selection_scores_bps")
        if not isinstance(scores, Mapping) or set(scores) != EXPECTED_SELECTIONS:
            raise LabValidationError(f"{task_id} does not contain all 56 portfolio scores")
        if any(
            len(str(selection)) != 3
            or len(set(str(selection))) != 3
            or not set(str(selection)) <= set("ABCDEFGH")
            or not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 10_000
            for selection, score in scores.items()
        ):
            raise LabValidationError(f"{task_id} has an invalid reward-map entry")
        normalized_names = {
            " ".join(str(name).replace("_", " ").casefold().split()) for name in choices.values()
        }
        if len(normalized_names) != 8:
            raise LabValidationError(f"{task_id} has duplicate normalized ingredient choices")
        optimum = str(task.get("optimal_selection") or "")
        if scores.get(optimum) != 10_000:
            raise LabValidationError(f"{task_id} does not bind its declared optimum")
        if str(task.get("family") or "") not in (*PRIMARY_FAMILIES, "cultural_composition"):
            raise LabValidationError(f"{task_id} has an unknown task family")
        chance = task.get("chance_score_bps")
        expected_chance = round(sum(int(score) for score in scores.values()) / len(scores))
        if not isinstance(chance, int) or isinstance(chance, bool) or chance != expected_chance:
            raise LabValidationError(f"{task_id} has an invalid exact-chance baseline")


def extract_answer(record: Mapping[str, Any]) -> str | None:
    """Extract answer text from the compact lab schema or a released response document."""

    status = str(record.get("status") or "completed")
    if status not in {"completed", "success", "ok"}:
        return None
    for key in ("response", "completion", "answer", "answer_markdown"):
        value = record.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            nested = value.get("answer_markdown")
            if isinstance(nested, str):
                return nested
            generation = value.get("generation")
            if isinstance(generation, Mapping) and isinstance(
                generation.get("answer_markdown"), str
            ):
                return str(generation["answer_markdown"])
    generation = record.get("generation")
    if isinstance(generation, Mapping) and isinstance(generation.get("answer_markdown"), str):
        return str(generation["answer_markdown"])
    return None


def reward(task: Mapping[str, Any], completion: str) -> float:
    """Return the deterministic Epicure reward in [0, 1] for one completion."""

    return float(score_answer_v3(task, completion)["score_bps"]) / 10_000.0


def reward_bps(task: Mapping[str, Any], completion: str) -> int:
    """Return the deterministic Epicure reward in integer basis points."""

    return int(score_answer_v3(task, completion)["score_bps"])


def _completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[-1], Mapping):
        content = value[-1].get("content")
        if isinstance(content, str):
            return content
    if isinstance(value, Mapping) and isinstance(value.get("content"), str):
        return str(value["content"])
    return ""


def trl_reward(
    completions: Sequence[Any],
    selection_scores_bps: Sequence[Mapping[str, int]],
    choices: Sequence[Mapping[str, str]],
    **_: Any,
) -> list[float]:
    """TRL-compatible dense reward function for GRPO/RLOO training.

    The GRPO dataset includes ``choices`` and ``selection_scores_bps`` columns. TRL passes those
    columns to this function together with generated completions, so training requires no network
    call and never touches the official leaderboard task set.
    """

    if not (len(completions) == len(selection_scores_bps) == len(choices)):
        raise LabValidationError("TRL reward inputs have different batch lengths")
    values: list[float] = []
    for completion, score_map, task_choices in zip(
        completions, selection_scores_bps, choices, strict=True
    ):
        task = {"choices": task_choices, "selection_scores_bps": score_map}
        selection = parse_final_selection_v3(task, _completion_text(completion))
        values.append(float(score_map.get(selection, 0)) / 10_000.0 if selection else 0.0)
    return values


def _family_macro(values: Sequence[float], families: Sequence[str]) -> float:
    means: list[float] = []
    array = np.asarray(values, dtype=np.float64)
    family_array = np.asarray(families, dtype=object)
    for family in PRIMARY_FAMILIES:
        indices = np.flatnonzero(family_array == family)
        if not len(indices):
            raise LabValidationError(f"score set is missing the {family} family")
        means.append(float(array[indices].mean()))
    return float(np.mean(means))


def _bootstrap_interval(
    scores: Sequence[float],
    families: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    array = np.asarray(scores, dtype=np.float64)
    family_array = np.asarray(families, dtype=object)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    indices = [np.flatnonzero(family_array == family) for family in PRIMARY_FAMILIES]
    for start in range(0, resamples, 1000):
        width = min(1000, resamples - start)
        batch = np.zeros(width, dtype=np.float64)
        for family_indices in indices:
            draws = rng.integers(0, len(family_indices), size=(width, len(family_indices)))
            batch += array[family_indices[draws]].mean(axis=1) / len(PRIMARY_FAMILIES)
        samples[start : start + width] = batch
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _sign_flip_pvalue(
    differences: Sequence[float], families: Sequence[str], *, resamples: int, seed: int
) -> float:
    array = np.asarray(differences, dtype=np.float64)
    family_array = np.asarray(families, dtype=object)
    family_indices = [np.flatnonzero(family_array == family) for family in PRIMARY_FAMILIES]
    observed = abs(_family_macro(differences, families))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, resamples, 2000):
        width = min(2000, resamples - start)
        signs = rng.integers(0, 2, size=(width, len(array)), dtype=np.int8) * 2 - 1
        signed = signs * array
        null = sum(
            signed[:, indices].mean(axis=1) / len(PRIMARY_FAMILIES) for indices in family_indices
        )
        exceed += int(np.count_nonzero(np.abs(null) >= observed - 1e-12))
    return float((exceed + 1) / (resamples + 1))


def score_submission(
    tasks: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    *,
    include_inference: bool = True,
    bootstrap_resamples: int = 50_000,
    sign_flip_resamples: int = 100_000,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Score a response artifact and return a self-contained lab report.

    A comparable FlavourBench Score is emitted only when all tasks are present and parseable.
    Incomplete runs retain coverage and per-task diagnostics but cannot be presented as leaderboard
    results.
    """

    if bootstrap_resamples <= 0 or sign_flip_resamples <= 0:
        raise LabValidationError("inference resample counts must be positive")
    validate_tasks(tasks)
    if any(str(task["family"]) not in PRIMARY_FAMILIES for task in tasks):
        raise LabValidationError("submission scoring accepts only the three primary families")
    task_by_id = {str(task["task_id"]): task for task in tasks}
    response_by_id: dict[str, Mapping[str, Any]] = {}
    for index, response in enumerate(responses):
        task_id = str(response.get("task_id") or "")
        if not task_id:
            raise LabValidationError(f"response {index} has no task_id")
        if task_id not in task_by_id:
            raise LabValidationError(f"response references unknown task_id: {task_id}")
        if task_id in response_by_id:
            raise LabValidationError(f"response task_id is duplicated: {task_id}")
        response_by_id[task_id] = response

    per_task: list[dict[str, Any]] = []
    valid_scores: list[float] = []
    valid_families: list[str] = []
    complete_scores: list[float] = []
    complete_families: list[str] = []
    chance_scores: list[float] = []
    for task in tasks:
        task_id = str(task["task_id"])
        family = str(task["family"])
        response = response_by_id.get(task_id)
        answer = extract_answer(response) if response is not None else None
        scoring = (
            score_answer_v3(task, answer)
            if answer is not None
            else {
                "observed_selection": None,
                "optimal_selection": task["optimal_selection"],
                "parseable": False,
                "score_bps": 0,
                "score": 0.0,
                "optimal": False,
            }
        )
        status = "valid" if scoring["parseable"] else ("missing" if response is None else "invalid")
        row = {
            "task_id": task_id,
            "family": family,
            "anchor_ingredient": task.get("anchor_ingredient"),
            "status": status,
            **scoring,
        }
        per_task.append(row)
        if scoring["parseable"]:
            valid_scores.append(float(scoring["score"]))
            valid_families.append(family)
        complete_scores.append(float(scoring["score"]))
        complete_families.append(family)
        chance_scores.append(float(task["chance_score_bps"]) / 100.0)

    valid_count = sum(row["parseable"] is True for row in per_task)
    missing_count = sum(row["status"] == "missing" for row in per_task)
    invalid_count = sum(row["status"] == "invalid" for row in per_task)
    complete = valid_count == len(tasks)
    family_rows: list[dict[str, Any]] = []
    for family in PRIMARY_FAMILIES:
        rows = [row for row in per_task if row["family"] == family]
        valid = [row for row in rows if row["parseable"]]
        family_rows.append(
            {
                "family": family,
                "tasks": len(rows),
                "valid": len(valid),
                "coverage": len(valid) / len(rows) if rows else 0.0,
                "score": (
                    float(np.mean([row["score"] for row in valid]))
                    if rows and len(valid) == len(rows)
                    else None
                ),
                "diagnostic_valid_score": (
                    float(np.mean([row["score"] for row in valid])) if valid else None
                ),
            }
        )

    diagnostic_score = None
    if valid_scores and set(PRIMARY_FAMILIES) <= set(valid_families):
        diagnostic_score = _family_macro(valid_scores, valid_families)
    exact_chance_score = _family_macro(chance_scores, complete_families)
    report: dict[str, Any] = {
        "schema_version": LAB_REPORT_SCHEMA_VERSION,
        "comparable": complete,
        "flavourbench_score": _family_macro(complete_scores, complete_families)
        if complete
        else None,
        "diagnostic_valid_score": diagnostic_score,
        "coverage": {
            "tasks": len(tasks),
            "submitted": len(response_by_id),
            "valid": valid_count,
            "missing": missing_count,
            "invalid": invalid_count,
            "fraction_valid": valid_count / len(tasks),
        },
        "families": family_rows,
        "exact_chance_score": exact_chance_score,
        "task_set_semantic_sha256": semantic_sha256(
            sorted((dict(task) for task in tasks), key=lambda row: str(row["task_id"]))
        ),
        "response_set_semantic_sha256": semantic_sha256(
            sorted((dict(row) for row in responses), key=lambda row: str(row.get("task_id")))
        ),
        "parser": "flavourbench-selection-set-parser-v3",
        "scoring": "exact-selection-lookup-bps-v1; equal-family macro mean",
        "per_task": per_task,
    }
    if complete and include_inference:
        anchors = [str(task.get("anchor_ingredient") or task["task_id"]) for task in tasks]
        if len(set(anchors)) != len(anchors):
            raise LabValidationError("inference requires one task per independent anchor")
        interval = _bootstrap_interval(
            complete_scores,
            complete_families,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        differences = np.asarray(complete_scores) - np.asarray(chance_scores)
        report["inference"] = {
            "confidence_interval_95": interval,
            "bootstrap_resamples": bootstrap_resamples,
            "mean_difference_from_exact_chance": _family_macro(
                differences.tolist(), complete_families
            ),
            "paired_sign_flip_p": _sign_flip_pvalue(
                differences,
                complete_families,
                resamples=sign_flip_resamples,
                seed=seed + 1,
            ),
            "sign_flip_resamples": sign_flip_resamples,
            "independence_unit": "anchor_ingredient",
            "note": "Single-model interval; not a simultaneous multi-model rank interval.",
        }
    else:
        report["inference"] = None
    report["artifact_sha256"] = semantic_sha256(report)
    return report
