#!/usr/bin/env python3
"""Recompute the Epicure-native FlavourBench leaderboard from the public release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


class ReplayError(RuntimeError):
    """The public release cannot reproduce its leaderboard."""


FINAL_CHOICE_PATTERN = re.compile(r"(?im)^\s*FINAL_CHOICE\s*:\s*([A-D])\s*$")
CONDITIONS = ("epicure_off", "epicure_on")
FAMILIES = ("substitution", "composition", "cookability", "evidence")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ReplayError(f"non-finite JSON number: {value}")


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"release must be a regular file: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_constant,
    )
    if not isinstance(value, dict):
        raise ReplayError("release root is not an object")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _wilson(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [round(max(0.0, centre - radius), 6), round(min(1.0, centre + radius), 6)]


def _parse_final_choice(answer: object) -> str | None:
    matches = FINAL_CHOICE_PATTERN.findall(str(answer or ""))
    return matches[-1].upper() if matches else None


def replay(document: dict[str, Any]) -> list[dict[str, Any]]:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    if len(recorded) != 64 or _sha256(payload) != recorded:
        raise ReplayError("release content address does not verify")
    models = document.get("models")
    tasks = document.get("tasks")
    observations = document.get("observations")
    leaderboard = document.get("leaderboard")
    counts = document.get("counts")
    if not (
        document.get("release_status") == "complete_public_automated_leaderboard"
        and document.get("human_judgments") == 0
        and isinstance(models, list)
        and len(models) == 20
        and isinstance(tasks, list)
        and len(tasks) == 32
        and isinstance(observations, list)
        and len(observations) == 1_280
        and isinstance(leaderboard, dict)
        and leaderboard.get("status") == "complete_automated_leaderboard"
        and leaderboard.get("official_track") is True
        and isinstance(counts, dict)
        and counts.get("models") == 20
        and counts.get("tasks") == 32
        and counts.get("assigned_pairs") == 640
        and counts.get("assigned_arms") == 1_280
    ):
        raise ReplayError("release grid is incomplete")
    model_ids = [str(model["model_id"]) for model in models]
    task_ids = [str(task["task_id"]) for task in tasks]
    if len(set(model_ids)) != 20 or len(set(task_ids)) != 32:
        raise ReplayError("release model or task identity is duplicated")
    models_by_id = {str(model["model_id"]): model for model in models}
    route_expectations = {
        "moonshotai/kimi-k3": ("kimi_direct", "kimi-code-direct", "k3", 64),
        "cohere/command-a-plus-05-2026": (
            "cohere_direct",
            "cohere-direct",
            "command-a-plus-05-2026",
            64,
        ),
        "cohere/command-a-reasoning-08-2025": (
            "cohere_direct",
            "cohere-direct",
            "command-a-reasoning-08-2025",
            64,
        ),
        "qwen/qwen3.8-max": (
            "openrouter",
            "Alibaba",
            "qwen/qwen3.8-max-20260803",
            60,
        ),
        "z-ai/glm-5.2": (
            "openrouter",
            "CoreWeave",
            "z-ai/glm-5.2-20260616",
            63,
        ),
    }
    if not set(route_expectations) <= set(model_ids):
        raise ReplayError("required direct and routed identities are missing")
    for model_id, (
        backend,
        provider,
        returned_id,
        expected_arms,
    ) in route_expectations.items():
        routed_arms = [
            row
            for row in observations
            if row.get("model_id") == model_id and row.get("response_artifact_sha256") is not None
        ]
        if (
            models_by_id[model_id].get("execution_backend") != backend
            or len(routed_arms) != expected_arms
            or {row.get("actual_provider") for row in routed_arms} != {provider}
            or {row.get("actual_model_id") for row in routed_arms} != {returned_id}
        ):
            raise ReplayError(f"route fidelity differs: {model_id}")
    task_family = {str(task["task_id"]): str(task["family"]) for task in tasks}
    task_expected = {str(task["task_id"]): str(task["expected_choice"]) for task in tasks}
    for task in tasks:
        if task.get("reference_tool_result_sha256") != _sha256(task.get("reference_tool_result")):
            raise ReplayError(f"reference result hash differs: {task.get('task_id')}")
    family_counts = defaultdict(int)
    for family in task_family.values():
        family_counts[family] += 1
    if dict(family_counts) != {family: 8 for family in FAMILIES}:
        raise ReplayError("release task families are not balanced")
    observation_keys = [
        (
            str(row.get("model_id") or ""),
            str(row.get("task_id") or ""),
            str(row.get("condition") or ""),
        )
        for row in observations
        if isinstance(row, dict)
    ]
    expected_keys = {
        (model_id, task_id, condition)
        for model_id in model_ids
        for task_id in task_ids
        for condition in CONDITIONS
    }
    if len(observation_keys) != 1_280 or set(observation_keys) != expected_keys:
        raise ReplayError("release observation keys do not form the exact paired grid")
    observed_response_arms = sum(
        row.get("response_artifact_sha256") is not None for row in observations
    )
    if counts.get("observed_response_arms") != observed_response_arms:
        raise ReplayError("observed-response count differs from the release grid")
    rows_by_model = {str(row["model_id"]): row for row in leaderboard.get("models") or []}
    if len(rows_by_model) != 20:
        raise ReplayError("published leaderboard model membership is incomplete")
    results: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["model_id"])
        selected = [row for row in observations if row.get("model_id") == model_id]
        if len(selected) != 64:
            raise ReplayError(f"model observation count differs from 64: {model_id}")
        conditions: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            arms = [row for row in selected if row.get("condition") == condition]
            if len(arms) != 32:
                raise ReplayError(f"condition count differs from 32: {model_id}/{condition}")
            family_values: dict[str, list[int]] = defaultdict(list)
            for arm in arms:
                expected = task_expected[str(arm["task_id"])]
                observed = _parse_final_choice(arm.get("answer_markdown"))
                normal = arm.get("finish_reason") == "stop"
                parseable = observed is not None and normal
                correct = observed == expected and normal
                if (
                    arm.get("expected_choice") != expected
                    or arm.get("observed_choice") != observed
                    or arm.get("parseable_normal_completion") is not parseable
                    or arm.get("correct") is not correct
                ):
                    raise ReplayError(
                        "released answer projection differs from recomputation: "
                        f"{model_id}/{arm.get('task_id')}/{condition}"
                    )
                family_values[task_family[str(arm["task_id"])]].append(int(correct))
            family_accuracy = {
                family: round(sum(values) / len(values), 6)
                for family, values in sorted(family_values.items())
            }
            correct = sum(
                int(
                    _parse_final_choice(arm.get("answer_markdown"))
                    == task_expected[str(arm["task_id"])]
                    and arm.get("finish_reason") == "stop"
                )
                for arm in arms
            )
            parseable = sum(
                int(
                    _parse_final_choice(arm.get("answer_markdown")) is not None
                    and arm.get("finish_reason") == "stop"
                )
                for arm in arms
            )
            conditions[condition] = {
                "correct": correct,
                "attempted_tasks": 32,
                "macro_accuracy": round(sum(family_accuracy.values()) / 4, 6),
                "accuracy_percent": round(100 * correct / 32, 3),
                "wilson_95": _wilson(correct, 32),
                "family_accuracy": family_accuracy,
                "reliability": round(parseable / 32, 6),
            }
        result = {
            "model_id": model_id,
            "epicure_benchmark_score": round(100 * conditions["epicure_off"]["macro_accuracy"], 3),
            "uplift_percentage_points": round(
                100
                * (
                    conditions["epicure_on"]["macro_accuracy"]
                    - conditions["epicure_off"]["macro_accuracy"]
                ),
                3,
            ),
            "conditions": conditions,
        }
        published = rows_by_model.get(model_id)
        if not isinstance(published, dict):
            raise ReplayError(f"leaderboard row missing: {model_id}")
        for key in ("epicure_benchmark_score", "uplift_percentage_points"):
            if published.get(key) != result[key]:
                raise ReplayError(f"published {key} differs: {model_id}")
        for condition, metrics in conditions.items():
            published_metrics = published.get("conditions", {}).get(condition, {})
            for key in (
                "correct",
                "attempted_tasks",
                "macro_accuracy",
                "accuracy_percent",
                "wilson_95",
                "family_accuracy",
                "reliability",
            ):
                if published_metrics.get(key) != metrics[key]:
                    raise ReplayError(f"published {condition}/{key} differs: {model_id}")
        results.append(result)
    results.sort(
        key=lambda item: (
            -float(item["epicure_benchmark_score"]),
            -float(item["conditions"]["epicure_on"]["accuracy_percent"]),
            -float(item["conditions"]["epicure_on"]["reliability"]),
            item["model_id"],
        )
    )
    published_rows = leaderboard.get("models") or []
    if [row.get("rank") for row in published_rows] != list(range(1, 21)) or [
        row.get("model_id") for row in published_rows
    ] != [row["model_id"] for row in results]:
        raise ReplayError("published rank order differs from recomputation")
    for rank, result in enumerate(results, start=1):
        if rows_by_model[result["model_id"]].get("rank") != rank:
            raise ReplayError(f"published rank differs: {result['model_id']}")
    aggregate = leaderboard.get("aggregate") or {}
    off_total = sum(item["conditions"]["epicure_off"]["correct"] for item in results)
    on_total = sum(item["conditions"]["epicure_on"]["correct"] for item in results)
    if (
        aggregate.get("epicure_off_correct") != off_total
        or aggregate.get("epicure_on_correct") != on_total
        or aggregate.get("attempted_pairs") != 640
        or aggregate.get("uplift_percentage_points") != round(100 * (on_total - off_total) / 640, 3)
    ):
        raise ReplayError("published aggregate differs from recomputation")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("generated/epicure-native/epicure-native-release.json"),
    )
    args = parser.parse_args()
    document = _read(args.release)
    results = replay(document)
    print(
        json.dumps(
            {
                "status": "verified",
                "release_artifact_sha256": document["artifact_sha256"],
                "models": len(results),
                "tasks_per_model": 32,
                "assigned_arms": 1_280,
                "top_score": results[0]["epicure_benchmark_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
