"""Freeze balanced Season 0 model-arena and Epicure-uplift comparisons."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json, sha256_text
from .season0_costs import _latest_arms, _load

SCHEMA_VERSION = "flavourbench-season0-comparison-manifest-v1"


class ComparisonManifestError(RuntimeError):
    """Scored response arms cannot support the frozen comparison design."""


IDENTITY_LEAK_PATTERNS = {
    "epicure_condition": re.compile(r"\bepicure(?:\s+mcp)?\b", re.IGNORECASE),
    "provider_route": re.compile(
        r"\b(?:amazon\s+bedrock|openrouter|google\s+vertex|google\s+ai\s+studio)\b",
        re.IGNORECASE,
    ),
    "model_name": re.compile(
        r"\b(?:claude|gpt[-\s]?5(?:\.6)?|gemini\s*3|qwen\s*3|devstral|"
        r"minimax\s*m2|nova\s*2|gpt[-\s]?oss)\b",
        re.IGNORECASE,
    ),
}


def identity_leak_tags(answer: str) -> list[str]:
    return [tag for tag, pattern in IDENTITY_LEAK_PATTERNS.items() if pattern.search(answer)]


def round_robin_rounds(model_ids: Sequence[str]) -> list[list[tuple[str, str]]]:
    ordered = list(model_ids)
    if len(ordered) < 2 or len(ordered) % 2:
        raise ComparisonManifestError("round robin requires an even model count")
    anchor = ordered[0]
    rotating = ordered[1:]
    rounds: list[list[tuple[str, str]]] = []
    for _ in range(len(ordered) - 1):
        lineup = [anchor, *rotating]
        rounds.append([(lineup[index], lineup[-(index + 1)]) for index in range(len(ordered) // 2)])
        rotating = [rotating[-1], *rotating[:-1]]
    pairs = [tuple(sorted(pair)) for round_pairs in rounds for pair in round_pairs]
    if len(set(pairs)) != len(pairs):
        raise ComparisonManifestError("round-robin schedule repeated a model pair")
    return rounds


def _side_order(pair_id_seed: str, first: str, second: str) -> tuple[str, str]:
    return (first, second) if int(sha256_text(pair_id_seed), 16) % 2 == 0 else (second, first)


def _arm_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    answer = result.get("answer_markdown") if isinstance(result, Mapping) else None
    leak_tags = identity_leak_tags(answer) if isinstance(answer, str) else []
    return {
        "arm_id": row["arm_id"],
        "arm_artifact_sha256": row["artifact_sha256"],
        "season_model_id": row["model"]["season_model_id"],
        "condition": row["condition"],
        "status": row["status"],
        "delivery_state": row["delivery_state"],
        "answer_sha256": sha256_text(answer) if isinstance(answer, str) and answer else None,
        "identity_leak_tags": leak_tags,
    }


def freeze_comparisons(
    *,
    arms_dir: Path,
    task_bank: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    arms = _latest_arms(arms_dir)
    manifest_sha = str(model_manifest["artifact_sha256"])
    tasks = list(task_bank["tasks"])
    models = list(model_manifest["models"])
    if len(tasks) != 120 or len(models) != 12 or len(arms) != 2_880:
        raise ComparisonManifestError("dense scored design requires 120 tasks and 2,880 arms")
    lookup: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for arm in arms:
        if arm.get("phase") != "scored" or arm.get("synthetic") is not False:
            raise ComparisonManifestError("comparison input includes a non-scored or synthetic arm")
        contracts = arm.get("contracts")
        if (
            not isinstance(contracts, Mapping)
            or contracts.get("model_manifest_artifact_sha256") != manifest_sha
        ):
            raise ComparisonManifestError("scored arm is bound to another model manifest")
        key = (
            str(arm["task"]["task_id"]),
            str(arm["model"]["season_model_id"]),
            str(arm["condition"]),
        )
        if key in lookup:
            raise ComparisonManifestError("duplicate task/model/condition arm")
        lookup[key] = arm

    model_ids = [str(model["season_model_id"]) for model in models]
    schedule = round_robin_rounds(model_ids)
    comparisons: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        for first_model, second_model in schedule[task_index % len(schedule)]:
            first = lookup[(task_id, first_model, "epicure_on")]
            second = lookup[(task_id, second_model, "epicure_on")]
            left_id, right_id = _side_order(
                f"arena:{task_id}:{first_model}:{second_model}", first_model, second_model
            )
            by_model = {first_model: first, second_model: second}
            left = by_model[left_id]
            right = by_model[right_id]
            execution_eligible = all(
                arm.get("status") == "success"
                and arm.get("delivery_state") == "reconciled"
                and arm.get("rank_eligible") is True
                for arm in (left, right)
            )
            leak_detected = any(
                identity_leak_tags(str((arm.get("result") or {}).get("answer_markdown") or ""))
                for arm in (left, right)
            )
            judgable = execution_eligible and not leak_detected
            identity = {
                "track": "model_arena",
                "task_id": task_id,
                "left_arm_id": left["arm_id"],
                "right_arm_id": right["arm_id"],
                "schedule_round": task_index % len(schedule),
            }
            comparisons.append(
                {
                    "comparison_id": sha256_json(identity),
                    **identity,
                    "task_family": task["family"],
                    "task_sha256": task["task_sha256"],
                    "left": _arm_ref(left),
                    "right": _arm_ref(right),
                    "judgable": judgable,
                    "exclusion_reason": (
                        None
                        if judgable
                        else "identity_leak"
                        if execution_eligible and leak_detected
                        else "one_or_both_arms_failed"
                    ),
                }
            )
        for model_id in model_ids:
            off = lookup[(task_id, model_id, "epicure_off")]
            on = lookup[(task_id, model_id, "epicure_on")]
            left_condition, right_condition = _side_order(
                f"uplift:{task_id}:{model_id}", "epicure_off", "epicure_on"
            )
            by_condition = {"epicure_off": off, "epicure_on": on}
            left = by_condition[left_condition]
            right = by_condition[right_condition]
            execution_eligible = all(
                arm.get("status") == "success"
                and arm.get("delivery_state") == "reconciled"
                and arm.get("rank_eligible") is True
                for arm in (left, right)
            )
            leak_detected = any(
                identity_leak_tags(str((arm.get("result") or {}).get("answer_markdown") or ""))
                for arm in (left, right)
            )
            judgable = execution_eligible and not leak_detected
            identity = {
                "track": "epicure_uplift",
                "task_id": task_id,
                "left_arm_id": left["arm_id"],
                "right_arm_id": right["arm_id"],
                "season_model_id": model_id,
            }
            comparisons.append(
                {
                    "comparison_id": sha256_json(identity),
                    **identity,
                    "task_family": task["family"],
                    "task_sha256": task["task_sha256"],
                    "left": _arm_ref(left),
                    "right": _arm_ref(right),
                    "judgable": judgable,
                    "exclusion_reason": (
                        None
                        if judgable
                        else "identity_leak"
                        if execution_eligible and leak_detected
                        else "one_or_both_arms_failed"
                    ),
                }
            )

    arena = [row for row in comparisons if row["track"] == "model_arena"]
    uplift = [row for row in comparisons if row["track"] == "epicure_uplift"]
    exposure = Counter(
        arm["season_model_id"] for row in arena for arm in (row["left"], row["right"])
    )
    pair_counts = Counter(
        tuple(sorted((row["left"]["season_model_id"], row["right"]["season_model_id"])))
        for row in arena
    )
    arena_left_exposure = Counter(str(row["left"]["season_model_id"]) for row in arena)
    uplift_on_left = Counter(
        str(row["season_model_id"]) for row in uplift if row["left"]["condition"] == "epicure_on"
    )
    if set(exposure.values()) != {120} or set(pair_counts.values()) - {10, 11}:
        raise ComparisonManifestError("model-arena exposure schedule is not balanced")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "season": "Season 0",
        "synthetic_comparisons": 0,
        "task_bank_artifact_sha256": task_bank["artifact_sha256"],
        "model_manifest_artifact_sha256": manifest_sha,
        "model_set_sha256": model_manifest["model_set_sha256"],
        "design": {
            "model_arena": "one perfect-matching round per task from an 11-round round robin",
            "epicure_uplift": "all same-model off/on pairs",
            "side_assignment": "deterministic cryptographic 1/2 randomization",
            "failed_arm_policy": "no preference accepted; retain in reliability metrics",
        },
        "counts": {
            "comparisons": len(comparisons),
            "model_arena": len(arena),
            "epicure_uplift": len(uplift),
            "judgable": sum(row["judgable"] for row in comparisons),
            "failed_arm_exclusions": sum(
                row["exclusion_reason"] == "one_or_both_arms_failed" for row in comparisons
            ),
            "identity_leak_exclusions": sum(
                row["exclusion_reason"] == "identity_leak" for row in comparisons
            ),
        },
        "arena_model_exposure": dict(sorted(exposure.items())),
        "side_balance": {
            "arena_left_exposure_by_model": dict(sorted(arena_left_exposure.items())),
            "arena_right_exposure_by_model": {
                model_id: exposure[model_id] - arena_left_exposure[model_id]
                for model_id in sorted(exposure)
            },
            "uplift_epicure_on_left_by_model": {
                model_id: uplift_on_left[model_id] for model_id in sorted(exposure)
            },
            "uplift_epicure_on_right_by_model": {
                model_id: 120 - uplift_on_left[model_id] for model_id in sorted(exposure)
            },
        },
        "arena_unordered_pair_count_range": [min(pair_counts.values()), max(pair_counts.values())],
        "comparisons": comparisons,
    }
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"season0-comparisons-{digest}.json"
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(destination)
    return {**payload, "summary_path": str(destination)}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze_comparisons(
        arms_dir=args.arms_dir,
        task_bank=_load(args.task_bank),
        model_manifest=_load(args.model_manifest),
        output_dir=args.output_dir,
    )
    print(json.dumps({**result, "comparisons": "omitted_from_console"}, indent=2))


if __name__ == "__main__":
    run()
