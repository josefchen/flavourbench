"""Build and score the deterministic Epicure-native FlavourBench track.

The track is intentionally small enough to rerun whenever the frontier roster
changes.  It contains 32 four-choice questions: eight nearest-neighbour
retrievals, eight exact pairing scores, eight flavour-axis comparisons, and
eight cuisine-direction projections.  The answer key is computed directly by
the pinned, read-only Epicure runtime; no model is used to author or judge it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "flavourbench-epicure-native-taskset-v1"
FAMILIES = ("substitution", "composition", "cookability", "evidence")
CHOICE_LABELS = ("A", "B", "C", "D")
FINAL_CHOICE_PATTERN = re.compile(r"(?im)^\s*FINAL_CHOICE\s*:\s*([A-D])\s*$")

NEIGHBOR_SEEDS = (
    "tomato",
    "miso",
    "chocolate",
    "coffee",
    "mango",
    "lamb",
    "apple",
    "salmon",
)
PAIRING_PAIRS = (
    ("tomato", "basil"),
    ("chocolate", "coffee"),
    ("lamb", "cumin"),
    ("miso", "mirin"),
    ("mango", "mint"),
    ("apple", "cinnamon"),
    ("salmon", "dill"),
    ("strawberry", "balsamic_vinegar"),
)
AXIS_COMPARISONS = (
    ("coffee", "chocolate", "cf_bitter"),
    ("tomato", "strawberry", "cf_fruity"),
    ("vanilla", "garlic", "cf_sweet"),
    ("mushroom", "lemon", "cf_earthy"),
    ("mint", "garlic", "cf_minty"),
    ("chili_pepper", "vanilla", "cf_spicy"),
    ("beef", "apple", "cf_meaty"),
    ("lemon", "cream", "cf_sour"),
)
CULTURAL_INGREDIENTS = (
    "miso",
    "tahini",
    "kimchi",
    "cumin",
    "oregano",
    "dill",
    "coconut_milk",
    "soy_sauce",
)


class EpicureNativeTaskError(RuntimeError):
    """The exact task set could not be built or verified."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _choice_map(task_id: str, values: Sequence[str]) -> tuple[dict[str, str], str]:
    if len(values) != 4 or len(set(values)) != 4:
        raise EpicureNativeTaskError(f"{task_id} must have four unique choices")
    target = values[0]
    ordered = list(values)
    seed = int(hashlib.sha256(task_id.encode()).hexdigest(), 16)
    random.Random(seed).shuffle(ordered)
    choices = dict(zip(CHOICE_LABELS, ordered, strict=True))
    expected = next(label for label, value in choices.items() if value == target)
    return choices, expected


def _prompt(question: str, choices: Mapping[str, str]) -> str:
    rendered = "\n".join(f"{label}. {value}" for label, value in choices.items())
    return (
        "FlavourBench Epicure-native exact task. Use the read-only culinary evidence "
        "tools if they are available; otherwise answer from your own knowledge.\n\n"
        f"{question}\n\nChoices:\n{rendered}\n\n"
        "In your final answer, explain briefly if useful and finish on its own line with "
        "exactly `FINAL_CHOICE: X`, replacing X with A, B, C, or D."
    )


def _task(
    *,
    task_id: str,
    family: str,
    scoring_family: str,
    question: str,
    choices: Mapping[str, str],
    expected_choice: str,
    tool_name: str,
    tool_arguments: Mapping[str, Any],
    tool_result: Mapping[str, Any],
) -> dict[str, Any]:
    prompt = _prompt(question, choices)
    return {
        "task_id": task_id,
        "family": family,
        "scoring_family": scoring_family,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "choices": dict(choices),
        "expected_choice": expected_choice,
        "chance_accuracy": 0.25,
        "reference_tool_call": {"name": tool_name, "arguments": dict(tool_arguments)},
        "reference_tool_result": dict(tool_result),
        "reference_tool_result_sha256": _sha256(tool_result),
        "scoring": {
            "method": "exact_final_choice_marker_v1",
            "pattern": FINAL_CHOICE_PATTERN.pattern,
            "case_sensitive": False,
            "points_correct": 1,
            "points_incorrect_or_unparseable": 0,
        },
    }


def build_taskset(
    *,
    neighbors: Callable[..., Mapping[str, Any]],
    pairing_score: Callable[..., Mapping[str, Any]],
    compare_on_axis: Callable[..., Mapping[str, Any]],
    cultural_profile: Callable[..., Mapping[str, Any]],
    epicure_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute all answer keys from the supplied deterministic Epicure calls."""

    tasks: list[dict[str, Any]] = []
    for index, seed in enumerate(NEIGHBOR_SEEDS, start=1):
        result = dict(neighbors(seed, top_k=4))
        rows = result.get("neighbors")
        if not isinstance(rows, list) or len(rows) != 4:
            raise EpicureNativeTaskError(f"neighbors failed for {seed}")
        values = [str(row.get("name") or "") for row in rows if isinstance(row, Mapping)]
        task_id = f"fb-native-substitution-{index:03d}"
        choices, expected = _choice_map(task_id, values)
        tasks.append(
            _task(
                task_id=task_id,
                family="substitution",
                scoring_family="nearest_neighbor_retrieval",
                question=(
                    f"According to Epicure's `neighbors` result for `{seed}` with top_k=4, "
                    "which ingredient is ranked first?"
                ),
                choices=choices,
                expected_choice=expected,
                tool_name="neighbors",
                tool_arguments={"ingredient": seed, "top_k": 4},
                tool_result=result,
            )
        )

    for index, (left, right) in enumerate(PAIRING_PAIRS, start=1):
        result = dict(pairing_score(left, right))
        if "error" in result or not isinstance(result.get("pairing_score"), (int, float)):
            raise EpicureNativeTaskError(f"pairing_score failed for {left}/{right}")
        score = float(result["pairing_score"])
        distractors = (score - 0.08, score - 0.03, score + 0.05)
        values = [f"{score:.4f}", *(f"{max(-1.0, min(1.0, v)):.4f}" for v in distractors)]
        task_id = f"fb-native-composition-{index:03d}"
        choices, expected = _choice_map(task_id, values)
        tasks.append(
            _task(
                task_id=task_id,
                family="composition",
                scoring_family="exact_pairing_affinity",
                question=(
                    f"What exact four-decimal `pairing_score` does Epicure return for "
                    f"`{left}` and `{right}`?"
                ),
                choices=choices,
                expected_choice=expected,
                tool_name="pairing_score",
                tool_arguments={"ingredient_a": left, "ingredient_b": right},
                tool_result=result,
            )
        )

    for index, (left, right, axis) in enumerate(AXIS_COMPARISONS, start=1):
        result = dict(compare_on_axis(left, right, axis))
        if "error" in result:
            raise EpicureNativeTaskError(f"compare_on_axis failed for {left}/{right}/{axis}")
        projection_a = float(result["projection_a"])
        projection_b = float(result["projection_b"])
        values = [
            f"{left} has the higher projection",
            f"{right} has the higher projection",
            "the projections are equal to four decimals",
            "Epicure cannot resolve one or both ingredients",
        ]
        if projection_a > projection_b:
            target = values[0]
        elif projection_b > projection_a:
            target = values[1]
        else:
            target = values[2]
        task_id = f"fb-native-cookability-{index:03d}"
        choices, _ = _choice_map(task_id, [target, *(value for value in values if value != target)])
        expected = next(label for label, value in choices.items() if value == target)
        tasks.append(
            _task(
                task_id=task_id,
                family="cookability",
                scoring_family="flavour_axis_comparison",
                question=(
                    f"Which statement matches Epicure's `compare_on_axis` result for "
                    f"`{left}` versus `{right}` on `{axis}`?"
                ),
                choices=choices,
                expected_choice=expected,
                tool_name="compare_on_axis",
                tool_arguments={"ingredient_a": left, "ingredient_b": right, "axis": axis},
                tool_result=result,
            )
        )

    for index, ingredient in enumerate(CULTURAL_INGREDIENTS, start=1):
        result = dict(cultural_profile(ingredient))
        cuisines = result.get("cuisines")
        if not isinstance(cuisines, Mapping) or len(cuisines) < 4:
            raise EpicureNativeTaskError(f"cultural_profile failed for {ingredient}")
        ordered = sorted(
            cuisines,
            key=lambda name: (-float(cuisines[name]["score"]), str(name)),
        )[:4]
        task_id = f"fb-native-evidence-{index:03d}"
        choices, expected = _choice_map(task_id, [str(value) for value in ordered])
        tasks.append(
            _task(
                task_id=task_id,
                family="evidence",
                scoring_family="cuisine_projection",
                question=(
                    f"Which cuisine direction has the highest cosine score in Epicure's "
                    f"`cultural_profile` result for `{ingredient}`?"
                ),
                choices=choices,
                expected_choice=expected,
                tool_name="cultural_profile",
                tool_arguments={"ingredient": ingredient},
                tool_result=result,
            )
        )

    if (
        len(tasks) != 32
        or len({task["task_id"] for task in tasks}) != 32
        or len({task["prompt_sha256"] for task in tasks}) != 32
        or {family: sum(task["family"] == family for task in tasks) for family in FAMILIES}
        != {family: 8 for family in FAMILIES}
    ):
        raise EpicureNativeTaskError("taskset is not a unique balanced 32-task design")
    provenance = {
        key: epicure_provenance.get(key)
        for key in (
            "schema_version",
            "release_id",
            "bundle_sha256",
            "application_sha256",
            "ingredient_count",
            "embedding_dimensions",
        )
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "FlavourBench",
        "track": "epicure_native_exact",
        "status": "ready_for_automated_common_task_execution",
        "source_class": "deterministic_questions_and_answer_keys_from_pinned_epicure_runtime",
        "epicure_provenance": provenance,
        "counts": {
            "tasks": 32,
            "choices_per_task": 4,
            "per_family": {family: 8 for family in FAMILIES},
            "model_authored_tasks": 0,
            "model_judgments": 0,
            "human_judgments_required": 0,
        },
        "metric_contract": {
            "primary": "macro_exact_choice_accuracy",
            "family_aggregation": "equal_weight_across_four_families",
            "task_weighting": "equal_within_family",
            "chance_accuracy": 0.25,
            "uplift": "epicure_on_accuracy_minus_epicure_off_accuracy_in_percentage_points",
            "reliability": "fraction_of_arms_with_a_parseable_final_choice_and_normal_completion",
            "missing_or_unparseable": "zero_points",
        },
        "task_set_sha256": _sha256(
            [
                {
                    "task_id": task["task_id"],
                    "prompt_sha256": task["prompt_sha256"],
                    "reference_tool_result_sha256": task["reference_tool_result_sha256"],
                    "expected_choice": task["expected_choice"],
                }
                for task in tasks
            ]
        ),
        "tasks": tasks,
    }
    payload["artifact_sha256"] = _sha256(payload)
    return payload


def verify_taskset(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    tasks = payload.get("tasks")
    return bool(
        document.get("schema_version") == SCHEMA_VERSION
        and recorded
        and recorded == _sha256(payload)
        and isinstance(tasks, list)
        and len(tasks) == 32
        and all(
            isinstance(task, Mapping)
            and hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
            == task.get("prompt_sha256")
            and task.get("expected_choice") in CHOICE_LABELS
            and _sha256(task.get("reference_tool_result"))
            == task.get("reference_tool_result_sha256")
            for task in tasks
        )
    )


def parse_final_choice(answer_markdown: str) -> str | None:
    matches = FINAL_CHOICE_PATTERN.findall(answer_markdown)
    return matches[-1].upper() if matches else None


def score_answer(task: Mapping[str, Any], answer_markdown: str) -> dict[str, Any]:
    observed = parse_final_choice(answer_markdown)
    expected = str(task.get("expected_choice") or "")
    return {
        "observed_choice": observed,
        "expected_choice": expected,
        "parseable": observed is not None,
        "correct": observed == expected,
        "score": int(observed == expected),
    }


def _load_epicure(epicure_root: Path) -> tuple[dict[str, Callable[..., Any]], dict[str, Any]]:
    source = epicure_root / "epicure-mcp" / "src"
    if not source.is_dir():
        raise EpicureNativeTaskError(f"Epicure MCP source is absent: {source}")
    sys.path.insert(0, str(source))
    try:
        tools = importlib.import_module("epicure_mcp.tools")
        config = importlib.import_module("epicure_mcp.config").load_config()
        provenance_module = importlib.import_module("epicure_mcp.provenance")
        provenance = provenance_module.build_provenance_payload(
            str(config.data_dir.resolve()),
            config.release_id,
            config.application_sha256,
            str(source / "epicure_mcp"),
        )
        calls = {
            name: getattr(tools, name).run
            for name in ("neighbors", "pairing_score", "compare_on_axis", "cultural_profile")
        }
        return calls, provenance
    finally:
        sys.path.pop(0)


def write_taskset(document: Mapping[str, Any], output_directory: Path) -> Path:
    if not verify_taskset(document):
        raise EpicureNativeTaskError("refusing to write an invalid taskset")
    output_directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["artifact_sha256"])
    destination = output_directory / f"epicure-native-taskset-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise EpicureNativeTaskError("content-addressed taskset conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_directory, delete=False
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.link(temporary, destination)
    destination.chmod(0o644)
    temporary.unlink()
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epicure-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    calls, provenance = _load_epicure(args.epicure_root)
    taskset = build_taskset(epicure_provenance=provenance, **calls)
    path = write_taskset(taskset, args.output_directory)
    print(path)


if __name__ == "__main__":
    run()
