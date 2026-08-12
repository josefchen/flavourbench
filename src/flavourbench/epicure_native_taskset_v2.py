"""Build the powered, hidden Epicure-native FlavourBench task panel.

Version 1 was deliberately small: 32 tasks, eight in each family.  This
additive successor freezes 640 tasks before any model call.  It uses 160 tasks
per family, four prespecified difficulty bands, balanced answer positions, and
one unique anchor ingredient per task.  Answer keys are computed only from a
pinned Epicure runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "flavourbench-epicure-native-taskset-v2"
DESIGN_VERSION = "flavourbench-powered-hidden-panel-v1"
SELECTION_SEED = "flavourbench-powered-hidden-panel-20260811"
FAMILIES = ("substitution", "composition", "cookability", "evidence")
CHOICE_LABELS = ("A", "B", "C", "D")
TASKS_PER_FAMILY = 160
TASK_COUNT = TASKS_PER_FAMILY * len(FAMILIES)
TASKS_PER_DIFFICULTY_BAND = TASKS_PER_FAMILY // 4
TASKS_PER_ANSWER_LABEL = TASKS_PER_FAMILY // len(CHOICE_LABELS)
MAX_TASKS_PER_CATEGORY_PER_FAMILY = 16
FINAL_CHOICE_PATTERN = re.compile(r"(?im)^\s*FINAL_CHOICE\s*:\s*([A-D])\s*$")

SENSORY_AXES = (
    "cf_balsamic",
    "cf_bitter",
    "cf_citrus",
    "cf_earthy",
    "cf_fatty",
    "cf_floral",
    "cf_fruity",
    "cf_green",
    "cf_meaty",
    "cf_minty",
    "cf_nutty",
    "cf_pungent",
    "cf_savory",
    "cf_sour",
    "cf_spicy",
    "cf_sweet",
    "cf_woody",
)

# These bands were fixed from the complete local Epicure ingredient inventory,
# before the powered panel was generated or sent to any model.  They prevent a
# 640-task expansion from becoming 640 near-duplicates at one difficulty.
DIFFICULTY_BANDS: dict[str, tuple[tuple[float, float], ...]] = {
    "substitution": ((0.002, 0.008), (0.008, 0.020), (0.020, 0.040), (0.040, 0.1801)),
    "composition": ((0.005, 0.020), (0.020, 0.035), (0.035, 0.060), (0.060, 0.201)),
    "cookability": ((0.005, 0.020), (0.020, 0.050), (0.050, 0.100), (0.100, 0.301)),
    "evidence": ((0.005, 0.035), (0.035, 0.080), (0.080, 0.150), (0.150, 0.551)),
}

EXCLUDED_ANCHORS = frozenset(
    {
        "tomato",
        "miso",
        "chocolate",
        "coffee",
        "mango",
        "lamb",
        "apple",
        "salmon",
        "basil",
        "cumin",
        "mirin",
        "mint",
        "cinnamon",
        "dill",
        "strawberry",
        "balsamic_vinegar",
        "vanilla",
        "garlic",
        "mushroom",
        "lemon",
        "chili_pepper",
        "beef",
        "cream",
        "tahini",
        "kimchi",
        "oregano",
        "coconut_milk",
        "soy_sauce",
    }
)


class PoweredTasksetError(RuntimeError):
    """The powered task set could not be built or verified."""


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


def _selection_key(*parts: str) -> str:
    return hashlib.sha256((SELECTION_SEED + "\0" + "\0".join(parts)).encode()).hexdigest()


def _band_index(family: str, margin: float) -> int | None:
    for index, (lower, upper) in enumerate(DIFFICULTY_BANDS[family], start=1):
        if lower <= margin < upper:
            return index
    return None


def _prompt(question: str, choices: Mapping[str, str]) -> str:
    rendered = "\n".join(f"{label}. {value}" for label, value in choices.items())
    return (
        "FlavourBench powered Epicure-native task. Answer against the pinned Epicure "
        "culinary representation; no external browsing is allowed.\n\n"
        f"{question}\n\nChoices:\n{rendered}\n\n"
        "Return exactly one line: `FINAL_CHOICE: X`, replacing X with A, B, C, or D."
    )


def _balanced_choices(
    *, task_id: str, target: str, distractors: Sequence[str], target_label: str
) -> tuple[dict[str, str], str]:
    if target_label not in CHOICE_LABELS or len(distractors) != 3:
        raise PoweredTasksetError(f"{task_id} has an invalid choice plan")
    values = [target, *distractors]
    if len(set(values)) != 4 or any(not value for value in values):
        raise PoweredTasksetError(f"{task_id} must have four unique non-empty choices")
    shuffled = list(distractors)
    random.Random(int(_selection_key("choice", task_id), 16)).shuffle(shuffled)
    choices: dict[str, str] = {}
    cursor = iter(shuffled)
    for label in CHOICE_LABELS:
        choices[label] = target if label == target_label else next(cursor)
    return choices, target_label


def _task(
    *,
    task_id: str,
    family: str,
    scoring_family: str,
    difficulty_band: int,
    difficulty_margin: float,
    anchor: str,
    primary_category: str,
    question: str,
    choices: Mapping[str, str],
    expected_choice: str,
    reference_tool_calls: Sequence[Mapping[str, Any]],
    reference_tool_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prompt = _prompt(question, choices)
    return {
        "task_id": task_id,
        "split": "hidden_primary",
        "family": family,
        "scoring_family": scoring_family,
        "difficulty_band": difficulty_band,
        "difficulty_margin": round(difficulty_margin, 6),
        "anchor_ingredient": anchor,
        "primary_category": primary_category,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "choices": dict(choices),
        "expected_choice": expected_choice,
        "chance_accuracy": 0.25,
        "reference_tool_calls": [dict(value) for value in reference_tool_calls],
        "reference_tool_results": [dict(value) for value in reference_tool_results],
        "reference_tool_results_sha256": _sha256(reference_tool_results),
        "scoring": {
            "method": "exact_final_choice_marker_v1",
            "pattern": FINAL_CHOICE_PATTERN.pattern,
            "case_sensitive": False,
            "points_correct": 1,
            "points_incorrect_or_unparseable": 0,
        },
    }


def _eligible_ingredients(records: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    eligible: list[dict[str, str]] = []
    for record in records:
        name = str(record.get("name") or "")
        category = str(record.get("primary_category") or "")
        if (
            name in EXCLUDED_ANCHORS
            or not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", name)
            or not (3 <= len(name) <= 36)
            or not category
        ):
            continue
        eligible.append({"name": name, "primary_category": category})
    counts = Counter(record["primary_category"] for record in eligible)
    eligible = [record for record in eligible if counts[record["primary_category"]] >= 8]
    if len(eligible) < TASK_COUNT:
        raise PoweredTasksetError("ingredient inventory is too small for 640 unique anchors")
    return eligible


def _candidate_substitution(
    record: Mapping[str, str], *, neighbors: Callable[..., Mapping[str, Any]]
) -> dict[str, Any] | None:
    anchor = record["name"]
    result = dict(neighbors(anchor, top_k=8))
    rows = result.get("neighbors")
    if not isinstance(rows, list) or len(rows) < 4:
        return None
    names = [str(row.get("name") or "") for row in rows[:4] if isinstance(row, Mapping)]
    if len(names) != 4 or len(set(names)) != 4:
        return None
    margin = float(rows[0]["sim"]) - float(rows[1]["sim"])
    band = _band_index("substitution", margin)
    if band is None:
        return None
    return {
        "record": dict(record),
        "band": band,
        "margin": margin,
        "target": names[0],
        "distractors": names[1:],
        "question": (
            f"Among these candidates, which ingredient is the nearest neighbour of `{anchor}` "
            "in the pinned Epicure embedding?"
        ),
        "calls": [{"name": "neighbors", "arguments": {"ingredient": anchor, "top_k": 8}}],
        "results": [result],
        "scoring_family": "nearest_neighbor_retrieval",
    }


def _candidate_composition(
    record: Mapping[str, str],
    *,
    neighbors: Callable[..., Mapping[str, Any]],
    pairing_score: Callable[..., Mapping[str, Any]],
) -> dict[str, Any] | None:
    anchor = record["name"]
    neighbor_result = dict(neighbors(anchor, top_k=16))
    rows = neighbor_result.get("neighbors")
    if not isinstance(rows, list) or len(rows) < 12:
        return None
    partners = [str(rows[index].get("name") or "") for index in (0, 3, 7, 11)]
    if len(set(partners)) != 4 or any(not value for value in partners):
        return None
    results = [dict(pairing_score(anchor, partner)) for partner in partners]
    if any(not isinstance(result.get("pairing_score"), (int, float)) for result in results):
        return None
    scores = [float(result["pairing_score"]) for result in results]
    ordered = sorted(range(4), key=lambda index: (-scores[index], partners[index]))
    margin = scores[ordered[0]] - scores[ordered[1]]
    band = _band_index("composition", margin)
    if band is None:
        return None
    target = partners[ordered[0]]
    return {
        "record": dict(record),
        "band": band,
        "margin": margin,
        "target": target,
        "distractors": [value for value in partners if value != target],
        "question": (f"Which candidate has the highest Epicure pairing affinity with `{anchor}`?"),
        "calls": [
            {"name": "pairing_score", "arguments": {"ingredient_a": anchor, "ingredient_b": p}}
            for p in partners
        ],
        "results": results,
        "scoring_family": "comparative_pairing_affinity",
    }


def _candidate_cookability(
    record: Mapping[str, str],
    *,
    category_members: Mapping[str, Sequence[str]],
    compare_on_axis: Callable[..., Mapping[str, Any]],
) -> dict[str, Any] | None:
    anchor = record["name"]
    category = record["primary_category"]
    alternatives = sorted(
        (name for name in category_members[category] if name != anchor),
        key=lambda name: _selection_key("cookability-choice", anchor, name),
    )
    if len(alternatives) < 3:
        return None
    ingredients = [anchor, *alternatives[:3]]
    axis = SENSORY_AXES[int(_selection_key("cookability-axis", anchor), 16) % len(SENSORY_AXES)]
    first = dict(compare_on_axis(ingredients[0], ingredients[1], axis))
    if "error" in first:
        return None
    results = [first]
    projections = [float(first["projection_a"]), float(first["projection_b"])]
    for ingredient in ingredients[2:]:
        result = dict(compare_on_axis(ingredients[0], ingredient, axis))
        if "error" in result:
            return None
        results.append(result)
        projections.append(float(result["projection_b"]))
    ordered = sorted(range(4), key=lambda index: (-projections[index], ingredients[index]))
    margin = projections[ordered[0]] - projections[ordered[1]]
    band = _band_index("cookability", margin)
    if band is None:
        return None
    target = ingredients[ordered[0]]
    return {
        "record": dict(record),
        "band": band,
        "margin": margin,
        "target": target,
        "distractors": [value for value in ingredients if value != target],
        "question": (
            f"Which `{category}` ingredient has the highest projection on Epicure's `{axis}` axis?"
        ),
        "calls": [
            {
                "name": "compare_on_axis",
                "arguments": {
                    "ingredient_a": ingredients[0],
                    "ingredient_b": ingredient,
                    "axis": axis,
                },
            }
            for ingredient in ingredients[1:]
        ],
        "results": results,
        "scoring_family": "four_way_flavour_axis_comparison",
    }


def _candidate_evidence(
    record: Mapping[str, str], *, cultural_profile: Callable[..., Mapping[str, Any]]
) -> dict[str, Any] | None:
    anchor = record["name"]
    result = dict(cultural_profile(anchor))
    cuisines = result.get("cuisines")
    if not isinstance(cuisines, Mapping) or len(cuisines) < 4:
        return None
    ordered = sorted(cuisines, key=lambda name: (-float(cuisines[name]["score"]), str(name)))
    margin = float(cuisines[ordered[0]]["score"]) - float(cuisines[ordered[1]]["score"])
    band = _band_index("evidence", margin)
    if band is None:
        return None
    choices = [str(value) for value in ordered[:4]]
    return {
        "record": dict(record),
        "band": band,
        "margin": margin,
        "target": choices[0],
        "distractors": choices[1:],
        "question": (
            f"Which cuisine direction is most aligned with `{anchor}` in Epicure's pinned "
            "cultural projection?"
        ),
        "calls": [{"name": "cultural_profile", "arguments": {"ingredient": anchor}}],
        "results": [result],
        "scoring_family": "cuisine_direction_projection",
    }


def _select_candidates(
    *, family: str, candidates: Sequence[Mapping[str, Any]], used_anchors: set[str]
) -> list[dict[str, Any]]:
    by_band: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        anchor = str(candidate["record"]["name"])
        if anchor not in used_anchors:
            by_band[int(candidate["band"])].append(candidate)
    selected: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for band in range(1, 5):
        ordered = sorted(
            by_band[band],
            key=lambda candidate: _selection_key(
                "candidate", family, str(band), str(candidate["record"]["name"])
            ),
        )
        picked = 0
        for candidate in ordered:
            category = str(candidate["record"]["primary_category"])
            if category_counts[category] >= MAX_TASKS_PER_CATEGORY_PER_FAMILY:
                continue
            selected.append(dict(candidate))
            category_counts[category] += 1
            used_anchors.add(str(candidate["record"]["name"]))
            picked += 1
            if picked == TASKS_PER_DIFFICULTY_BAND:
                break
        if picked != TASKS_PER_DIFFICULTY_BAND:
            raise PoweredTasksetError(f"{family} difficulty band {band} has insufficient tasks")
    if len(selected) != TASKS_PER_FAMILY or len(category_counts) < 10:
        raise PoweredTasksetError(f"{family} selection lacks breadth")
    return selected


def build_taskset(
    *,
    ingredient_records: Sequence[Mapping[str, str]],
    neighbors: Callable[..., Mapping[str, Any]],
    pairing_score: Callable[..., Mapping[str, Any]],
    compare_on_axis: Callable[..., Mapping[str, Any]],
    cultural_profile: Callable[..., Mapping[str, Any]],
    epicure_provenance: Mapping[str, Any],
    ingredient_inventory_sha256: str,
) -> dict[str, Any]:
    """Build the complete powered panel before any model execution."""

    ingredients = _eligible_ingredients(ingredient_records)
    category_members: dict[str, list[str]] = defaultdict(list)
    for record in ingredients:
        category_members[record["primary_category"]].append(record["name"])
    used_anchors: set[str] = set()
    selected_by_family: dict[str, list[dict[str, Any]]] = {}
    builders = {
        "substitution": lambda record: _candidate_substitution(record, neighbors=neighbors),
        "composition": lambda record: _candidate_composition(
            record, neighbors=neighbors, pairing_score=pairing_score
        ),
        "cookability": lambda record: _candidate_cookability(
            record,
            category_members=category_members,
            compare_on_axis=compare_on_axis,
        ),
        "evidence": lambda record: _candidate_evidence(record, cultural_profile=cultural_profile),
    }
    for family in FAMILIES:
        generated = []
        for record in ingredients:
            candidate = builders[family](record)
            if candidate is not None:
                generated.append(candidate)
        selected_by_family[family] = _select_candidates(
            family=family, candidates=generated, used_anchors=used_anchors
        )

    tasks: list[dict[str, Any]] = []
    for family in FAMILIES:
        selected = sorted(
            selected_by_family[family],
            key=lambda candidate: (
                int(candidate["band"]),
                _selection_key("task-order", family, str(candidate["record"]["name"])),
            ),
        )
        for index, candidate in enumerate(selected, start=1):
            task_id = f"fb-powered-{family}-{index:03d}"
            target_label = CHOICE_LABELS[(index - 1) % len(CHOICE_LABELS)]
            choices, expected = _balanced_choices(
                task_id=task_id,
                target=str(candidate["target"]),
                distractors=[str(value) for value in candidate["distractors"]],
                target_label=target_label,
            )
            record = candidate["record"]
            tasks.append(
                _task(
                    task_id=task_id,
                    family=family,
                    scoring_family=str(candidate["scoring_family"]),
                    difficulty_band=int(candidate["band"]),
                    difficulty_margin=float(candidate["margin"]),
                    anchor=str(record["name"]),
                    primary_category=str(record["primary_category"]),
                    question=str(candidate["question"]),
                    choices=choices,
                    expected_choice=expected,
                    reference_tool_calls=candidate["calls"],
                    reference_tool_results=candidate["results"],
                )
            )

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
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "benchmark": "FlavourBench",
        "track": "epicure_native_powered_exact",
        "status": "frozen_before_model_execution",
        "selection_seed": SELECTION_SEED,
        "source_class": "deterministic_hidden_questions_and_answer_keys_from_pinned_epicure",
        "epicure_provenance": provenance,
        "ingredient_inventory_sha256": ingredient_inventory_sha256,
        "counts": {
            "tasks": TASK_COUNT,
            "choices_per_task": 4,
            "per_family": {family: TASKS_PER_FAMILY for family in FAMILIES},
            "per_family_difficulty_band": TASKS_PER_DIFFICULTY_BAND,
            "per_family_answer_label": TASKS_PER_ANSWER_LABEL,
            "unique_anchor_ingredients": TASK_COUNT,
            "model_authored_tasks": 0,
            "model_judgments": 0,
            "human_judgments_required": 0,
        },
        "selection_contract": {
            "all_tasks_retained_after_execution": True,
            "posthoc_item_pruning": False,
            "excluded_predecessor_anchors": sorted(EXCLUDED_ANCHORS),
            "maximum_tasks_per_primary_category_per_family": (MAX_TASKS_PER_CATEGORY_PER_FAMILY),
            "minimum_primary_categories_per_family": 10,
            "difficulty_bands": {
                family: [list(value) for value in DIFFICULTY_BANDS[family]] for family in FAMILIES
            },
        },
        "metric_contract": {
            "primary": "flavourbench_score_model_only_macro_exact_accuracy_times_100",
            "family_aggregation": "equal_weight_across_four_families",
            "task_weighting": "equal_within_family",
            "chance_accuracy": 0.25,
            "missing_or_unparseable": "zero_points",
            "assisted_condition": "separate_diagnostic_not_in_primary_score",
        },
        "task_set_sha256": _sha256(
            [
                {
                    "task_id": task["task_id"],
                    "prompt_sha256": task["prompt_sha256"],
                    "reference_tool_results_sha256": task["reference_tool_results_sha256"],
                    "expected_choice": task["expected_choice"],
                }
                for task in tasks
            ]
        ),
        "tasks": tasks,
    }
    payload["artifact_sha256"] = _sha256(payload)
    if not verify_taskset(payload):
        raise PoweredTasksetError("constructed powered taskset failed self-verification")
    return payload


def verify_taskset(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    tasks = payload.get("tasks")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or recorded != _sha256(payload)
        or not isinstance(tasks, list)
        or len(tasks) != TASK_COUNT
    ):
        return False
    if (
        Counter(task.get("family") for task in tasks)
        != Counter({family: TASKS_PER_FAMILY for family in FAMILIES})
        or len({task.get("task_id") for task in tasks}) != TASK_COUNT
        or len({task.get("prompt_sha256") for task in tasks}) != TASK_COUNT
        or len({task.get("anchor_ingredient") for task in tasks}) != TASK_COUNT
    ):
        return False
    for family in FAMILIES:
        family_tasks = [task for task in tasks if task.get("family") == family]
        if Counter(task.get("difficulty_band") for task in family_tasks) != Counter(
            {index: TASKS_PER_DIFFICULTY_BAND for index in range(1, 5)}
        ):
            return False
        if Counter(task.get("expected_choice") for task in family_tasks) != Counter(
            {label: TASKS_PER_ANSWER_LABEL for label in CHOICE_LABELS}
        ):
            return False
        if len({task.get("primary_category") for task in family_tasks}) < 10:
            return False
        if (
            max(Counter(task.get("primary_category") for task in family_tasks).values())
            > MAX_TASKS_PER_CATEGORY_PER_FAMILY
        ):
            return False
    return all(
        isinstance(task, Mapping)
        and hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
        == task.get("prompt_sha256")
        and isinstance(task.get("choices"), Mapping)
        and set(task["choices"]) == set(CHOICE_LABELS)
        and task.get("expected_choice") in CHOICE_LABELS
        and isinstance(task.get("reference_tool_results"), list)
        and _sha256(task["reference_tool_results"]) == task.get("reference_tool_results_sha256")
        for task in tasks
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


def _load_epicure(
    epicure_root: Path,
) -> tuple[dict[str, Callable[..., Any]], dict[str, Any], list[dict[str, str]], str]:
    source = epicure_root / "epicure-mcp" / "src"
    if not source.is_dir():
        raise PoweredTasksetError(f"Epicure MCP source is absent: {source}")
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
        inventory_path = Path(config.data_dir) / "ingredient_list.csv"
        with inventory_path.open(newline="", encoding="utf-8") as handle:
            ingredients = [dict(row) for row in csv.DictReader(handle)]
        return calls, provenance, ingredients, _sha256_file(inventory_path)
    finally:
        sys.path.pop(0)


def write_taskset(document: Mapping[str, Any], output_directory: Path) -> Path:
    if not verify_taskset(document):
        raise PoweredTasksetError("refusing to write an invalid powered taskset")
    output_directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["artifact_sha256"])
    destination = output_directory / f"epicure-native-powered-taskset-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise PoweredTasksetError("content-addressed powered taskset conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_directory, delete=False
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epicure-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    calls, provenance, ingredients, inventory_sha = _load_epicure(args.epicure_root)
    document = build_taskset(
        ingredient_records=ingredients,
        epicure_provenance=provenance,
        ingredient_inventory_sha256=inventory_sha,
        **calls,
    )
    print(write_taskset(document, args.output_directory))


if __name__ == "__main__":
    run()
