"""Build the construct-validated powered FlavourBench task panel.

This successor was designed after a calibration task showed that unrestricted
nearest-neighbour retrieval could reward guessing idiosyncratic latent geometry.
Every item here has an interpretable culinary question and an explicit validity
check: role and region concordance for substitution/pairing, exact dietary or
processing metadata for constraints, and agreement between an ingredient's
recorded region and Epicure's learned cuisine direction for provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "flavourbench-epicure-native-taskset-v3"
DESIGN_VERSION = "flavourbench-construct-validated-powered-panel-v2"
SELECTION_SEED = "flavourbench-construct-validated-panel-20260811"
FAMILIES = ("substitution", "pairing", "constraint", "provenance")
CHOICE_LABELS = ("A", "B", "C", "D")
TASKS_PER_FAMILY = 160
TASK_COUNT = TASKS_PER_FAMILY * len(FAMILIES)
TASKS_PER_STRATUM = 40
MAX_TASKS_PER_CATEGORY_PER_FAMILY = 24
MAX_TARGET_REUSE_PER_FAMILY = 4
FINAL_CHOICE_PATTERN = re.compile(r"(?im)^\s*FINAL_CHOICE\s*:\s*([A-D])\s*$")

CUISINES = (
    "Japanese",
    "East_Asian",
    "Southeast_Asian",
    "South_Asian",
    "Latin_American",
    "Mediterranean",
    "Eastern_European",
    "Western_Atlantic",
)
CORE_CATEGORIES = frozenset(
    {
        "Beverage",
        "Dairy",
        "Fat/Oil",
        "Fish",
        "Fruit",
        "Grain",
        "Herb",
        "Legume",
        "Meat",
        "Nut/Seed",
        "Seafood",
        "Spice",
        "Sweet",
        "Vegetable",
    }
)
COMPLEMENTS: dict[str, frozenset[str]] = {
    "Beverage": frozenset({"Fruit", "Herb", "Spice", "Sweet"}),
    "Dairy": frozenset({"Fruit", "Grain", "Herb", "Spice", "Sweet"}),
    "Fat/Oil": frozenset({"Herb", "Spice", "Vegetable", "Grain"}),
    "Fish": frozenset({"Herb", "Spice", "Vegetable", "Grain"}),
    "Fruit": frozenset({"Dairy", "Herb", "Spice", "Sweet", "Nut/Seed"}),
    "Grain": frozenset({"Dairy", "Legume", "Meat", "Vegetable", "Spice"}),
    "Herb": frozenset({"Meat", "Fish", "Seafood", "Vegetable", "Dairy"}),
    "Legume": frozenset({"Grain", "Herb", "Spice", "Vegetable", "Fat/Oil"}),
    "Meat": frozenset({"Herb", "Spice", "Vegetable", "Grain"}),
    "Nut/Seed": frozenset({"Fruit", "Sweet", "Grain", "Vegetable", "Spice"}),
    "Seafood": frozenset({"Herb", "Spice", "Vegetable", "Grain"}),
    "Spice": frozenset({"Meat", "Fish", "Seafood", "Vegetable", "Fruit"}),
    "Sweet": frozenset({"Fruit", "Dairy", "Grain", "Nut/Seed", "Beverage"}),
    "Vegetable": frozenset({"Herb", "Spice", "Grain", "Legume", "Dairy"}),
}


class ConstructValidatedTasksetError(RuntimeError):
    """The construct-validated task set could not be built or verified."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


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


def _regions(value: str) -> frozenset[str]:
    return frozenset(part for part in value.split(";") if part and part != "nan")


def _region_overlap(left: str, right: str) -> bool:
    left_regions = _regions(left)
    right_regions = _regions(right)
    return bool(left_regions & right_regions) or "universal" in left_regions | right_regions


def _word_overlap(left: str, right: str) -> bool:
    return bool(set(left.split("_")) & set(right.split("_")))


def _prompt(question: str, choices: Mapping[str, str]) -> str:
    rendered = "\n".join(f"{label}. {choices[label].replace('_', ' ')}" for label in CHOICE_LABELS)
    return (
        "FlavourBench culinary reasoning task. Use culinary knowledge only; do not browse or "
        "call external tools.\n\n"
        f"{question}\n\nChoices:\n{rendered}\n\n"
        "Return exactly one line: `FINAL_CHOICE: X`, replacing X with A, B, C, or D."
    )


def _balanced_choices(
    *, task_id: str, target: str, distractors: Sequence[str], target_label: str
) -> tuple[dict[str, str], str]:
    if len(distractors) != 3 or len({target, *distractors}) != 4:
        raise ConstructValidatedTasksetError(f"{task_id} has invalid choices")
    ordered_distractors = sorted(
        distractors, key=lambda value: _selection_key("choice", task_id, value)
    )
    choices: dict[str, str] = {}
    cursor = iter(ordered_distractors)
    for label in CHOICE_LABELS:
        choices[label] = target if label == target_label else next(cursor)
    return choices, target_label


def _task(
    *,
    task_id: str,
    family: str,
    stratum: int,
    margin: float,
    anchor: str,
    category: str,
    question: str,
    target: str,
    distractors: Sequence[str],
    reference: Mapping[str, Any],
    validity: Mapping[str, Any],
) -> dict[str, Any]:
    label = CHOICE_LABELS[(int(task_id.rsplit("-", 1)[1]) - 1) % 4]
    choices, expected = _balanced_choices(
        task_id=task_id,
        target=target,
        distractors=distractors,
        target_label=label,
    )
    prompt = _prompt(question, choices)
    return {
        "task_id": task_id,
        "split": "hidden_primary",
        "family": family,
        "scoring_family": family,
        "validation_stratum": stratum,
        "oracle_margin": round(margin, 6),
        "anchor_ingredient": anchor,
        "primary_category": category,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "choices": choices,
        "expected_choice": expected,
        "chance_accuracy": 0.25,
        "oracle_reference": dict(reference),
        "oracle_reference_sha256": _sha256(reference),
        "construct_validity": dict(validity),
        "scoring": {
            "method": "exact_final_choice_marker_v1",
            "pattern": FINAL_CHOICE_PATTERN.pattern,
            "points_correct": 1,
            "points_incorrect_unparseable_or_failed": 0,
        },
    }


def _rank_strata(candidates: list[dict[str, Any]]) -> None:
    ordered = sorted(candidates, key=lambda value: (-float(value["margin"]), value["anchor"]))
    for index, candidate in enumerate(ordered):
        candidate["stratum"] = min(4, index * 4 // len(ordered) + 1)


def _candidate_substitution(
    *, record: Mapping[str, Any], records: Sequence[Mapping[str, Any]], similarity: np.ndarray
) -> dict[str, Any] | None:
    anchor = str(record["name"])
    pool = [
        other
        for other in records
        if other["name"] != anchor
        and other["category"] == record["category"]
        and _region_overlap(str(record["region"]), str(other["region"]))
        and not _word_overlap(anchor, str(other["name"]))
    ]
    if len(pool) < 8:
        return None
    ranked = sorted(
        (
            (float(similarity[int(other["row"])]), str(other["name"]), str(other["region"]))
            for other in pool
        ),
        reverse=True,
    )
    margin = ranked[0][0] - ranked[1][0]
    if ranked[0][0] < 0.15 or margin < 0.002:
        return None
    chosen = [ranked[index] for index in (0, 1, 3, 7)]
    return {
        "anchor": anchor,
        "category": record["category"],
        "margin": margin,
        "target": chosen[0][1],
        "distractors": [value[1] for value in chosen[1:]],
        "question": (
            f"Which ingredient is the strongest substitute for `{anchor.replace('_', ' ')}` "
            f"while preserving its {str(record['category']).lower()} role and regional context?"
        ),
        "reference": {
            "method": "highest_cosine_within_same_role_and_overlapping_region",
            "anchor_region": record["region"],
            "ranked_candidates": [
                {"name": name, "cosine": round(score, 6), "region": region}
                for score, name, region in chosen
            ],
        },
        "validity": {
            "same_primary_category_for_all_choices": True,
            "regional_overlap_for_all_choices": True,
            "lexical_overlap_excluded": True,
        },
    }


def _candidate_pairing(
    *, record: Mapping[str, Any], records: Sequence[Mapping[str, Any]], similarity: np.ndarray
) -> dict[str, Any] | None:
    anchor = str(record["name"])
    complement_categories = COMPLEMENTS.get(str(record["category"]), frozenset())
    pool = [
        other
        for other in records
        if other["category"] in complement_categories
        and _region_overlap(str(record["region"]), str(other["region"]))
        and not _word_overlap(anchor, str(other["name"]))
    ]
    if len(pool) < 8:
        return None
    ranked = sorted(
        (
            (
                float(similarity[int(other["row"])]),
                str(other["name"]),
                str(other["category"]),
                str(other["region"]),
            )
            for other in pool
        ),
        reverse=True,
    )
    margin = ranked[0][0] - ranked[1][0]
    if ranked[0][0] < 0.15 or margin < 0.002:
        return None
    chosen = [ranked[index] for index in (0, 1, 3, 7)]
    return {
        "anchor": anchor,
        "category": record["category"],
        "margin": margin,
        "target": chosen[0][1],
        "distractors": [value[1] for value in chosen[1:]],
        "question": (
            f"Which ingredient has the strongest learned culinary pairing with "
            f"`{anchor.replace('_', ' ')}`?"
        ),
        "reference": {
            "method": "highest_cosine_among_role_complements_with_overlapping_region",
            "anchor_region": record["region"],
            "allowed_complement_categories": sorted(complement_categories),
            "ranked_candidates": [
                {
                    "name": name,
                    "cosine": round(score, 6),
                    "category": category,
                    "region": region,
                }
                for score, name, category, region in chosen
            ],
        },
        "validity": {
            "all_choices_are_prespecified_role_complements": True,
            "regional_overlap_for_all_choices": True,
            "lexical_overlap_excluded": True,
        },
    }


def _constraint_candidates(
    records: Sequence[Mapping[str, Any]], used_anchors: set[str]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    vegan = [record for record in records if record["vegan"] is True]
    not_vegan = [record for record in records if record["vegan"] is False]
    for stratum, source, distractor_source, desired, question in (
        (1, vegan, not_vegan, True, "Which ingredient is suitable for a strictly vegan dish?"),
        (2, not_vegan, vegan, False, "Which ingredient is not suitable for a vegan dish?"),
    ):
        for record in source:
            anchor = str(record["name"])
            if anchor in used_anchors:
                continue
            pool = sorted(
                (
                    other
                    for other in distractor_source
                    if not _word_overlap(anchor, str(other["name"]))
                ),
                key=lambda other: _selection_key("constraint", anchor, str(other["name"])),
            )
            if len(pool) < 3:
                continue
            distractors = [str(other["name"]) for other in pool[:3]]
            candidates.append(
                {
                    "anchor": anchor,
                    "category": record["category"],
                    "margin": 1.0,
                    "stratum": stratum,
                    "target": anchor,
                    "distractors": distractors,
                    "question": question,
                    "reference": {
                        "method": "exact_ingredient_metadata",
                        "property": "is_vegan",
                        "target_value": desired,
                        "choices": [
                            {"name": anchor, "is_vegan": desired},
                            *[{"name": value, "is_vegan": not desired} for value in distractors],
                        ],
                    },
                    "validity": {
                        "single_choice_satisfies_predicate": True,
                        "metadata_field": "is_vegan",
                    },
                }
            )
    with_nova = [record for record in records if record["nova"] is not None]
    for record in with_nova:
        anchor = str(record["name"])
        if anchor in used_anchors:
            continue
        comparable = [
            other
            for other in with_nova
            if other["name"] != anchor
            and other["category"] == record["category"]
            and not _word_overlap(anchor, str(other["name"]))
        ]
        if len(comparable) < 3:
            continue
        for stratum, reverse, question in (
            (3, False, "Which ingredient is the least processed option?"),
            (4, True, "Which ingredient is the most processed option?"),
        ):
            anchor_level = float(record["nova"])
            pool = sorted(
                (
                    other
                    for other in comparable
                    if (
                        float(other["nova"]) >= anchor_level + 1.0
                        if not reverse
                        else float(other["nova"]) <= anchor_level - 1.0
                    )
                ),
                key=lambda other: _selection_key("nova", anchor, str(other["name"])),
            )
            if len(pool) < 3:
                continue
            distractors = [str(other["name"]) for other in pool[:3]]
            candidates.append(
                {
                    "anchor": anchor,
                    "category": record["category"],
                    "margin": min(
                        abs(float(other["nova"]) - float(record["nova"])) for other in pool[:3]
                    ),
                    "stratum": stratum,
                    "target": anchor,
                    "distractors": distractors,
                    "question": question,
                    "reference": {
                        "method": "exact_ingredient_metadata",
                        "property": "nova_level",
                        "direction": "minimum" if stratum == 3 else "maximum",
                        "choices": [
                            {"name": anchor, "nova_level": record["nova"]},
                            *[
                                {
                                    "name": other["name"],
                                    "nova_level": other["nova"],
                                }
                                for other in pool[:3]
                            ],
                        ],
                    },
                    "validity": {
                        "single_choice_satisfies_predicate": True,
                        "metadata_field": "nova_level",
                        "minimum_level_separation": 1.0,
                    },
                }
            )
    return candidates


def _candidate_provenance(
    *, record: Mapping[str, Any], normed: np.ndarray, directions: Mapping[str, np.ndarray]
) -> dict[str, Any] | None:
    regions = _regions(str(record["region"]))
    if len(regions) != 1 or next(iter(regions)) not in CUISINES:
        return None
    recorded_region = next(iter(regions))
    scores = sorted(
        (
            (float(normed[int(record["row"])] @ directions[cuisine]), cuisine)
            for cuisine in CUISINES
        ),
        reverse=True,
    )
    if scores[0][1] != recorded_region:
        return None
    margin = scores[0][0] - scores[1][0]
    if margin < 0.01:
        return None
    chosen = [scores[index] for index in (0, 1, 2, 3)]
    return {
        "anchor": record["name"],
        "category": record["category"],
        "margin": margin,
        "target": recorded_region,
        "distractors": [value[1] for value in chosen[1:]],
        "question": (
            f"Which culinary region is most strongly associated with "
            f"`{str(record['name']).replace('_', ' ')}`?"
        ),
        "reference": {
            "method": "recorded_region_confirmed_by_highest_epicure_cuisine_projection",
            "recorded_region": recorded_region,
            "ranked_directions": [
                {"region": cuisine, "projection": round(score, 6)} for score, cuisine in chosen
            ],
        },
        "validity": {
            "recorded_region_matches_oracle_top_direction": True,
            "single_recorded_region": True,
        },
    }


def _select(
    *,
    family: str,
    candidates: Sequence[Mapping[str, Any]],
    used_anchors: set[str],
    target_reuse_cap: int = MAX_TARGET_REUSE_PER_FAMILY,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for stratum in range(1, 5):
        eligible = sorted(
            (
                candidate
                for candidate in candidates
                if int(candidate["stratum"]) == stratum
                and str(candidate["anchor"]) not in used_anchors
            ),
            key=lambda value: _selection_key("select", family, str(stratum), str(value["anchor"])),
        )
        picked = 0
        for candidate in eligible:
            category = str(candidate["category"])
            target = str(candidate["target"])
            if (
                category_counts[category] >= MAX_TASKS_PER_CATEGORY_PER_FAMILY
                or target_counts[target] >= target_reuse_cap
            ):
                continue
            selected.append(dict(candidate))
            category_counts[category] += 1
            target_counts[target] += 1
            used_anchors.add(str(candidate["anchor"]))
            picked += 1
            if picked == TASKS_PER_STRATUM:
                break
        if picked != TASKS_PER_STRATUM:
            raise ConstructValidatedTasksetError(
                f"{family} validation stratum {stratum} has only {picked} selectable items"
            )
    if len(category_counts) < 8:
        raise ConstructValidatedTasksetError(f"{family} selection lacks category breadth")
    return selected


def build_taskset(
    *,
    records: Sequence[Mapping[str, Any]],
    normed: np.ndarray,
    directions: Mapping[str, np.ndarray],
    epicure_provenance: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    excluded_anchors: Sequence[str] = (),
) -> dict[str, Any]:
    if normed.shape != (len(records), 300):
        raise ConstructValidatedTasksetError("Epicure embedding matrix has an invalid shape")
    eligible = [
        dict(record)
        for record in records
        if record["category"] in CORE_CATEGORIES
        and record["name"] not in set(excluded_anchors)
        and re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", str(record["name"]))
    ]
    used_anchors = set(excluded_anchors)
    substitution: list[dict[str, Any]] = []
    pairing: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for record in eligible:
        similarity = normed @ normed[int(record["row"])]
        sub = _candidate_substitution(record=record, records=eligible, similarity=similarity)
        pair = _candidate_pairing(record=record, records=eligible, similarity=similarity)
        prov = _candidate_provenance(record=record, normed=normed, directions=directions)
        if sub:
            substitution.append(sub)
        if pair:
            pairing.append(pair)
        if prov:
            provenance.append(prov)
    _rank_strata(substitution)
    _rank_strata(pairing)
    _rank_strata(provenance)
    selected = {
        "substitution": _select(
            family="substitution", candidates=substitution, used_anchors=used_anchors
        ),
        "pairing": _select(family="pairing", candidates=pairing, used_anchors=used_anchors),
    }
    constraints = _constraint_candidates(eligible, used_anchors)
    selected["constraint"] = _select(
        family="constraint", candidates=constraints, used_anchors=used_anchors
    )
    selected["provenance"] = _select(
        family="provenance",
        candidates=provenance,
        used_anchors=used_anchors,
        target_reuse_cap=40,
    )

    tasks: list[dict[str, Any]] = []
    for family in FAMILIES:
        ordered = sorted(
            selected[family],
            key=lambda value: (
                int(value["stratum"]),
                _selection_key("task", family, str(value["anchor"])),
            ),
        )
        for index, candidate in enumerate(ordered, start=1):
            tasks.append(
                _task(
                    task_id=f"fb-validated-{family}-{index:03d}",
                    family=family,
                    stratum=int(candidate["stratum"]),
                    margin=float(candidate["margin"]),
                    anchor=str(candidate["anchor"]),
                    category=str(candidate["category"]),
                    question=str(candidate["question"]),
                    target=str(candidate["target"]),
                    distractors=[str(value) for value in candidate["distractors"]],
                    reference=candidate["reference"],
                    validity=candidate["validity"],
                )
            )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "status": "frozen_before_successor_model_execution",
        "selection_seed": SELECTION_SEED,
        "benchmark": "FlavourBench",
        "track": "construct_validated_epicure_native_exact",
        "calibration_predecessor": {
            "status": "superseded_after_construct_validity_failure",
            "taskset_sha256": ("d64f606052bf5d4766a01f76d73655d0df605b82fc2164232479921c1d15e996"),
            "powered_plan_sha256": (
                "8faf32ac364e5b7cf08c1cdf0ca95d30cc3048803f10a480a8d03a923bf99fe0"
            ),
            "provider_calls": 20,
            "used_as_primary_data": False,
        },
        "epicure_provenance": dict(epicure_provenance),
        "source_hashes": dict(source_hashes),
        "counts": {
            "tasks": TASK_COUNT,
            "choices_per_task": 4,
            "per_family": {family: TASKS_PER_FAMILY for family in FAMILIES},
            "per_family_validation_stratum": TASKS_PER_STRATUM,
            "unique_anchor_ingredients": TASK_COUNT,
            "model_authored_tasks": 0,
            "human_judgments_required": 0,
        },
        "construct_validity_contract": {
            "substitution": "same role, overlapping region, no lexical overlap",
            "pairing": "prespecified complementary roles, overlapping region, no lexical overlap",
            "constraint": "single exact dietary or NOVA metadata predicate",
            "provenance": "recorded single region equals top learned cuisine direction",
            "all_items_retained_after_successor_execution": True,
            "posthoc_item_pruning": False,
        },
        "metric_contract": {
            "primary": "flavourbench_score_equal_family_macro_exact_accuracy_times_100",
            "chance_accuracy": 0.25,
            "missing_unparseable_or_failed": "zero_points",
            "epicure_assisted_condition": "separate_diagnostic_not_in_primary_score",
        },
        "task_set_sha256": _sha256(
            [
                {
                    "task_id": task["task_id"],
                    "prompt_sha256": task["prompt_sha256"],
                    "oracle_reference_sha256": task["oracle_reference_sha256"],
                    "expected_choice": task["expected_choice"],
                }
                for task in tasks
            ]
        ),
        "tasks": tasks,
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_taskset(document):
        raise ConstructValidatedTasksetError("constructed taskset failed verification")
    return document


def verify_taskset(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    tasks = document.get("tasks")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or recorded != _sha256(payload)
        or not isinstance(tasks, list)
        or len(tasks) != TASK_COUNT
        or len({task.get("task_id") for task in tasks}) != TASK_COUNT
        or len({task.get("prompt_sha256") for task in tasks}) != TASK_COUNT
        or len({task.get("anchor_ingredient") for task in tasks}) != TASK_COUNT
        or Counter(task.get("family") for task in tasks)
        != Counter({family: TASKS_PER_FAMILY for family in FAMILIES})
    ):
        return False
    for family in FAMILIES:
        family_tasks = [task for task in tasks if task.get("family") == family]
        if Counter(task.get("validation_stratum") for task in family_tasks) != Counter(
            {stratum: TASKS_PER_STRATUM for stratum in range(1, 5)}
        ) or Counter(task.get("expected_choice") for task in family_tasks) != Counter(
            {label: 40 for label in CHOICE_LABELS}
        ):
            return False
    return all(
        isinstance(task, Mapping)
        and hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
        == task.get("prompt_sha256")
        and set(task.get("choices") or {}) == set(CHOICE_LABELS)
        and task.get("expected_choice") in CHOICE_LABELS
        and _sha256(task.get("oracle_reference")) == task.get("oracle_reference_sha256")
        and all(
            value is True
            for value in (task.get("construct_validity") or {}).values()
            if isinstance(value, bool)
        )
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


def _load_inputs(
    epicure_root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    data_directory = epicure_root / "epicure-mcp" / "data"
    source_directory = epicure_root / "epicure-mcp" / "src"
    tags_path = data_directory / "ingredient_tags.csv"
    embeddings_path = data_directory / "embeddings.csv"
    directions_path = data_directory / "supervised_directions.npz"
    with tags_path.open(newline="", encoding="utf-8") as handle:
        tags = {int(row["node_id"]): row for row in csv.DictReader(handle)}
    with embeddings_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        dimension_fields = [field for field in reader.fieldnames or [] if field.startswith("dim_")]
        embedding_rows = list(reader)
    matrix = np.asarray(
        [[float(row[field]) for field in dimension_fields] for row in embedding_rows],
        dtype=np.float32,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = matrix / norms
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(embedding_rows):
        tag = tags[int(row["node_id"])]
        try:
            nova: float | None = float(tag["nova_level"])
            if not np.isfinite(nova):
                nova = None
        except (TypeError, ValueError):
            nova = None
        records.append(
            {
                "row": row_index,
                "node_id": int(row["node_id"]),
                "name": tag["name"],
                "category": tag["primary_category"],
                "region": tag["cuisine_region"],
                "vegan": str(tag["is_vegan"]).lower() == "true",
                "vegetarian": str(tag["is_vegetarian"]).lower() == "true",
                "nova": nova,
            }
        )
    raw_directions = np.load(directions_path, allow_pickle=False)
    directions = {name: raw_directions[name].astype(np.float32) for name in CUISINES}
    directions = {
        name: vector / (float(np.linalg.norm(vector)) or 1.0) for name, vector in directions.items()
    }
    sys.path.insert(0, str(source_directory))
    try:
        config = importlib.import_module("epicure_mcp.config").load_config()
        provenance_module = importlib.import_module("epicure_mcp.provenance")
        provenance = provenance_module.build_provenance_payload(
            str(config.data_dir.resolve()),
            config.release_id,
            config.application_sha256,
            str(source_directory / "epicure_mcp"),
        )
    finally:
        sys.path.pop(0)
    provenance = {
        key: provenance[key]
        for key in (
            "schema_version",
            "release_id",
            "bundle_sha256",
            "application_sha256",
            "ingredient_count",
            "embedding_dimensions",
        )
    }
    hashes = {
        "ingredient_tags_physical_sha256": _sha256_file(tags_path),
        "embeddings_physical_sha256": _sha256_file(embeddings_path),
        "supervised_directions_physical_sha256": _sha256_file(directions_path),
    }
    return records, normed, directions, provenance, hashes


def write_taskset(document: Mapping[str, Any], output_directory: Path) -> Path:
    if not verify_taskset(document):
        raise ConstructValidatedTasksetError("refusing to write invalid taskset")
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = (
        output_directory / f"epicure-native-validated-taskset-{document['artifact_sha256']}.json"
    )
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise ConstructValidatedTasksetError("content-addressed taskset conflict")
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
    parser.add_argument("--exclude-taskset", type=Path)
    args = parser.parse_args(argv)
    excluded: list[str] = []
    if args.exclude_taskset:
        prior = json.loads(args.exclude_taskset.read_text(encoding="utf-8"))
        pilot_id = "fb-powered-substitution-097"
        excluded = [
            str(task["anchor_ingredient"]) for task in prior["tasks"] if task["task_id"] == pilot_id
        ]
    records, normed, directions, provenance, hashes = _load_inputs(args.epicure_root)
    document = build_taskset(
        records=records,
        normed=normed,
        directions=directions,
        epicure_provenance=provenance,
        source_hashes=hashes,
        excluded_anchors=excluded,
    )
    print(write_taskset(document, args.output_directory))


if __name__ == "__main__":
    run()
