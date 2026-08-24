"""Build the Epicure-scored combinatorial FlavourBench task set.

Each item asks a model to select three ingredients from eight candidates.
Epicure evaluates every one of the 56 possible selections before any model is
called, so responses receive continuous, deterministic partial credit rather
than a brittle all-or-nothing label.  Four families cover substitution,
pairing, dietary/process constraints, and cultural composition.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .epicure_native_taskset_v3 import (
    COMPLEMENTS,
    CORE_CATEGORIES,
    CUISINES,
    _load_inputs,
    _region_overlap,
    _word_overlap,
)

SCHEMA_VERSION = "flavourbench-epicure-selection-taskset-v5"
DESIGN_VERSION = "flavourbench-executable-culinary-selection-v5"
SELECTION_SEED = "flavourbench-executable-selection-20260811"
FAMILIES = ("substitution", "pairing", "constraint", "cultural_composition")
LABELS = tuple("ABCDEFGH")
SELECTION_SIZE = 3
PARSER_SCHEMA_VERSION = "flavourbench-selection-set-parser-v2"
TASKS_PER_FAMILY = 160
TASKS_PER_STRATUM = 40
TASK_COUNT = TASKS_PER_FAMILY * len(FAMILIES)
ALL_SELECTION_KEYS = tuple(
    "".join(values) for values in itertools.combinations(LABELS, SELECTION_SIZE)
)
FINAL_SELECTION_PATTERN = re.compile(
    r"(?im)^\s*(?:[`*_]+\s*)?FINAL_SELECTION\s*:\s*([A-H])\s*,\s*([A-H])\s*,\s*"
    r"([A-H])\s*(?:[`*_]+\s*)?$"
)
CALIBRATION_EXCLUDED_ANCHORS = (
    "sweet_potato_starch",
    "birds_eye_chili",
    "perilla_oil",
    "squid",
    "adjika",
)


class EpicureSelectionTasksetError(RuntimeError):
    """The executable selection task set failed verification."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def selection_parser_contract() -> dict[str, Any]:
    return {
        "schema_version": PARSER_SCHEMA_VERSION,
        "marker": "FINAL_SELECTION",
        "allowed_labels": list(LABELS),
        "selection_size": SELECTION_SIZE,
        "match_rule": "last exact marker line",
        "distinct_labels_required": True,
        "normalization": "uppercase then sort as an unordered ingredient set",
        "invalid_result": "zero score",
    }


def selection_parser_sha256() -> str:
    return _sha256(selection_parser_contract())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_key(*parts: str) -> str:
    return hashlib.sha256((SELECTION_SEED + "\0" + "\0".join(parts)).encode()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _pairwise_mean(rows: Sequence[int], normed: np.ndarray) -> float:
    return _mean(
        [float(normed[left] @ normed[right]) for left, right in itertools.combinations(rows, 2)]
    )


def _spaced(values: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    if len(values) < count:
        raise EpicureSelectionTasksetError("candidate pool is too small")
    if count == 1:
        return [values[0]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]


def _choice_permutation(task_id: str, candidates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ordered = sorted(
        (str(candidate["name"]) for candidate in candidates),
        key=lambda name: _selection_key("choice", task_id, name),
    )
    return dict(zip(LABELS, ordered, strict=True))


def _normalize_scores(
    *,
    choices: Mapping[str, str],
    utility: Callable[[tuple[Mapping[str, Any], ...]], float | None],
    records_by_name: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, int], str, int, int]:
    raw: dict[str, float | None] = {}
    for labels in itertools.combinations(LABELS, SELECTION_SIZE):
        selected = tuple(records_by_name[choices[label]] for label in labels)
        raw["".join(labels)] = utility(selected)
    valid_values = [value for value in raw.values() if value is not None and math.isfinite(value)]
    if len(valid_values) < 4 or max(valid_values) - min(valid_values) < 1e-6:
        raise EpicureSelectionTasksetError("selection utility lacks a usable range")
    low = min(valid_values)
    high = max(valid_values)
    scores = {
        key: (
            0
            if value is None
            else int(round(10_000 * max(0.0, min(1.0, (value - low) / (high - low)))))
        )
        for key, value in raw.items()
    }
    best = max(scores.values())
    optimal = [key for key, value in scores.items() if value == best]
    if len(optimal) != 1 or best != 10_000:
        raise EpicureSelectionTasksetError("selection task lacks a unique optimum")
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    chance = int(round(sum(scores.values()) / len(scores)))
    return scores, optimal[0], chance, margin


def _render_prompt(
    *, family: str, question: str, choices: Mapping[str, str], selection_size: int
) -> str:
    rendered = "\n".join(f"{label}. {choices[label].replace('_', ' ')}" for label in LABELS)
    displayed_family = (
        "regional cuisine selection"
        if family == "cultural_composition"
        else family.replace("_", " ")
    )
    return (
        "FlavourBench executable culinary decision task. Epicure is the deterministic scoring "
        "environment. Select the best portfolio using culinary knowledge; do not browse or call "
        "external tools. Partial credit reflects the quality of the complete selection.\n\n"
        f"Family: {displayed_family}\n{question}\n\nCandidates:\n{rendered}\n\n"
        f"Choose exactly {selection_size} distinct labels. Return exactly one line in alphabetical "
        "label order: `FINAL_SELECTION: A,C,F`."
    )


def _task_from_candidate(
    *,
    family: str,
    index: int,
    candidate: Mapping[str, Any],
    records_by_name: Mapping[str, Mapping[str, Any]],
    normed: np.ndarray,
    directions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    task_id = f"fb-executable-{family}-{index:03d}"
    choices = _choice_permutation(task_id, candidate["candidates"])
    anchor_row = int(records_by_name[str(candidate["anchor"])]["row"])
    target_region = candidate.get("target_region")

    def utility(selected: tuple[Mapping[str, Any], ...]) -> float | None:
        rows = [int(value["row"]) for value in selected]
        anchor_similarity = _mean([float(normed[anchor_row] @ normed[row]) for row in rows])
        coherence = _pairwise_mean(rows, normed)
        if family == "substitution":
            return 0.8 * anchor_similarity + 0.2 * coherence
        if family == "pairing":
            return 0.65 * anchor_similarity + 0.35 * coherence
        if family == "constraint":
            variant = str(candidate["constraint_variant"])
            maximum_nova = int(candidate["maximum_nova"])
            if any(
                value.get("nova") is None or float(value["nova"]) > maximum_nova
                for value in selected
            ):
                return None
            if variant == "vegan" and any(value["vegan"] is not True for value in selected):
                return None
            if variant == "vegetarian" and any(
                value["vegetarian"] is not True for value in selected
            ):
                return None
            return 0.7 * anchor_similarity + 0.3 * coherence
        if family == "cultural_composition":
            if len({str(value["category"]) for value in selected}) != SELECTION_SIZE:
                return None
            direction = directions[str(target_region)]
            cultural = _mean([float(normed[row] @ direction) for row in rows])
            return 0.75 * cultural + 0.25 * coherence
        raise EpicureSelectionTasksetError("unknown task family")

    scores, optimal, chance, score_margin = _normalize_scores(
        choices=choices, utility=utility, records_by_name=records_by_name
    )
    prompt = _render_prompt(
        family=family,
        question=str(candidate["question"]),
        choices=choices,
        selection_size=SELECTION_SIZE,
    )
    reference = {
        "method": str(candidate["method"]),
        "anchor": candidate["anchor"],
        "target_region": target_region,
        "constraint_variant": candidate.get("constraint_variant"),
        "maximum_nova": candidate.get("maximum_nova"),
        "selection_size": SELECTION_SIZE,
        "all_combinations_scored": len(ALL_SELECTION_KEYS),
        "normalization": "within_task_min_max_over_valid_selections",
        "invalid_selection_score_bps": 0,
    }
    return {
        "task_id": task_id,
        "split": "hidden_primary",
        "family": family,
        "validation_stratum": int(candidate["stratum"]),
        "anchor_ingredient": candidate["anchor"],
        "primary_category": candidate["category"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "choices": choices,
        "selection_size": SELECTION_SIZE,
        "selection_scores_bps": scores,
        "optimal_selection": optimal,
        "chance_score_bps": chance,
        "optimal_margin_bps": score_margin,
        "oracle_reference": reference,
        "oracle_reference_sha256": _sha256(reference),
        "scoring": {
            "method": "exact_selection_lookup_bps_v1",
            "pattern": FINAL_SELECTION_PATTERN.pattern,
            "minimum_bps": 0,
            "maximum_bps": 10_000,
            "failed_unparseable_duplicate_or_wrong_cardinality_bps": 0,
        },
    }


def _rank_strata(candidates: list[dict[str, Any]]) -> None:
    ordered = sorted(candidates, key=lambda value: (-float(value["difficulty"]), value["anchor"]))
    for index, candidate in enumerate(ordered):
        candidate["stratum"] = min(4, index * 4 // len(ordered) + 1)


def _substitution_candidate(
    record: Mapping[str, Any], records: Sequence[Mapping[str, Any]], normed: np.ndarray
) -> dict[str, Any] | None:
    anchor = str(record["name"])
    row = int(record["row"])
    pool = [
        other
        for other in records
        if other["name"] != anchor
        and other["category"] == record["category"]
        and _region_overlap(str(record["region"]), str(other["region"]))
        and not _word_overlap(anchor, str(other["name"]))
    ]
    if len(pool) < 12:
        return None
    ranked = sorted(
        pool, key=lambda other: float(normed[row] @ normed[int(other["row"])]), reverse=True
    )
    considered = ranked[: min(40, len(ranked))]
    chosen = _spaced(considered, 8)
    spread = float(normed[row] @ normed[int(chosen[0]["row"])]) - float(
        normed[row] @ normed[int(chosen[-1]["row"])]
    )
    if spread < 0.08:
        return None
    return {
        "anchor": anchor,
        "category": record["category"],
        "difficulty": 1.0 / spread,
        "candidates": chosen,
        "method": "epicure_anchor_similarity_80pct_plus_portfolio_coherence_20pct",
        "question": (
            f"Select three alternatives to `{anchor.replace('_', ' ')}` that best preserve its "
            f"{str(record['category']).lower()} role, regional context, and mutual portfolio "
            "coherence."
        ),
    }


def _pairing_candidate(
    record: Mapping[str, Any], records: Sequence[Mapping[str, Any]], normed: np.ndarray
) -> dict[str, Any] | None:
    anchor = str(record["name"])
    row = int(record["row"])
    complements = COMPLEMENTS.get(str(record["category"]), frozenset())
    pool = [
        other
        for other in records
        if other["category"] in complements
        and _region_overlap(str(record["region"]), str(other["region"]))
        and not _word_overlap(anchor, str(other["name"]))
    ]
    if len(pool) < 16:
        return None
    ranked = sorted(
        pool, key=lambda other: float(normed[row] @ normed[int(other["row"])]), reverse=True
    )
    chosen = _spaced(ranked[: min(48, len(ranked))], 8)
    spread = float(normed[row] @ normed[int(chosen[0]["row"])]) - float(
        normed[row] @ normed[int(chosen[-1]["row"])]
    )
    if spread < 0.08:
        return None
    return {
        "anchor": anchor,
        "category": record["category"],
        "difficulty": 1.0 / spread,
        "candidates": chosen,
        "method": "epicure_anchor_pairing_65pct_plus_bundle_coherence_35pct",
        "question": (
            f"Select the three-ingredient bundle with the strongest learned pairing to "
            f"`{anchor.replace('_', ' ')}` and the best internal coherence."
        ),
    }


def _constraint_candidates(
    record: Mapping[str, Any], records: Sequence[Mapping[str, Any]], normed: np.ndarray
) -> list[dict[str, Any]]:
    anchor = str(record["name"])
    row = int(record["row"])
    complements = COMPLEMENTS.get(str(record["category"]), frozenset())
    pool = [
        other
        for other in records
        if other["name"] != anchor
        and other["category"] in complements
        and other.get("nova") is not None
        and not _word_overlap(anchor, str(other["name"]))
    ]
    ranked = sorted(
        pool, key=lambda other: float(normed[row] @ normed[int(other["row"])]), reverse=True
    )
    outputs: list[dict[str, Any]] = []
    for variant, maximum_nova in (("vegan", 2), ("vegetarian", 2), ("vegan", 3), ("vegetarian", 3)):
        valid = [
            value
            for value in ranked
            if float(value["nova"]) <= maximum_nova and value[variant] is True
        ]
        invalid = [
            value
            for value in ranked
            if not (float(value["nova"]) <= maximum_nova and value[variant] is True)
        ]
        if len(valid) < 4 or len(invalid) < 4:
            continue
        chosen = [*valid[:4], *_spaced(invalid[: min(32, len(invalid))], 4)]
        outputs.append(
            {
                "anchor": anchor,
                "category": record["category"],
                "difficulty": float(maximum_nova) + (0.2 if variant == "vegetarian" else 0.0),
                "candidates": chosen,
                "method": "hard_dietary_and_nova_filter_then_epicure_coherence_optimization",
                "constraint_variant": variant,
                "maximum_nova": maximum_nova,
                "question": (
                    f"For a dish centered on `{anchor.replace('_', ' ')}`, select three "
                    f"{variant} complements with NOVA processing level at most {maximum_nova}; "
                    "among valid portfolios, maximize learned pairing and coherence."
                ),
            }
        )
    return outputs


def _cultural_candidate(
    record: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    normed: np.ndarray,
    directions: Mapping[str, np.ndarray],
) -> dict[str, Any] | None:
    anchor = str(record["name"])
    regions = [part for part in str(record["region"]).split(";") if part in CUISINES]
    if len(regions) != 1:
        return None
    target = regions[0]
    direction = directions[target]
    pool = [
        other
        for other in records
        if other["name"] != anchor and not _word_overlap(anchor, str(other["name"]))
    ]
    ranked = sorted(
        pool, key=lambda other: float(normed[int(other["row"])] @ direction), reverse=True
    )
    high: list[Mapping[str, Any]] = []
    categories: set[str] = set()
    for value in ranked:
        category = str(value["category"])
        if category not in categories:
            high.append(value)
            categories.add(category)
        if len(high) == 4:
            break
    low_pool = [
        value for value in reversed(ranked) if value["name"] not in {x["name"] for x in high}
    ]
    low = _spaced(low_pool[: min(80, len(low_pool))], 4) if len(low_pool) >= 4 else []
    if len(high) != 4 or len(low) != 4:
        return None
    chosen = [*high, *low]
    spread = _mean([float(normed[int(value["row"])] @ direction) for value in high]) - _mean(
        [float(normed[int(value["row"])] @ direction) for value in low]
    )
    if spread < 0.15:
        return None
    return {
        "anchor": anchor,
        "category": record["category"],
        "difficulty": 1.0 / spread,
        "candidates": chosen,
        "target_region": target,
        "method": "epicure_cuisine_projection_75pct_plus_bundle_coherence_25pct",
        "question": (
            f"Target cuisine label: {target.replace('_', ' ')}. Select three ingredients that "
            "best fit this culinary profile. The selected ingredients must represent three "
            "distinct primary roles."
        ),
    }


def _select_family(
    *, family: str, candidates: Sequence[Mapping[str, Any]], used_anchors: set[str]
) -> list[dict[str, Any]]:
    mutable = [dict(value) for value in candidates]
    _rank_strata(mutable)
    selected: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for stratum in range(1, 5):
        eligible = sorted(
            (
                value
                for value in mutable
                if value["stratum"] == stratum and value["anchor"] not in used_anchors
            ),
            key=lambda value: _selection_key("select", family, str(stratum), str(value["anchor"])),
        )
        picked = 0
        for value in eligible:
            category = str(value["category"])
            if category_counts[category] >= 24:
                continue
            selected.append(value)
            category_counts[category] += 1
            used_anchors.add(str(value["anchor"]))
            picked += 1
            if picked == TASKS_PER_STRATUM:
                break
        if picked != TASKS_PER_STRATUM:
            raise EpicureSelectionTasksetError(
                f"{family} stratum {stratum} has only {picked} selectable candidates"
            )
    if len(category_counts) < 8:
        raise EpicureSelectionTasksetError(f"{family} lacks category breadth")
    return selected


def build_taskset(
    *,
    records: Sequence[Mapping[str, Any]],
    normed: np.ndarray,
    directions: Mapping[str, np.ndarray],
    epicure_provenance: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    eligible = [
        dict(record)
        for record in records
        if record["category"] in CORE_CATEGORIES
        and record["name"] not in CALIBRATION_EXCLUDED_ANCHORS
        and re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", str(record["name"]))
    ]
    records_by_name = {str(record["name"]): record for record in eligible}
    if len(records_by_name) != len(eligible) or normed.shape != (len(records), 300):
        raise EpicureSelectionTasksetError("Epicure source matrix is inconsistent")
    candidates: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    for record in eligible:
        substitution = _substitution_candidate(record, eligible, normed)
        pairing = _pairing_candidate(record, eligible, normed)
        cultural = _cultural_candidate(record, eligible, normed, directions)
        if substitution:
            candidates["substitution"].append(substitution)
        if pairing:
            candidates["pairing"].append(pairing)
        candidates["constraint"].extend(_constraint_candidates(record, eligible, normed))
        if cultural:
            candidates["cultural_composition"].append(cultural)

    used_anchors = set(CALIBRATION_EXCLUDED_ANCHORS)
    selected = {
        family: _select_family(
            family=family, candidates=candidates[family], used_anchors=used_anchors
        )
        for family in FAMILIES
    }
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
                _task_from_candidate(
                    family=family,
                    index=index,
                    candidate=candidate,
                    records_by_name=records_by_name,
                    normed=normed,
                    directions=directions,
                )
            )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "status": "frozen_before_model_execution",
        "selection_seed": SELECTION_SEED,
        "benchmark": "FlavourBench",
        "track": "epicure_scored_combinatorial_culinary_decisions",
        "prompt_revision": {
            "scope": "all cultural-composition items",
            "change": (
                "replace culturally coherent wording with a neutral target-cuisine-label "
                "selection instruction and display the track as regional cuisine selection"
            ),
            "score_maps_changed": False,
            "reason": "remove normative phrasing and avoid construct-irrelevant refusals",
        },
        "calibration_predecessors": {
            "powered_v2": {
                "provider_calls": 20,
                "used_as_primary_data": False,
                "reason": "unrestricted latent-neighbor key failed construct calibration",
            },
            "validated_mcq_v3": {
                "taskset_sha256": (
                    "95aae66e8180510dc446a5f20c54aba00f18f07f3459b3245fbce41904b139da"
                ),
                "plan_sha256": ("409cc17770ae854dda45c8eeebe127206050b18ed35d635701cf9add1e72049e"),
                "provider_calls": 80,
                "used_as_primary_data": False,
                "reason": (
                    "single-key geometry remained brittle; endpoint pilot exposed two stale routes"
                ),
            },
        },
        "excluded_calibration_anchors": list(CALIBRATION_EXCLUDED_ANCHORS),
        "epicure_provenance": dict(epicure_provenance),
        "source_hashes": dict(source_hashes),
        "counts": {
            "tasks": TASK_COUNT,
            "tasks_per_family": TASKS_PER_FAMILY,
            "tasks_per_family_stratum": TASKS_PER_STRATUM,
            "choices_per_task": len(LABELS),
            "selection_size": SELECTION_SIZE,
            "scored_combinations_per_task": len(ALL_SELECTION_KEYS),
            "total_prefrozen_selection_scores": TASK_COUNT * len(ALL_SELECTION_KEYS),
            "unique_anchor_ingredients": TASK_COUNT,
            "human_judgments_required": 0,
        },
        "metric_contract": {
            "primary": "equal_family_macro_mean_epicure_selection_score_0_to_100",
            "per_task": "prefrozen_exact_lookup_over_all_56_three_of_eight_selections",
            "partial_credit": True,
            "invalid_failed_or_unparseable": 0,
            "best_selection": 100,
            "chance_baseline": "exact_taskwise_mean_over_all_56_selections",
            "posthoc_item_exclusion": False,
            "response_normalization": (
                "extract the last exact FINAL_SELECTION line, require three distinct A-H "
                "labels, and sort the selected set before score lookup"
            ),
        },
        "task_set_sha256": _sha256(
            [
                {
                    "task_id": task["task_id"],
                    "prompt_sha256": task["prompt_sha256"],
                    "oracle_reference_sha256": task["oracle_reference_sha256"],
                    "score_map_sha256": _sha256(task["selection_scores_bps"]),
                }
                for task in tasks
            ]
        ),
        "tasks": tasks,
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_taskset(document):
        raise EpicureSelectionTasksetError("constructed task set failed verification")
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
        or len({task.get("anchor_ingredient") for task in tasks}) != TASK_COUNT
        or Counter(task.get("family") for task in tasks)
        != Counter({family: TASKS_PER_FAMILY for family in FAMILIES})
    ):
        return False
    for family in FAMILIES:
        family_tasks = [task for task in tasks if task.get("family") == family]
        if Counter(task.get("validation_stratum") for task in family_tasks) != Counter(
            {stratum: TASKS_PER_STRATUM for stratum in range(1, 5)}
        ):
            return False
    return all(
        isinstance(task, Mapping)
        and hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
        == task.get("prompt_sha256")
        and set(task.get("choices") or {}) == set(LABELS)
        and len(set((task.get("choices") or {}).values())) == len(LABELS)
        and set(task.get("selection_scores_bps") or {}) == set(ALL_SELECTION_KEYS)
        and all(
            isinstance(value, int) and 0 <= value <= 10_000
            for value in (task.get("selection_scores_bps") or {}).values()
        )
        and task.get("selection_scores_bps", {}).get(task.get("optimal_selection")) == 10_000
        and sum(value == 10_000 for value in (task.get("selection_scores_bps") or {}).values()) == 1
        and _sha256(task.get("oracle_reference")) == task.get("oracle_reference_sha256")
        for task in tasks
    )


def parse_final_selection(answer_markdown: str) -> str | None:
    matches = FINAL_SELECTION_PATTERN.findall(answer_markdown)
    if not matches:
        return None
    values = tuple(value.upper() for value in matches[-1])
    if len(set(values)) != SELECTION_SIZE:
        return None
    return "".join(sorted(values))


def score_answer(task: Mapping[str, Any], answer_markdown: str) -> dict[str, Any]:
    observed = parse_final_selection(answer_markdown)
    score_bps = int(task["selection_scores_bps"].get(observed, 0)) if observed else 0
    return {
        "observed_selection": observed,
        "optimal_selection": task["optimal_selection"],
        "parseable": observed is not None,
        "score_bps": score_bps,
        "score": score_bps / 100,
        "optimal": score_bps == 10_000,
    }


def _write(document: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"epicure-selection-taskset-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise EpicureSelectionTasksetError("content-addressed task set conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
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
    records, normed, directions, provenance, source_hashes = _load_inputs(args.epicure_root)
    document = build_taskset(
        records=records,
        normed=normed,
        directions=directions,
        epicure_provenance=provenance,
        source_hashes=source_hashes,
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
