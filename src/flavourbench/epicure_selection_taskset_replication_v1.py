"""Build a response-blind, anchor-free second FlavourBench replication.

The first anchor-free panel retains the 640 task geometries selected for the
original powered study.  This module independently selects another balanced
640-task panel from the same frozen Epicure source.  It maximizes novel anchors,
prohibits duplicate anchors within the replication, and records unavoidable
overlap with the first panel.  No model response is inspected or reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .epicure_native_taskset_v3 import CORE_CATEGORIES, _load_inputs
from .epicure_selection_taskset_v1 import (
    ALL_SELECTION_KEYS,
    CALIBRATION_EXCLUDED_ANCHORS,
    FAMILIES,
    LABELS,
    SELECTION_SIZE,
    TASK_COUNT,
    TASKS_PER_FAMILY,
    TASKS_PER_STRATUM,
    EpicureSelectionTasksetError,
    _constraint_candidates,
    _cultural_candidate,
    _normalize_scores,
    _pairing_candidate,
    _rank_strata,
    _substitution_candidate,
    parse_final_selection,
)
from .epicure_selection_taskset_v2 import (
    CONCRETE_SELECTION_EXAMPLE,
    PROMPT_PROTOCOL,
)
from .epicure_selection_taskset_v2 import (
    verify_taskset as verify_first_panel,
)

SCHEMA_VERSION = "flavourbench-epicure-selection-taskset-v7-replication-2"
DESIGN_VERSION = "flavourbench-executable-culinary-selection-v7-anchor-free-replication-2"
SELECTION_SEED = "flavourbench-executable-selection-replication-2-20260815"


class EpicureSelectionReplicationError(RuntimeError):
    """The independent task-set replication failed verification."""


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


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EpicureSelectionReplicationError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EpicureSelectionReplicationError("task-set input is not a JSON object")
    return value


def _selection_key(*parts: str) -> str:
    return hashlib.sha256((SELECTION_SEED + "\0" + "\0".join(parts)).encode()).hexdigest()


def _choice_permutation(task_id: str, candidates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ordered = sorted(
        (str(candidate["name"]) for candidate in candidates),
        key=lambda name: _selection_key("choice", task_id, name),
    )
    return dict(zip(LABELS, ordered, strict=True))


def _select_family(
    *,
    family: str,
    candidates: Sequence[Mapping[str, Any]],
    selected_anchors: set[str],
    first_panel_anchors: frozenset[str],
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
                if value["stratum"] == stratum and value["anchor"] not in selected_anchors
            ),
            key=lambda value: (
                str(value["anchor"]) in first_panel_anchors,
                _selection_key("select", family, str(stratum), str(value["anchor"])),
            ),
        )
        picked = 0
        for value in eligible:
            category = str(value["category"])
            if category_counts[category] >= 24:
                continue
            selected.append(value)
            category_counts[category] += 1
            selected_anchors.add(str(value["anchor"]))
            picked += 1
            if picked == TASKS_PER_STRATUM:
                break
        if picked != TASKS_PER_STRATUM:
            raise EpicureSelectionReplicationError(
                f"{family} stratum {stratum} has only {picked} unique candidates"
            )
    if len(category_counts) < 8:
        raise EpicureSelectionReplicationError(f"{family} lacks category breadth")
    return selected


def _render_prompt(*, family: str, question: str, choices: Mapping[str, str]) -> str:
    rendered = "\n".join(f"{label}. {choices[label].replace('_', ' ')}" for label in LABELS)
    displayed_family = (
        "regional cuisine selection"
        if family == "cultural_composition"
        else family.replace("_", " ")
    )
    prompt = (
        "FlavourBench executable culinary decision task. Epicure is the deterministic scoring "
        "environment. Select the best portfolio using culinary knowledge; do not browse or call "
        "external tools. Partial credit reflects the quality of the complete selection.\n\n"
        f"Family: {displayed_family}\n{question}\n\nCandidates:\n{rendered}\n\n"
        f"Choose exactly {SELECTION_SIZE} distinct labels. Return exactly one line beginning "
        "with the marker `FINAL_SELECTION:`, followed by the three selected A-through-H labels "
        "separated by commas and ordered alphabetically."
    )
    if CONCRETE_SELECTION_EXAMPLE.search(prompt):
        raise EpicureSelectionReplicationError("replication prompt contains a concrete answer")
    return prompt


def _task_from_candidate(
    *,
    family: str,
    index: int,
    candidate: Mapping[str, Any],
    records_by_name: Mapping[str, Mapping[str, Any]],
    normed: np.ndarray,
    directions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    task_id = f"fb-executable-rep2-{family}-{index:03d}"
    choices = _choice_permutation(task_id, candidate["candidates"])
    anchor_row = int(records_by_name[str(candidate["anchor"])]["row"])
    target_region = candidate.get("target_region")

    def utility(selected: tuple[Mapping[str, Any], ...]) -> float | None:
        rows = [int(value["row"]) for value in selected]
        anchor_similarity = float(
            sum(float(normed[anchor_row] @ normed[row]) for row in rows) / len(rows)
        )
        coherence = float(
            sum(
                float(normed[rows[left]] @ normed[rows[right]])
                for left in range(len(rows))
                for right in range(left + 1, len(rows))
            )
            / math.comb(len(rows), 2)
        )
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
            cultural = float(sum(float(normed[row] @ direction) for row in rows) / len(rows))
            return 0.75 * cultural + 0.25 * coherence
        raise EpicureSelectionReplicationError("unknown task family")

    scores, optimal, chance, margin = _normalize_scores(
        choices=choices, utility=utility, records_by_name=records_by_name
    )
    prompt = _render_prompt(family=family, question=str(candidate["question"]), choices=choices)
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
        "split": "hidden_primary_replication_2",
        "family": family,
        "validation_stratum": int(candidate["stratum"]),
        "anchor_ingredient": candidate["anchor"],
        "primary_category": candidate["category"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_protocol": PROMPT_PROTOCOL,
        "choices": choices,
        "selection_size": SELECTION_SIZE,
        "selection_scores_bps": scores,
        "optimal_selection": optimal,
        "chance_score_bps": chance,
        "optimal_margin_bps": margin,
        "oracle_reference": reference,
        "oracle_reference_sha256": _sha256(reference),
        "scoring": {
            "method": "exact_selection_lookup_bps_v1",
            "minimum_bps": 0,
            "maximum_bps": 10_000,
            "quality_score_inclusion": "successful_and_parseable_only",
            "transport_or_parse_failure": "coverage_only_not_quality_score",
        },
    }


def _task_set_sha256(tasks: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(
        [
            {
                "task_id": task["task_id"],
                "prompt_sha256": task["prompt_sha256"],
                "oracle_reference_sha256": task["oracle_reference_sha256"],
                "score_map_sha256": _sha256(task["selection_scores_bps"]),
            }
            for task in tasks
        ]
    )


def build_taskset(
    *,
    records: Sequence[Mapping[str, Any]],
    normed: np.ndarray,
    directions: Mapping[str, np.ndarray],
    epicure_provenance: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    first_panel: Mapping[str, Any],
    first_panel_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_first_panel(first_panel):
        raise EpicureSelectionReplicationError("replication requires the exact valid first panel")
    first_anchors = {str(task["anchor_ingredient"]) for task in first_panel["tasks"]}
    if len(first_anchors) != TASK_COUNT:
        raise EpicureSelectionReplicationError("first panel anchor set is not unique")
    eligible = [
        dict(record)
        for record in records
        if record["category"] in CORE_CATEGORIES
        and record["name"] not in CALIBRATION_EXCLUDED_ANCHORS
        and re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", str(record["name"]))
    ]
    records_by_name = {str(record["name"]): record for record in eligible}
    if len(records_by_name) != len(eligible) or normed.shape != (len(records), 300):
        raise EpicureSelectionReplicationError("Epicure source matrix is inconsistent")
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

    for family in FAMILIES:
        scoreable: list[dict[str, Any]] = []
        for candidate in candidates[family]:
            candidate_for_check = {**candidate, "stratum": 1}
            try:
                _task_from_candidate(
                    family=family,
                    index=1,
                    candidate=candidate_for_check,
                    records_by_name=records_by_name,
                    normed=normed,
                    directions=directions,
                )
            except EpicureSelectionTasksetError:
                continue
            scoreable.append(candidate)
        candidates[family] = scoreable

    selected_anchors = set(CALIBRATION_EXCLUDED_ANCHORS)
    selected = {
        family: _select_family(
            family=family,
            candidates=candidates[family],
            selected_anchors=selected_anchors,
            first_panel_anchors=frozenset(first_anchors),
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

    second_anchors = {str(task["anchor_ingredient"]) for task in tasks}
    shared_anchors = sorted(first_anchors & second_anchors)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "status": "frozen_before_replication_2_model_execution",
        "selection_seed": SELECTION_SEED,
        "benchmark": "FlavourBench",
        "track": "epicure_scored_combinatorial_culinary_decisions",
        "prompt_protocol": PROMPT_PROTOCOL,
        "replication": {
            "index": 2,
            "selection_is_response_blind": True,
            "first_panel_semantic_sha256": first_panel["artifact_sha256"],
            "first_panel_physical_sha256": first_panel_physical_sha256,
            "first_panel_task_set_sha256": first_panel["task_set_sha256"],
            "first_panel_anchor_count": TASK_COUNT,
            "novel_anchor_count": TASK_COUNT - len(shared_anchors),
            "shared_anchor_count": len(shared_anchors),
            "shared_anchor_sha256": _sha256(shared_anchors),
            "analysis_dependency": (
                "joint-panel uncertainty must cluster first-panel and replication-2 tasks by "
                "anchor ingredient when an anchor is shared"
            ),
            "responses_reused": False,
        },
        "epicure_provenance": dict(epicure_provenance),
        "source_hashes": dict(source_hashes),
        "excluded_calibration_anchors": list(CALIBRATION_EXCLUDED_ANCHORS),
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
            "primary": "equal_family_macro_mean_over_successful_parseable_responses",
            "per_task": "prefrozen_exact_lookup_over_all_56_three_of_eight_selections",
            "partial_credit": True,
            "invalid_failed_or_unparseable": "excluded_from_quality_score",
            "coverage": "valid_scored_divided_by_scheduled_reported_separately",
            "minimum_coverage_for_score": None,
            "dnf_classification": False,
            "posthoc_item_exclusion": False,
        },
        "task_set_sha256": _task_set_sha256(tasks),
        "tasks": tasks,
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_taskset(document, first_panel=first_panel):
        raise EpicureSelectionReplicationError("constructed replication failed verification")
    return document


def verify_taskset(
    document: Mapping[str, Any], *, first_panel: Mapping[str, Any] | None = None
) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    tasks = document.get("tasks")
    metric = document.get("metric_contract") or {}
    replication = document.get("replication") or {}
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("design_version") != DESIGN_VERSION
        or document.get("prompt_protocol") != PROMPT_PROTOCOL
        or recorded != _sha256(payload)
        or not isinstance(tasks, list)
        or len(tasks) != TASK_COUNT
        or len({task.get("task_id") for task in tasks}) != TASK_COUNT
        or len({task.get("anchor_ingredient") for task in tasks}) != TASK_COUNT
        or Counter(task.get("family") for task in tasks)
        != Counter({family: TASKS_PER_FAMILY for family in FAMILIES})
        or replication.get("index") != 2
        or replication.get("selection_is_response_blind") is not True
        or not isinstance(replication.get("shared_anchor_count"), int)
        or metric.get("invalid_failed_or_unparseable") != "excluded_from_quality_score"
        or metric.get("dnf_classification") is not False
        or metric.get("minimum_coverage_for_score") is not None
        or document.get("task_set_sha256") != _task_set_sha256(tasks)
    ):
        return False
    if first_panel is not None:
        first_anchors = {str(task["anchor_ingredient"]) for task in first_panel["tasks"]}
        second_anchors = {str(task["anchor_ingredient"]) for task in tasks}
        shared_anchors = sorted(first_anchors & second_anchors)
        if (
            replication.get("first_panel_semantic_sha256") != first_panel.get("artifact_sha256")
            or replication.get("shared_anchor_count") != len(shared_anchors)
            or replication.get("novel_anchor_count") != TASK_COUNT - len(shared_anchors)
            or replication.get("shared_anchor_sha256") != _sha256(shared_anchors)
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
        and task.get("prompt_protocol") == PROMPT_PROTOCOL
        and not CONCRETE_SELECTION_EXAMPLE.search(str(task.get("prompt") or ""))
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
        if destination.is_symlink() or destination.read_text(encoding="utf-8") != data:
            raise EpicureSelectionReplicationError("content-addressed task-set conflict")
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
    parser.add_argument("--first-panel", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    first_panel = _load(args.first_panel)
    records, normed, directions, provenance, source_hashes = _load_inputs(args.epicure_root)
    document = build_taskset(
        records=records,
        normed=normed,
        directions=directions,
        epicure_provenance=provenance,
        source_hashes=source_hashes,
        first_panel=first_panel,
        first_panel_physical_sha256=_sha256_file(args.first_panel),
    )
    print(_write(document, args.output_directory))


if __name__ == "__main__":
    run()
