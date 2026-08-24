"""Freeze an outcome-blind human-judgment frame for the blocked v6 design.

This module is deliberately offline and fail closed.  It consumes only the exact
content-addressed v6 study-design candidate, the exact human/rank Gate audit, and
the exact stale workload prose being reconciled.  It does not read responses,
scores, preferences, reviewer identities, or live state, and it cannot authorize
generation, contact, judgment collection, ranking, or publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json
from .release_human_gate_audit import verify_audit as verify_gate_audit
from .task_campaign_study_design_successor import (
    verify_successor_design as verify_source_design,
)

SCHEMA_VERSION = "flavourbench-season1-human-judgment-sampling-v1-candidate"
STATUS = "blocked_outcome_blind_sampling_frame_not_authorized"
RECIPE_VERSION = "flavourbench-human-judgment-exact-balance-recipe-v1"

FLAVOURBENCH_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PAPER_ROOT = FLAVOURBENCH_ROOT.parent

SOURCE_DESIGN_SEMANTIC_SHA256 = (
    "e9d31fffbd0e6a7791c04e0cc0b0c4308bfac91745099e0e685c38224479f59e"
)
SOURCE_DESIGN_PHYSICAL_SHA256 = (
    "6affdc8f80e59476254834d8edc588c471a5bd7e86145e66448e4fb7b90118af"
)
GATE_AUDIT_SEMANTIC_SHA256 = (
    "14ed060e870e59713d982b5d533d5fe2d41a71bdc3e32332a4f5ffc97919329a"
)
GATE_AUDIT_PHYSICAL_SHA256 = (
    "a2b80731cfaee70125dc646b7eba7cc1a059170a979481bccdf8a3efbb1c9d20"
)
STALE_REVIEW_PHYSICAL_SHA256 = (
    "33c5fd525dc5f071fef7c829dc81ae9d9063ed6b8ccc16b0fdf9f05cc198b8ee"
)

SOURCE_DESIGN_REFERENCE = (
    "flavourbench/artifacts/season1/study-design-v6-candidate/"
    f"study-design-v6-candidate-{SOURCE_DESIGN_SEMANTIC_SHA256}.json"
)
GATE_AUDIT_REFERENCE = (
    "flavourbench/artifacts/season1/readiness/release-human-gate-v1/"
    f"release-human-gate-audit-{GATE_AUDIT_SEMANTIC_SHA256}.json"
)
STALE_REVIEW_REFERENCE = (
    "governance/reviews/FLAVOURBENCH-HUMAN-STUDY-GO-READINESS-20260809.md"
)

DEFAULT_SOURCE_DESIGN = EVALUATION_PAPER_ROOT / SOURCE_DESIGN_REFERENCE
DEFAULT_GATE_AUDIT = EVALUATION_PAPER_ROOT / GATE_AUDIT_REFERENCE
DEFAULT_STALE_REVIEW = EVALUATION_PAPER_ROOT / STALE_REVIEW_REFERENCE
DEFAULT_OUTPUT_DIR = FLAVOURBENCH_ROOT / (
    "artifacts/season1/human-judgment-sampling-v1-candidate"
)

FAMILIES = ("substitution", "composition", "cookability", "evidence")
TRACKS = ("model_arena", "epicure_uplift")
RATER_SLOTS = (1, 2)
UPLIFT_FAMILY_OFFSETS = {
    "substitution": 0,
    "composition": 4,
    "cookability": 8,
    "evidence": 12,
}
SELECTION_SEED = (
    f"{RECIPE_VERSION}|source={SOURCE_DESIGN_SEMANTIC_SHA256}|"
    f"gate={GATE_AUDIT_SEMANTIC_SHA256}"
)

# Zero-based positions in the source's canonical 35-pair task ordering.  This is
# a reviewed exact-balance recipe, not a PRNG result.  Each row corresponds to
# source design_slot_ordinal 1..80 and contains ten unique positions.
ARENA_PAIR_POSITIONS_ZERO_BASED: tuple[tuple[int, ...], ...] = (
    (2, 4, 6, 7, 9, 13, 14, 29, 32, 34),
    (2, 10, 15, 19, 20, 24, 28, 29, 30, 33),
    (1, 3, 4, 5, 14, 18, 19, 20, 24, 28),
    (1, 3, 8, 14, 18, 20, 25, 28, 29, 32),
    (0, 3, 10, 11, 12, 13, 23, 28, 30, 33),
    (0, 2, 3, 5, 6, 12, 16, 23, 25, 28),
    (0, 1, 4, 10, 13, 19, 20, 24, 29, 30),
    (0, 3, 4, 7, 8, 13, 17, 18, 31, 33),
    (5, 8, 9, 15, 18, 19, 20, 24, 28, 33),
    (0, 1, 2, 4, 5, 19, 21, 22, 24, 25),
    (0, 1, 5, 6, 8, 15, 20, 21, 22, 34),
    (1, 2, 3, 6, 20, 24, 28, 29, 32, 33),
    (1, 4, 12, 14, 25, 29, 30, 31, 32, 34),
    (0, 4, 8, 12, 17, 19, 26, 29, 30, 31),
    (1, 4, 9, 13, 16, 21, 23, 27, 30, 34),
    (2, 4, 9, 13, 21, 23, 24, 31, 32, 34),
    (2, 4, 5, 9, 13, 14, 19, 23, 24, 27),
    (2, 7, 17, 20, 22, 24, 26, 27, 29, 31),
    (8, 13, 15, 17, 18, 21, 27, 28, 31, 32),
    (2, 3, 5, 6, 7, 8, 10, 12, 25, 27),
    (1, 2, 4, 11, 12, 21, 22, 25, 26, 32),
    (7, 13, 16, 21, 23, 25, 27, 30, 32, 34),
    (3, 5, 7, 8, 9, 10, 12, 19, 23, 26),
    (4, 11, 14, 25, 26, 27, 29, 30, 31, 32),
    (1, 12, 13, 14, 15, 16, 17, 18, 19, 26),
    (1, 3, 4, 5, 8, 12, 18, 19, 21, 22),
    (1, 5, 6, 7, 8, 9, 10, 12, 13, 27),
    (2, 6, 13, 15, 17, 22, 27, 28, 32, 34),
    (2, 6, 9, 13, 16, 17, 23, 24, 27, 28),
    (1, 2, 15, 16, 19, 24, 26, 28, 29, 33),
    (4, 5, 11, 13, 14, 15, 21, 24, 25, 34),
    (0, 6, 15, 17, 25, 26, 27, 28, 29, 30),
    (7, 8, 15, 17, 18, 20, 25, 27, 28, 30),
    (2, 3, 8, 13, 17, 24, 27, 28, 31, 33),
    (4, 7, 10, 13, 14, 17, 18, 19, 24, 29),
    (2, 7, 9, 13, 16, 24, 27, 28, 29, 30),
    (0, 2, 3, 4, 5, 6, 14, 16, 23, 33),
    (4, 5, 8, 10, 21, 23, 24, 25, 27, 34),
    (0, 3, 17, 22, 26, 28, 29, 30, 31, 33),
    (1, 2, 4, 9, 11, 13, 24, 27, 28, 32),
    (3, 7, 8, 13, 17, 18, 24, 28, 33, 34),
    (1, 4, 5, 14, 19, 21, 24, 26, 27, 30),
    (0, 4, 11, 13, 15, 18, 21, 22, 24, 25),
    (2, 11, 12, 13, 15, 22, 23, 24, 28, 33),
    (1, 2, 12, 17, 19, 21, 22, 23, 26, 32),
    (5, 7, 8, 18, 22, 26, 27, 29, 31, 33),
    (1, 6, 15, 22, 25, 26, 27, 28, 29, 32),
    (0, 10, 11, 13, 15, 16, 25, 27, 30, 33),
    (1, 2, 3, 4, 5, 6, 12, 14, 16, 34),
    (1, 2, 3, 5, 6, 7, 16, 27, 29, 33),
    (1, 3, 5, 12, 18, 19, 21, 22, 28, 32),
    (4, 7, 9, 17, 19, 20, 30, 31, 32, 34),
    (1, 8, 11, 12, 13, 15, 28, 29, 30, 34),
    (6, 10, 14, 16, 19, 21, 23, 27, 32, 33),
    (0, 3, 9, 10, 13, 17, 23, 27, 30, 31),
    (0, 3, 6, 7, 10, 13, 17, 26, 30, 33),
    (1, 3, 8, 17, 18, 21, 27, 28, 31, 32),
    (14, 15, 17, 18, 19, 23, 28, 30, 32, 33),
    (0, 2, 3, 5, 9, 13, 18, 22, 23, 27),
    (2, 3, 7, 9, 11, 18, 22, 31, 32, 33),
    (0, 1, 2, 6, 15, 20, 24, 27, 28, 29),
    (6, 7, 8, 9, 10, 18, 20, 22, 27, 29),
    (4, 5, 8, 9, 12, 13, 15, 18, 19, 28),
    (0, 3, 7, 11, 12, 22, 25, 26, 31, 32),
    (0, 1, 2, 19, 20, 21, 22, 24, 29, 34),
    (0, 4, 9, 11, 12, 17, 19, 26, 30, 31),
    (6, 10, 20, 21, 22, 23, 24, 25, 32, 34),
    (1, 3, 13, 14, 17, 18, 22, 24, 27, 29),
    (2, 9, 11, 13, 20, 23, 24, 28, 30, 33),
    (1, 5, 8, 10, 16, 18, 19, 20, 21, 33),
    (4, 5, 7, 9, 18, 22, 24, 25, 27, 31),
    (0, 1, 4, 8, 11, 12, 21, 25, 26, 34),
    (0, 9, 10, 11, 15, 17, 18, 19, 20, 30),
    (5, 6, 8, 9, 19, 23, 26, 28, 30, 33),
    (1, 14, 17, 18, 19, 20, 21, 22, 24, 32),
    (2, 7, 9, 10, 17, 19, 20, 21, 24, 32),
    (5, 8, 18, 20, 21, 23, 24, 25, 28, 34),
    (2, 5, 12, 14, 16, 19, 20, 21, 23, 25),
    (10, 14, 17, 19, 20, 21, 22, 23, 24, 25),
    (0, 3, 4, 13, 16, 17, 26, 30, 31, 33),
)


class HumanSamplingError(RuntimeError):
    """A bound source or exact sampling invariant failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanSamplingError(message)


def _physical_sha256(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"source is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, *, physical_sha256: str, semantic_sha256: str) -> dict[str, Any]:
    _require(_physical_sha256(path) == physical_sha256, f"physical digest mismatch: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HumanSamplingError(f"invalid JSON source: {path}") from error
    _require(isinstance(value, dict), f"JSON source is not an object: {path}")
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    _require(
        value.get("artifact_sha256") == semantic_sha256
        and sha256_json(body) == semantic_sha256,
        f"semantic digest mismatch: {path}",
    )
    return value


def _load_sources(
    design_path: Path,
    gate_path: Path,
    stale_review_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    design = _load_json(
        design_path,
        physical_sha256=SOURCE_DESIGN_PHYSICAL_SHA256,
        semantic_sha256=SOURCE_DESIGN_SEMANTIC_SHA256,
    )
    gate = _load_json(
        gate_path,
        physical_sha256=GATE_AUDIT_PHYSICAL_SHA256,
        semantic_sha256=GATE_AUDIT_SEMANTIC_SHA256,
    )
    verify_source_design(design)
    _require(verify_gate_audit(gate), "human/rank Gate audit does not verify")
    _verify_design_boundary(design)
    _verify_gate_boundary(gate)

    _require(
        _physical_sha256(stale_review_path) == STALE_REVIEW_PHYSICAL_SHA256,
        "stale workload review physical digest mismatch",
    )
    stale_text = stale_review_path.read_text(encoding="utf-8")
    _require(
        stale_text.count("| 3,072 output judgments at ") == 2,
        "the exact stale 3,072-judgment wording is absent or changed",
    )
    return design, gate


def _verify_design_boundary(design: Mapping[str, Any]) -> None:
    review = design.get("prospective_human_review_floor")
    primary = design.get("primary_schedule")
    panel = design.get("candidate_model_panel")
    boundary = design.get("claim_boundary")
    _require(
        design.get("schema_version") == "flavourbench-season1-study-design-v6-candidate",
        "source study-design schema changed",
    )
    _require(
        isinstance(review, Mapping)
        and review.get("unique_comparisons")
        == {"model_arena": 800, "epicure_uplift": 800}
        and review.get("independent_raters_per_comparison") == 2
        and review.get("primary_judgments") == 3200
        and review.get("additional_repeat_presentations") == 400
        and review.get("total_rating_presentations") == 3600
        and review.get("authorized") is False,
        "source human-review floor changed",
    )
    _require(
        isinstance(primary, Mapping)
        and primary.get("power_validated") is False
        and primary.get("model_arena", {}).get("total_battles") == 2800
        and primary.get("epicure_uplift", {}).get("total_pairs") == 2800,
        "source generation schedule changed",
    )
    _require(
        isinstance(panel, Mapping)
        and panel.get("model_count") == 14
        and panel.get("official_roster") is False
        and panel.get("quality_observations_used") == 0,
        "source candidate panel crossed its evidence boundary",
    )
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("official") is False
        and boundary.get("rank_eligible") is False
        and boundary.get("human_contact_authorized") is False
        and boundary.get("model_or_epicure_calls_authorized") is False
        and boundary.get("human_judgments") == 0
        and boundary.get("quality_observations") == 0,
        "source study-design claim boundary changed",
    )


def _verify_gate_boundary(gate: Mapping[str, Any]) -> None:
    human = gate.get("human_rank_readiness")
    boundary = gate.get("claim_boundary")
    _require(
        gate.get("overall_decision") == "NO_GO_OFFICIAL_RANKING"
        and isinstance(human, Mapping)
        and human.get("decision") == "NO_GO",
        "human/rank Gate is not NO-GO",
    )
    thresholds = human.get("prospective_expert_judgment_thresholds")
    _require(
        isinstance(thresholds, Mapping)
        and thresholds.get("minimum_unique_comparisons")
        == {"epicure_uplift": 800, "model_arena": 800}
        and thresholds.get("distinct_independent_raters_per_comparison") == 2
        and thresholds.get("reliability_repeat_rate") == 0.125
        and thresholds.get("current_admissible_judgments") == 0,
        "human/rank Gate thresholds changed",
    )
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("official_result_supported") is False
        and boundary.get("rank_eligible") is False
        and boundary.get("provider_calls_made") == 0
        and boundary.get("epicure_calls_made") == 0
        and boundary.get("reviewers_invented") == 0
        and boundary.get("judgments_invented") == 0,
        "human/rank Gate claim boundary changed",
    )


def _models(design: Mapping[str, Any]) -> list[str]:
    panel = design["candidate_model_panel"]
    rows = panel["models"]
    model_ids = [str(row["model_id"]) for row in rows]
    _require(len(model_ids) == 14 and len(set(model_ids)) == 14, "model panel is not exact")
    return model_ids


def _arena_candidates(design: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    arena = design["primary_schedule"]["model_arena"]
    factorization = arena["factorization"]["matchings"]
    matchings: dict[int, list[tuple[str, str]]] = {}
    for matching in factorization:
        index = int(matching["matching_index_zero_based"])
        matchings[index] = [tuple(sorted(pair)) for pair in matching["model_pairs"]]
    _require(set(matchings) == set(range(13)), "source one-factorization indices changed")
    _require(all(len(rows) == 7 for rows in matchings.values()), "source matching size changed")

    result: list[list[dict[str, Any]]] = []
    slots = arena["abstract_task_schedule"]
    _require(len(slots) == 80, "source arena task count changed")
    for expected_ordinal, slot in enumerate(slots, start=1):
        _require(slot["design_slot_ordinal"] == expected_ordinal, "arena slot order changed")
        candidates: list[dict[str, Any]] = []
        for matching_index in slot["matching_indices_zero_based"]:
            for matching_pair_index, pair in enumerate(matchings[matching_index]):
                candidates.append(
                    {
                        "model_ids": pair,
                        "source_matching_index_zero_based": matching_index,
                        "source_matching_pair_index_zero_based": matching_pair_index,
                    }
                )
        _require(
            len(candidates) == 35
            and len({row["model_ids"] for row in candidates}) == 35,
            "source arena task does not have 35 distinct generated pairs",
        )
        result.append(candidates)
    return result


def _comparison_id(identity: Mapping[str, Any]) -> str:
    return f"comparison-{sha256_json(identity)}"


def _judgment_slot_id(comparison_id: str, rater_slot: int) -> str:
    identity = {"comparison_id": comparison_id, "rater_slot": rater_slot}
    return f"judgment-slot-{sha256_json(identity)}"


def _arena_frame(design: Mapping[str, Any], recipe_sha256: str) -> list[dict[str, Any]]:
    arena = design["primary_schedule"]["model_arena"]
    slots = arena["abstract_task_schedule"]
    candidates_by_task = _arena_candidates(design)
    _require(len(ARENA_PAIR_POSITIONS_ZERO_BASED) == 80, "arena recipe must have 80 rows")
    rows: list[dict[str, Any]] = []
    for slot, candidates, positions in zip(
        slots,
        candidates_by_task,
        ARENA_PAIR_POSITIONS_ZERO_BASED,
        strict=True,
    ):
        _require(
            len(positions) == 10
            and len(set(positions)) == 10
            and tuple(sorted(positions)) == positions
            and min(positions) >= 0
            and max(positions) < 35,
            "arena position recipe is malformed",
        )
        for position in positions:
            source = candidates[position]
            identity = {
                "recipe_sha256": recipe_sha256,
                "source_design_semantic_sha256": SOURCE_DESIGN_SEMANTIC_SHA256,
                "track": "model_arena",
                "design_slot_ordinal": slot["design_slot_ordinal"],
                "source_generated_pair_index_zero_based": position,
                "model_ids": list(source["model_ids"]),
            }
            comparison_id = _comparison_id(identity)
            rows.append(
                {
                    **identity,
                    "comparison_id": comparison_id,
                    "family": slot["family"],
                    "family_slot_ordinal": slot["family_slot_ordinal"],
                    "source_matching_index_zero_based": source[
                        "source_matching_index_zero_based"
                    ],
                    "source_matching_pair_index_zero_based": source[
                        "source_matching_pair_index_zero_based"
                    ],
                    "required_distinct_raters": 2,
                    "judgment_slot_ids": [
                        _judgment_slot_id(comparison_id, rater_slot)
                        for rater_slot in RATER_SLOTS
                    ],
                }
            )
    return rows


def _uplift_repetition_index(
    *,
    design_slot_ordinal: int,
    model_id: str,
    available_repetitions: int,
) -> int:
    payload = (
        f"{SELECTION_SEED}|uplift-existing-repetition|"
        f"task={design_slot_ordinal}|model={model_id}"
    ).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return 1 + value % available_repetitions


def _uplift_frame(design: Mapping[str, Any], recipe_sha256: str) -> list[dict[str, Any]]:
    models = _models(design)
    uplift = design["primary_schedule"]["epicure_uplift"]
    slots = uplift["abstract_task_schedule"]
    _require(len(slots) == 80, "source uplift task count changed")
    rows: list[dict[str, Any]] = []
    for expected_ordinal, slot in enumerate(slots, start=1):
        _require(slot["design_slot_ordinal"] == expected_ordinal, "uplift slot order changed")
        family = slot["family"]
        offset = UPLIFT_FAMILY_OFFSETS[family]
        selected_indices = sorted(
            {
                (10 * (slot["family_slot_ordinal"] - 1) + offset + delta) % 14
                for delta in range(10)
            }
        )
        _require(len(selected_indices) == 10, "uplift task does not select ten models")
        third_repetition = set(slot["models_with_third_repetition"])
        for model_index in selected_indices:
            model_id = models[model_index]
            available_repetitions = 2 + int(model_id in third_repetition)
            repetition_index = _uplift_repetition_index(
                design_slot_ordinal=expected_ordinal,
                model_id=model_id,
                available_repetitions=available_repetitions,
            )
            identity = {
                "recipe_sha256": recipe_sha256,
                "source_design_semantic_sha256": SOURCE_DESIGN_SEMANTIC_SHA256,
                "track": "epicure_uplift",
                "design_slot_ordinal": expected_ordinal,
                "model_id": model_id,
                "source_generated_repetition_index_one_based": repetition_index,
            }
            comparison_id = _comparison_id(identity)
            rows.append(
                {
                    **identity,
                    "comparison_id": comparison_id,
                    "family": family,
                    "family_slot_ordinal": slot["family_slot_ordinal"],
                    "conditions": ["epicure_off", "epicure_on"],
                    "source_generated_repetitions_available": available_repetitions,
                    "required_distinct_raters": 2,
                    "judgment_slot_ids": [
                        _judgment_slot_id(comparison_id, rater_slot)
                        for rater_slot in RATER_SLOTS
                    ],
                }
            )
    return rows


def _repeat_frame(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[tuple[int, str, int], list[tuple[str, str]]] = defaultdict(list)
    for row in comparisons:
        for rater_slot, judgment_slot_id in zip(
            RATER_SLOTS,
            row["judgment_slot_ids"],
            strict=True,
        ):
            key = (int(row["design_slot_ordinal"]), str(row["track"]), rater_slot)
            by_category[key].append((str(row["comparison_id"]), str(judgment_slot_id)))

    category_order = (
        ("model_arena", 1),
        ("model_arena", 2),
        ("epicure_uplift", 1),
        ("epicure_uplift", 2),
    )
    repeats: list[dict[str, Any]] = []
    for task_ordinal in range(1, 81):
        extra_category = (task_ordinal - 1) % len(category_order)
        for category_index, (track, rater_slot) in enumerate(category_order):
            candidates = by_category[task_ordinal, track, rater_slot]
            _require(len(candidates) == 10, "repeat source category does not contain ten slots")
            count = 1 + int(category_index == extra_category)
            ranked = sorted(
                candidates,
                key=lambda item: hashlib.sha256(
                    f"{SELECTION_SEED}|concealed-repeat|{item[1]}".encode()
                ).hexdigest(),
            )
            for comparison_id, judgment_slot_id in ranked[:count]:
                repeat_identity = {
                    "recipe_version": RECIPE_VERSION,
                    "source_judgment_slot_id": judgment_slot_id,
                    "repeat_ordinal_for_source": 1,
                }
                repeats.append(
                    {
                        "repeat_presentation_id": (
                            f"repeat-presentation-{sha256_json(repeat_identity)}"
                        ),
                        "source_comparison_id": comparison_id,
                        "source_judgment_slot_id": judgment_slot_id,
                        "design_slot_ordinal": task_ordinal,
                        "track": track,
                        "rater_slot": rater_slot,
                        "same_rater_as_source_required": True,
                    }
                )
    return repeats


def _recipe() -> dict[str, Any]:
    body = {
        "recipe_version": RECIPE_VERSION,
        "source_design_semantic_sha256": SOURCE_DESIGN_SEMANTIC_SHA256,
        "selection_seed": SELECTION_SEED,
        "arena": {
            "source_candidate_order": (
                "for each task, source matching_indices_zero_based order, then source "
                "factorization model_pairs order"
            ),
            "source_candidates_per_task": 35,
            "selected_per_task": 10,
            "positions_are_zero_based": True,
            "positions_by_design_slot_ordinal": [
                list(row) for row in ARENA_PAIR_POSITIONS_ZERO_BASED
            ],
        },
        "uplift": {
            "source_model_order": "candidate_model_panel.models order",
            "selected_distinct_models_per_task": 10,
            "model_index_formula": (
                "(10 * (family_slot_ordinal - 1) + family_offset + delta) mod 14, "
                "for delta=0..9"
            ),
            "family_offsets": UPLIFT_FAMILY_OFFSETS,
            "repetition_index_formula": (
                "1 + uint64_be(sha256(selection_seed | uplift-existing-repetition | "
                "task | model)[0:8]) mod source_generated_repetitions_available"
            ),
        },
        "rater_slots": {
            "abstract_slots_per_comparison": list(RATER_SLOTS),
            "distinct_people_required_across_slots": True,
            "reviewer_identities_assigned": 0,
        },
        "concealed_repeats": {
            "rate_of_primary_judgment_slots": "0.125",
            "repeats_per_task": 5,
            "category_order": [
                {"track": track, "rater_slot": rater_slot}
                for track, rater_slot in (
                    ("model_arena", 1),
                    ("model_arena", 2),
                    ("epicure_uplift", 1),
                    ("epicure_uplift", 2),
                )
            ],
            "selection_rule": (
                "select one SHA-256-ranked source slot per category and a second from category "
                "(design_slot_ordinal - 1) mod 4"
            ),
            "same_rater_as_source_required": True,
        },
    }
    return {**body, "recipe_sha256": sha256_json(body)}


def _is_connected(models: Sequence[str], pairs: Sequence[tuple[str, str]]) -> bool:
    adjacency: dict[str, set[str]] = {model: set() for model in models}
    for left, right in pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited: set[str] = set()
    pending = [models[0]]
    while pending:
        model = pending.pop()
        if model in visited:
            continue
        visited.add(model)
        pending.extend(adjacency[model] - visited)
    return visited == set(models)


def _distribution(values: Sequence[int]) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(values).items())}


def _balance_certificate(
    design: Mapping[str, Any],
    arena: Sequence[Mapping[str, Any]],
    uplift: Sequence[Mapping[str, Any]],
    repeats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    models = _models(design)
    arena_by_task: Counter[int] = Counter()
    arena_by_family: Counter[str] = Counter()
    arena_pairs: Counter[tuple[str, str]] = Counter()
    arena_family_pairs: dict[str, Counter[tuple[str, str]]] = {
        family: Counter() for family in FAMILIES
    }
    arena_models: Counter[str] = Counter()
    arena_family_models: dict[str, Counter[str]] = {
        family: Counter() for family in FAMILIES
    }
    arena_task_models: dict[int, Counter[str]] = defaultdict(Counter)
    for row in arena:
        task = int(row["design_slot_ordinal"])
        family = str(row["family"])
        pair = tuple(row["model_ids"])
        arena_by_task[task] += 1
        arena_by_family[family] += 1
        arena_pairs[pair] += 1
        arena_family_pairs[family][pair] += 1
        arena_models.update(pair)
        arena_family_models[family].update(pair)
        arena_task_models[task].update(pair)

    _require(len(arena) == 800, "arena selection count is not 800")
    _require(set(arena_by_task.values()) == {10}, "arena task count is not exactly ten")
    _require(arena_by_family == Counter({family: 200 for family in FAMILIES}), "arena family count")
    _require(len(arena_pairs) == 91 and set(arena_pairs.values()) == {8, 9}, "arena pair balance")
    _require(set(arena_models.values()) == {114, 115}, "arena global model balance")
    _require(
        all(
            len(counts) == 91 and set(counts.values()) == {2, 3}
            for counts in arena_family_pairs.values()
        ),
        "arena family-pair balance",
    )
    _require(
        all(set(counts.values()) == {28, 29} for counts in arena_family_models.values()),
        "arena family-model balance",
    )
    _require(
        all(
            set(counts.values()) == {1, 2}
            and Counter(counts.values()) == Counter({1: 8, 2: 6})
            for counts in arena_task_models.values()
        ),
        "arena task-model degree balance",
    )
    _require(
        _is_connected(models, list(arena_pairs))
        and all(
            _is_connected(models, list(arena_family_pairs[family])) for family in FAMILIES
        ),
        "arena graph connectivity",
    )

    uplift_by_task: Counter[int] = Counter()
    uplift_by_family: Counter[str] = Counter()
    uplift_models: Counter[str] = Counter()
    uplift_family_models: dict[str, Counter[str]] = {
        family: Counter() for family in FAMILIES
    }
    uplift_task_models: dict[int, set[str]] = defaultdict(set)
    for row in uplift:
        task = int(row["design_slot_ordinal"])
        family = str(row["family"])
        model = str(row["model_id"])
        repetition = int(row["source_generated_repetition_index_one_based"])
        available = int(row["source_generated_repetitions_available"])
        _require(1 <= repetition <= available and available in {2, 3}, "uplift repetition index")
        uplift_by_task[task] += 1
        uplift_by_family[family] += 1
        uplift_models[model] += 1
        uplift_family_models[family][model] += 1
        uplift_task_models[task].add(model)

    _require(len(uplift) == 800, "uplift selection count is not 800")
    _require(set(uplift_by_task.values()) == {10}, "uplift task count is not exactly ten")
    _require(
        all(len(selected) == 10 for selected in uplift_task_models.values()),
        "uplift models are not distinct within task",
    )
    _require(
        uplift_by_family == Counter({family: 200 for family in FAMILIES}),
        "uplift family count",
    )
    _require(set(uplift_models.values()) == {57, 58}, "uplift global model balance")
    _require(
        all(set(counts.values()) == {14, 15} for counts in uplift_family_models.values()),
        "uplift family-model balance",
    )

    comparisons = [*arena, *uplift]
    comparison_ids = [str(row["comparison_id"]) for row in comparisons]
    judgment_slot_ids = [
        str(slot_id) for row in comparisons for slot_id in row["judgment_slot_ids"]
    ]
    _require(len(set(comparison_ids)) == 1600, "comparison IDs are not unique")
    _require(len(judgment_slot_ids) == 3200, "primary judgment-slot count is not 3,200")
    _require(len(set(judgment_slot_ids)) == 3200, "judgment-slot IDs are not unique")
    _require(len(repeats) == 400, "concealed repeat count is not 400")
    _require(
        len({row["repeat_presentation_id"] for row in repeats}) == 400,
        "repeat presentation IDs are not unique",
    )
    repeat_by_task = Counter(int(row["design_slot_ordinal"]) for row in repeats)
    repeat_by_track = Counter(str(row["track"]) for row in repeats)
    repeat_by_rater_slot = Counter(int(row["rater_slot"]) for row in repeats)
    _require(set(repeat_by_task.values()) == {5}, "concealed repeats are not five per task")
    _require(repeat_by_track == Counter({track: 200 for track in TRACKS}), "repeat track balance")
    _require(
        repeat_by_rater_slot == Counter({slot: 200 for slot in RATER_SLOTS}),
        "repeat abstract-rater-slot balance",
    )

    return {
        "arena": {
            "unique_comparisons": 800,
            "comparisons_per_task": 10,
            "comparisons_per_family": {family: 200 for family in FAMILIES},
            "is_subset_of_35_generated_pairs_per_task": True,
            "task_model_degree_distribution_each_task": {"1": 8, "2": 6},
            "global_pair_repetition_distribution": _distribution(list(arena_pairs.values())),
            "family_pair_repetition_distribution": {
                family: _distribution(list(arena_family_pairs[family].values()))
                for family in FAMILIES
            },
            "global_model_appearance_distribution": _distribution(list(arena_models.values())),
            "family_model_appearance_distribution": {
                family: _distribution(list(arena_family_models[family].values()))
                for family in FAMILIES
            },
            "full_graph": {"nodes": 14, "distinct_edges": 91, "connected": True},
            "family_graphs": {
                family: {"nodes": 14, "distinct_edges": 91, "connected": True}
                for family in FAMILIES
            },
            "exact_pair_counts_sha256": sha256_json(
                {" || ".join(pair): count for pair, count in sorted(arena_pairs.items())}
            ),
            "exact_family_pair_counts_sha256": sha256_json(
                {
                    family: {
                        " || ".join(pair): count
                        for pair, count in sorted(arena_family_pairs[family].items())
                    }
                    for family in FAMILIES
                }
            ),
            "exact_selected_comparison_ids_sha256": sha256_json(sorted(comparison_ids[:800])),
        },
        "uplift": {
            "unique_comparisons": 800,
            "distinct_models_per_task": 10,
            "comparisons_per_family": {family: 200 for family in FAMILIES},
            "global_model_appearance_distribution": _distribution(list(uplift_models.values())),
            "family_model_appearance_distribution": {
                family: _distribution(list(uplift_family_models[family].values()))
                for family in FAMILIES
            },
            "selected_repetition_exists_in_source_generation_schedule": True,
            "exact_selected_comparison_ids_sha256": sha256_json(sorted(comparison_ids[800:])),
        },
        "human_presentations": {
            "unique_comparisons": 1600,
            "distinct_rater_slots_per_comparison": 2,
            "primary_judgment_slots": 3200,
            "concealed_repeat_presentations": 400,
            "total_rating_presentations": 3600,
            "repeats_per_task": 5,
            "repeat_presentations_per_track": {track: 200 for track in TRACKS},
            "repeat_presentations_per_abstract_rater_slot": {"1": 200, "2": 200},
            "exact_primary_judgment_slot_ids_sha256": sha256_json(sorted(judgment_slot_ids)),
            "exact_repeat_presentation_ids_sha256": sha256_json(
                sorted(str(row["repeat_presentation_id"]) for row in repeats)
            ),
        },
    }


def _source_commitments() -> list[dict[str, Any]]:
    return [
        {
            "role": "source_14_model_80_scored_task_study_design_v6_candidate",
            "reference_path": SOURCE_DESIGN_REFERENCE,
            "schema_version": "flavourbench-season1-study-design-v6-candidate",
            "semantic_sha256": SOURCE_DESIGN_SEMANTIC_SHA256,
            "physical_sha256": SOURCE_DESIGN_PHYSICAL_SHA256,
        },
        {
            "role": "authoritative_human_and_rank_gate",
            "reference_path": GATE_AUDIT_REFERENCE,
            "schema_version": "flavourbench-release-human-gate-audit-v1",
            "semantic_sha256": GATE_AUDIT_SEMANTIC_SHA256,
            "physical_sha256": GATE_AUDIT_PHYSICAL_SHA256,
        },
        {
            "role": "stale_non_authoritative_workload_wording_reconciled_not_modified",
            "reference_path": STALE_REVIEW_REFERENCE,
            "schema_version": "plain_markdown_no_semantic_address",
            "semantic_sha256": None,
            "physical_sha256": STALE_REVIEW_PHYSICAL_SHA256,
        },
    ]


def _build_body(design: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    del gate  # The exact Gate was already checked; only its frozen thresholds are consumed below.
    recipe = _recipe()
    recipe_sha256 = str(recipe["recipe_sha256"])
    arena = _arena_frame(design, recipe_sha256)
    uplift = _uplift_frame(design, recipe_sha256)
    comparisons = [*arena, *uplift]
    repeats = _repeat_frame(comparisons)
    certificate = _balance_certificate(design, arena, uplift, repeats)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "artifact_role": "compressed_exact_sampling_frame_and_balance_certificate_only",
        "source_commitments": _source_commitments(),
        "selection_timing_and_inputs": {
            "outcome_blind": True,
            "must_be_frozen_before_any_model_output_or_quality_outcome": True,
            "allowed_inputs": [
                "source model order and immutable model IDs",
                "source abstract scored-task slot ordinals and families",
                "source arena one-factorization and five-matchings-per-task schedule",
                "source uplift per-model generated repetition counts",
                "authoritative Gate comparison and independent-rater floors",
                "this versioned coordinate recipe and SHA-256 rules",
            ],
            "response_texts_used": 0,
            "model_scores_used": 0,
            "preference_labels_used": 0,
            "quality_observations_used": 0,
            "reviewer_identities_used": 0,
            "outcome_dependent_reselection_allowed": False,
            "task_binding": (
                "the v6 source contains 80 abstract scored-task slots, not admitted task IDs; "
                "a future pre-output task-split freeze may bind one task ID to each ordinal "
                "without changing any selected comparison coordinate"
            ),
        },
        "sampling_recipe": recipe,
        "balance_certificate": certificate,
        "materialization_contract": {
            "implementation": (
                "flavourbench.task_campaign_human_sampling_successor.materialize_sampling_frame"
            ),
            "compressed_frame_is_exact": True,
            "materialized_arena_comparisons": 800,
            "materialized_uplift_comparisons": 800,
            "materialized_primary_judgment_slots": 3200,
            "materialized_concealed_repeat_presentations": 400,
            "actual_reviewer_assignments": 0,
        },
        "judgment_count_reconciliation": {
            "authoritative_gate_arithmetic": "(800 arena + 800 uplift) * 2 raters = 3,200",
            "authoritative_primary_judgments": 3200,
            "source_design_primary_judgments": 3200,
            "stale_workload_wording_judgments": 3072,
            "difference": 128,
            "disposition": (
                "3,072 is stale non-authoritative budget-scenario prose; it is not a sampling "
                "floor. The machine Gate and source v6 design agree on 3,200 primary judgments."
            ),
            "stale_document_modified": False,
        },
        "validation_status": {
            "combinatorial_feasibility_validated": True,
            "source_subset_and_existing_repetition_membership_validated": True,
            "power_validated": False,
            "precision_validated": False,
            "type_i_error_validated": False,
            "missingness_validated": False,
            "cost_validated": False,
        },
        "remaining_simulation_power_and_cost_work": {
            "simulation_and_power": [
                "define primary and family estimands for the 800/800 subsample",
                "simulate minimum detectable effects and interval width under task clustering",
                "validate type-I error and multiplicity control for all planned contrasts",
                "validate coverage under ties, rater dependence, disagreement, and heterogeneity",
                "stress-test missing responses, missing ratings, and nonrandom reviewer dropout",
                "compare the sampled arena graph with the full 2,800-battle generation graph",
                "validate the 12.5% repeat design and prespecify reliability exclusion rules",
            ],
            "cost_and_operations": [
                "measure compensated rating, training, calibration, and adjudication time",
                "set jurisdiction-aware wage floors and a funded compensation authority",
                "budget recruitment, platform fees, taxes, withdrawal, retention, and contingency",
                "reprice all 13,300 planned generation arms against exact frozen routes",
                "approve separate provider and human-study cost envelopes before activation",
            ],
        },
        "claim_boundary": {
            "activation_effect": "none",
            "official": False,
            "rank_eligible": False,
            "model_calls_authorized": False,
            "epicure_calls_authorized": False,
            "human_contact_authorized": False,
            "human_judgment_collection_authorized": False,
            "compensation_or_spend_authorized": False,
            "quality_evidence_observed": False,
            "quality_observations": 0,
            "human_judgments": 0,
            "reviewer_identities_assigned": 0,
            "research_result": False,
            "paper_or_public_claim_authorized": False,
        },
    }


def build_sampling_artifact(
    *,
    design_path: Path = DEFAULT_SOURCE_DESIGN,
    gate_path: Path = DEFAULT_GATE_AUDIT,
    stale_review_path: Path = DEFAULT_STALE_REVIEW,
) -> dict[str, Any]:
    """Build the deterministic blocked sampling artifact from exact local sources."""

    design, gate = _load_sources(design_path, gate_path, stale_review_path)
    body = _build_body(design, gate)
    document = {**body, "artifact_sha256": sha256_json(body)}
    verify_sampling_artifact(
        document,
        design_path=design_path,
        gate_path=gate_path,
        stale_review_path=stale_review_path,
    )
    return document


def verify_sampling_artifact(
    document: Mapping[str, Any],
    *,
    design_path: Path = DEFAULT_SOURCE_DESIGN,
    gate_path: Path = DEFAULT_GATE_AUDIT,
    stale_review_path: Path = DEFAULT_STALE_REVIEW,
) -> None:
    """Fail unless the artifact exactly matches its sources and reviewed recipe."""

    _require(isinstance(document, Mapping), "sampling artifact must be an object")
    recorded = document.get("artifact_sha256")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(
        isinstance(recorded, str) and recorded == sha256_json(body),
        "sampling artifact semantic digest mismatch",
    )
    design, gate = _load_sources(design_path, gate_path, stale_review_path)
    _require(body == _build_body(design, gate), "sampling artifact differs from exact recipe")


def materialize_sampling_frame(
    document: Mapping[str, Any],
    *,
    design_path: Path = DEFAULT_SOURCE_DESIGN,
    gate_path: Path = DEFAULT_GATE_AUDIT,
    stale_review_path: Path = DEFAULT_STALE_REVIEW,
) -> dict[str, list[dict[str, Any]]]:
    """Return all exact comparisons, abstract judgment slots, and repeats."""

    verify_sampling_artifact(
        document,
        design_path=design_path,
        gate_path=gate_path,
        stale_review_path=stale_review_path,
    )
    design, _ = _load_sources(design_path, gate_path, stale_review_path)
    recipe_sha256 = str(document["sampling_recipe"]["recipe_sha256"])
    arena = _arena_frame(design, recipe_sha256)
    uplift = _uplift_frame(design, recipe_sha256)
    comparisons = [*arena, *uplift]
    judgment_slots = [
        {
            "judgment_slot_id": judgment_slot_id,
            "comparison_id": row["comparison_id"],
            "track": row["track"],
            "design_slot_ordinal": row["design_slot_ordinal"],
            "rater_slot": rater_slot,
            "distinct_from_other_rater_slot_for_comparison": True,
        }
        for row in comparisons
        for rater_slot, judgment_slot_id in zip(
            RATER_SLOTS,
            row["judgment_slot_ids"],
            strict=True,
        )
    ]
    return {
        "arena_comparisons": arena,
        "uplift_comparisons": uplift,
        "primary_judgment_slots": judgment_slots,
        "concealed_repeat_presentations": _repeat_frame(comparisons),
    }


def write_sampling_artifact(document: Mapping[str, Any], output_dir: Path) -> Path:
    """Atomically write one verified, content-addressed blocked artifact."""

    verify_sampling_artifact(document)
    _require(not output_dir.is_symlink(), "output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(output_dir.is_dir() and not output_dir.is_symlink(), "invalid output directory")
    digest = str(document["artifact_sha256"])
    destination = output_dir / f"human-judgment-sampling-v1-candidate-{digest}.json"
    rendered = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if destination.exists():
        _require(
            not destination.is_symlink() and destination.read_text(encoding="utf-8") == rendered,
            "existing content-addressed artifact differs",
        )
        return destination
    descriptor, temporary_name = tempfile.mkstemp(prefix=".human-sampling-v1-", dir=output_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_SOURCE_DESIGN)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE_AUDIT)
    parser.add_argument("--stale-review", type=Path, default=DEFAULT_STALE_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-candidate", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_sampling_artifact(
        design_path=args.design,
        gate_path=args.gate,
        stale_review_path=args.stale_review,
    )
    if args.write_candidate:
        print(write_sampling_artifact(document, args.output_dir))
    else:
        print(document["artifact_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
