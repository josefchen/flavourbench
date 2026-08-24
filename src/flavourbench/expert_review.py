from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

PROTOCOL_VERSION = "flavourbench-blinded-pair-review-v6"
RELIABILITY_REPEAT_INTERVAL = 8
TASK_FAMILIES = ("substitution", "composition", "cookability", "evidence")
AUTHOR_EVALUATOR_PRIMARY_JUDGMENTS = 32
AUTHOR_EVALUATOR_RELIABILITY_REPEATS = 4
AUTHOR_EVALUATOR_TOTAL_PRESENTATIONS = (
    AUTHOR_EVALUATOR_PRIMARY_JUDGMENTS + AUTHOR_EVALUATOR_RELIABILITY_REPEATS
)

REQUIRED_ACKNOWLEDGEMENTS = frozenset(
    {
        "conflict_disclosed",
        "culinary_competence",
        "identity_blinding",
        "no_external_model_identification",
        "no_active_batch_discussion",
        "voluntary_participation",
        "sealed_prompt_confidentiality",
    }
)

RUBRIC_DIMENSIONS: dict[str, dict[str, Any]] = {
    "task_completion": {
        "label": "Task completion",
        "question": "Does the response answer the requested culinary problem?",
        "anchors": {
            "1": "Misses the central request or is unusable.",
            "3": "Addresses the request but leaves material gaps.",
            "5": "Fully answers the request at the requested level of detail.",
        },
    },
    "constraint_compliance": {
        "label": "Constraint compliance",
        "question": "Does the response respect every explicit requirement?",
        "anchors": {
            "1": "Breaks a central explicit constraint.",
            "3": "Meets most constraints but has a meaningful omission or ambiguity.",
            "5": "Meets all explicit constraints without evasive substitutions.",
        },
    },
    "coherence": {
        "label": "Culinary coherence",
        "question": "Do the ingredients, techniques, and explanation form a coherent whole?",
        "anchors": {
            "1": "Contains contradictions or incompatible culinary logic.",
            "3": "Mostly coherent with one weak link or underexplained choice.",
            "5": "The culinary logic is internally consistent and well integrated.",
        },
    },
    "sensory_promise": {
        "label": "Sensory promise",
        "question": "How convincing is the likely taste, aroma, texture, and balance?",
        "anchors": {
            "1": "Likely unpleasant, badly imbalanced, or sensorially incoherent.",
            "3": "Plausibly enjoyable but ordinary, uncertain, or imperfectly balanced.",
            "5": "Strong, specific sensory potential with convincing balance.",
        },
    },
    "cookability": {
        "label": "Cookability",
        "question": "Could a competent cook execute this successfully from the response?",
        "anchors": {
            "1": "Materially impractical, underspecified, or technically unsound.",
            "3": "Cookable with interpretation or minor correction.",
            "5": "Practical, technically sound, and equipped with useful process cues.",
        },
    },
    "clarity": {
        "label": "Clarity",
        "question": "Is the response easy to understand and appropriately organized?",
        "anchors": {
            "1": "Confusing, contradictory, or difficult to use.",
            "3": "Understandable but wordy, uneven, or locally ambiguous.",
            "5": "Clear, precise, and proportionate to the task.",
        },
    },
    "originality": {
        "label": "Originality",
        "question": "Does the response offer useful, non-generic culinary thinking?",
        "anchors": {
            "1": "Generic boilerplate or an obvious restatement of the prompt.",
            "3": "Competent with at least one specific or useful idea.",
            "5": "Distinctive and useful without novelty for its own sake.",
        },
    },
    "evidence_use": {
        "label": "Evidence use",
        "question": "Are claims supported and is Epicure-style evidence interpreted correctly?",
        "anchors": {
            "1": "Invents evidence, misstates it, or substitutes scores for culinary judgment.",
            "3": "Uses relevant evidence but with limited explanation or traceability.",
            "5": "Uses evidence accurately, specifically, and in proportion to what it supports.",
        },
    },
    "calibration": {
        "label": "Calibration",
        "question": "Does confidence match the uncertainty and evidence available?",
        "anchors": {
            "1": "Makes categorical claims that the evidence cannot support.",
            "3": "Mostly calibrated but misses a meaningful uncertainty or caveat.",
            "5": "Clearly distinguishes evidence, inference, uncertainty, and practical judgment.",
        },
    },
}

TASK_ISSUE_TAGS = {
    "missing_context": "Required context is absent.",
    "ambiguous_constraint": "A central constraint admits materially different readings.",
    "specialist_scope": "The task belongs in a separately governed specialist track.",
    "answer_leakage": "The prompt reveals or strongly cues an answer.",
    "non_culinary": "The task does not measure the intended culinary construct.",
    "rights_or_privacy": "The task raises a rights, privacy, or identity concern.",
    "other": "Another task-quality issue is described in the rationale.",
}

LEGACY_RESPONSE_FAILURE_TAGS = {
    "invented_evidence",
    "unsafe_or_impractical",
}

RESPONSE_FAILURE_TAGS = {
    "ignored_constraint",
    "weak_flavour_logic",
    "unclear",
    "generic",
    "overconfident",
    "safety_hazard",
    "unsupported_safety_claim",
    "allergen_or_dietary_risk",
    "impractical",
    "evidence_trace_mismatch",
    "entity_resolution_mismatch",
    "similarity_as_functional_proof",
    "similarity_as_mechanism",
    "axis_as_measured_quantity",
    "score_as_normative_truth",
    "selective_evidence",
    "irrelevant_evidence",
    "false_precision",
}

SPECIALIST_DOMAINS = {
    "nutrition",
    "allergen",
    "food_safety",
    "cultural_authenticity",
    "medical_dietary",
    "other",
}

WORKLOAD_TARGET = {
    "season_models": 16,
    "model_arena_exposures_per_model": 40,
    "epicure_uplift_pairs_per_model": 40,
    "primary_model_arena_judgments": 320,
    "primary_epicure_uplift_judgments": 640,
    "primary_judgments": 960,
    "reliability_repeat_rate": 0.125,
    "reliability_repeats": 120,
    "total_presentations": 1080,
    "target_per_family": 270,
    "recommended_daily_limit": 32,
    "recommended_session_minutes": 60,
}


def _spread(total: int, labels: tuple[str, ...]) -> dict[str, int]:
    quotient, remainder = divmod(total, len(labels))
    return {label: quotient + (1 if index < remainder else 0) for index, label in enumerate(labels)}


def workload_cell_targets(total_presentations: int) -> dict[str, Any]:
    if total_presentations < 1:
        raise ValueError("total presentations must be positive")
    reliability_repeats = round(
        total_presentations
        * WORKLOAD_TARGET["reliability_repeats"]
        / WORKLOAD_TARGET["total_presentations"]
    )
    primary_judgments = total_presentations - reliability_repeats
    arena_primary = round(
        primary_judgments
        * WORKLOAD_TARGET["primary_model_arena_judgments"]
        / WORKLOAD_TARGET["primary_judgments"]
    )
    uplift_primary = primary_judgments - arena_primary
    arena_repeats = round(
        reliability_repeats
        * WORKLOAD_TARGET["primary_model_arena_judgments"]
        / WORKLOAD_TARGET["primary_judgments"]
    )
    uplift_repeats = reliability_repeats - arena_repeats
    primary = {
        "model_arena": _spread(arena_primary, TASK_FAMILIES),
        "epicure_uplift": _spread(uplift_primary, TASK_FAMILIES),
    }
    reliability = {
        "model_arena": _spread(arena_repeats, TASK_FAMILIES),
        "epicure_uplift": _spread(uplift_repeats, TASK_FAMILIES),
    }
    return {
        "primary": primary,
        "reliability": reliability,
        "primary_judgments": primary_judgments,
        "reliability_repeats": reliability_repeats,
        "total_presentations": total_presentations,
    }


def author_evaluator_workload_cell_targets(
    primary_judgments: int = AUTHOR_EVALUATOR_PRIMARY_JUDGMENTS,
) -> dict[str, Any]:
    if primary_judgments < len(TASK_FAMILIES) * 2:
        raise ValueError("author-evaluator workload requires at least two items per family")
    reliability_repeats = max(
        len(TASK_FAMILIES),
        primary_judgments // RELIABILITY_REPEAT_INTERVAL,
    )
    return {
        "primary": {
            "model_arena": {family: 0 for family in TASK_FAMILIES},
            "epicure_uplift": _spread(primary_judgments, TASK_FAMILIES),
        },
        "reliability": {
            "model_arena": {family: 0 for family in TASK_FAMILIES},
            "epicure_uplift": _spread(reliability_repeats, TASK_FAMILIES),
        },
        "primary_judgments": primary_judgments,
        "reliability_repeats": reliability_repeats,
        "total_presentations": primary_judgments + reliability_repeats,
    }


def isolated_uplift_workload_cell_targets(
    primary_by_family: Mapping[str, int],
) -> dict[str, Any]:
    """Build an isolated uplift workload without inventing unavailable cells.

    Development review pools can be unbalanced after real provider failures.  This
    contract keeps every reviewable pair, requires useful coverage in each family,
    and allocates concealed repeats separately from the primary observations.
    """

    if set(primary_by_family) != set(TASK_FAMILIES):
        raise ValueError("isolated uplift workload requires all four task families")
    normalized: dict[str, int] = {}
    for family in TASK_FAMILIES:
        value = primary_by_family[family]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("isolated uplift family counts must be integers")
        if value < 2:
            raise ValueError("isolated uplift workload requires at least two items per family")
        normalized[family] = value
    primary_judgments = sum(normalized.values())
    reliability_repeats = max(
        len(TASK_FAMILIES),
        primary_judgments // RELIABILITY_REPEAT_INTERVAL,
    )
    return {
        "primary": {
            "model_arena": {family: 0 for family in TASK_FAMILIES},
            "epicure_uplift": normalized,
        },
        "reliability": {
            "model_arena": {family: 0 for family in TASK_FAMILIES},
            "epicure_uplift": _spread(reliability_repeats, TASK_FAMILIES),
        },
        "primary_judgments": primary_judgments,
        "primary_by_family": normalized,
        "reliability_repeats": reliability_repeats,
        "total_presentations": primary_judgments + reliability_repeats,
    }


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def protocol_payload() -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "cohortUse": {
            "primaryLabel": "pathway_specific",
            "admissibleClaim": (
                "Blinded judgments reported with their reviewer pathway, "
                "qualification basis, and sample size."
            ),
            "inadmissibleClaim": (
                "Independent expert consensus without verified qualifications "
                "and more than one independent rater."
            ),
            "poolingRule": (
                "Never silently pool author, anonymous external, verified independent-expert, "
                "provider-affiliated, public, or automated judgments."
            ),
        },
        "blinding": {
            "modelProviderAndConditionHidden": True,
            "batchRevealOnly": True,
            "identityGuessingProhibited": True,
            "reliabilityRepeatsUnannounced": True,
        },
        "preferenceChoices": {
            "left": "Answer A is materially better for this task.",
            "right": "Answer B is materially better for this task.",
            "tie": "The answers are practically equivalent in overall quality.",
            "both_bad": "Neither answer reaches a minimally acceptable response.",
        },
        "rubricScale": {
            "minimum": 1,
            "maximum": 5,
            "dimensions": RUBRIC_DIMENSIONS,
        },
        "taskValidity": {
            "assessmentStage": (
                "The prompt is assessed and sealed before either response is released."
            ),
            "values": {
                "valid": "The task is suitable for the intended family and comparison.",
                "minor_issue": "The task remains usable but has a documented limitation.",
                "invalid": "The task should not enter preference fitting.",
            },
            "issueTags": TASK_ISSUE_TAGS,
        },
        "taskScope": {
            "assessmentStage": (
                "Answerability, family fit, and specialist scope are sealed before "
                "either response is released."
            ),
            "answerabilityValues": {
                "answerable": "The prompt supplies enough information for a defensible answer.",
                "minor_ambiguity": (
                    "A bounded ambiguity remains, but comparison is still informative."
                ),
                "unanswerable": "Missing information prevents a defensible comparison.",
            },
            "familyFitValues": {
                "in_family": "The prompt measures the assigned culinary task family.",
                "borderline": "The prompt only partly measures the assigned task family.",
                "out_of_family": "The prompt measures a different construct.",
            },
            "scopeEligibilityValues": {
                "general_track": "Eligible for the general culinary benchmark.",
                "specialist_track": (
                    "Requires separately governed expertise and reference evidence."
                ),
                "exclude": "Not eligible for either current track.",
            },
            "specialistDomains": sorted(SPECIALIST_DOMAINS),
            "generalTrackRule": (
                "Eligible only when task validity is not invalid, the prompt is not "
                "unanswerable or out of family, and scope eligibility is general track."
            ),
        },
        "responseFailureTags": sorted(RESPONSE_FAILURE_TAGS),
        "reliability": {
            "repeatInterval": RELIABILITY_REPEAT_INTERVAL,
            "presentation": "Previously judged pairs return later with left and right swapped.",
            "primaryStatistic": "Exact normalized preference agreement.",
            "secondaryStatistic": "Mean absolute within-dimension score difference.",
        },
        "workload": WORKLOAD_TARGET,
        "workloadCellTargets": workload_cell_targets(WORKLOAD_TARGET["total_presentations"]),
        "fatigueControls": {
            "recommendedDailyLimit": WORKLOAD_TARGET["recommended_daily_limit"],
            "recommendedSessionMinutes": WORKLOAD_TARGET["recommended_session_minutes"],
            "enforcedRolling24HourLimit": WORKLOAD_TARGET["recommended_daily_limit"],
            "minimumBreakMinutesAfterHour": 15,
            "stoppingRule": (
                "Pause for fatigue, interruption, or loss of concentration; never stop based "
                "on emerging model or Epicure results."
            ),
        },
        "requiredAcknowledgements": sorted(REQUIRED_ACKNOWLEDGEMENTS),
    }


PROTOCOL_SHA256 = canonical_sha256(protocol_payload())


def validate_acknowledgements(values: list[str]) -> list[str]:
    normalized = list(dict.fromkeys(values))
    if set(normalized) != REQUIRED_ACKNOWLEDGEMENTS:
        missing = sorted(REQUIRED_ACKNOWLEDGEMENTS - set(normalized))
        extra = sorted(set(normalized) - REQUIRED_ACKNOWLEDGEMENTS)
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if extra:
            detail.append(f"unsupported: {', '.join(extra)}")
        raise ValueError("acknowledgements must match the protocol (" + "; ".join(detail) + ")")
    return sorted(normalized)


def presentation_sha256(
    *,
    battle_id: str,
    prompt_sha256: str,
    answer_sha256_by_canonical_side: Mapping[str, str],
    presented_side_map: Mapping[str, str],
    protocol_sha256: str = PROTOCOL_SHA256,
) -> str:
    return canonical_sha256(
        {
            "battleId": battle_id,
            "promptSha256": prompt_sha256,
            "answerSha256ByCanonicalSide": dict(answer_sha256_by_canonical_side),
            "presentedSideMap": dict(presented_side_map),
            "protocolSha256": protocol_sha256,
        }
    )


def normalize_choice(choice: str, presented_side_map: Mapping[str, str]) -> str:
    if choice in {"tie", "both_bad"}:
        return choice
    if choice not in {"left", "right"}:
        raise ValueError("unsupported preference choice")
    canonical = presented_side_map.get(choice)
    if canonical not in {"left", "right"}:
        raise ValueError("presented side map is invalid")
    return canonical


def normalize_rubric(
    rubric: Mapping[str, Any], presented_side_map: Mapping[str, str]
) -> dict[str, Any]:
    review_metadata = dict(rubric["review_metadata"])
    output = {
        "review_metadata": review_metadata,
        "rubric_version": str(rubric["rubric_version"]),
    }
    for presented_side in ("left", "right"):
        canonical_side = presented_side_map.get(presented_side)
        if canonical_side not in {"left", "right"}:
            raise ValueError("presented side map is invalid")
        output[canonical_side] = dict(rubric[presented_side])
        failure_tags = rubric["review_metadata"].get(
            f"{presented_side}_failure_tags",
            [],
        )
        output["review_metadata"][f"{canonical_side}_failure_tags"] = list(failure_tags)
    return output


def reliability_summary(
    primary_votes_by_battle: Mapping[str, Mapping[str, Any]],
    repeat_reviews: list[Mapping[str, Any]],
) -> dict[str, Any]:
    comparable = []
    preference_pairs: list[tuple[str, str]] = []
    dimension_differences: list[float] = []
    for repeat in repeat_reviews:
        original = primary_votes_by_battle.get(str(repeat.get("battle_id", "")))
        if original is None:
            continue
        original_choice = str(original.get("choice"))
        repeat_choice = str(repeat.get("normalized_choice"))
        comparable.append(original_choice == repeat_choice)
        preference_pairs.append((original_choice, repeat_choice))
        original_rubric = original.get("rubric")
        repeat_rubric = repeat.get("normalized_rubric")
        if not isinstance(original_rubric, Mapping) or not isinstance(repeat_rubric, Mapping):
            continue
        for side in ("left", "right"):
            original_side = original_rubric.get(side)
            repeat_side = repeat_rubric.get(side)
            if not isinstance(original_side, Mapping) or not isinstance(repeat_side, Mapping):
                continue
            for dimension in RUBRIC_DIMENSIONS:
                first = original_side.get(dimension)
                second = repeat_side.get(dimension)
                if isinstance(first, int) and isinstance(second, int):
                    dimension_differences.append(abs(first - second))
    agreement = sum(comparable) / len(comparable) if comparable else None
    agreement_interval: list[float] | None = None
    cohens_kappa: float | None = None
    if comparable:
        z = 1.959963984540054
        sample_size = len(comparable)
        denominator = 1 + (z**2 / sample_size)
        center = (agreement + z**2 / (2 * sample_size)) / denominator
        half_width = (
            z
            * math.sqrt(agreement * (1 - agreement) / sample_size + z**2 / (4 * sample_size**2))
            / denominator
        )
        agreement_interval = [
            round(max(0.0, center - half_width), 4),
            round(min(1.0, center + half_width), 4),
        ]
        original_counts = Counter(first for first, _ in preference_pairs)
        repeat_counts = Counter(second for _, second in preference_pairs)
        expected_agreement = sum(
            (original_counts[choice] / sample_size) * (repeat_counts[choice] / sample_size)
            for choice in set(original_counts) | set(repeat_counts)
        )
        if expected_agreement < 1:
            cohens_kappa = round(
                (agreement - expected_agreement) / (1 - expected_agreement),
                4,
            )
    return {
        "completedRepeats": len(repeat_reviews),
        "comparableRepeats": len(comparable),
        "exactPreferenceAgreement": (round(agreement, 4) if agreement is not None else None),
        "preferenceAgreementInterval95": agreement_interval,
        "cohensKappa": cohens_kappa,
        "meanAbsoluteDimensionDifference": (
            round(sum(dimension_differences) / len(dimension_differences), 4)
            if dimension_differences
            else None
        ),
        "dimensionComparisons": len(dimension_differences),
        "provisional": len(comparable) < 40,
    }
