"""Descriptive statistics for the public human task-validation campaign.

The functions in this module consume only sealed human-review state. They do
not infer missing labels, impute decisions, or treat the validation packet as
evidence. Undefined rates remain ``None`` until their denominator exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

DECISIONS = ("valid", "revise", "exclude")
ISSUE_TAGS = (
    "construct_mismatch",
    "missing_context",
    "multiple_unrelated_questions",
    "specialist_scope",
    "answer_leakage",
    "low_discrimination",
    "duplicate_or_contaminated",
    "other",
)
BLIND_CHECKS = (
    "construct_fit",
    "context_complete",
    "coherent_question",
    "general_track_scope",
    "answer_leakage_absent",
    "discrimination_value",
)
REFERENCE_ADEQUACY = ("adequate", "partial", "misleading")


class DevelopmentTaskStatisticsError(ValueError):
    """Sealed task-validation state is internally inconsistent."""


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _ordered_counts(counter: Counter[str], labels: Sequence[str]) -> dict[str, int]:
    return {label: int(counter[label]) for label in labels}


def _fleiss_kappa(decision_counts: Sequence[Mapping[str, int]]) -> tuple[float, float | None]:
    """Return mean pairwise agreement and Fleiss' kappa for fixed three-label rows."""

    task_count = len(decision_counts)
    if not task_count:
        raise DevelopmentTaskStatisticsError("Fleiss' kappa requires at least one task")
    ratings_per_task = 3
    label_totals: Counter[str] = Counter()
    item_agreement: list[float] = []
    for counts in decision_counts:
        normalized = {decision: int(counts.get(decision, 0)) for decision in DECISIONS}
        if any(value < 0 for value in normalized.values()) or sum(normalized.values()) != 3:
            raise DevelopmentTaskStatisticsError(
                "each complete task must contain exactly three recognized decisions"
            )
        label_totals.update(normalized)
        numerator = sum(value * value for value in normalized.values()) - ratings_per_task
        item_agreement.append(numerator / (ratings_per_task * (ratings_per_task - 1)))

    mean_pairwise_agreement = sum(item_agreement) / task_count
    total_ratings = task_count * ratings_per_task
    expected_agreement = sum(
        (label_totals[decision] / total_ratings) ** 2 for decision in DECISIONS
    )
    if expected_agreement >= 1.0:
        kappa = None
    else:
        kappa = (mean_pairwise_agreement - expected_agreement) / (1 - expected_agreement)
    return round(mean_pairwise_agreement, 6), None if kappa is None else round(kappa, 6)


def summarize_task_validation(
    *,
    task_rows: Sequence[Mapping[str, Any]],
    blind_records: Sequence[Mapping[str, Any]],
    criterion_records: Sequence[Mapping[str, Any]],
    required_reviews_per_task: int = 3,
) -> dict[str, Any]:
    """Summarize coverage, agreement, and observed defects without imputation."""

    if required_reviews_per_task != 3:
        raise DevelopmentTaskStatisticsError("this statistic contract requires three reviewers")
    task_ids: set[str] = set()
    family_task_counts: Counter[str] = Counter()
    family_completed_reviews: Counter[str] = Counter()
    family_three_label_tasks: Counter[str] = Counter()
    family_validated_tasks: Counter[str] = Counter()
    family_adjudication_tasks: Counter[str] = Counter()
    decision_totals: Counter[str] = Counter()
    complete_decision_rows: list[Mapping[str, int]] = []
    unanimous_decision_tasks = 0
    unanimous_valid_tasks = 0

    for row in task_rows:
        task_id = str(row.get("taskId") or "")
        family = str(row.get("family") or "")
        if not task_id or task_id in task_ids or not family:
            raise DevelopmentTaskStatisticsError("task rows require unique IDs and families")
        task_ids.add(task_id)
        family_task_counts[family] += 1
        complete_reviews = int(row.get("completeIndependentReviews", 0))
        if complete_reviews < 0 or complete_reviews > required_reviews_per_task:
            raise DevelopmentTaskStatisticsError("task review coverage is outside the contract")
        family_completed_reviews[family] += complete_reviews
        counts_value = row.get("decisionCounts", {})
        if not isinstance(counts_value, Mapping):
            raise DevelopmentTaskStatisticsError("decisionCounts must be an object")
        unknown = set(counts_value) - set(DECISIONS)
        counts = {decision: int(counts_value.get(decision, 0)) for decision in DECISIONS}
        if unknown or any(value < 0 for value in counts.values()):
            raise DevelopmentTaskStatisticsError("decisionCounts contain invalid labels")
        if sum(counts.values()) != complete_reviews:
            raise DevelopmentTaskStatisticsError("decisionCounts disagree with review coverage")
        decision_totals.update(counts)
        status = str(row.get("status") or "")
        if complete_reviews == required_reviews_per_task:
            family_three_label_tasks[family] += 1
            complete_decision_rows.append(counts)
            if max(counts.values()) == required_reviews_per_task:
                unanimous_decision_tasks += 1
            if counts["valid"] == required_reviews_per_task:
                unanimous_valid_tasks += 1
        if status in {"validated_unanimous", "adjudicated_valid"}:
            family_validated_tasks[family] += 1
        if status == "awaiting_independent_adjudication":
            family_adjudication_tasks[family] += 1

    blind_decisions: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    failed_check_counts: Counter[str] = Counter()
    for record in blind_records:
        decision = str(record.get("decision") or "")
        if decision not in DECISIONS:
            raise DevelopmentTaskStatisticsError("blind record has an invalid decision")
        blind_decisions[decision] += 1
        tags = record.get("issue_tags", [])
        if not isinstance(tags, list) or any(tag not in ISSUE_TAGS for tag in tags):
            raise DevelopmentTaskStatisticsError("blind record has invalid issue tags")
        issue_counts.update(str(tag) for tag in tags)
        for check in BLIND_CHECKS:
            value = record.get(check)
            if not isinstance(value, bool):
                raise DevelopmentTaskStatisticsError("blind record has a non-boolean check")
            failed_check_counts[check] += int(not value)

    if blind_decisions != decision_totals:
        raise DevelopmentTaskStatisticsError("task rows and sealed blind records disagree")

    adequacy_counts: Counter[str] = Counter()
    for record in criterion_records:
        adequacy = str(record.get("reference_adequacy") or "")
        if adequacy not in REFERENCE_ADEQUACY:
            raise DevelopmentTaskStatisticsError("criterion record has invalid reference adequacy")
        adequacy_counts[adequacy] += 1

    complete_task_count = len(complete_decision_rows)
    if complete_task_count:
        mean_pairwise_agreement, fleiss_kappa = _fleiss_kappa(complete_decision_rows)
    else:
        mean_pairwise_agreement = None
        fleiss_kappa = None
    families = sorted(family_task_counts)
    by_family = {
        family: {
            "tasks": family_task_counts[family],
            "requiredIndependentReviews": (
                family_task_counts[family] * required_reviews_per_task
            ),
            "completeIndependentReviews": family_completed_reviews[family],
            "tasksWithThreeCompleteReviews": family_three_label_tasks[family],
            "independentlyValidatedTasks": family_validated_tasks[family],
            "tasksAwaitingAdjudication": family_adjudication_tasks[family],
        }
        for family in families
    }
    completed_reviews = sum(decision_totals.values())
    required_reviews = len(task_rows) * required_reviews_per_task
    return {
        "schemaVersion": "flavourbench-development-task-validation-statistics-v1",
        "coverage": {
            "tasks": len(task_rows),
            "requiredIndependentReviews": required_reviews,
            "completeIndependentReviews": completed_reviews,
            "reviewCoverageRate": _rate(completed_reviews, required_reviews),
            "tasksWithThreeCompleteReviews": complete_task_count,
            "threeReviewTaskCoverageRate": _rate(complete_task_count, len(task_rows)),
            "criterionPacks": len(criterion_records),
        },
        "agreement": {
            "population": "tasks_with_exactly_three_complete_independent_reviews",
            "categories": list(DECISIONS),
            "taskCount": complete_task_count,
            "unanimousDecisionTasks": unanimous_decision_tasks,
            "unanimousDecisionRate": _rate(unanimous_decision_tasks, complete_task_count),
            "unanimousValidTasks": unanimous_valid_tasks,
            "meanPairwiseAgreement": mean_pairwise_agreement,
            "fleissKappa": fleiss_kappa,
            "undefinedMetricsRemainNull": True,
        },
        "observedDefects": {
            "decisionCounts": _ordered_counts(decision_totals, DECISIONS),
            "issueTagCounts": _ordered_counts(issue_counts, ISSUE_TAGS),
            "failedBlindCheckCounts": _ordered_counts(failed_check_counts, BLIND_CHECKS),
            "referenceAdequacyCounts": _ordered_counts(
                adequacy_counts, REFERENCE_ADEQUACY
            ),
        },
        "byFamily": by_family,
        "claimBoundary": {
            "realSealedHumanRecordsOnly": True,
            "missingReviewsImputed": False,
            "packetRowsCountAsHumanEvidence": False,
            "descriptiveNotConfirmatory": True,
        },
    }
