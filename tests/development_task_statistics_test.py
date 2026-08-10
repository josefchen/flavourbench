from __future__ import annotations

import pytest

from flavourbench.development_task_statistics import (
    DevelopmentTaskStatisticsError,
    summarize_task_validation,
)


def _row(
    task_id: str,
    *,
    family: str = "composition",
    status: str = "awaiting_independent_review",
    counts: dict[str, int] | None = None,
) -> dict:
    decisions = counts or {}
    return {
        "taskId": task_id,
        "family": family,
        "status": status,
        "completeIndependentReviews": sum(decisions.values()),
        "decisionCounts": decisions,
    }


def _blind(decision: str) -> dict:
    nonvalid = decision != "valid"
    return {
        "decision": decision,
        "construct_fit": not nonvalid,
        "context_complete": True,
        "coherent_question": True,
        "general_track_scope": True,
        "answer_leakage_absent": True,
        "discrimination_value": True,
        "issue_tags": ["construct_mismatch"] if nonvalid else [],
    }


def test_empty_campaign_keeps_agreement_null_and_denominators_visible() -> None:
    rows = [_row(f"task-{ordinal}") for ordinal in range(1, 41)]

    summary = summarize_task_validation(
        task_rows=rows,
        blind_records=[],
        criterion_records=[],
    )

    assert summary["coverage"] == {
        "tasks": 40,
        "requiredIndependentReviews": 120,
        "completeIndependentReviews": 0,
        "reviewCoverageRate": 0.0,
        "tasksWithThreeCompleteReviews": 0,
        "threeReviewTaskCoverageRate": 0.0,
        "criterionPacks": 0,
    }
    assert summary["agreement"]["taskCount"] == 0
    assert summary["agreement"]["unanimousDecisionRate"] is None
    assert summary["agreement"]["meanPairwiseAgreement"] is None
    assert summary["agreement"]["fleissKappa"] is None
    assert summary["claimBoundary"]["packetRowsCountAsHumanEvidence"] is False


def test_known_three_rater_labels_recover_agreement_and_defect_counts() -> None:
    rows = [
        _row(
            "task-1",
            status="validated_unanimous",
            counts={"valid": 3},
        ),
        _row(
            "task-2",
            status="awaiting_independent_adjudication",
            counts={"valid": 2, "revise": 1},
        ),
        _row(
            "task-3",
            family="cookability",
            status="awaiting_independent_adjudication",
            counts={"revise": 3},
        ),
    ]
    blind_records = [
        *[_blind("valid") for _ in range(3)],
        _blind("valid"),
        _blind("valid"),
        _blind("revise"),
        *[_blind("revise") for _ in range(3)],
    ]
    criterion_records = [
        {"reference_adequacy": "adequate"} for _ in range(5)
    ]

    summary = summarize_task_validation(
        task_rows=rows,
        blind_records=blind_records,
        criterion_records=criterion_records,
    )

    assert summary["agreement"]["taskCount"] == 3
    assert summary["agreement"]["unanimousDecisionTasks"] == 2
    assert summary["agreement"]["unanimousDecisionRate"] == pytest.approx(2 / 3)
    assert summary["agreement"]["unanimousValidTasks"] == 1
    assert summary["agreement"]["meanPairwiseAgreement"] == pytest.approx(7 / 9)
    assert summary["agreement"]["fleissKappa"] == 0.55
    defects = summary["observedDefects"]
    assert defects["decisionCounts"] == {"valid": 5, "revise": 4, "exclude": 0}
    assert defects["issueTagCounts"]["construct_mismatch"] == 4
    assert defects["failedBlindCheckCounts"]["construct_fit"] == 4
    assert defects["referenceAdequacyCounts"] == {
        "adequate": 5,
        "partial": 0,
        "misleading": 0,
    }
    assert summary["byFamily"]["composition"]["tasksAwaitingAdjudication"] == 1
    assert summary["byFamily"]["cookability"]["tasksAwaitingAdjudication"] == 1


def test_summary_fails_closed_when_rows_and_sealed_records_disagree() -> None:
    with pytest.raises(
        DevelopmentTaskStatisticsError,
        match="task rows and sealed blind records disagree",
    ):
        summarize_task_validation(
            task_rows=[_row("task-1", counts={"valid": 3})],
            blind_records=[_blind("valid"), _blind("valid")],
            criterion_records=[],
        )
