"""Build a portable, source-backed report for the contained human-review QA batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from flavourbench.expert_calibration import (
    TASK_SCOPE_QUARANTINE,
    TASK_SCOPE_REVIEW_SHA256,
)

QA_SCHEMA_VERSION = "flavourbench-human-review-operational-qa-v3"
EXPECTED_REVIEWED_QUARANTINE_TASKS = 7


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("QA evidence is not a JSON object")
    embedded = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if embedded != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise ValueError("QA evidence digest does not match its content")
    if value.get("schema_version") != QA_SCHEMA_VERSION:
        raise ValueError("unsupported QA evidence schema")
    claim = value.get("claim_boundary", {})
    if not isinstance(claim, dict) or not all(
        claim.get(field) is False
        for field in ("research_use", "paper_use", "rank_eligible", "leaderboard_use")
    ):
        raise ValueError("QA evidence is not fully restricted")
    _assert_scope_contract(value)
    return value


def _assert_scope_contract(report: dict[str, Any]) -> None:
    scope = report.get("scope_audit")
    if not isinstance(scope, dict):
        raise ValueError("QA evidence has no scope audit")
    task_ids = scope.get("task_public_ids")
    if (
        not isinstance(task_ids, list)
        or len(task_ids) != EXPECTED_REVIEWED_QUARANTINE_TASKS
        or not all(isinstance(task_id, str) for task_id in task_ids)
        or len(set(task_ids)) != EXPECTED_REVIEWED_QUARANTINE_TASKS
        or not set(task_ids).issubset(TASK_SCOPE_QUARANTINE)
        or scope.get("general_track_quarantine_tasks_reviewed")
        != EXPECTED_REVIEWED_QUARANTINE_TASKS
        or scope.get("governed_quarantine_tasks") != len(TASK_SCOPE_QUARANTINE)
    ):
        raise ValueError("QA evidence does not contain the governed 17-task/7-reviewed scope")

    governance = scope.get("governance_review")
    if (
        not isinstance(governance, dict)
        or governance.get("schema_version") != "flavourbench-specialist-scope-review-v1"
        or governance.get("artifact_sha256") != TASK_SCOPE_REVIEW_SHA256
        or governance.get("quarantined_task_count") != len(TASK_SCOPE_QUARANTINE)
    ):
        raise ValueError("QA evidence is not bound to the authoritative scope review")


def write_projection(path: Path, datasets: dict[str, list[dict[str, Any]]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing projection: {path}")
    connection = sqlite3.connect(path)
    try:
        for table, rows in datasets.items():
            if not rows:
                continue
            fields = list(rows[0])
            types: dict[str, str] = {}
            for field in fields:
                values = [row.get(field) for row in rows if row.get(field) is not None]
                if values and all(isinstance(value, bool | int) for value in values):
                    types[field] = "INTEGER"
                elif values and all(isinstance(value, bool | int | float) for value in values):
                    types[field] = "REAL"
                else:
                    types[field] = "TEXT"
            columns = ", ".join(f'"{field}" {types[field]}' for field in fields)
            connection.execute(f'CREATE TABLE "{table}" ({columns})')
            placeholders = ", ".join("?" for _ in fields)
            connection.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [
                    tuple(
                        int(row[field]) if isinstance(row.get(field), bool) else row.get(field)
                        for field in fields
                    )
                    for row in rows
                ],
            )
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o600)


def source(
    source_id: str,
    label: str,
    *,
    sql: str,
    table: str,
    description: str,
    filters: list[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": "report_projection.sqlite",
        "query": {
            "engine": "sqlite",
            "sql": sql,
            "description": description,
            "executed_at": "2026-07-31T00:00:00Z",
            "tables_used": [table],
            "filters": filters,
            "metric_definitions": [
                "Counts are deterministic projections of the content-addressed QA evidence.",
                "No preference estimate in this report is eligible for research or publication.",
            ],
        },
    }


def build(report: dict[str, Any]) -> dict[str, Any]:
    _assert_scope_contract(report)
    funnel = report["quality_funnel"]
    progress = report["review_progress"]
    scope = report["scope_audit"]
    scope_reviewed_count = scope["general_track_quarantine_tasks_reviewed"]
    repeat = report["repeat_check"]
    completion = report["completion_audit"]["replacement_candidate"]
    evidence = report["evidence_use_signal"]
    claim = report["claim_boundary"]

    if funnel != {
        "candidate_pairs": 32,
        "unique_primary_pairs_reviewed": 29,
        "finish_clean_primary_pairs": 26,
        "finish_clean_non_both_bad_preferences": 25,
        "unseen_candidate_pairs": 3,
        "non_normal_response_arms": 4,
        "affected_primary_pairs": 3,
    }:
        raise ValueError("QA funnel does not match the frozen containment evidence")
    if repeat.get("reliability_interpretable") is not False:
        raise ValueError("recognized repeats cannot be presented as reliability")
    if report["provenance"]["consent"].get("collection_permitted") is not False:
        raise ValueError("the contained consent record must remain inactive")

    metrics = [
        {
            "presentations": progress["completed_presentations"],
            "unique_pairs": funnel["unique_primary_pairs_reviewed"],
            "finish_clean_pairs": funnel["finish_clean_primary_pairs"],
            "affected_pairs": funnel["affected_primary_pairs"],
            "potential_safety_reports": scope["response_specific_safety_reports"],
            "verified_safety_errors": scope["safety_reports_verified"],
            "synthetic_arms": report["source_pool"]["synthetic_arms"],
        }
    ]
    funnel_rows = [
        {"order": 1, "stage": "Candidate pairs", "pairs": funnel["candidate_pairs"]},
        {
            "order": 2,
            "stage": "Primary review completed",
            "pairs": funnel["unique_primary_pairs_reviewed"],
        },
        {
            "order": 3,
            "stage": "Normal final responses",
            "pairs": funnel["finish_clean_primary_pairs"],
        },
        {
            "order": 4,
            "stage": "Usable preference record",
            "pairs": funnel["finish_clean_non_both_bad_preferences"],
        },
    ]
    failure_rows = [
        {
            "tag": tag.replace("_", " "),
            "condition": condition.replace("epicure_", "Epicure "),
            "count": count,
        }
        for tag, conditions in report["failure_tag_audit"]["counts_by_tag_and_condition"].items()
        for condition, count in conditions.items()
        if count
    ]
    score_rows = [
        {
            "analysis": analysis,
            "condition": condition.replace("epicure_", "Epicure "),
            "mean_score": score,
        }
        for analysis, values in (
            ("All reviewed arms", evidence["mean_evidence_use_score"]),
            ("Finish-clean pairs", evidence["finish_clean_mean_evidence_use_score"]),
        )
        for condition, score in values.items()
    ]
    scope_rows = [
        {
            "task_id": task_id,
            "status": "Specialist review required",
            "general_track_eligible": False,
        }
        for task_id in scope["task_public_ids"]
    ]
    control_rows = [
        {
            "control": "Human-review batch",
            "observed_state": "Restricted operational QA",
            "required_remedy": "Active consent and institutional determination before reuse",
        },
        {
            "control": "Final-response completion",
            "observed_state": "4 non-normal arms across 3 pairs",
            "required_remedy": "Normal finish state required at collection, import, and assignment",
        },
        {
            "control": "Task scope",
            "observed_state": (
                f"{scope_reviewed_count} reviewed tasks require specialist governance"
            ),
            "required_remedy": "Separate answerability, family fit, and scope eligibility",
        },
        {
            "control": "Repeat reliability",
            "observed_state": "3 of 3 repeats recognized",
            "required_remedy": "Do not report agreement as rater reliability",
        },
        {
            "control": "Safety",
            "observed_state": "2 reviewer reports; 0 specialist verifications",
            "required_remedy": "Qualified food-safety adjudication before any error claim",
        },
        {
            "control": "Replacement reserve",
            "observed_state": (
                f"{completion['pairs']} pairs; {completion['source_arms']} real arms; "
                f"{completion['synthetic_arms']} synthetic"
            ),
            "required_remedy": "Use only after active consent and a new review session",
        },
    ]
    use_rows = [
        {"use": "Research analysis", "permitted": claim["research_use"]},
        {"use": "Paper result", "permitted": claim["paper_use"]},
        {"use": "Leaderboard", "permitted": claim["leaderboard_use"]},
        {"use": "Operational QA", "permitted": True},
    ]

    datasets = {
        "metrics": metrics,
        "quality_funnel": funnel_rows,
        "failure_tags": failure_rows,
        "evidence_scores": score_rows,
        "scope_quarantine": scope_rows,
        "control_actions": control_rows,
        "permitted_use": use_rows,
    }
    sources = [
        source(
            "qa_metrics",
            "Contained human-review QA evidence",
            sql="SELECT * FROM metrics LIMIT 1;",
            table="metrics",
            description="Headline counts from the append-only containment snapshot.",
            filters=["One restricted review session", "No research or ranking use"],
        ),
        source(
            "qa_funnel",
            "Finish-state audit projection",
            sql='SELECT "order", stage, pairs FROM quality_funnel ORDER BY "order";',
            table="quality_funnel",
            description="Pair survival after primary review and completion checks.",
            filters=["Unique primary comparisons only", "Normal finish states only"],
        ),
        source(
            "qa_failure_tags",
            "Response-tag audit projection",
            sql=(
                "SELECT tag, condition, count FROM failure_tags "
                "ORDER BY count DESC, tag, condition;"
            ),
            table="failure_tags",
            description="Legacy response failure-tag counts by Epicure condition.",
            filters=["Primary comparisons only", "Post hoc diagnostic use only"],
        ),
        source(
            "qa_evidence_scores",
            "Evidence-use rubric projection",
            sql=(
                "SELECT analysis, condition, mean_score FROM evidence_scores "
                "ORDER BY analysis, condition;"
            ),
            table="evidence_scores",
            description="Mean evidence-use scores before and after finish-state exclusion.",
            filters=["Single rater", "Descriptive operational signal only"],
        ),
        source(
            "qa_controls",
            "Containment and remediation projection",
            sql="SELECT control, observed_state, required_remedy FROM control_actions;",
            table="control_actions",
            description="Defect, observed state, and implemented or required control.",
            filters=["Append-only QA containment", "No deletion of raw records"],
        ),
        source(
            "qa_scope",
            "Specialist-scope quarantine projection",
            sql=(
                "SELECT task_id, status, general_track_eligible FROM scope_quarantine "
                "ORDER BY task_id;"
            ),
            table="scope_quarantine",
            description="Reviewed tasks already covered by the governed scope quarantine.",
            filters=["General culinary track", "Specialist adjudication pending"],
        ),
        source(
            "qa_use",
            "Permitted-use projection",
            sql="SELECT use, permitted FROM permitted_use;",
            table="permitted_use",
            description="Use restrictions derived from consent and quality containment.",
            filters=["Current evidence state"],
        ),
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "FlavourBench human-review QA audit",
            "description": (
                "Completion, consent, scope, and tagging audit for a restricted review batch."
            ),
            "generatedAt": report["observed_at"],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# FlavourBench human-review QA audit",
                },
                {
                    "id": "boundary",
                    "type": "markdown",
                    "sourceId": "qa_use",
                    "body": (
                        "## Containment decision\n\n"
                        "**This batch is not a benchmark result.** Its consent document is marked "
                        "inactive, four response arms ended abnormally, "
                        f"{scope_reviewed_count} reviewed tasks require "
                        "specialist governance, and every observed repeat was recognized. The raw "
                        "records are preserved, but research, paper, ranking, and leaderboard use "
                        "are disabled."
                    ),
                },
                {
                    "id": "metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "presentations_card",
                        "primary_card",
                        "clean_card",
                        "affected_card",
                        "safety_card",
                        "synthetic_card",
                    ],
                },
                {
                    "id": "funnel_text",
                    "type": "markdown",
                    "sourceId": "qa_funnel",
                    "body": (
                        "## Completion is a data-validity gate\n\n"
                        "The review covered 29 unique pairs. Three pairs contained at least one "
                        "non-normal final response, leaving 26 finish-clean pairs and 25 usable "
                        "preference records after excluding one both-bad judgment. The plot uses "
                        "pair counts, not scores."
                    ),
                },
                {
                    "id": "funnel_chart_block",
                    "type": "chart",
                    "chartId": "funnel_chart",
                    "layout": "full",
                },
                {
                    "id": "evidence_text",
                    "type": "markdown",
                    "sourceId": "qa_evidence_scores",
                    "body": (
                        "## Evidence misuse is a prospective measurement target\n\n"
                        "The broad legacy invented-evidence tag occurred on 9 Epicure-on arms and "
                        "2 Epicure-off arms. Mean evidence-use scores were also lower for the "
                        "Epicure-on condition in both the full and finish-clean subsets. These "
                        "post hoc, single-rater diagnostics cannot identify a causal effect. They "
                        "motivate a preregistered taxonomy that separates trace mismatch, entity "
                        "resolution, similarity-to-mechanism, similarity-to-function, normative "
                        "overreach, selective use, irrelevance, and false precision."
                    ),
                },
                {
                    "id": "evidence_chart_block",
                    "type": "chart",
                    "chartId": "evidence_score_chart",
                    "layout": "full",
                },
                {
                    "id": "tag_chart_block",
                    "type": "chart",
                    "chartId": "failure_tag_chart",
                    "layout": "full",
                },
                {
                    "id": "repeat_text",
                    "type": "markdown",
                    "body": (
                        "## The repeat check is not a reliability estimate\n\n"
                        "All three repeats were recognized and deliberately scored to mirror the "
                        "earlier presentation. The observed agreement of 3 out of 3 and rubric "
                        "difference of 0 are manipulation-check outcomes only. They are not "
                        "reported as within-rater reliability."
                    ),
                },
                {
                    "id": "scope_table_block",
                    "type": "table",
                    "tableId": "scope_table",
                    "layout": "full",
                },
                {
                    "id": "safety_text",
                    "type": "markdown",
                    "body": (
                        "## Safety reports remain unverified\n\n"
                        "The reviewer reported two potentially unsafe dosing recommendations, one "
                        "in each Epicure condition. No qualified food-safety adjudicator has "
                        "verified either report. They remain response-level incidents and do not "
                        "support a condition-level safety claim."
                    ),
                },
                {
                    "id": "controls_table_block",
                    "type": "table",
                    "tableId": "controls_table",
                    "layout": "full",
                },
                {
                    "id": "use_table_block",
                    "type": "table",
                    "tableId": "use_table",
                    "layout": "full",
                },
            ],
            "cards": [
                {
                    "id": "presentations_card",
                    "description": "All submitted presentations, including repeats.",
                    "dataset": "metrics",
                    "sourceId": "qa_metrics",
                    "metrics": [
                        {"label": "Presentations", "field": "presentations", "format": "number"}
                    ],
                },
                {
                    "id": "primary_card",
                    "description": "Unique primary pair judgments.",
                    "dataset": "metrics",
                    "sourceId": "qa_metrics",
                    "metrics": [
                        {"label": "Unique pairs", "field": "unique_pairs", "format": "number"}
                    ],
                },
                {
                    "id": "clean_card",
                    "description": "Pairs whose two arms ended normally.",
                    "dataset": "metrics",
                    "sourceId": "qa_metrics",
                    "metrics": [
                        {
                            "label": "Finish-clean pairs",
                            "field": "finish_clean_pairs",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "affected_card",
                    "description": "Reviewed pairs excluded for non-normal completion.",
                    "dataset": "metrics",
                    "sourceId": "qa_metrics",
                    "metrics": [
                        {"label": "Affected pairs", "field": "affected_pairs", "format": "number"}
                    ],
                },
                {
                    "id": "safety_card",
                    "description": "Reviewer reports pending specialist adjudication.",
                    "dataset": "metrics",
                    "sourceId": "qa_metrics",
                    "metrics": [
                        {
                            "label": "Potential safety reports",
                            "field": "potential_safety_reports",
                            "format": "number",
                        },
                        {
                            "label": "Verified",
                            "field": "verified_safety_errors",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "synthetic_card",
                    "description": "Synthetic response arms in the source pool.",
                    "dataset": "metrics",
                    "sourceId": "qa_metrics",
                    "metrics": [
                        {"label": "Synthetic arms", "field": "synthetic_arms", "format": "number"}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "funnel_chart",
                    "title": "Pair survival through the QA checks",
                    "subtitle": "Counts only; the batch is excluded from research and ranking",
                    "type": "bar",
                    "dataset": "quality_funnel",
                    "sourceId": "qa_funnel",
                    "layout": "full",
                    "question": "How many candidate pairs survived each quality gate?",
                    "rationale": "Ordered bars preserve a common pair denominator.",
                    "encodings": {
                        "x": {
                            "field": "stage",
                            "type": "nominal",
                            "label": "Stage",
                            "sort": "order",
                        },
                        "y": {
                            "field": "pairs",
                            "type": "quantitative",
                            "label": "Pairs",
                            "format": "number",
                        },
                        "tooltip": [
                            {"field": "stage", "type": "nominal", "label": "Stage"},
                            {
                                "field": "pairs",
                                "type": "quantitative",
                                "label": "Pairs",
                                "format": "number",
                            },
                        ],
                    },
                },
                {
                    "id": "evidence_score_chart",
                    "title": "Mean evidence-use score by condition",
                    "subtitle": "Single-rater post hoc diagnostic; 1 to 5 rubric scale",
                    "type": "bar",
                    "dataset": "evidence_scores",
                    "sourceId": "qa_evidence_scores",
                    "layout": "full",
                    "question": (
                        "Does the evidence-use difference persist after finish-state exclusion?"
                    ),
                    "rationale": (
                        "Grouped means show the full and finish-clean diagnostics without "
                        "implying causality."
                    ),
                    "encodings": {
                        "x": {"field": "analysis", "type": "nominal", "label": "Subset"},
                        "y": {
                            "field": "mean_score",
                            "type": "quantitative",
                            "label": "Mean evidence-use score",
                            "format": "number",
                        },
                        "color": {"field": "condition", "type": "nominal", "label": "Condition"},
                        "tooltip": [
                            {"field": "analysis", "type": "nominal", "label": "Subset"},
                            {"field": "condition", "type": "nominal", "label": "Condition"},
                            {
                                "field": "mean_score",
                                "type": "quantitative",
                                "label": "Mean score",
                                "format": "number",
                            },
                        ],
                    },
                },
                {
                    "id": "failure_tag_chart",
                    "title": "Legacy response failure tags by condition",
                    "subtitle": "Arm-level counts; tags are post hoc and non-exclusive",
                    "type": "bar",
                    "dataset": "failure_tags",
                    "sourceId": "qa_failure_tags",
                    "layout": "full",
                    "question": "Which response failure labels appeared most often?",
                    "rationale": (
                        "Condition-grouped counts expose taxonomy pressure points for the next "
                        "protocol."
                    ),
                    "encodings": {
                        "x": {
                            "field": "tag",
                            "type": "nominal",
                            "label": "Failure tag",
                            "sort": "-y",
                        },
                        "y": {
                            "field": "count",
                            "type": "quantitative",
                            "label": "Tagged arms",
                            "format": "number",
                        },
                        "color": {"field": "condition", "type": "nominal", "label": "Condition"},
                        "tooltip": [
                            {"field": "tag", "type": "nominal", "label": "Tag"},
                            {"field": "condition", "type": "nominal", "label": "Condition"},
                            {
                                "field": "count",
                                "type": "quantitative",
                                "label": "Tagged arms",
                                "format": "number",
                            },
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "scope_table",
                    "title": "Reviewed tasks held outside the general track",
                    "subtitle": "Specialist review is required before any scoring use",
                    "dataset": "scope_quarantine",
                    "sourceId": "qa_scope",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "task_id", "direction": "asc"},
                    "columns": [
                        {"field": "task_id", "label": "Task", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                        {
                            "field": "general_track_eligible",
                            "label": "General-track eligible",
                            "type": "boolean",
                        },
                    ],
                },
                {
                    "id": "controls_table",
                    "title": "Containment and remediation",
                    "subtitle": "Observed defect and the control required before reuse",
                    "dataset": "control_actions",
                    "sourceId": "qa_controls",
                    "layout": "full",
                    "density": "spacious",
                    "columns": [
                        {"field": "control", "label": "Control", "type": "text"},
                        {"field": "observed_state", "label": "Observed state", "type": "text"},
                        {"field": "required_remedy", "label": "Required remedy", "type": "text"},
                    ],
                },
                {
                    "id": "use_table",
                    "title": "Permitted use",
                    "subtitle": "The restriction is enforced in the database and application",
                    "dataset": "permitted_use",
                    "sourceId": "qa_use",
                    "layout": "full",
                    "density": "dense",
                    "columns": [
                        {"field": "use", "label": "Use", "type": "text"},
                        {"field": "permitted", "label": "Permitted", "type": "boolean"},
                    ],
                },
            ],
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": report["observed_at"],
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": sources,
        "lineage": {
            "input_schema_version": report["schema_version"],
            "input_artifact_sha256": report["artifact_sha256"],
            "scope_governance_artifact_sha256": scope["governance_review"]["artifact_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build(load_report(args.input))
    projection_path = args.output.parent / "report_projection.sqlite"
    existing = [path for path in (args.output, projection_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing report artifact: "
            + ", ".join(str(path) for path in existing)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.chmod(0o700)
    write_projection(projection_path, artifact["snapshot"]["datasets"])
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    args.output.chmod(0o600)


if __name__ == "__main__":
    main()
