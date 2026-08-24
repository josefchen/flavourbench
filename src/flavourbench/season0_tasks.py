"""Freeze the independently curated, real-human Season 0 task bank."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json, sha256_text

FAMILIES = ("substitution", "composition", "cookability", "evidence")
TASK_BANK_SCHEMA = "flavourbench-season0-real-task-bank-v1"
REVIEW_QUEUE_SCHEMA = "flavourbench-season0-pi-task-review-queue-v1"


class TaskFreezeError(RuntimeError):
    """The selected task bank did not satisfy the frozen real-data contract."""


@dataclass(frozen=True)
class CuratedSource:
    pool_path: Path
    audit_path: Path
    pool: Mapping[str, Any]
    audit: Mapping[str, Any]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TaskFreezeError(f"expected a JSON object: {path}")
    return value


def _atomic_write(directory: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise TaskFreezeError(f"content-address conflict at {destination}")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _verify_artifact(document: Mapping[str, Any], *, label: str) -> str:
    claimed = document.get("artifact_sha256")
    if not isinstance(claimed, str):
        raise TaskFreezeError(f"{label} has no artifact_sha256")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if sha256_json(body) != claimed:
        raise TaskFreezeError(f"{label} content hash is invalid")
    return claimed


def load_curated_source(pool_path: Path, audit_path: Path) -> CuratedSource:
    pool = _load(pool_path)
    audit = _load(audit_path)
    if int(pool.get("synthetic_tasks", -1)) != 0:
        raise TaskFreezeError(f"synthetic or unspecified tasks in {pool_path}")
    if pool.get("curation_answers_visible_to_models") is not False:
        raise TaskFreezeError(f"curation answer-blinding is not attested in {pool_path}")
    if int(audit.get("synthetic_tasks", -1)) != 0:
        raise TaskFreezeError(f"synthetic or unspecified tasks in {audit_path}")
    audit_sha256 = _verify_artifact(audit, label=str(audit_path))
    expected_pool_sha256 = audit.get("candidate_pool_sha256")
    if expected_pool_sha256 != pool.get("content_sha256"):
        raise TaskFreezeError("curation audit does not bind its candidate pool")
    if len(audit.get("curator_ids") or []) != 2:
        raise TaskFreezeError("exactly two independent curators are required")
    if audit_sha256 != audit["artifact_sha256"]:
        raise TaskFreezeError("unreachable audit hash mismatch")
    return CuratedSource(pool_path=pool_path, audit_path=audit_path, pool=pool, audit=audit)


def _strict_rows(source: CuratedSource) -> list[dict[str, Any]]:
    candidates = {
        int(candidate["question_id"]): candidate
        for candidate in source.pool.get("candidates", [])
        if isinstance(candidate, Mapping)
    }
    output: list[dict[str, Any]] = []
    strict = source.audit.get("strict_consensus")
    if not isinstance(strict, Mapping):
        raise TaskFreezeError("curation audit has no strict-consensus mapping")
    for family in FAMILIES:
        family_rows = strict.get(family)
        if not isinstance(family_rows, list):
            raise TaskFreezeError(f"curation audit has no strict rows for {family}")
        for audit_row in family_rows:
            if not isinstance(audit_row, Mapping):
                raise TaskFreezeError("invalid strict-consensus row")
            question_id = int(audit_row["question_id"])
            candidate = candidates.get(question_id)
            if candidate is None:
                raise TaskFreezeError(f"strict question {question_id} is absent from its pool")
            judgments = audit_row.get("judgments")
            if not isinstance(judgments, Mapping) or len(judgments) != 2:
                raise TaskFreezeError(f"question {question_id} lacks two curator judgments")
            for judgment in judgments.values():
                if not isinstance(judgment, Mapping):
                    raise TaskFreezeError(f"question {question_id} has an invalid judgment")
                if (
                    judgment.get("include") is not True
                    or judgment.get("family") != family
                    or judgment.get("specialist_risk") != "none"
                    or judgment.get("self_contained") is not True
                ):
                    raise TaskFreezeError(f"question {question_id} violates strict consensus")
            output.append(
                {
                    "family": family,
                    "candidate": candidate,
                    "audit_row": dict(audit_row),
                    "source_pool_sha256": source.pool["content_sha256"],
                    "curation_audit_sha256": source.audit["artifact_sha256"],
                }
            )
    return output


def _rank_key(row: Mapping[str, Any]) -> tuple[float, int, int, int, int]:
    audit_row = row["audit_row"]
    candidate = row["candidate"]
    judgments = list(audit_row["judgments"].values())
    multi_step_votes = sum(judgment.get("requires_multi_step") is True for judgment in judgments)
    source = candidate.get("source") or {}
    reference = candidate.get("human_reference") or {}
    recent = int(bool((candidate.get("heuristics") or {}).get("recent")))
    return (
        float(audit_row.get("mean_curation_score") or 0),
        multi_step_votes,
        int(reference.get("score_at_import") or 0),
        int(source.get("score_at_import") or 0),
        recent,
    )


def freeze_task_bank(
    sources: Sequence[CuratedSource],
    *,
    per_family: int = 30,
    observed_through_utc: str = "2026-07-16T00:00:00Z",
) -> dict[str, Any]:
    if per_family <= 0:
        raise TaskFreezeError("per_family must be positive")
    rows = [row for source in sources for row in _strict_rows(source)]
    question_ids = [int(row["candidate"]["question_id"]) for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise TaskFreezeError("candidate pools overlap by source question ID")

    selected: list[dict[str, Any]] = []
    for family in FAMILIES:
        eligible = [row for row in rows if row["family"] == family]
        eligible.sort(
            key=lambda row: (
                tuple(-value for value in _rank_key(row)),
                int(row["candidate"]["question_id"]),
            )
        )
        if len(eligible) < per_family:
            raise TaskFreezeError(
                f"{family} has {len(eligible)} strict tasks; {per_family} are required"
            )
        for ordinal, row in enumerate(eligible[:per_family], start=1):
            candidate = row["candidate"]
            prompt = str(candidate["prompt"])
            task_id = f"fb-s0-{family}-{ordinal:03d}"
            task = {
                "task_id": task_id,
                "family": family,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "source_question_id": int(candidate["question_id"]),
                "source": candidate["source"],
                "human_reference": candidate["human_reference"],
                "curation": {
                    "answers_visible_to_curators": False,
                    "judgments": row["audit_row"]["judgments"],
                    "mean_curation_score": row["audit_row"]["mean_curation_score"],
                    "source_pool_sha256": row["source_pool_sha256"],
                    "curation_audit_sha256": row["curation_audit_sha256"],
                },
                "task_version": 1,
                "split": "season0_scored",
                "synthetic": False,
            }
            task["task_sha256"] = sha256_json(task)
            selected.append(task)

    selected.sort(key=lambda task: (FAMILIES.index(task["family"]), task["task_id"]))
    if len(selected) != per_family * len(FAMILIES):
        raise TaskFreezeError("selected task count is inconsistent")
    if len({task["prompt_sha256"] for task in selected}) != len(selected):
        raise TaskFreezeError("selected task prompts are not unique")
    if len({task["source_question_id"] for task in selected}) != len(selected):
        raise TaskFreezeError("selected source questions are not unique")

    body: dict[str, Any] = {
        "schema_version": TASK_BANK_SCHEMA,
        "benchmark": "FlavourBench",
        "season": "Season 0",
        "status": "frozen_for_generation_pi_task_audit_pending",
        "official_score_eligibility": "pending_pi_task_audit_and_model_contract_freeze",
        "source_class": "licensed_real_human_authored_questions_with_accepted_human_answers",
        "synthetic_tasks": 0,
        "legacy_development_items_included": 0,
        "observed_through_utc": observed_through_utc,
        "selection_policy": {
            "independent_curators_required": 2,
            "exact_family_agreement_required": True,
            "specialist_risk_required": "none",
            "self_contained_required": True,
            "minimum_family_fit_each": 3,
            "minimum_difficulty_each": 2,
            "minimum_specificity_each": 2,
            "minimum_mean_epicure_relevance": 2,
            "rank_order": [
                "mean_curation_score_desc",
                "multi_step_votes_desc",
                "accepted_answer_score_desc",
                "question_score_desc",
                "recent_desc",
                "source_question_id_asc",
            ],
            "human_reference_use": "hidden_reference_not_automatic_ground_truth",
        },
        "counts": {
            "total": len(selected),
            "per_family": {family: per_family for family in FAMILIES},
            "accepted_human_references": len(selected),
            "synthetic": 0,
        },
        "input_artifacts": [
            {
                "candidate_pool_sha256": source.pool["content_sha256"],
                "curation_audit_sha256": source.audit["artifact_sha256"],
            }
            for source in sources
        ],
        "tasks": selected,
    }
    body["task_set_sha256"] = sha256_json(
        [
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "prompt_sha256": task["prompt_sha256"],
                "task_sha256": task["task_sha256"],
            }
            for task in selected
        ]
    )
    return body


def build_pi_review_queue(task_bank: Mapping[str, Any]) -> dict[str, Any]:
    tasks = task_bank.get("tasks")
    if not isinstance(tasks, list):
        raise TaskFreezeError("task bank contains no task list")
    task_set_sha256 = task_bank.get("task_set_sha256")
    if not isinstance(task_set_sha256, str):
        raise TaskFreezeError("task bank contains no task-set hash")
    return {
        "schema_version": REVIEW_QUEUE_SCHEMA,
        "benchmark": "FlavourBench",
        "season": "Season 0",
        "reviewer": {
            "name": "Josef Chen",
            "affiliation": "KAIKAKU",
            "relevant_experience_years": 8,
            "qualified_families": list(FAMILIES),
            "compensation": "unpaid_volunteer",
            "conflict_disclosure": "benchmark_creator_and_Epicure_product_affiliate",
        },
        "task_set_sha256": task_set_sha256,
        "blinding": {
            "human_reference_hidden_until_initial_task_assessment": True,
            "curator_rationales_hidden": True,
        },
        "rubric": {
            "decision": ["accept", "revise", "exclude"],
            "family": list(FAMILIES),
            "specialist_risk": [
                "none",
                "nutrition",
                "allergen",
                "food_safety",
                "cultural_authenticity",
                "other",
            ],
            "scores": {
                "clarity": [1, 2, 3, 4, 5],
                "difficulty": [1, 2, 3, 4, 5],
                "answerability": [1, 2, 3, 4, 5],
                "culinary_value": [1, 2, 3, 4, 5],
                "epicure_relevance": [1, 2, 3, 4, 5],
            },
        },
        "items": [
            {
                "task_id": task["task_id"],
                "family_proposed": task["family"],
                "prompt": task["prompt"],
                "prompt_sha256": task["prompt_sha256"],
                "review": None,
            }
            for task in tasks
        ],
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", nargs=2, metavar=("POOL", "AUDIT"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/season0/frozen"))
    parser.add_argument("--per-family", type=int, default=30)
    args = parser.parse_args(argv)
    if not args.source:
        parser.error("at least one --source POOL AUDIT pair is required")
    sources = [load_curated_source(Path(pool), Path(audit)) for pool, audit in args.source]
    task_bank = freeze_task_bank(sources, per_family=args.per_family)
    task_path = _atomic_write(args.output_dir, "season0-real-task-bank", task_bank)
    review_queue = build_pi_review_queue(task_bank)
    review_path = _atomic_write(args.output_dir, "season0-pi-task-review-queue", review_queue)
    print(
        json.dumps(
            {
                "task_bank": str(task_path),
                "review_queue": str(review_path),
                "task_set_sha256": task_bank["task_set_sha256"],
                "counts": task_bank["counts"],
                "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
