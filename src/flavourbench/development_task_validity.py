"""Verify and freeze a real, public development task set.

This module closes a narrow but important evidence gap.  It verifies the
content-addressed Season 0 source bank, removes the governed specialist-scope
quarantine, and selects a balanced set for current-model development runs.

The resulting artifact is deliberately not a confirmatory task bank.  Public
Stack Exchange questions are contamination-susceptible, model curation is not
independent human validation, and accepted answers are non-binding references.
Those boundaries are recorded in the artifact rather than hidden behind a
single task-validity label.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .real_task_bank import sha256_json, sha256_text

SCHEMA_VERSION = "flavourbench-development-task-validity-v1"
FAMILIES = ("substitution", "composition", "cookability", "evidence")
RECENT_CUTOFF = "2025-01-01T00:00:00Z"
DEFAULT_TASKS_PER_FAMILY = 10
DEFAULT_RECENT_FLOOR_PER_FAMILY = 2


class DevelopmentTaskValidityError(RuntimeError):
    """A source bank or scope decision failed closed verification."""


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DevelopmentTaskValidityError(f"input must be a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise DevelopmentTaskValidityError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise DevelopmentTaskValidityError(f"expected a JSON object: {path}")
    return value


def _verify_embedded_digest(document: Mapping[str, Any], *, label: str) -> str:
    recorded = str(document.get("artifact_sha256") or "")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    actual = sha256_json(payload)
    if recorded != actual:
        raise DevelopmentTaskValidityError(f"{label} content address does not verify")
    return actual


def _require_sha256(value: object, *, field: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(character not in "0123456789abcdef" for character in rendered):
        raise DevelopmentTaskValidityError(f"{field} must be a lowercase SHA-256")
    return rendered


def _source_url_is_attributable(value: object, *, answer: bool) -> bool:
    parsed = urlparse(str(value or ""))
    if parsed.scheme != "https" or parsed.netloc != "cooking.stackexchange.com":
        return False
    prefix = "/a/" if answer else "/questions/"
    return parsed.path.startswith(prefix)


def _task_set_digest(tasks: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(
        [
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "prompt_sha256": task["prompt_sha256"],
                "task_sha256": task["task_sha256"],
            }
            for task in tasks
        ]
    )


def verify_task_bank(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return verified task records or fail before partial use."""

    _verify_embedded_digest(document, label="task bank")
    if document.get("schema_version") != "flavourbench-season0-real-task-bank-v1":
        raise DevelopmentTaskValidityError("unexpected task-bank schema")
    if document.get("source_class") != (
        "licensed_real_human_authored_questions_with_accepted_human_answers"
    ):
        raise DevelopmentTaskValidityError("task bank is not the licensed human source class")
    if document.get("synthetic_tasks") != 0:
        raise DevelopmentTaskValidityError("synthetic tasks are prohibited")
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 120:
        raise DevelopmentTaskValidityError("expected the frozen 120-task source bank")

    tasks: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    prompt_hashes: set[str] = set()
    question_ids: set[int] = set()
    for raw in raw_tasks:
        if not isinstance(raw, Mapping):
            raise DevelopmentTaskValidityError("task bank contains a non-object task")
        task = dict(raw)
        task_id = str(task.get("task_id") or "")
        family = str(task.get("family") or "")
        prompt = str(task.get("prompt") or "")
        prompt_sha256 = _require_sha256(task.get("prompt_sha256"), field="prompt_sha256")
        if family not in FAMILIES or not task_id.startswith(f"fb-s0-{family}-"):
            raise DevelopmentTaskValidityError(f"invalid family binding for {task_id!r}")
        if task.get("synthetic") is not False:
            raise DevelopmentTaskValidityError(f"{task_id} is not explicitly non-synthetic")
        if prompt_sha256 != sha256_text(prompt):
            raise DevelopmentTaskValidityError(f"{task_id} prompt hash does not verify")
        recorded_task_sha = _require_sha256(task.get("task_sha256"), field="task_sha256")
        task_payload = {key: value for key, value in task.items() if key != "task_sha256"}
        if recorded_task_sha != sha256_json(task_payload):
            raise DevelopmentTaskValidityError(f"{task_id} task hash does not verify")

        source = task.get("source")
        reference = task.get("human_reference")
        curation = task.get("curation")
        if not all(isinstance(item, Mapping) for item in (source, reference, curation)):
            raise DevelopmentTaskValidityError(f"{task_id} has incomplete provenance")
        assert isinstance(source, Mapping)
        assert isinstance(reference, Mapping)
        assert isinstance(curation, Mapping)
        if source.get("human_origin") is not True:
            raise DevelopmentTaskValidityError(f"{task_id} lacks explicit human origin")
        if not str(source.get("license") or "").startswith("CC BY-SA "):
            raise DevelopmentTaskValidityError(f"{task_id} has no supported source licence")
        if not str(reference.get("license") or "").startswith("CC BY-SA "):
            raise DevelopmentTaskValidityError(f"{task_id} has no supported reference licence")
        if not _source_url_is_attributable(source.get("url"), answer=False):
            raise DevelopmentTaskValidityError(f"{task_id} has an invalid source URL")
        if not _source_url_is_attributable(reference.get("url"), answer=True):
            raise DevelopmentTaskValidityError(f"{task_id} has an invalid reference URL")
        if reference.get("accepted") is not True or reference.get("use") != (
            "hidden_reference_not_automatic_ground_truth"
        ):
            raise DevelopmentTaskValidityError(f"{task_id} has an invalid reference boundary")
        reference_text = str(reference.get("text") or "")
        if _require_sha256(reference.get("text_sha256"), field="reference.text_sha256") != (
            sha256_text(reference_text)
        ):
            raise DevelopmentTaskValidityError(f"{task_id} reference hash does not verify")
        if curation.get("answers_visible_to_curators") is not False:
            raise DevelopmentTaskValidityError(f"{task_id} curation was not answer-blind")
        judgments = curation.get("judgments")
        if not isinstance(judgments, Mapping) or len(judgments) != 2:
            raise DevelopmentTaskValidityError(f"{task_id} lacks two curation judgments")
        for judgment in judgments.values():
            if not isinstance(judgment, Mapping) or not (
                judgment.get("include") is True
                and judgment.get("family") == family
                and judgment.get("self_contained") is True
                and judgment.get("specialist_risk") == "none"
            ):
                raise DevelopmentTaskValidityError(f"{task_id} lacks strict curation consensus")

        question_id = int(task.get("source_question_id") or 0)
        if question_id <= 0 or int(source.get("question_id") or 0) != question_id:
            raise DevelopmentTaskValidityError(f"{task_id} source identity is inconsistent")
        if task_id in identifiers or prompt_sha256 in prompt_hashes or question_id in question_ids:
            raise DevelopmentTaskValidityError("task bank contains duplicate identities")
        identifiers.add(task_id)
        prompt_hashes.add(prompt_sha256)
        question_ids.add(question_id)
        tasks.append(task)

    expected_counts = {family: 30 for family in FAMILIES}
    actual_counts = {family: sum(task["family"] == family for task in tasks) for family in FAMILIES}
    if actual_counts != expected_counts:
        raise DevelopmentTaskValidityError("task bank is not balanced at 30 tasks per family")
    if document.get("task_set_sha256") != _task_set_digest(tasks):
        raise DevelopmentTaskValidityError("task-set coordinate hash does not verify")
    return tasks


def verify_scope_review(document: Mapping[str, Any], *, task_ids: set[str]) -> dict[str, str]:
    _verify_embedded_digest(document, label="specialist-scope review")
    if document.get("schema_version") != "flavourbench-specialist-scope-review-v1":
        raise DevelopmentTaskValidityError("unexpected specialist-scope schema")
    boundary = document.get("decision_boundary")
    decisions = document.get("quarantine_decisions")
    if not isinstance(boundary, Mapping) or not isinstance(decisions, list):
        raise DevelopmentTaskValidityError("specialist-scope review is incomplete")
    if boundary.get("general_track_scope_only") is not True:
        raise DevelopmentTaskValidityError("scope review is not for the general track")
    quarantined: dict[str, str] = {}
    for item in decisions:
        if not isinstance(item, Mapping):
            raise DevelopmentTaskValidityError("scope review contains a non-object decision")
        task_id = str(item.get("task_id") or "")
        reason = str(item.get("reason_code") or "")
        if task_id not in task_ids or not reason or task_id in quarantined:
            raise DevelopmentTaskValidityError("scope quarantine is invalid or duplicated")
        quarantined[task_id] = reason
    if len(quarantined) != 17:
        raise DevelopmentTaskValidityError("expected all 17 governed specialist quarantines")
    return quarantined


def _recent(task: Mapping[str, Any]) -> bool:
    source = task.get("source")
    return isinstance(source, Mapping) and str(source.get("created_utc") or "") >= RECENT_CUTOFF


def _rank_key(task: Mapping[str, Any]) -> tuple[float, int, int, int, str]:
    curation = task["curation"]
    judgments = curation["judgments"]
    reference = task["human_reference"]
    source = task["source"]
    return (
        -float(curation.get("mean_curation_score") or 0),
        -sum(
            isinstance(judgment, Mapping) and judgment.get("requires_multi_step") is True
            for judgment in judgments.values()
        ),
        -int(reference.get("score_at_import") or 0),
        -int(source.get("score_at_import") or 0),
        str(task["task_id"]),
    )


def select_development_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    quarantined: Mapping[str, str],
    tasks_per_family: int = DEFAULT_TASKS_PER_FAMILY,
    recent_floor_per_family: int = DEFAULT_RECENT_FLOOR_PER_FAMILY,
) -> list[dict[str, Any]]:
    if tasks_per_family <= 0 or recent_floor_per_family < 0:
        raise DevelopmentTaskValidityError("selection counts must be non-negative")
    if recent_floor_per_family > tasks_per_family:
        raise DevelopmentTaskValidityError("recent floor cannot exceed the family size")
    selected: list[dict[str, Any]] = []
    for family in FAMILIES:
        eligible = [
            dict(task)
            for task in tasks
            if task["family"] == family and task["task_id"] not in quarantined
        ]
        recent = sorted((task for task in eligible if _recent(task)), key=_rank_key)
        if len(recent) < recent_floor_per_family:
            raise DevelopmentTaskValidityError(
                f"{family} has only {len(recent)} recent general-track tasks"
            )
        chosen = recent[:recent_floor_per_family]
        chosen_ids = {str(task["task_id"]) for task in chosen}
        remainder = sorted(
            (task for task in eligible if str(task["task_id"]) not in chosen_ids),
            key=_rank_key,
        )
        chosen.extend(remainder[: tasks_per_family - len(chosen)])
        if len(chosen) != tasks_per_family:
            raise DevelopmentTaskValidityError(
                f"{family} has only {len(chosen)} admissible development tasks"
            )
        selected.extend(chosen)
    return sorted(
        selected,
        key=lambda task: (
            FAMILIES.index(str(task["family"])),
            str(task["task_id"]),
        ),
    )


def build_validity_artifact(
    *,
    task_bank: Mapping[str, Any],
    scope_review: Mapping[str, Any],
    tasks_per_family: int = DEFAULT_TASKS_PER_FAMILY,
    recent_floor_per_family: int = DEFAULT_RECENT_FLOOR_PER_FAMILY,
) -> dict[str, Any]:
    tasks = verify_task_bank(task_bank)
    quarantined = verify_scope_review(
        scope_review,
        task_ids={str(task["task_id"]) for task in tasks},
    )
    selected = select_development_tasks(
        tasks,
        quarantined=quarantined,
        tasks_per_family=tasks_per_family,
        recent_floor_per_family=recent_floor_per_family,
    )
    selected_records = [
        {
            "task_id": task["task_id"],
            "family": task["family"],
            "prompt": task["prompt"],
            "prompt_sha256": task["prompt_sha256"],
            "task_sha256": task["task_sha256"],
            "source_question_id": task["source_question_id"],
            "source_url": task["source"]["url"],
            "source_created_utc": task["source"]["created_utc"],
            "source_license": task["source"]["license"],
            "source_question_score": int(task["source"].get("score_at_import") or 0),
            "hidden_reference_sha256": task["human_reference"]["text_sha256"],
            "hidden_reference_url": task["human_reference"]["url"],
            "hidden_reference_score": int(task["human_reference"].get("score_at_import") or 0),
            "recent": _recent(task),
            "global_machine_checks": [
                "response_nonempty",
                "normal_final_completion",
                "identity_contract_match",
                "no_pre_vote_identity_leakage",
                "mcp_trace_integrity_when_epicure_on",
            ],
            "task_specific_criterion_status": "pending_independent_human_authoring",
            "task_specific_success_criteria": [],
            "task_specific_failure_criteria": [],
            "accepted_reference_use": "human_review_aid_not_automatic_ground_truth",
            "confirmatory_eligible": False,
            "rank_eligible": False,
        }
        for task in selected
    ]
    candidate_coordinate_sha256 = sha256_json(
        [
            {
                "task_id": record["task_id"],
                "task_sha256": record["task_sha256"],
                "prompt_sha256": record["prompt_sha256"],
            }
            for record in selected_records
        ]
    )
    general_track = [task for task in tasks if task["task_id"] not in quarantined]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "FlavourBench",
        "artifact_role": "real_public_development_task_validity_dossier",
        "status": "source_verified_development_candidate_not_confirmatory",
        "source_observed_through_utc": task_bank["observed_through_utc"],
        "source_task_bank_sha256": task_bank["artifact_sha256"],
        "source_task_set_sha256": task_bank["task_set_sha256"],
        "specialist_scope_review_sha256": scope_review["artifact_sha256"],
        "candidate_coordinate_sha256": candidate_coordinate_sha256,
        "counts": {
            "source_tasks": len(tasks),
            "source_integrity_verified": len(tasks),
            "licensed_human_origin": len(tasks),
            "accepted_human_references": len(tasks),
            "synthetic_tasks": 0,
            "dual_model_curated": len(tasks),
            "specialist_scope_quarantined": len(quarantined),
            "general_track_candidates": len(general_track),
            "selected_development_tasks": len(selected_records),
            "selected_recent_tasks": sum(record["recent"] for record in selected_records),
            "independently_human_validated_tasks": 0,
            "tasks_with_human_authored_criterion_packs": 0,
            "private_contamination_resistant_tasks": 0,
            "per_family": {
                family: sum(record["family"] == family for record in selected_records)
                for family in FAMILIES
            },
        },
        "selection_policy": {
            "general_track_only": True,
            "tasks_per_family": tasks_per_family,
            "recent_cutoff_utc": RECENT_CUTOFF,
            "minimum_recent_per_family": recent_floor_per_family,
            "recent_quota_selected_first": True,
            "remaining_rank_order": [
                "mean_dual_curator_score_desc",
                "multi_step_votes_desc",
                "accepted_reference_score_desc",
                "source_question_score_desc",
                "task_id_asc",
            ],
            "quality_observations_used_for_selection": 0,
        },
        "validity_layers": {
            "provenance_integrity": {
                "status": "verified",
                "evidence": (
                    "content addresses, prompt/reference hashes, identities, licences, "
                    "and uniqueness"
                ),
            },
            "human_origin": {
                "status": "verified",
                "evidence": (
                    "licensed public questions and accepted human answers; zero synthetic tasks"
                ),
            },
            "construct_triage": {
                "status": "model_curated_only",
                "evidence": (
                    "two answer-blind curation judgments with strict family and scope consensus"
                ),
            },
            "specialist_scope": {
                "status": "governed_quarantine_applied",
                "evidence": (
                    "17 nutrition, allergen, food-safety, chemical-dosing, or "
                    "authenticity tasks excluded"
                ),
            },
            "task_specific_criteria": {
                "status": "missing",
                "evidence": (
                    "no independent human-authored success/failure criterion packs are sealed"
                ),
            },
            "independent_human_validation": {
                "status": "missing",
                "evidence": "zero tasks have two author-independent qualified approvals",
            },
            "contamination_resistance": {
                "status": "not_met",
                "evidence": (
                    "all selected tasks and references are public and must remain development-only"
                ),
            },
        },
        "claim_boundary": {
            "supports_real_current_model_development_runs": True,
            "supports_machine-checked_reliability_metrics": True,
            "supports_confirmatory_task_validity": False,
            "supports_official_leaderboard": False,
            "supports_model_quality_ranking_without_new_judgments": False,
            "official": False,
            "rank_eligible": False,
        },
        "specialist_quarantine": [
            {"task_id": task_id, "reason_code": quarantined[task_id]}
            for task_id in sorted(quarantined)
        ],
        "tasks": selected_records,
    }
    return payload


def _atomic_write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"development-task-validity-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise DevelopmentTaskValidityError("content-addressed output conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--scope-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks-per-family", type=int, default=DEFAULT_TASKS_PER_FAMILY)
    parser.add_argument(
        "--recent-floor-per-family",
        type=int,
        default=DEFAULT_RECENT_FLOOR_PER_FAMILY,
    )
    arguments = parser.parse_args(argv)
    payload = build_validity_artifact(
        task_bank=_load_object(arguments.task_bank),
        scope_review=_load_object(arguments.scope_review),
        tasks_per_family=arguments.tasks_per_family,
        recent_floor_per_family=arguments.recent_floor_per_family,
    )
    path = _atomic_write(arguments.output_dir, payload)
    print(
        json.dumps(
            {
                "output": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "counts": payload["counts"],
                "claim_boundary": payload["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
