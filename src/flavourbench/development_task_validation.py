"""Build the blinded human-validation packet for the 40-task calibration pool.

The packet contains only licensed human questions and their accepted human
references. References remain sealed until a reviewer records an answer-blind
validity decision. Three distinct, qualification-matched reviewers are required
for each task. Unanimous valid decisions complete the source-review stage;
disagreement requires a fourth independent adjudicator.

The empty packet is a review instrument, not task-validity evidence. Human
decisions and criterion packs are stored separately as append-only database
events.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .development_task_validity import DevelopmentTaskValidityError, verify_task_bank
from .development_task_validity_v2 import SCHEMA_VERSION as TASK_VALIDITY_SCHEMA_VERSION
from .real_task_bank import sha256_json, sha256_text

SCHEMA_VERSION = "flavourbench-development-task-human-validation-packet-v2"
REQUIRED_INDEPENDENT_REVIEWERS = 3
ASSIGNMENT_POLICY = "unfinished-criteria-then-least-reviewed-hmac-tiebreak-v1"
STATISTICS_POLICY = "three-label-null-safe-descriptive-statistics-v1"
EXPECTED_TASKS = 40
EXPECTED_TASKS_PER_FAMILY = 10
FAMILIES = ("substitution", "composition", "cookability", "evidence")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DevelopmentTaskValidationError(RuntimeError):
    """The validation packet could not be verified or assembled safely."""


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DevelopmentTaskValidationError(f"input must be a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise DevelopmentTaskValidationError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise DevelopmentTaskValidationError(f"expected a JSON object: {path}")
    return value


def _verify_embedded_digest(document: Mapping[str, Any], *, field: str, label: str) -> str:
    recorded = str(document.get(field) or "")
    payload = {key: value for key, value in document.items() if key != field}
    actual = sha256_json(payload)
    if recorded != actual:
        raise DevelopmentTaskValidationError(f"{label} content address does not verify")
    return actual


def _selected_task_coordinates(task_validity: Mapping[str, Any]) -> list[dict[str, str]]:
    if task_validity.get("schema_version") != TASK_VALIDITY_SCHEMA_VERSION:
        raise DevelopmentTaskValidationError("unexpected development task-validity schema")
    _verify_embedded_digest(task_validity, field="artifact_sha256", label="task-validity dossier")
    tasks = task_validity.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_TASKS:
        raise DevelopmentTaskValidationError(
            f"task-validity dossier must contain {EXPECTED_TASKS} tasks"
        )
    coordinates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_task_hashes: set[str] = set()
    seen_prompt_hashes: set[str] = set()
    families: Counter[str] = Counter()
    for record in tasks:
        if not isinstance(record, Mapping):
            raise DevelopmentTaskValidationError("task-validity coordinate is not an object")
        task_id = str(record.get("task_id") or "")
        family = str(record.get("family") or "")
        task_sha256 = str(record.get("task_sha256") or "")
        prompt_sha256 = str(record.get("prompt_sha256") or "")
        if family not in FAMILIES or not task_id.startswith(f"fb-s0-{family}-"):
            raise DevelopmentTaskValidationError("task-validity coordinate has invalid identity")
        if not _SHA256.fullmatch(task_sha256) or not _SHA256.fullmatch(prompt_sha256):
            raise DevelopmentTaskValidationError("task-validity coordinate has invalid hashes")
        if (
            task_id in seen_ids
            or task_sha256 in seen_task_hashes
            or prompt_sha256 in seen_prompt_hashes
        ):
            raise DevelopmentTaskValidationError("task-validity coordinates are not unique")
        seen_ids.add(task_id)
        seen_task_hashes.add(task_sha256)
        seen_prompt_hashes.add(prompt_sha256)
        families[family] += 1
        coordinates.append(
            {
                "task_id": task_id,
                "family": family,
                "task_sha256": task_sha256,
                "prompt_sha256": prompt_sha256,
            }
        )
    if families != Counter({family: EXPECTED_TASKS_PER_FAMILY for family in FAMILIES}):
        raise DevelopmentTaskValidationError("task-validity dossier is not family balanced")
    coordinate_payload = [
        {
            "task_id": row["task_id"],
            "task_sha256": row["task_sha256"],
            "prompt_sha256": row["prompt_sha256"],
        }
        for row in coordinates
    ]
    if sha256_json(coordinate_payload) != task_validity.get("candidate_coordinate_sha256"):
        raise DevelopmentTaskValidationError("task-validity coordinate digest does not verify")
    return coordinates


def build_validation_packet(
    *,
    task_bank: Mapping[str, Any],
    task_validity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a content-addressable review packet without human judgments."""

    try:
        bank_tasks = verify_task_bank(task_bank)
    except DevelopmentTaskValidityError as error:
        raise DevelopmentTaskValidationError(str(error)) from error
    task_validity_sha256 = _verify_embedded_digest(
        task_validity,
        field="artifact_sha256",
        label="task-validity dossier",
    )
    if task_validity.get("source_task_bank_sha256") != task_bank.get("artifact_sha256"):
        raise DevelopmentTaskValidationError("task-validity dossier does not bind the task bank")
    coordinates = _selected_task_coordinates(task_validity)
    task_by_id = {str(task["task_id"]): task for task in bank_tasks}

    tasks: list[dict[str, Any]] = []
    for ordinal, coordinate in enumerate(coordinates, start=1):
        task = task_by_id.get(coordinate["task_id"])
        if task is None:
            raise DevelopmentTaskValidationError("selected task is absent from the source bank")
        if (
            task["family"] != coordinate["family"]
            or task["task_sha256"] != coordinate["task_sha256"]
            or task["prompt_sha256"] != coordinate["prompt_sha256"]
            or sha256_text(str(task["prompt"])) != coordinate["prompt_sha256"]
        ):
            raise DevelopmentTaskValidationError("selected task coordinate does not match the bank")
        source = task["source"]
        reference = task["human_reference"]
        tasks.append(
            {
                "ordinal": ordinal,
                "task_id": task["task_id"],
                "family": task["family"],
                "prompt": task["prompt"],
                "prompt_sha256": task["prompt_sha256"],
                "task_sha256": task["task_sha256"],
                "blind_validity_stage": {
                    "answer_visible": False,
                    "source_url_visible": False,
                    "required_checks": [
                        "construct_fit",
                        "context_complete",
                        "single_primary_question_or_coherent_question_family",
                        "general_track_scope",
                        "answer_leakage_absent",
                        "culinary_discrimination_value",
                    ],
                    "allowed_decisions": ["valid", "revise", "exclude"],
                    "required_note_for_nonvalid_decision": True,
                },
                "sealed_human_reference_stage": {
                    "unlock_condition": "blind_validity_decision_sealed",
                    "reference_text": reference["text"],
                    "reference_text_sha256": reference["text_sha256"],
                    "reference_url": reference["url"],
                    "reference_license": reference["license"],
                    "reference_author": {
                        "display_name": reference["author"]["display_name"],
                        "profile_url": reference["author"]["profile_url"],
                    },
                    "reference_accepted": True,
                    "reference_use": "review_aid_not_automatic_ground_truth",
                    "source_url": source["url"],
                    "source_license": source["license"],
                    "source_author": {
                        "display_name": source["author"]["display_name"],
                        "profile_url": source["author"]["profile_url"],
                    },
                    "source_created_utc": source["created_utc"],
                    "criterion_authoring_requirements": {
                        "required_success_criteria": {"minimum": 2, "maximum": 8},
                        "permitted_variations": {"minimum": 1, "maximum": 6},
                        "disqualifying_errors": {"minimum": 1, "maximum": 8},
                        "objective_checks": {"minimum": 0, "maximum": 8},
                        "reference_adequacy_decision_required": True,
                        "criteria_must_be_answerable_from_prompt": True,
                        "criteria_must_not_require_reference_wording": True,
                    },
                },
                "review_status": "awaiting_three_independent_human_reviews",
                "independent_review_count": 0,
                "criteria_status": "not_authored",
                "confirmatory_eligible": False,
                "rank_eligible": False,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "FlavourBench",
        "artifact_role": "answer_blind_three_label_human_task_validation_packet",
        "source_task_bank_sha256": task_bank["artifact_sha256"],
        "source_task_validity_sha256": task_validity_sha256,
        "source_candidate_coordinate_sha256": task_validity["candidate_coordinate_sha256"],
        "task_coordinate_sha256": sha256_json(coordinates),
        "tasks": tasks,
        "counts": {
            "tasks": len(tasks),
            "per_family": {
                family: sum(task["family"] == family for task in tasks)
                for family in FAMILIES
            },
            "synthetic_tasks": 0,
            "human_questions": len(tasks),
            "accepted_human_references": len(tasks),
            "sealed_human_reviews": 0,
            "independently_validated_tasks": 0,
            "human_criterion_packs": 0,
        },
        "review_policy": {
            "required_independent_reviewers_per_task": REQUIRED_INDEPENDENT_REVIEWERS,
            "assignment_policy": ASSIGNMENT_POLICY,
            "assignment_tiebreaker_visible_to_reviewer": False,
            "unanimous_valid_completes_source_review": True,
            "adjudication_trigger": "any_nonunanimous_decision",
            "adjudicator_must_be_distinct_from_source_reviewers": True,
            "reviewer_must_be_qualified_for_family": True,
            "reviewer_identity_publication": "pseudonymous_or_named_at_reviewer_choice",
            "author_affiliated_reviews_reported_separately": True,
            "model_outputs_visible_during_task_validation": False,
            "reference_hidden_until_blind_validity_decision": True,
            "raw_reviews_append_only": True,
        },
        "statistics_policy": {
            "version": STATISTICS_POLICY,
            "agreement_population": "tasks_with_exactly_three_complete_independent_reviews",
            "decision_categories": ["valid", "revise", "exclude"],
            "agreement_metrics": [
                "unanimous_decision_rate",
                "mean_pairwise_agreement",
                "fleiss_kappa",
            ],
            "defect_metrics": [
                "decision_counts",
                "issue_tag_counts",
                "failed_blind_check_counts",
                "reference_adequacy_counts",
            ],
            "missing_reviews_imputed": False,
            "undefined_metrics_remain_null": True,
            "packet_rows_count_as_human_evidence": False,
        },
        "claim_boundary": {
            "packet_itself_is_human_validation_evidence": False,
            "criteria_are_model_authored": False,
            "criteria_are_human_authored_before_review": False,
            "three_independent_sealed_reviews_required": True,
            "public_tasks_remain_contamination_susceptible": True,
            "official": False,
            "rank_eligible": False,
        },
    }


def verify_validation_packet(document: Mapping[str, Any]) -> None:
    _verify_embedded_digest(document, field="artifact_sha256", label="validation packet")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise DevelopmentTaskValidationError("unexpected validation packet schema")
    if not _SHA256.fullmatch(str(document.get("source_task_validity_sha256") or "")):
        raise DevelopmentTaskValidationError("validation packet has no task-validity binding")
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_TASKS:
        raise DevelopmentTaskValidationError(
            f"validation packet must contain {EXPECTED_TASKS} tasks"
        )
    counts = document.get("counts")
    if not isinstance(counts, Mapping) or counts.get("synthetic_tasks") != 0:
        raise DevelopmentTaskValidationError("validation packet contains synthetic tasks")
    if counts.get("human_questions") != EXPECTED_TASKS:
        raise DevelopmentTaskValidationError("validation packet human-task count is inconsistent")
    policy = document.get("review_policy")
    if not isinstance(policy, Mapping) or (
        policy.get("required_independent_reviewers_per_task")
        != REQUIRED_INDEPENDENT_REVIEWERS
    ):
        raise DevelopmentTaskValidationError("validation packet has the wrong reviewer threshold")
    if policy.get("assignment_policy") != ASSIGNMENT_POLICY:
        raise DevelopmentTaskValidationError("validation packet has the wrong assignment policy")
    statistics_policy = document.get("statistics_policy")
    if not isinstance(statistics_policy, Mapping) or (
        statistics_policy.get("version") != STATISTICS_POLICY
        or statistics_policy.get("missing_reviews_imputed") is not False
        or statistics_policy.get("undefined_metrics_remain_null") is not True
        or statistics_policy.get("packet_rows_count_as_human_evidence") is not False
    ):
        raise DevelopmentTaskValidationError("validation packet has the wrong statistics policy")
    seen_ids: set[str] = set()
    seen_task_hashes: set[str] = set()
    seen_prompt_hashes: set[str] = set()
    family_counts: Counter[str] = Counter()
    coordinates: list[dict[str, str]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            raise DevelopmentTaskValidationError("validation packet contains a non-object task")
        task_id = str(task.get("task_id") or "")
        family = str(task.get("family") or "")
        task_sha256 = str(task.get("task_sha256") or "")
        prompt_sha256 = str(task.get("prompt_sha256") or "")
        if family not in FAMILIES or not task_id.startswith(f"fb-s0-{family}-"):
            raise DevelopmentTaskValidationError("validation packet task identity is invalid")
        if sha256_text(str(task.get("prompt") or "")) != prompt_sha256:
            raise DevelopmentTaskValidationError("validation packet prompt hash does not verify")
        if not _SHA256.fullmatch(task_sha256):
            raise DevelopmentTaskValidationError("validation packet task hash is invalid")
        if (
            task_id in seen_ids
            or task_sha256 in seen_task_hashes
            or prompt_sha256 in seen_prompt_hashes
        ):
            raise DevelopmentTaskValidationError(
                "validation packet task coordinates are not unique"
            )
        seen_ids.add(task_id)
        seen_task_hashes.add(task_sha256)
        seen_prompt_hashes.add(prompt_sha256)
        family_counts[family] += 1
        coordinates.append(
            {
                "task_id": task_id,
                "family": family,
                "task_sha256": task_sha256,
                "prompt_sha256": prompt_sha256,
            }
        )
        reference = task.get("sealed_human_reference_stage")
        if not isinstance(reference, Mapping) or (
            sha256_text(str(reference.get("reference_text") or ""))
            != reference.get("reference_text_sha256")
        ):
            raise DevelopmentTaskValidationError("validation packet reference hash does not verify")
        if (
            task.get("independent_review_count") != 0
            or task.get("criteria_status") != "not_authored"
        ):
            raise DevelopmentTaskValidationError("unreviewed packet contains review claims")
    if family_counts != Counter({family: EXPECTED_TASKS_PER_FAMILY for family in FAMILIES}):
        raise DevelopmentTaskValidationError("validation packet is not family balanced")
    if sha256_json(coordinates) != document.get("task_coordinate_sha256"):
        raise DevelopmentTaskValidationError("validation packet coordinate hash does not verify")


def _atomic_write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"development-task-human-validation-v2-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise DevelopmentTaskValidationError("content-addressed output conflict")
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
    parser.add_argument("--task-validity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    packet = build_validation_packet(
        task_bank=_load_object(arguments.task_bank),
        task_validity=_load_object(arguments.task_validity),
    )
    path = _atomic_write(arguments.output_dir, packet)
    print(
        json.dumps(
            {
                "output": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "counts": packet["counts"],
                "review_policy": packet["review_policy"],
                "claim_boundary": packet["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
