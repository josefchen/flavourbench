"""Freeze a cleaner, real-human development task set with surface-dependency gates.

Version 1 verified provenance and specialist-scope quarantine.  Version 2 keeps
those guarantees and additionally removes prompts that visibly depend on URLs,
photographs, images, or diagrams that are absent from the text-only benchmark.
This remains development evidence: public source material is contamination-prone
and no machine rule substitutes for independent answer-blind human validation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .development_task_validity import (
    DEFAULT_RECENT_FLOOR_PER_FAMILY,
    DEFAULT_TASKS_PER_FAMILY,
    FAMILIES,
    RECENT_CUTOFF,
    DevelopmentTaskValidityError,
    _load_object,
    _rank_key,
    _recent,
    verify_scope_review,
    verify_task_bank,
)
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-development-task-validity-v2"
_EXTERNAL_URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_VISUAL_CONTEXT = re.compile(
    r"\b(?:image|images|photo|photos|picture|pictures|diagram|diagrams|attached)\b",
    re.IGNORECASE,
)


def surface_dependency_reasons(prompt: str) -> list[str]:
    """Return stable, high-precision reasons that a text task is not self-contained."""

    reasons: list[str] = []
    if _EXTERNAL_URL.search(prompt):
        reasons.append("external_url_dependency_signal")
    if _VISUAL_CONTEXT.search(prompt):
        reasons.append("visual_context_dependency_signal")
    return reasons


def select_surface_clean_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    quarantined: Mapping[str, str],
    tasks_per_family: int = DEFAULT_TASKS_PER_FAMILY,
    recent_floor_per_family: int = DEFAULT_RECENT_FLOOR_PER_FAMILY,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if tasks_per_family <= 0 or recent_floor_per_family < 0:
        raise DevelopmentTaskValidityError("selection counts must be non-negative")
    if recent_floor_per_family > tasks_per_family:
        raise DevelopmentTaskValidityError("recent floor cannot exceed the family size")

    surface_quarantine = {
        str(task["task_id"]): surface_dependency_reasons(str(task["prompt"]))
        for task in tasks
        if task["task_id"] not in quarantined
        and surface_dependency_reasons(str(task["prompt"]))
    }
    selected: list[dict[str, Any]] = []
    for family in FAMILIES:
        eligible = [
            dict(task)
            for task in tasks
            if task["family"] == family
            and task["task_id"] not in quarantined
            and task["task_id"] not in surface_quarantine
        ]
        recent = sorted((task for task in eligible if _recent(task)), key=_rank_key)
        if len(recent) < recent_floor_per_family:
            raise DevelopmentTaskValidityError(
                f"{family} has only {len(recent)} recent surface-clean tasks"
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
                f"{family} has only {len(chosen)} surface-clean development tasks"
            )
        selected.extend(chosen)
    return (
        sorted(
            selected,
            key=lambda task: (FAMILIES.index(str(task["family"])), str(task["task_id"])),
        ),
        surface_quarantine,
    )


def build_validity_artifact(
    *,
    task_bank: Mapping[str, Any],
    scope_review: Mapping[str, Any],
    tasks_per_family: int = DEFAULT_TASKS_PER_FAMILY,
    recent_floor_per_family: int = DEFAULT_RECENT_FLOOR_PER_FAMILY,
) -> dict[str, Any]:
    tasks = verify_task_bank(task_bank)
    specialist_quarantine = verify_scope_review(
        scope_review,
        task_ids={str(task["task_id"]) for task in tasks},
    )
    selected, surface_quarantine = select_surface_clean_tasks(
        tasks,
        quarantined=specialist_quarantine,
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
            "source_answer_count": int(task["source"].get("answer_count_at_import") or 0),
            "source_question_score": int(task["source"].get("score_at_import") or 0),
            "hidden_reference_sha256": task["human_reference"]["text_sha256"],
            "hidden_reference_url": task["human_reference"]["url"],
            "hidden_reference_score": int(
                task["human_reference"].get("score_at_import") or 0
            ),
            "recent": _recent(task),
            "surface_dependency_screen": {
                "status": "pass",
                "failure_reasons": [],
                "screen_version": SCHEMA_VERSION,
            },
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
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "FlavourBench",
        "artifact_role": "surface_clean_real_public_development_task_validity_dossier",
        "status": "surface_clean_source_verified_development_candidate_not_confirmatory",
        "source_observed_through_utc": task_bank["observed_through_utc"],
        "source_task_bank_sha256": task_bank["artifact_sha256"],
        "source_task_set_sha256": task_bank["task_set_sha256"],
        "specialist_scope_review_sha256": scope_review["artifact_sha256"],
        "candidate_coordinate_sha256": candidate_coordinate_sha256,
        "counts": {
            "source_tasks": len(tasks),
            "synthetic_tasks": 0,
            "specialist_scope_quarantined": len(specialist_quarantine),
            "surface_dependency_quarantined": len(surface_quarantine),
            "surface_clean_general_track_candidates": sum(
                task["task_id"] not in specialist_quarantine
                and task["task_id"] not in surface_quarantine
                for task in tasks
            ),
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
            "text_self_containment_surface_gate": True,
            "tasks_per_family": tasks_per_family,
            "recent_cutoff_utc": RECENT_CUTOFF,
            "minimum_recent_per_family": recent_floor_per_family,
            "quality_outcomes_used_for_selection": 0,
            "human_answers_visible_during_surface_screen": False,
        },
        "validity_layers": {
            "provenance_integrity": "verified",
            "licensed_human_origin": "verified_zero_synthetic",
            "specialist_scope": "governed_quarantine_applied",
            "text_self_containment_surface": "machine_screen_passed",
            "independent_answer_blind_human_validation": "missing",
            "task_specific_human_criterion_packs": "missing",
            "contamination_resistance": "not_met_public_development_source",
        },
        "specialist_quarantine": [
            {"task_id": task_id, "reason_code": specialist_quarantine[task_id]}
            for task_id in sorted(specialist_quarantine)
        ],
        "surface_dependency_quarantine": [
            {"task_id": task_id, "reason_codes": surface_quarantine[task_id]}
            for task_id in sorted(surface_quarantine)
        ],
        "claim_boundary": {
            "real_human_authored_tasks": True,
            "synthetic_tasks": 0,
            "supports_real_current_model_development_runs": True,
            "supports_confirmatory_task_validity": False,
            "supports_official_leaderboard": False,
            "machine_surface_screen_is_ground_truth": False,
            "official": False,
            "rank_eligible": False,
        },
        "tasks": selected_records,
    }


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
    destination = output_dir / f"development-task-validity-v2-{digest}.json"
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
    path = _atomic_write(arguments.output_dir.resolve(), payload)
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
