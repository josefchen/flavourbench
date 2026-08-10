"""Content-addressed quarantine for suspect exact-frontier development tasks.

The records in this module do not mutate the frozen Seasoned Advice task bank or
the real provider outputs generated from it.  They are a conservative admission
overlay: affected tasks stay available for reliability and audit work, but are
not admitted to candidate review pools or an official preference fit unless a
later, qualified human adjudication artifact explicitly supersedes this hold.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-current-frontier-task-quarantine-v1"
SOURCE_TASK_BANK_SHA256 = (
    "1ce969bdee4124fa44bab46a04feda2a0ebeddf4d37c49c0264b48b3833a4313"
)
SOURCE_TASK_BANK_FILE_SHA256 = (
    "023cb9cee80bd2b3d21d2b94cd01c824cf26d0938f9ed398ccb0fd873dd525e3"
)
GOVERNANCE_REVIEW_PATH = (
    "governance/reviews/FLAVOURBENCH-TASK-DATA-QUALITY-20260721.md"
)

# These four records are the exact intersection between the current 28-task
# exact-frontier collection and the conservative manual screening findings.
# Prompt and task hashes bind the hold to immutable task bytes rather than to a
# reusable human-readable identifier alone.
QUARANTINE_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "fb-s0-composition-006",
        "family": "composition",
        "task_sha256": "f74b9cb3c752c0d546e6ff783f5a07c77fa5e042eac25a1b66766a89efb46ce0",
        "prompt_sha256": "5389038b6cacb310e3a77e691ed9b7c6e3ef9dee7bd120279c3088480b7e50f0",
        "reason_codes": [
            "stripped_external_recipe_context",
            "stripped_external_product_context",
            "context_completeness_pending_human_adjudication",
        ],
        "evidence_note": (
            "The stored text refers to 'this recipe', a garlic product, and a specific "
            "olive oil, but the linked material is absent from the text-only task."
        ),
        "finding_class": "current_conservative_screening",
    },
    {
        "task_id": "fb-s0-composition-008",
        "family": "composition",
        "task_sha256": "c78eab5b370d541d23170105fc23700b8107e4fdf99e0e869d039bd2371870d5",
        "prompt_sha256": "12312a34bb2a6b17280a1f241a5523a05cba08a661422c12c040ce3c694d951a",
        "reason_codes": [
            "composition_construct_family_ambiguity",
            "published_recipe_rights_review_pending",
        ],
        "evidence_note": (
            "The prompt compares published recipes while omitting their full procedures, "
            "and explicitly flags uncertainty about permission to reproduce them."
        ),
        "finding_class": "current_conservative_screening",
    },
    {
        "task_id": "fb-s0-composition-009",
        "family": "composition",
        "task_sha256": "b0f4b733e5157c41d6b3a5ca7ba0b4db8452e2255452338bd6a7b4020c437751",
        "prompt_sha256": "1d02fd9d0a801d0a01c22cd3fa61a63bd3c95992e646fdc766f15bce13e14c6c",
        "reason_codes": [
            "composition_construct_drift",
            "conceptual_sensory_language_without_executable_composition_decision",
        ],
        "evidence_note": (
            "The primary request is terminology for a subjective sensory experience, not "
            "an executable multi-ingredient composition decision."
        ),
        "finding_class": "current_conservative_screening",
    },
    {
        "task_id": "fb-s0-cookability-003",
        "family": "cookability",
        "task_sha256": "663fa0c7c873bcd9a2d4dacb15c7b7ed7241d2bfa40e87f1d780f2e0fa04dfa5",
        "prompt_sha256": "eb62c31c9b472c0ea328ca2213844716226d43ccfe810e3c1028e7f09e093b2e",
        "reason_codes": [
            "four_absent_images",
            "embedded_edit_advances_author_causal_resolution",
        ],
        "evidence_note": (
            "Four referenced images are absent, and a later edit advances shorter proofing "
            "as the author's own causal resolution before the model answers."
        ),
        "finding_class": "documented_task_bank_audit_and_current_screening",
    },
)


class CurrentFrontierTaskQuarantineError(ValueError):
    """The quarantine overlay or its attempted application is malformed."""


def build_quarantine_artifact() -> dict[str, Any]:
    records = [dict(record) for record in QUARANTINE_RECORDS]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "superseding_current_frontier_development_task_admission_overlay",
        "status": "quarantine_pending_qualified_adjudication",
        "source_task_bank_sha256": SOURCE_TASK_BANK_SHA256,
        "source_task_bank_file_sha256": SOURCE_TASK_BANK_FILE_SHA256,
        "scope": {
            "collection": "current_exact_frontier_development_study",
            "tracks": ["model_arena", "epicure_uplift"],
            "raw_tasks_mutated": False,
            "raw_response_arms_mutated": False,
            "synthetic_tasks": 0,
            "synthetic_arms": 0,
        },
        "policy": {
            "candidate_review_pool_admission": "exclude",
            "official_fit_admission": "exclude",
            "reliability_and_audit_use": "retain",
            "supersession_requirement": (
                "content-addressed qualification-matched human adjudication record"
            ),
            "machine_screen_is_ground_truth": False,
        },
        "evidence": {
            "governance_review_path": GOVERNANCE_REVIEW_PATH,
            "qualification_boundary": (
                "The review documents cookability-003 and the composition-family review "
                "risk. The first three task-level dispositions are conservative current "
                "screening findings and require qualified adjudication."
            ),
        },
        "records": records,
        "record_count": len(records),
        "task_set_sha256": sha256_json(
            [
                {
                    "task_id": record["task_id"],
                    "task_sha256": record["task_sha256"],
                    "prompt_sha256": record["prompt_sha256"],
                }
                for record in records
            ]
        ),
        "claim_boundary": {
            "declares_tasks_definitively_invalid": False,
            "supports_quality_ranking": False,
            "supports_epicure_uplift_estimation": False,
            "development_admission_control_only": True,
        },
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def verify_quarantine_artifact(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CurrentFrontierTaskQuarantineError("unexpected quarantine schema")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != sha256_json(payload):
        raise CurrentFrontierTaskQuarantineError("quarantine content address is invalid")
    if value.get("status") != "quarantine_pending_qualified_adjudication":
        raise CurrentFrontierTaskQuarantineError("quarantine status changed")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != len(QUARANTINE_RECORDS):
        raise CurrentFrontierTaskQuarantineError("quarantine record set changed")
    if len({str(record.get("task_id")) for record in records}) != len(records):
        raise CurrentFrontierTaskQuarantineError("quarantine task IDs are not unique")


def quarantine_task_ids() -> frozenset[str]:
    return frozenset(str(record["task_id"]) for record in QUARANTINE_RECORDS)


def quarantine_binding() -> dict[str, Any]:
    artifact = build_quarantine_artifact()
    return {
        "status": artifact["status"],
        "artifact_sha256": artifact["artifact_sha256"],
        "task_set_sha256": artifact["task_set_sha256"],
        "task_ids": sorted(quarantine_task_ids()),
        "candidate_review_pool_admission": "exclude",
        "official_fit_admission": "exclude",
    }


def write_quarantine_artifact(output_dir: Path) -> Path:
    artifact = build_quarantine_artifact()
    verify_quarantine_artifact(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (
        f"current-frontier-task-quarantine-{artifact['artifact_sha256']}.json"
    )
    rendered = (
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise CurrentFrontierTaskQuarantineError(
                "content-addressed quarantine output conflict"
            )
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
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    path = write_quarantine_artifact(arguments.output_dir.resolve())
    artifact = build_quarantine_artifact()
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "artifact_sha256": artifact["artifact_sha256"],
                "record_count": artifact["record_count"],
                "output": str(path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
