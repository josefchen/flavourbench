"""Append deterministic validator receipts to completed real pilot arms.

Historical paid pilot arms pre-date the current worker-side validator pipeline.
This command verifies their immutable hashes, evaluates the same deterministic
validators used prospectively, inserts only missing versioned receipts, and
writes a restricted content-addressed aggregate.  It never edits model outputs,
votes, task labels, or ranking eligibility.
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

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from .database import session_scope
from .models import Battle, CatalogModel, ResponseArm, ToolCall, ValidatorResult
from .validators import VALIDATOR_VERSION, Validation, validate_output

SCHEMA_VERSION = "flavourbench-real-arm-validator-audit-v1"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "flavourbench/artifacts/season1/validators/real-pilot-v1"
)


class ValidatorBackfillError(RuntimeError):
    """The immutable source rows or existing validator receipts disagree."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _runtime_json_sha256(value: object) -> str:
    """Match the persisted ORM/worker JSON digest convention exactly."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validator_payload(validation: Validation) -> dict[str, Any]:
    return {
        "validator_name": validation.name,
        "validator_version": VALIDATOR_VERSION,
        "status": validation.status,
        "score_milli": validation.score_milli,
        "detail": validation.detail,
    }


def _source_rows(session: Session) -> list[dict[str, Any]]:
    tool_counts = (
        select(
            ToolCall.arm_id.label("arm_id"),
            func.count(ToolCall.id).label("tool_calls"),
            func.sum(case((ToolCall.is_error.is_(True), 1), else_=0)).label("tool_errors"),
        )
        .group_by(ToolCall.arm_id)
        .subquery()
    )
    statement = (
        select(
            ResponseArm,
            Battle,
            CatalogModel,
            func.coalesce(tool_counts.c.tool_calls, 0),
            func.coalesce(tool_counts.c.tool_errors, 0),
        )
        .join(Battle, Battle.id == ResponseArm.battle_id)
        .join(CatalogModel, CatalogModel.model_id == ResponseArm.model_id)
        .outerjoin(tool_counts, tool_counts.c.arm_id == ResponseArm.id)
        .where(
            ResponseArm.status == "complete",
            Battle.run_class == "pilot",
            Battle.data_stratum == "development",
            Battle.rank_eligible.is_(False),
        )
        .order_by(Battle.created_at, Battle.id, ResponseArm.side)
    )
    rows: list[dict[str, Any]] = []
    for arm, battle, model, tool_calls, tool_errors in session.execute(statement):
        if not arm.answer_markdown:
            raise ValidatorBackfillError(f"complete arm {arm.id} has no answer")
        if arm.answer_markdown_sha256 != _sha256_text(arm.answer_markdown):
            raise ValidatorBackfillError(f"arm {arm.id} answer digest does not verify")
        if not isinstance(arm.output_json, Mapping):
            raise ValidatorBackfillError(f"arm {arm.id} output is not an object")
        if arm.output_json_sha256 != _runtime_json_sha256(arm.output_json):
            raise ValidatorBackfillError(f"arm {arm.id} output digest does not verify")
        if battle.prompt is None or battle.prompt_redacted:
            raise ValidatorBackfillError(f"pilot battle {battle.id} prompt is unavailable")
        if battle.prompt_sha256 != _sha256_text(battle.prompt):
            raise ValidatorBackfillError(f"battle {battle.id} prompt digest does not verify")
        if arm.prompt_sha256 != battle.prompt_sha256:
            raise ValidatorBackfillError(f"arm {arm.id} prompt binding does not verify")
        if (
            arm.execution_backend == "mock"
            or arm.provider_slug == "mock"
            or arm.model_id.startswith("flavourbench/mock-")
        ):
            raise ValidatorBackfillError("real pilot scope contains a synthetic/mock arm")
        validations = validate_output(
            prompt=battle.prompt,
            output=dict(arm.output_json),
            answer=arm.answer_markdown,
            model_name=model.name,
            tool_errors=int(tool_errors),
            tool_calls=int(tool_calls),
            finish_reason=arm.finish_reason,
        )
        rows.append(
            {
                "arm": arm,
                "battle": battle,
                "model": model,
                "tool_calls": int(tool_calls),
                "tool_errors": int(tool_errors),
                "validations": validations,
            }
        )
    if not rows:
        raise ValidatorBackfillError("no completed real development-pilot arms were found")
    return rows


def backfill_and_build_report(session: Session) -> dict[str, Any]:
    rows = _source_rows(session)
    arm_ids = [str(row["arm"].id) for row in rows]
    existing = {
        (result.arm_id, result.validator_name): result
        for result in session.scalars(
            select(ValidatorResult).where(
                ValidatorResult.arm_id.in_(arm_ids),
                ValidatorResult.validator_version == VALIDATOR_VERSION,
            )
        )
    }
    inserted = 0
    already_present = 0
    status_by_validator: dict[str, Counter[str]] = defaultdict(Counter)
    status_by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    status_by_model: dict[str, Counter[str]] = defaultdict(Counter)
    coordinates: list[dict[str, Any]] = []

    for row in rows:
        arm: ResponseArm = row["arm"]
        battle: Battle = row["battle"]
        validation_coordinates: list[dict[str, Any]] = []
        for validation in row["validations"]:
            payload = _validator_payload(validation)
            current = existing.get((arm.id, validation.name))
            if current is None:
                session.add(
                    ValidatorResult(
                        arm_id=arm.id,
                        validator_name=validation.name,
                        validator_version=VALIDATOR_VERSION,
                        status=validation.status,
                        score_milli=validation.score_milli,
                        detail_json=validation.detail,
                    )
                )
                inserted += 1
            else:
                expected = {
                    "validator_name": current.validator_name,
                    "validator_version": current.validator_version,
                    "status": current.status,
                    "score_milli": current.score_milli,
                    "detail": current.detail_json,
                }
                if expected != payload:
                    raise ValidatorBackfillError(
                        f"existing validator receipt disagrees for {arm.id}/{validation.name}"
                    )
                already_present += 1
            status_by_validator[validation.name][validation.status] += 1
            status_by_condition[arm.condition][f"{validation.name}:{validation.status}"] += 1
            status_by_model[arm.model_id][f"{validation.name}:{validation.status}"] += 1
            validation_coordinates.append(
                {
                    "name": validation.name,
                    "version": VALIDATOR_VERSION,
                    "status": validation.status,
                    "detail_sha256": _runtime_json_sha256({"detail": validation.detail}),
                }
            )
        coordinates.append(
            {
                "battle_id": battle.id,
                "arm_id": arm.id,
                "side": arm.side,
                "condition": arm.condition,
                "model_id": arm.model_id,
                "execution_backend": arm.execution_backend,
                "actual_model_id": arm.actual_model_id,
                "actual_provider_slug": arm.actual_provider_slug,
                "generation_id": arm.generation_id,
                "prompt_sha256": battle.prompt_sha256,
                "answer_sha256": arm.answer_markdown_sha256,
                "output_sha256": arm.output_json_sha256,
                "finish_reason": arm.finish_reason,
                "tool_calls": row["tool_calls"],
                "tool_errors": row["tool_errors"],
                "validations": validation_coordinates,
            }
        )
    session.flush()

    non_normal = sum(
        any(
            validation.name == "semantic_completion"
            and "non_normal_finish_reason" in validation.detail["failure_reasons"]
            for validation in row["validations"]
        )
        for row in rows
    )
    evidence_warnings = status_by_validator["evidence_claim_boundary"]["warn"]
    surface_warnings = status_by_validator["task_surface_integrity"]["warn"]
    source_coordinate_sha256 = _sha256_json(coordinates)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "restricted_real_arm_validator_audit",
        "scope": "development_quality_assurance_not_benchmark_results",
        "validator_version": VALIDATOR_VERSION,
        "source_coordinate_sha256": source_coordinate_sha256,
        "observed": {
            "real_response_arms": len(rows),
            "synthetic_or_mock_arms": 0,
            "distinct_battles": len({str(row["battle"].id) for row in rows}),
            "distinct_models": len({str(row["arm"].model_id) for row in rows}),
            "distinct_conditions": sorted({str(row["arm"].condition) for row in rows}),
            "validator_receipts_verified": inserted + already_present,
            "non_normal_final_completion_arms": non_normal,
            "evidence_claim_boundary_warnings": evidence_warnings,
            "task_surface_integrity_warnings": surface_warnings,
        },
        "status_by_validator": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(status_by_validator.items())
        },
        "status_by_condition": {
            condition: dict(sorted(counts.items()))
            for condition, counts in sorted(status_by_condition.items())
        },
        "status_by_exact_model": {
            model_id: dict(sorted(counts.items()))
            for model_id, counts in sorted(status_by_model.items())
        },
        "claim_boundary": {
            "real_model_outputs": True,
            "real_provider_calls": True,
            "real_epicure_traces_when_recorded": True,
            "synthetic_observations": 0,
            "deterministic_shape_and_audit_triggers_only": True,
            "evidence_warning_is_culinary_ground_truth": False,
            "task_surface_warning_is_construct_invalidity_ground_truth": False,
            "human_quality_judgments_added": 0,
            "paper_result_use": False,
            "official_leaderboard_use": False,
            "model_ranking_use": False,
        },
        "source_coordinates": coordinates,
    }
    return report


def _write_report(report: Mapping[str, Any], output_dir: Path) -> Path:
    digest = _sha256_json(report)
    document = {**report, "artifact_sha256": digest}
    rendered = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"real-arm-validator-audit-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise ValidatorBackfillError("content-addressed validator report conflicts with disk")
    else:
        with tempfile.NamedTemporaryFile(
            "w", dir=output_dir, delete=False, encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    with session_scope() as session:
        report = backfill_and_build_report(session)
    path = _write_report(report, arguments.output_dir.resolve())
    print(
        json.dumps(
            {
                "output": str(path),
                "artifact_sha256": path.stem.rsplit("-", 1)[-1],
                "observed": report["observed"],
                "claim_boundary": report["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
