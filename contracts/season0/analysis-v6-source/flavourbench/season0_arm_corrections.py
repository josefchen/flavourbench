"""Validate append-only interpretation corrections for immutable Season 0 arms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-season0-arm-interpretation-correction-v1"
RECORD_TYPE = "superseding_delivery_state_interpretation"
REASON_CODE = "read_timeout_has_ambiguous_delivery"
SOURCE_SCOPE = "season0-scored-v1"


class ArmInterpretationCorrectionError(RuntimeError):
    """A correction is missing, malformed, or not bound to the immutable arms."""


@dataclass(frozen=True)
class ValidatedArmInterpretationCorrection:
    artifact_sha256: str
    arm_ids: tuple[str, ...]
    source_arm_set_sha256: str
    conservative_reservation_usd: Decimal


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ArmInterpretationCorrectionError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm_paths_by_id(arms_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(arms_dir.glob("*.json")):
        arm = _load(path)
        arm_id = arm.get("arm_id")
        if not isinstance(arm_id, str) or not arm_id:
            raise ArmInterpretationCorrectionError(f"arm has no identity: {path}")
        if arm_id in paths:
            raise ArmInterpretationCorrectionError(
                f"multiple immutable records found for corrected arm {arm_id}"
            )
        paths[arm_id] = path
    return paths


def validate_arm_interpretation_correction(
    *,
    correction: dict[str, Any] | None,
    arms_dir: Path,
) -> ValidatedArmInterpretationCorrection | None:
    """Validate the sole active delivery-state overlay without changing raw records."""

    paths = _arm_paths_by_id(arms_dir)
    candidate_ids: set[str] = set()
    arms: dict[str, dict[str, Any]] = {}
    for arm_id, path in paths.items():
        arm = _load(path)
        if (
            arm.get("status") == "failed"
            and arm.get("error_type") == "ReadTimeoutError"
            and arm.get("delivery_state") == "safe_pre_inference"
        ):
            claimed = arm.get("artifact_sha256")
            actual = sha256_json(
                {key: value for key, value in arm.items() if key != "artifact_sha256"}
            )
            if claimed != actual:
                raise ArmInterpretationCorrectionError(
                    f"source arm artifact hash mismatch: {arm_id}"
                )
            candidate_ids.add(arm_id)
            arms[arm_id] = arm

    if correction is None:
        if candidate_ids:
            raise ArmInterpretationCorrectionError(
                "ambiguous read-timeout arms require an interpretation correction"
            )
        return None

    claimed = correction.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in correction.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise ArmInterpretationCorrectionError("correction artifact hash mismatch")
    if (
        correction.get("schema_version") != SCHEMA_VERSION
        or correction.get("record_type") != RECORD_TYPE
        or correction.get("scope") != SOURCE_SCOPE
        or correction.get("reason_code") != REASON_CODE
        or correction.get("supersedes_correction_artifact_sha256") is not None
    ):
        raise ArmInterpretationCorrectionError("unsupported correction contract")

    rows = correction.get("corrections")
    if not isinstance(rows, list) or correction.get("source_arm_count") != len(rows):
        raise ArmInterpretationCorrectionError("correction count does not reconcile")
    if len(rows) != len(candidate_ids):
        raise ArmInterpretationCorrectionError(
            "correction does not cover the complete ambiguous-timeout population"
        )

    expected_assertions = {
        "provider": "bedrock",
        "season_model_id": "fb-s0-model-10",
        "status": "failed",
        "error_type": "ReadTimeoutError",
        "delivery_state": "safe_pre_inference",
        "result": None,
        "rank_eligible": False,
        "reservation_usd": "0.033968880",
        "synthetic": False,
    }
    expected_field_correction = {
        "json_pointer": "/delivery_state",
        "from": "safe_pre_inference",
        "to": "uncertain",
    }
    expected_interpretation = {
        "failure_class": "uncertain_delivery",
        "cost_disposition": "retain_conservative_reservation",
        "replay_authorized": False,
        "rank_eligible": False,
    }

    seen_ids: set[str] = set()
    seen_source_hashes: set[str] = set()
    source_hashes: list[str] = []
    total_reservation = Decimal(0)
    for row in rows:
        if not isinstance(row, dict):
            raise ArmInterpretationCorrectionError("correction row is not an object")
        arm_id = row.get("arm_id")
        source_hash = row.get("source_arm_artifact_sha256")
        if not isinstance(arm_id, str) or not isinstance(source_hash, str):
            raise ArmInterpretationCorrectionError("correction row lacks source identity")
        if arm_id in seen_ids or source_hash in seen_source_hashes:
            raise ArmInterpretationCorrectionError("correction repeats a source head")
        seen_ids.add(arm_id)
        seen_source_hashes.add(source_hash)
        source_hashes.append(source_hash)

        arm = arms.get(arm_id)
        path = paths.get(arm_id)
        if arm is None or path is None or arm_id not in candidate_ids:
            raise ArmInterpretationCorrectionError(
                f"correction references a non-candidate arm: {arm_id}"
            )
        if (
            source_hash != arm.get("artifact_sha256")
            or row.get("source_arm_file_sha256") != _file_sha256(path)
        ):
            raise ArmInterpretationCorrectionError(
                f"correction source hash mismatch: {arm_id}"
            )
        model = arm.get("model")
        assertions = {
            "provider": model.get("provider") if isinstance(model, dict) else None,
            "season_model_id": (
                model.get("season_model_id") if isinstance(model, dict) else None
            ),
            "status": arm.get("status"),
            "error_type": arm.get("error_type"),
            "delivery_state": arm.get("delivery_state"),
            "result": arm.get("result"),
            "rank_eligible": arm.get("rank_eligible"),
            "reservation_usd": arm.get("reservation_usd"),
            "synthetic": arm.get("synthetic"),
        }
        if row.get("source_assertions") != assertions or assertions != expected_assertions:
            raise ArmInterpretationCorrectionError(
                f"correction source assertions do not match: {arm_id}"
            )
        if row.get("field_correction") != expected_field_correction:
            raise ArmInterpretationCorrectionError(
                f"correction changes an unsupported field: {arm_id}"
            )
        if row.get("derived_interpretation") != expected_interpretation:
            raise ArmInterpretationCorrectionError(
                f"correction changes a derived disposition: {arm_id}"
            )
        total_reservation += Decimal(str(arm["reservation_usd"]))

    if seen_ids != candidate_ids:
        raise ArmInterpretationCorrectionError("correction leaves a source arm unbound")
    source_set_sha = sha256_json({"artifact_sha256s": sorted(source_hashes)})
    if correction.get("source_arm_set_sha256") != source_set_sha:
        raise ArmInterpretationCorrectionError("correction source-arm set hash mismatch")
    if correction.get("conservative_reservation_usd") != format(
        total_reservation, ".9f"
    ):
        raise ArmInterpretationCorrectionError(
            "correction conservative reservation does not reconcile"
        )

    return ValidatedArmInterpretationCorrection(
        artifact_sha256=actual,
        arm_ids=tuple(sorted(seen_ids)),
        source_arm_set_sha256=source_set_sha,
        conservative_reservation_usd=total_reservation,
    )
