"""Build and validate the Season 0 final-completion interpretation overlay.

The immutable collector records are not rewritten. This module identifies every
arm that the historical collector marked rank eligible despite a non-normal
provider finish reason, then binds a conservative derived interpretation to the
complete candidate population.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json, sha256_text

SCHEMA_VERSION = "flavourbench-season0-completion-interpretation-correction-v1"
RECORD_TYPE = "superseding_final_completion_interpretation"
REASON_CODE = "non_normal_finish_is_not_a_complete_answer"
SOURCE_SCOPE = "season0-scored-v1"
ACCEPTED_FINISH_REASONS = frozenset({"completed", "end_turn", "stop", "stop_sequence"})


class CompletionInterpretationCorrectionError(RuntimeError):
    """The completion overlay is missing, malformed, or not bound to its arms."""


@dataclass(frozen=True)
class ValidatedCompletionInterpretationCorrection:
    artifact_sha256: str
    arm_ids: tuple[str, ...]
    source_arm_set_sha256: str
    counts_by_finish_reason: dict[str, int]


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CompletionInterpretationCorrectionError(
            f"invalid completion-correction input: {path}"
        ) from error
    if not isinstance(value, dict):
        raise CompletionInterpretationCorrectionError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finish_reason(arm: Mapping[str, Any]) -> str:
    result = arm.get("result")
    if not isinstance(result, Mapping):
        return "missing"
    return str(result.get("finish_reason") or "missing").strip().lower()


def _candidate(arm: Mapping[str, Any]) -> bool:
    return bool(
        arm.get("status") == "success"
        and arm.get("delivery_state") == "reconciled"
        and arm.get("rank_eligible") is True
        and _finish_reason(arm) not in ACCEPTED_FINISH_REASONS
    )


def _arm_paths(arms_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(arms_dir.glob("*.json")):
        arm = _load(path)
        arm_id = arm.get("arm_id")
        if not isinstance(arm_id, str) or not arm_id:
            raise CompletionInterpretationCorrectionError(f"arm has no identity: {path}")
        if arm_id in paths:
            raise CompletionInterpretationCorrectionError(
                f"multiple immutable records found for arm {arm_id}"
            )
        paths[arm_id] = path
    return paths


def _source_assertions(arm: Mapping[str, Any]) -> dict[str, Any]:
    result = arm.get("result")
    model = arm.get("model")
    answer = result.get("answer_markdown") if isinstance(result, Mapping) else None
    return {
        "status": arm.get("status"),
        "delivery_state": arm.get("delivery_state"),
        "rank_eligible": arm.get("rank_eligible"),
        "finish_reason": _finish_reason(arm),
        "answer_sha256": sha256_text(answer) if isinstance(answer, str) else None,
        "provider": model.get("provider") if isinstance(model, Mapping) else None,
        "season_model_id": (model.get("season_model_id") if isinstance(model, Mapping) else None),
        "condition": arm.get("condition"),
        "synthetic": arm.get("synthetic"),
    }


def build_completion_interpretation_correction(arms_dir: Path) -> dict[str, Any]:
    paths = _arm_paths(arms_dir)
    rows: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    by_finish: Counter[str] = Counter()
    by_condition: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    for arm_id, path in paths.items():
        arm = _load(path)
        claimed = arm.get("artifact_sha256")
        actual = sha256_json({key: value for key, value in arm.items() if key != "artifact_sha256"})
        if claimed != actual:
            raise CompletionInterpretationCorrectionError(
                f"source arm artifact hash mismatch: {arm_id}"
            )
        if not _candidate(arm):
            continue
        assertions = _source_assertions(arm)
        source_hashes.append(actual)
        by_finish[str(assertions["finish_reason"])] += 1
        by_condition[str(assertions["condition"])] += 1
        by_model[str(assertions["season_model_id"])] += 1
        rows.append(
            {
                "arm_id": arm_id,
                "source_arm_artifact_sha256": actual,
                "source_arm_file_sha256": _file_sha256(path),
                "source_assertions": assertions,
                "derived_interpretation": {
                    "completion_status": "incomplete",
                    "failure_class": "incomplete_final_response",
                    "rank_eligible": False,
                    "cost_disposition": "retain_recorded_or_reserved_cost",
                    "replay_authorized": False,
                },
            }
        )
    rows.sort(key=lambda row: str(row["arm_id"]))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "scope": SOURCE_SCOPE,
        "reason_code": REASON_CODE,
        "policy": {
            "accepted_finish_reasons": sorted(ACCEPTED_FINISH_REASONS),
            "rule": (
                "A rank-eligible final answer must end with a normal provider completion "
                "reason. Nonempty text does not override truncation or filtering."
            ),
        },
        "source_arm_count": len(rows),
        "source_arm_set_sha256": sha256_json({"artifact_sha256s": sorted(source_hashes)}),
        "counts_by_finish_reason": dict(sorted(by_finish.items())),
        "counts_by_condition": dict(sorted(by_condition.items())),
        "counts_by_season_model_id": dict(sorted(by_model.items())),
        "corrections": rows,
    }
    return {**payload, "artifact_sha256": sha256_json(payload)}


def validate_completion_interpretation_correction(
    *, correction: Mapping[str, Any], arms_dir: Path
) -> ValidatedCompletionInterpretationCorrection:
    claimed = correction.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in correction.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise CompletionInterpretationCorrectionError(
            "completion-correction artifact hash mismatch"
        )
    if (
        correction.get("schema_version") != SCHEMA_VERSION
        or correction.get("record_type") != RECORD_TYPE
        or correction.get("scope") != SOURCE_SCOPE
        or correction.get("reason_code") != REASON_CODE
        or correction.get("policy")
        != {
            "accepted_finish_reasons": sorted(ACCEPTED_FINISH_REASONS),
            "rule": (
                "A rank-eligible final answer must end with a normal provider completion "
                "reason. Nonempty text does not override truncation or filtering."
            ),
        }
    ):
        raise CompletionInterpretationCorrectionError("unsupported completion-correction contract")

    expected = build_completion_interpretation_correction(arms_dir)
    if correction != expected:
        raise CompletionInterpretationCorrectionError(
            "completion correction does not cover the exact immutable candidate population"
        )
    rows = correction.get("corrections")
    if not isinstance(rows, list):
        raise CompletionInterpretationCorrectionError("corrections must be an array")
    return ValidatedCompletionInterpretationCorrection(
        artifact_sha256=actual,
        arm_ids=tuple(str(row["arm_id"]) for row in rows),
        source_arm_set_sha256=str(correction["source_arm_set_sha256"]),
        counts_by_finish_reason={
            str(key): int(value)
            for key, value in dict(correction["counts_by_finish_reason"]).items()
        },
    )


def apply_completion_interpretation(
    arms: Mapping[str, Mapping[str, Any]],
    correction: ValidatedCompletionInterpretationCorrection,
) -> dict[str, dict[str, Any]]:
    corrected = {arm_id: copy.deepcopy(dict(arm)) for arm_id, arm in arms.items()}
    for arm_id in correction.arm_ids:
        arm = corrected[arm_id]
        arm["source_status"] = arm.get("status")
        arm["source_rank_eligible"] = arm.get("rank_eligible")
        arm["status"] = "failed"
        arm["rank_eligible"] = False
        arm["error_type"] = "IncompleteFinalResponse"
        arm["completion_interpretation"] = "incomplete_final_response"
    return corrected


def _atomic_write(directory: Path, document: Mapping[str, Any]) -> Path:
    digest = str(document["artifact_sha256"])
    destination = directory / f"completion-interpretation-correction-{digest}.json"
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        .replace(b"</", b"<\\/")
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != data:
            raise CompletionInterpretationCorrectionError(
                "content-addressed completion correction conflicts with an existing file"
            )
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_completion_interpretation_correction(args.arms_dir)
    path = _atomic_write(args.output_dir, document)
    print(
        json.dumps(
            {
                "path": str(path),
                "artifact_sha256": document["artifact_sha256"],
                "source_arm_count": document["source_arm_count"],
                "counts_by_finish_reason": document["counts_by_finish_reason"],
                "external_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
