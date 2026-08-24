import hashlib
import json
from pathlib import Path

import pytest

from flavourbench.real_task_bank import sha256_json
from flavourbench.season0_arm_corrections import (
    ArmInterpretationCorrectionError,
    validate_arm_interpretation_correction,
)


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    arms_dir = tmp_path / "arms"
    arms_dir.mkdir()
    arm_payload = {
        "arm_id": "arm-one",
        "status": "failed",
        "error_type": "ReadTimeoutError",
        "delivery_state": "safe_pre_inference",
        "result": None,
        "rank_eligible": False,
        "reservation_usd": "0.033968880",
        "synthetic": False,
        "model": {
            "provider": "bedrock",
            "season_model_id": "fb-s0-model-10",
        },
    }
    arm = {**arm_payload, "artifact_sha256": sha256_json(arm_payload)}
    arm_path = arms_dir / "arm.json"
    arm_path.write_text(json.dumps(arm, sort_keys=True), encoding="utf-8")
    literal_sha = hashlib.sha256(arm_path.read_bytes()).hexdigest()
    correction_payload = {
        "schema_version": "flavourbench-season0-arm-interpretation-correction-v1",
        "record_type": "superseding_delivery_state_interpretation",
        "recorded_at_utc": "2026-07-26T17:00:00Z",
        "scope": "season0-scored-v1",
        "reason_code": "read_timeout_has_ambiguous_delivery",
        "source_arm_count": 1,
        "source_arm_set_sha256": sha256_json(
            {"artifact_sha256s": [arm["artifact_sha256"]]}
        ),
        "conservative_reservation_usd": "0.033968880",
        "corrections": [
            {
                "arm_id": "arm-one",
                "source_arm_artifact_sha256": arm["artifact_sha256"],
                "source_arm_file_sha256": literal_sha,
                "source_assertions": {
                    "provider": "bedrock",
                    "season_model_id": "fb-s0-model-10",
                    "status": "failed",
                    "error_type": "ReadTimeoutError",
                    "delivery_state": "safe_pre_inference",
                    "result": None,
                    "rank_eligible": False,
                    "reservation_usd": "0.033968880",
                    "synthetic": False,
                },
                "field_correction": {
                    "json_pointer": "/delivery_state",
                    "from": "safe_pre_inference",
                    "to": "uncertain",
                },
                "derived_interpretation": {
                    "failure_class": "uncertain_delivery",
                    "cost_disposition": "retain_conservative_reservation",
                    "replay_authorized": False,
                    "rank_eligible": False,
                },
            }
        ],
        "supersedes_correction_artifact_sha256": None,
    }
    return arms_dir, {
        **correction_payload,
        "artifact_sha256": sha256_json(correction_payload),
    }


def test_delivery_correction_is_bound_without_mutating_raw_arm(tmp_path: Path) -> None:
    arms_dir, correction = _fixture(tmp_path)
    before = (arms_dir / "arm.json").read_bytes()
    validated = validate_arm_interpretation_correction(
        correction=correction,
        arms_dir=arms_dir,
    )
    assert validated is not None
    assert validated.arm_ids == ("arm-one",)
    assert str(validated.conservative_reservation_usd) == "0.033968880"
    assert (arms_dir / "arm.json").read_bytes() == before


def test_delivery_correction_rejects_an_unbound_field_change(tmp_path: Path) -> None:
    arms_dir, correction = _fixture(tmp_path)
    correction["corrections"][0]["field_correction"]["json_pointer"] = "/status"  # type: ignore[index]
    body = {key: value for key, value in correction.items() if key != "artifact_sha256"}
    correction["artifact_sha256"] = sha256_json(body)
    with pytest.raises(
        ArmInterpretationCorrectionError,
        match="unsupported field",
    ):
        validate_arm_interpretation_correction(
            correction=correction,
            arms_dir=arms_dir,
        )


def test_delivery_correction_is_required_for_ambiguous_timeout(tmp_path: Path) -> None:
    arms_dir, _ = _fixture(tmp_path)
    with pytest.raises(
        ArmInterpretationCorrectionError,
        match="require an interpretation correction",
    ):
        validate_arm_interpretation_correction(
            correction=None,
            arms_dir=arms_dir,
        )
