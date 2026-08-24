from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.real_task_bank import sha256_json
from flavourbench.season0_completion_corrections import (
    CompletionInterpretationCorrectionError,
    apply_completion_interpretation,
    build_completion_interpretation_correction,
    validate_completion_interpretation_correction,
)


def _write_arm(directory: Path, arm_id: str, finish_reason: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "arm_id": arm_id,
        "status": "success",
        "delivery_state": "reconciled",
        "rank_eligible": True,
        "condition": "epicure_on",
        "synthetic": False,
        "model": {"provider": "openrouter", "season_model_id": "model-1"},
        "result": {
            "finish_reason": finish_reason,
            "answer_markdown": "A recorded answer.",
        },
    }
    document = {**payload, "artifact_sha256": sha256_json(payload)}
    (directory / f"{arm_id}.json").write_text(json.dumps(document))
    return document


def test_completion_correction_covers_every_non_normal_finish(tmp_path: Path) -> None:
    normal = _write_arm(tmp_path, "normal", "stop")
    truncated = _write_arm(tmp_path, "truncated", "length")
    filtered = _write_arm(tmp_path, "filtered", "content_filter")

    document = build_completion_interpretation_correction(tmp_path)
    validated = validate_completion_interpretation_correction(
        correction=document,
        arms_dir=tmp_path,
    )
    effective = apply_completion_interpretation(
        {"normal": normal, "truncated": truncated, "filtered": filtered},
        validated,
    )

    assert validated.arm_ids == ("filtered", "truncated")
    assert effective["normal"]["status"] == "success"
    assert effective["truncated"]["status"] == "failed"
    assert effective["truncated"]["rank_eligible"] is False
    assert effective["filtered"]["error_type"] == "IncompleteFinalResponse"


def test_completion_correction_rejects_partial_population(tmp_path: Path) -> None:
    _write_arm(tmp_path, "truncated", "max_tokens")
    document = build_completion_interpretation_correction(tmp_path)
    document["corrections"] = []
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    document["artifact_sha256"] = sha256_json(payload)

    with pytest.raises(
        CompletionInterpretationCorrectionError,
        match="exact immutable candidate population",
    ):
        validate_completion_interpretation_correction(
            correction=document,
            arms_dir=tmp_path,
        )
