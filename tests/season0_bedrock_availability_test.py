from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.real_task_bank import sha256_json
from flavourbench.season0_bedrock_availability import (
    BedrockAvailabilityError,
    _atomic_write,
    collect_availability,
)


class FakeControl:
    def get_foundation_model_availability(self, *, modelId: str):
        return {
            "modelId": modelId,
            "agreementAvailability": {"status": "AVAILABLE"},
            "authorizationStatus": "AUTHORIZED",
            "entitlementAvailability": "AVAILABLE",
            "regionAvailability": "AVAILABLE",
            "ResponseMetadata": {"RequestId": "must-not-be-retained"},
        }


def test_availability_artifact_retains_only_safe_contract_fields(tmp_path: Path) -> None:
    payload = collect_availability(
        control=FakeControl(),
        model_ids=["model.a", "model.b"],
        region="eu-west-1",
        observed_at="2026-07-16T16:00:00+00:00",
    )
    path = _atomic_write(tmp_path, payload)
    saved = json.loads(path.read_bytes())

    assert saved["artifact_sha256"] == sha256_json(payload)
    assert saved["inference_calls"] == 0
    assert [row["model_id"] for row in saved["models"]] == ["model.a", "model.b"]
    assert saved["models"][0]["authorization_status"] == "AUTHORIZED"
    assert "RequestId" not in path.read_text()
    assert saved["privacy"]["contains_credentials"] is False


def test_availability_rejects_duplicate_or_incomplete_models() -> None:
    with pytest.raises(BedrockAvailabilityError, match="unique"):
        collect_availability(
            control=FakeControl(),
            model_ids=["model.a", "model.a"],
            region="eu-west-1",
            observed_at="2026-07-16T16:00:00+00:00",
        )

    class IncompleteControl:
        def get_foundation_model_availability(self, *, modelId: str):
            return {"modelId": modelId, "agreementAvailability": {}}

    with pytest.raises(BedrockAvailabilityError, match="incomplete"):
        collect_availability(
            control=IncompleteControl(),
            model_ids=["model.a"],
            region="eu-west-1",
            observed_at="2026-07-16T16:00:00+00:00",
        )
