from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from flavourbench.real_task_bank import sha256_json
from flavourbench.structured_parity_audit import (
    StructuredParityAuditError,
    build_audit,
)


def _write_addressed(path: Path, payload: dict[str, object]) -> Path:
    document = {**payload, "artifact_sha256": sha256_json(payload)}
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    models: list[dict[str, object]] = []
    for index in range(8):
        receipt = _write_addressed(
            tmp_path / f"pass-{index}.json",
            {
                "status": "smoke_passed",
                "requested_model_id": f"lab/model-{index}",
                "canonical_slug": f"lab/model-{index}-20260731",
                "returned_provider_name": "Lab",
                "provider_structured_output_required": True,
                "finish_reason": "stop",
                "real_provider_calls": 2,
                "real_epicure_calls": 1,
                "generation_costs_reconciled": True,
                "cost_usd": "0.01",
            },
        )
        models.append(
            {
                "execution_backend": "openrouter",
                "smoke_artifact_path": str(receipt),
            }
        )
    panel = _write_addressed(tmp_path / "panel.json", {"models": models})
    failed = _write_addressed(
        tmp_path / "failed.json",
        {
            "status": "failed",
            "requested_model_id": "lab/model-0",
            "error_type": "HTTPStatusError",
            "error": "429",
        },
    )
    return panel, failed


def test_audit_separates_selected_cost_from_failed_attempt_reserve(tmp_path: Path) -> None:
    panel, failed = _fixture(tmp_path)

    audit = build_audit(panel, [(failed, Decimal("2"))])

    assert audit["selected_contract_evidence"]["provider_generations"] == 16
    assert audit["selected_contract_evidence"]["epicure_calls"] == 8
    assert audit["selected_contract_evidence"]["reconciled_cost_usd"] == "0.08"
    assert audit["failed_attempt_exposure"]["conservative_upper_bound_usd"] == "2"
    assert audit["total_qualification_exposure_interval_usd"] == {
        "lower": "0.08",
        "upper": "2.08",
    }


def test_audit_rejects_a_selected_non_normal_completion(tmp_path: Path) -> None:
    panel, failed = _fixture(tmp_path)
    document = json.loads(panel.read_text(encoding="utf-8"))
    source = Path(document["models"][0]["smoke_artifact_path"])
    receipt = json.loads(source.read_text(encoding="utf-8"))
    receipt.pop("artifact_sha256")
    receipt["finish_reason"] = "length"
    _write_addressed(source, receipt)

    with pytest.raises(StructuredParityAuditError, match="incomplete"):
        build_audit(panel, [(failed, Decimal("2"))])
