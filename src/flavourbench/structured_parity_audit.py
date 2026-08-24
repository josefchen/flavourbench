"""Account for selected and failed Season 1 structured-parity qualification calls."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json


class StructuredParityAuditError(RuntimeError):
    """The structured-parity evidence could not be reconciled safely."""


def _load_verified(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StructuredParityAuditError(f"evidence must be a regular file: {path}")
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise StructuredParityAuditError(f"invalid JSON evidence: {path}") from error
    if not isinstance(document, dict):
        raise StructuredParityAuditError(f"evidence is not an object: {path}")
    recorded = str(document.get("artifact_sha256") or "")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if recorded != sha256_json(payload):
        raise StructuredParityAuditError(f"content address does not verify: {path}")
    return document


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def build_audit(
    panel_path: Path,
    failed_attempts: Sequence[tuple[Path, Decimal]],
) -> dict[str, Any]:
    panel = _load_verified(panel_path)
    models = panel.get("models")
    if not isinstance(models, list):
        raise StructuredParityAuditError("panel has no model records")
    selected = [
        model
        for model in models
        if isinstance(model, Mapping) and model.get("execution_backend") == "openrouter"
    ]
    if len(selected) != 8:
        raise StructuredParityAuditError("panel must contain eight OpenRouter routes")

    selected_records: list[dict[str, Any]] = []
    selected_cost = Decimal(0)
    for model in selected:
        source = Path(str(model.get("smoke_artifact_path") or ""))
        receipt = _load_verified(source)
        if (
            receipt.get("status") != "smoke_passed"
            or receipt.get("provider_structured_output_required") is not True
            or receipt.get("finish_reason") != "stop"
            or int(receipt.get("real_provider_calls") or 0) != 2
            or int(receipt.get("real_epicure_calls") or 0) != 1
            or receipt.get("generation_costs_reconciled") is not True
        ):
            raise StructuredParityAuditError(
                f"selected structured-parity receipt is incomplete: {source}"
            )
        cost = Decimal(str(receipt.get("cost_usd") or "0"))
        selected_cost += cost
        selected_records.append(
            {
                "requested_model_id": receipt.get("requested_model_id"),
                "canonical_model_slug": receipt.get("canonical_slug"),
                "provider": receipt.get("returned_provider_name"),
                "provider_calls": 2,
                "epicure_calls": 1,
                "finish_reason": "stop",
                "structured_output_required": True,
                "reconciled_cost_usd": _decimal_text(cost),
                "artifact_path": str(source),
                "artifact_sha256": receipt["artifact_sha256"],
            }
        )

    failure_records: list[dict[str, Any]] = []
    unknown_upper = Decimal(0)
    seen: set[str] = set()
    for source, reserve in failed_attempts:
        if reserve <= 0:
            raise StructuredParityAuditError("failed-attempt reserve must be positive")
        receipt = _load_verified(source)
        digest = str(receipt["artifact_sha256"])
        if digest in seen:
            raise StructuredParityAuditError("failed-attempt evidence is duplicated")
        seen.add(digest)
        if receipt.get("status") == "smoke_passed":
            raise StructuredParityAuditError("failed-attempt evidence unexpectedly passed")
        unknown_upper += reserve
        failure_records.append(
            {
                "requested_model_id": receipt.get("requested_model_id"),
                "error_type": receipt.get("error_type"),
                "error": receipt.get("error"),
                "reconciled_cost_usd": None,
                "conservative_unreconciled_reserve_usd": _decimal_text(reserve),
                "artifact_path": str(source),
                "artifact_sha256": digest,
                "interpretation": (
                    "The receipt proves a failed client-observed attempt but does not retain "
                    "enough provider identity metadata to establish billable generation count."
                ),
            }
        )

    upper = selected_cost + unknown_upper
    return {
        "schema_version": "flavourbench-season1-structured-parity-cost-audit-v1",
        "status": "qualified_selected_receipts_with_separate_failed_attempt_reserve",
        "panel_artifact_path": str(panel_path),
        "panel_artifact_sha256": panel["artifact_sha256"],
        "selected_contract_evidence": {
            "routes": len(selected_records),
            "provider_generations": sum(row["provider_calls"] for row in selected_records),
            "epicure_calls": sum(row["epicure_calls"] for row in selected_records),
            "all_normal_stop": True,
            "all_strict_structured_output": True,
            "reconciled_cost_usd": _decimal_text(selected_cost),
            "records": selected_records,
        },
        "failed_attempt_exposure": {
            "receipts": len(failure_records),
            "reconciled_cost_usd": None,
            "conservative_upper_bound_usd": _decimal_text(unknown_upper),
            "records": failure_records,
        },
        "total_qualification_exposure_interval_usd": {
            "lower": _decimal_text(selected_cost),
            "upper": _decimal_text(upper),
        },
        "claim_boundary": {
            "quality_observations": 0,
            "rank_eligible": False,
            "failed_attempts_excluded_from_contract_qualification": True,
            "failed_attempts_included_in_cost_exposure": True,
        },
    }


def _write(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"structured-parity-cost-audit-{digest}.json"
    if destination.exists():
        if destination.read_bytes() != data:
            raise StructuredParityAuditError("content-addressed audit conflict")
        return destination
    with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o600)
    return destination


def _failed_attempt(value: str) -> tuple[Path, Decimal]:
    path, separator, reserve = value.rpartition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("failed attempt must be PATH=RESERVE_USD")
    try:
        parsed = Decimal(reserve)
    except Exception as error:  # noqa: BLE001 - argparse conversion boundary
        raise argparse.ArgumentTypeError("invalid failed-attempt reserve") from error
    return Path(path), parsed


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--failed-attempt",
        action="append",
        type=_failed_attempt,
        default=[],
        help="Content-addressed failed receipt and conservative reserve as PATH=USD",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = build_audit(arguments.panel, arguments.failed_attempt)
    destination = _write(arguments.output_dir, payload)
    print(
        json.dumps(
            {
                "audit": str(destination),
                "artifact_sha256": destination.stem.rsplit("-", 1)[-1],
                "selected": payload["selected_contract_evidence"],
                "failed_attempt_exposure": payload["failed_attempt_exposure"],
                "inference_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
