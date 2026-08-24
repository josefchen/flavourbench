"""Audit the paid, real-provider Season 0 calibration collection."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json
from .season0_collection import ARM_SCHEMA, CONDITIONS, FAMILIES

SCHEMA_VERSION = "flavourbench-season0-calibration-audit-v2"


class CalibrationAuditError(RuntimeError):
    """Calibration records are incomplete, mixed, or not reproducible."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise CalibrationAuditError(f"expected a JSON object: {path}")
    return value


def _artifact_sha(document: Mapping[str, Any], *, label: str) -> str:
    expected = document.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if expected != actual:
        raise CalibrationAuditError(f"{label} artifact hash mismatch")
    return actual


def _atomic_write(directory: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"calibration-audit-{digest}.json"
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(destination)
    return destination


def _quantiles(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, (95 * len(ordered) + 99) // 100 - 1))
    return {
        "min": ordered[0],
        "median": round(statistics.median(ordered)),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _latest_arms(directory: Path) -> tuple[dict[str, dict[str, Any]], int]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for path in directory.glob("arm-*.json"):
        value = _load(path)
        arm_id = value.get("arm_id")
        if isinstance(arm_id, str):
            by_id.setdefault(arm_id, []).append(value)
    latest = {
        arm_id: sorted(rows, key=lambda row: str(row.get("completed_at") or ""))[-1]
        for arm_id, rows in by_id.items()
    }
    return latest, sum(max(0, len(rows) - 1) for rows in by_id.values())


def _coordinate(record: Mapping[str, Any]) -> tuple[str, str, str]:
    task = record.get("task")
    model = record.get("model")
    return (
        str(task.get("task_id") if isinstance(task, Mapping) else ""),
        str(model.get("season_model_id") if isinstance(model, Mapping) else ""),
        str(record.get("condition") or ""),
    )


def _expected_coordinates(
    task_bank: Mapping[str, Any], model_manifest: Mapping[str, Any]
) -> set[tuple[str, str, str]]:
    tasks = task_bank.get("tasks")
    models = model_manifest.get("models")
    if not isinstance(tasks, list) or not isinstance(models, list):
        raise CalibrationAuditError("task bank or model manifest is malformed")
    selected: list[Mapping[str, Any]] = []
    for family in FAMILIES:
        family_tasks = [
            task for task in tasks if isinstance(task, Mapping) and task.get("family") == family
        ]
        if not family_tasks:
            raise CalibrationAuditError(f"task bank has no tasks for {family}")
        selected.append(family_tasks[len(family_tasks) // 2])
    return {
        (str(task["task_id"]), str(model["season_model_id"]), condition)
        for task in selected
        for model in models
        if isinstance(model, Mapping)
        for condition in CONDITIONS
    }


def _arm_identity_scheme(record: Mapping[str, Any]) -> str | None:
    task = record.get("task")
    model = record.get("model")
    contracts = record.get("contracts")
    if not all(isinstance(value, Mapping) for value in (task, model, contracts)):
        return None
    base = {
        "schema_version": record.get("schema_version"),
        "phase": record.get("phase"),
        "task_set_sha256": contracts.get("task_set_sha256"),
        "model_set_sha256": contracts.get("model_set_sha256"),
        "task_id": task.get("task_id"),
        "task_sha256": task.get("task_sha256"),
        "season_model_id": model.get("season_model_id"),
        "canonical_model_id": model.get("canonical_model_id"),
        "condition": record.get("condition"),
        "seed": 20260716,
    }
    arm_id = record.get("arm_id")
    bound = {
        **base,
        "execution_contract_sha256": contracts.get("execution_contract_sha256"),
        "epicure_intervention_artifact_sha256": contracts.get(
            "epicure_intervention_artifact_sha256"
        ),
        "system_prompt_sha256": contracts.get("system_prompt_sha256"),
    }
    if sha256_json(bound) == arm_id:
        return "contract_bound_v2"
    if sha256_json(base) == arm_id:
        return "legacy_coordinate_v1"
    return None


def audit_calibration(
    *,
    task_bank: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    arms_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    task_bank_sha = _artifact_sha(task_bank, label="task bank")
    model_manifest_sha = _artifact_sha(model_manifest, label="model manifest")
    expected = _expected_coordinates(task_bank, model_manifest)
    records, superseded_records = _latest_arms(arms_dir)
    coordinates = [_coordinate(record) for record in records.values()]
    coordinate_counts = Counter(coordinates)
    unexpected = sorted(set(coordinates) - expected)
    missing = sorted(expected - set(coordinates))
    duplicates = sorted(coordinate for coordinate, count in coordinate_counts.items() if count > 1)
    if unexpected:
        raise CalibrationAuditError("calibration directory contains unexpected coordinates")
    if duplicates:
        raise CalibrationAuditError("calibration directory contains duplicate coordinates")

    verified: list[dict[str, Any]] = []
    integrity_errors: list[dict[str, str]] = []
    identity_schemes: Counter[str] = Counter()
    for arm_id, record in records.items():
        errors: list[str] = []
        if record.get("schema_version") != ARM_SCHEMA:
            errors.append("schema")
        if record.get("phase") != "calibration":
            errors.append("phase")
        if record.get("synthetic") is not False:
            errors.append("synthetic")
        contracts = record.get("contracts")
        if not isinstance(contracts, Mapping):
            errors.append("contracts")
        else:
            if contracts.get("task_bank_artifact_sha256") != task_bank_sha:
                errors.append("task_bank_hash")
            if contracts.get("model_manifest_artifact_sha256") != model_manifest_sha:
                errors.append("model_manifest_hash")
        try:
            _artifact_sha(record, label=f"arm {arm_id}")
        except CalibrationAuditError:
            errors.append("artifact_hash")
        identity_scheme = _arm_identity_scheme(record)
        if identity_scheme is None:
            errors.append("arm_identity")
        if errors:
            integrity_errors.append({"arm_id": arm_id, "errors": ",".join(errors)})
        else:
            verified.append(record)
            identity_schemes[identity_scheme] += 1

    by_model: dict[str, dict[str, Any]] = {}
    for model in model_manifest["models"]:
        model_id = str(model["season_model_id"])
        rows = [row for row in verified if row["model"]["season_model_id"] == model_id]
        successes = [row for row in rows if row.get("status") == "success"]
        failures = [row for row in rows if row.get("status") != "success"]
        length_finishes = [
            row
            for row in successes
            if str((row.get("result") or {}).get("finish_reason") or "").lower()
            in {"length", "max_tokens", "max_tokens_reached"}
        ]
        reconciled_rows = [
            row
            for row in rows
            if row.get("delivery_state") == "reconciled"
            and isinstance(row.get("result"), Mapping)
            and isinstance(row["result"].get("usage"), Mapping)
        ]
        infrastructure_failures = [row for row in rows if row not in reconciled_rows]
        openrouter_costs = [
            Decimal(str(row["result"]["actual_cost_usd"]))
            for row in reconciled_rows
            if model["provider"] == "openrouter"
            and (row.get("result") or {}).get("actual_cost_usd") is not None
        ]
        if model["provider"] == "openrouter":
            reservation = max(
                Decimal("0.10"),
                Decimal("2") * max(openrouter_costs, default=Decimal("0")),
            )
            cost_reconciled = len(openrouter_costs) == len(reconciled_rows)
        else:
            reservation = Decimal("1")
            cost_reconciled = True
        output_tokens = [
            int(row["result"]["usage"].get("output_tokens") or 0) for row in reconciled_rows
        ]
        latencies = [
            int(row["result"].get("wall_clock_latency_ms") or 0) for row in reconciled_rows
        ]
        infrastructure_ready = len(rows) == 8 and not infrastructure_failures and cost_reconciled
        by_model[model_id] = {
            "display_name": model["display_name"],
            "provider": model["provider"],
            "planned_arms": 8,
            "terminal_arms": len(rows),
            "success": len(successes),
            "failed": len(failures),
            "reconciled_terminal_arms": len(reconciled_rows),
            "infrastructure_failure_count": len(infrastructure_failures),
            "success_by_condition": dict(Counter(row["condition"] for row in successes)),
            "failure_reasons": dict(
                Counter(str(row.get("error") or "unspecified") for row in failures)
            ),
            "length_finish_count": len(length_finishes),
            "provider_calls": sum(
                int(row["result"].get("provider_calls") or 0) for row in reconciled_rows
            ),
            "real_epicure_calls": sum(
                int(row["result"].get("real_epicure_calls") or 0) for row in reconciled_rows
            ),
            "output_tokens": _quantiles(output_tokens),
            "wall_clock_latency_ms": _quantiles(latencies),
            "openrouter_actual_cost_usd": format(sum(openrouter_costs, Decimal(0)), "f"),
            "max_actual_arm_cost_usd": (
                format(max(openrouter_costs), "f") if openrouter_costs else None
            ),
            "scored_arm_reservation_usd": format(reservation, "f"),
            "cost_reconciled": cost_reconciled,
            "model_behavior_passed_all_arms": (len(successes) == 8 and not length_finishes),
            "infrastructure_ready_for_scored_collection": infrastructure_ready,
        }

    success = sum(row.get("status") == "success" for row in verified)
    failed = len(verified) - success
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "FlavourBench",
        "season": "Season 0",
        "run_class": "paid_real_calibration_excluded_from_scoring",
        "synthetic_arms": 0,
        "task_bank_artifact_sha256": task_bank_sha,
        "model_manifest_artifact_sha256": model_manifest_sha,
        "counts": {
            "planned_arms": len(expected),
            "terminal_arms": len(records),
            "verified_arms": len(verified),
            "missing_arms": len(missing),
            "superseded_records": superseded_records,
            "integrity_errors": len(integrity_errors),
            "arm_identity_schemes": dict(identity_schemes),
            "success": success,
            "failed": failed,
            "real_provider_calls": sum(
                int((row.get("result") or {}).get("provider_calls") or 0)
                for row in verified
                if row.get("delivery_state") == "reconciled"
            ),
            "real_epicure_calls": sum(
                int((row.get("result") or {}).get("real_epicure_calls") or 0)
                for row in verified
                if row.get("delivery_state") == "reconciled"
            ),
        },
        "failure_reasons": dict(
            Counter(
                str(row.get("error") or "unspecified")
                for row in verified
                if row.get("status") != "success"
            )
        ),
        "missing_coordinates": [
            {"task_id": task_id, "season_model_id": model_id, "condition": condition}
            for task_id, model_id, condition in missing
        ],
        "integrity_errors": integrity_errors,
        "models": by_model,
        "ready_for_scored_collection": (
            not missing
            and not integrity_errors
            and all(
                value["infrastructure_ready_for_scored_collection"] for value in by_model.values()
            )
        ),
    }
    path = _atomic_write(output_dir, payload)
    return {**payload, "summary_path": str(path)}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_calibration(
        task_bank=_load(args.task_bank),
        model_manifest=_load(args.model_manifest),
        arms_dir=args.arms_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
