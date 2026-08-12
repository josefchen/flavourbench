"""Seal an interrupted powered-selection arm as a conservative ITT failure.

This utility performs no provider I/O. It is only for a cell whose durable
attempt journal proves that execution began but no response artifact was
published. Reissuing such a cell could double-submit an ambiguously delivered
request, so the scored panel instead retains the cell at zero.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .epicure_native_powered_runner import (
    RESPONSE_SCHEMA_VERSION,
    _reserve_micros,
    _sha256,
    _task_reference_payload,
    _write_content_addressed,
    build_generation_spec,
)
from .epicure_selection_powered_plan_v31 import selection_execution_policy_v31
from .epicure_selection_powered_runner import RUNNER_SCHEMA_VERSION, build_cells, validate_inputs
from .epicure_selection_taskset_v1 import score_answer


class InterruptionSealerError(RuntimeError):
    """The interrupted arm cannot be sealed without ambiguity."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_journal(path: Path, *, plan_sha256: str) -> dict[str, list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_file():
        raise InterruptionSealerError("attempt journal must be a regular file")
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        payload = dict(row)
        recorded = str(payload.pop("event_sha256", ""))
        if recorded != _sha256(payload) or payload.get("plan_sha256") != plan_sha256:
            raise InterruptionSealerError("attempt journal failed integrity validation")
        event = payload.get("event")
        if not isinstance(event, dict):
            raise InterruptionSealerError("attempt journal event is malformed")
        by_arm.setdefault(str(event.get("arm_id") or ""), []).append(row)
    return by_arm


def _event_projection(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    event_types = [str(row["event"]["event_type"]) for row in rows]
    if event_types == ["request_started"]:
        return event_types, {
            "cost_micros": 0,
            "cost_reconciled": False,
            "generation_ids": [],
            "generation_metadata": [],
            "billing_reconciliation_status": (
                "operator_interrupted_after_request_start_delivery_unknown"
            ),
        }
    if event_types == ["request_started", "response_received"]:
        response = rows[-1]["event"]
        metadata = response.get("metadata") or {}
        if response.get("http_status") != 200 or not response.get("generation_id"):
            raise InterruptionSealerError("received response is not an accepted generation")
        return event_types, {
            "actual_model_id": str(metadata.get("response_model") or ""),
            "actual_provider": str((metadata.get("response_envelope") or {}).get("provider") or ""),
            "generation_id": str(response["generation_id"]),
            "generation_ids": [str(response["generation_id"])],
            "cost_micros": 0,
            "cost_reconciled": False,
            "generation_metadata": [
                {
                    "finish_reason": metadata.get("finish_reason"),
                    "native_finish_reason": metadata.get("native_finish_reason"),
                    "response_payload_sha256": response.get("payload_sha256"),
                }
            ],
            "billing_reconciliation_status": (
                "accepted_response_interrupted_before_accounting_and_materialization"
            ),
        }
    raise InterruptionSealerError("arm has an unsupported attempt-event sequence")


def seal(
    *,
    arm_id: str,
    manifest_path: Path,
    manifest_sha256: str,
    taskset_path: Path,
    repeat_panel_path: Path,
    plan_path: Path,
    predecessor_release_path: Path,
    run_directory: Path,
    interruption_reason: str,
) -> Path:
    manifest, taskset, repeat, plan, predecessor, candidates = validate_inputs(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        taskset_path=taskset_path,
        repeat_panel_path=repeat_panel_path,
        plan_path=plan_path,
        predecessor_release_path=predecessor_release_path,
    )
    del manifest
    matching = [
        cell
        for cell in build_cells(
            plan=plan,
            taskset=taskset,
            repeat_panel=repeat,
            candidates=candidates,
            phase="all",
        )
        if cell.arm_id == arm_id
    ]
    if len(matching) != 1:
        raise InterruptionSealerError("arm does not identify exactly one frozen cell")
    cell = matching[0]
    response_directory = run_directory / "responses" / cell.panel / cell.candidate.slot_id
    if list(response_directory.glob(f"response-{cell.cell_id}-*.json")):
        raise InterruptionSealerError("interrupted cell already has a response artifact")
    journal = _load_journal(
        run_directory / "attempts" / "provider-attempts.jsonl",
        plan_sha256=str(plan["artifact_sha256"]),
    )
    rows = journal.get(arm_id) or []
    event_types, generation = _event_projection(rows)
    policy = selection_execution_policy_v31()
    reserve = _reserve_micros(
        cell.candidate,
        predecessor,
        max_output_tokens=policy.max_output_tokens,
    )
    spec, protocol_bundle = build_generation_spec(
        cell=cell,
        plan=plan,
        manifest_sha256=manifest_sha256,
        taskset=taskset,
        reserve_micros=reserve,
        execution_policy=policy,
    )
    recorded_at = _utc_now()
    artifact: dict[str, Any] = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "status": "failed",
        "recorded_at": recorded_at,
        "started_at": str(rows[0]["recorded_at"]),
        "wall_time_ms": 0,
        "plan_sha256": plan["artifact_sha256"],
        "manifest_sha256": manifest_sha256,
        "taskset_sha256": taskset["artifact_sha256"],
        "repeat_panel_sha256": repeat["artifact_sha256"],
        "cell_id": cell.cell_id,
        "panel": cell.panel,
        "arm_id": cell.arm_id,
        "slot_id": cell.candidate.slot_id,
        "model_id": cell.candidate.model_id,
        "model_name": cell.candidate.model_name,
        "canonical_model_slug": cell.candidate.canonical_model_slug,
        "execution_backend": cell.candidate.execution_backend,
        "provider_route": cell.candidate.provider_tag,
        "endpoint_execution_sha256": cell.candidate.endpoint_execution_sha256,
        "backend_contract_sha256": cell.candidate.backend_contract_sha256,
        "task_id": cell.task["task_id"],
        "original_task_id": cell.task.get("original_task_id"),
        "family": cell.task["family"],
        "prompt_sha256": cell.task["prompt_sha256"],
        **_task_reference_payload(cell.task),
        "protocol_bundle": protocol_bundle,
        "protocol_bundle_sha256": spec.protocol_bundle_sha256,
        "attempt_event_sha256s": [str(row["event_sha256"]) for row in rows],
        "generation": generation,
        "scoring": score_answer(cell.task, ""),
        "error": {
            "type": "OperatorInterruptedAmbiguousDelivery",
            "message": interruption_reason,
            "attempt_event_types": event_types,
        },
        "budget": {
            "reserved_micros": reserve,
            "actual_micros": 0,
            "global_cap_micros": int(float(plan["budget"]["hard_cap"]) * 1_000_000),
        },
    }
    return _write_content_addressed(
        artifact,
        directory=response_directory,
        filename_prefix=f"response-{cell.cell_id}",
    )


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-semantic-sha256", required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--repeat-panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predecessor-release", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    outputs = [
        seal(
            arm_id=arm_id,
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_semantic_sha256,
            taskset_path=args.taskset,
            repeat_panel_path=args.repeat_panel,
            plan_path=args.plan,
            predecessor_release_path=args.predecessor_release,
            run_directory=args.run_directory,
            interruption_reason=args.reason,
        )
        for arm_id in args.arm_id
    ]
    print(json.dumps({"sealed": [str(path) for path in outputs]}, sort_keys=True))


if __name__ == "__main__":
    run()
