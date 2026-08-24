"""Recover only documented pre-inference Bedrock throttles in Season 0 judging.

The primary collector intentionally never replays a request with ambiguous delivery. This
module is narrower: AWS documents ``ThrottlingException`` as a denied request and explicitly
instructs clients to resubmit later. A content-addressed plan freezes the eligible rejection
registry before preference outcomes are analyzed. Each judgment identity receives at most one
recovery attempt, and every superseded record remains immutable on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients
from .bedrock_provider import structured_output_config
from .real_task_bank import sha256_json, sha256_text
from .season0_judge_manifest import MAX_OUTPUT_TOKENS
from .season0_judge_protocol import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT_SHA256,
    JUDGMENT_SCHEMA,
    JUDGMENT_SCHEMA_SHA256,
    PROTOCOL_VERSION,
    normalize_choice,
    validate_judgment,
)
from .season0_judging import (
    SCHEMA_VERSION as JUDGMENT_SCHEMA_VERSION,
)
from .season0_judging import (
    JudgmentBudget,
    JudgmentWorkItem,
    _arms_by_id,
    _atomic_write,
    _bedrock_transport_config,
    _cost,
    _latest_by_id,
    _load,
    _prompt_for,
    _reservation,
    _response_text,
    _usage,
    _verify_artifact,
    build_work_items,
)

PLAN_SCHEMA_VERSION = "flavourbench-season0-judge-throttle-recovery-plan-v1"
FINAL_SUMMARY_SCHEMA_VERSION = "flavourbench-season0-automated-judgment-final-summary-v1"
RECOVERY_PROTOCOL_VERSION = "flavourbench-season0-bedrock-throttle-recovery-v1"
CONFIRMATION = "RUN_REAL_SEASON0_DOCUMENTED_THROTTLE_RECOVERY_V1"
RECOVERABLE_ERROR_TYPE = "ThrottlingException"
AWS_ERROR_SEMANTICS_URL = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html"
)


class JudgmentRecoveryError(RuntimeError):
    """A recovery plan or execution violates the frozen retry boundary."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _all_records(directory: Path) -> list[dict[str, Any]]:
    return [_load(path) for path in sorted(directory.glob("judgment-*.json"))]


def _is_documented_throttle_rejection(record: Mapping[str, Any]) -> bool:
    result = record.get("result")
    return (
        record.get("status") == "failed"
        and record.get("error_type") == RECOVERABLE_ERROR_TYPE
        and isinstance(result, Mapping)
        and result.get("estimated_cost_usd") is None
        and result.get("usage") is None
        and result.get("raw_output_sha256") is None
        and result.get("request_id_sha256") is None
    )


def _plan_payload(
    *,
    original_summary: Mapping[str, Any],
    latest: Mapping[str, Mapping[str, Any]],
    original_summary_sha256: str,
) -> dict[str, Any]:
    if (
        original_summary.get("status") != "collection_complete"
        or original_summary.get("synthetic_judgments") != 0
    ):
        raise JudgmentRecoveryError("recovery requires the complete real first-pass summary")
    counts = original_summary.get("counts")
    if not isinstance(counts, Mapping):
        raise JudgmentRecoveryError("first-pass judgment counts are missing")
    planned = int(counts.get("planned_judgments") or 0)
    terminal = int(counts.get("terminal_judgments") or 0)
    if planned <= 0 or terminal != planned or len(latest) != terminal:
        raise JudgmentRecoveryError("first-pass judgment registry is incomplete")
    summary_hashes = original_summary.get("judgment_artifact_sha256s")
    latest_hashes = sorted(str(row.get("artifact_sha256") or "") for row in latest.values())
    if not isinstance(summary_hashes, list) or sorted(summary_hashes) != latest_hashes:
        raise JudgmentRecoveryError("first-pass summary does not bind the latest judgment registry")

    items = []
    by_judge: Counter[str] = Counter()
    for judgment_id, record in sorted(latest.items()):
        if not _is_documented_throttle_rejection(record):
            continue
        if record.get("recovery") is not None or record.get("attempt_number") is not None:
            raise JudgmentRecoveryError(
                "a recovery plan cannot include an earlier recovery attempt"
            )
        judge = record.get("judge")
        judge_id = str(judge.get("judge_id") or "") if isinstance(judge, Mapping) else ""
        if not judge_id:
            raise JudgmentRecoveryError("throttled judgment has no frozen judge identity")
        by_judge[judge_id] += 1
        items.append(
            {
                "judgment_id": judgment_id,
                "superseded_artifact_sha256": record["artifact_sha256"],
                "comparison_id": record["comparison_id"],
                "judge_id": judge_id,
                "orientation": record["orientation"],
                "reservation_usd": record["reservation_usd"],
            }
        )

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "season": "Season 0",
        "status": "frozen_before_preference_outcome_analysis",
        "synthetic_judgments": 0,
        "preference_outcomes_inspected": False,
        "recovery_protocol_version": RECOVERY_PROTOCOL_VERSION,
        "eligible_error_type": RECOVERABLE_ERROR_TYPE,
        "eligibility": (
            "failed Bedrock Converse call with ThrottlingException, no returned usage, "
            "no response body hash, and no request identifier"
        ),
        "excluded_error_types": [
            "ReadTimeoutError",
            "OrphanedRequestEvent",
            "ServiceUnavailableException",
            "InternalServerException",
            "JSONDecodeError",
            "JudgmentProtocolError",
        ],
        "maximum_total_provider_attempts_per_judgment": 2,
        "recovery_global_concurrency": 4,
        "recovery_per_judge_concurrency": 2,
        "minimum_start_interval_seconds_per_judge": 1.0,
        "adaptive_throttle_backoff_seconds": [30, 60, 120, 240, 300],
        "aws_error_semantics": {
            "url": AWS_ERROR_SEMANTICS_URL,
            "interpretation": (
                "AWS defines ThrottlingException as a denied request and directs the client "
                "to retry later"
            ),
        },
        "original_collection_summary_artifact_sha256": original_summary_sha256,
        "task_bank_artifact_sha256": original_summary["task_bank_artifact_sha256"],
        "comparison_manifest_artifact_sha256": original_summary[
            "comparison_manifest_artifact_sha256"
        ],
        "judge_manifest_artifact_sha256": original_summary["judge_manifest_artifact_sha256"],
        "judge_cost_envelope_artifact_sha256": original_summary[
            "judge_cost_envelope_artifact_sha256"
        ],
        "counts": {
            "first_pass_judgment_identities": terminal,
            "eligible_documented_throttle_rejections": len(items),
            "by_judge": dict(sorted(by_judge.items())),
        },
        "recovery_items": items,
        "implementation_source_sha256": _source_sha256(),
    }


def freeze_recovery_plan(
    *, original_summary: Mapping[str, Any], judgments_dir: Path, output_dir: Path
) -> dict[str, Any]:
    original_summary_sha = _verify_artifact(original_summary, "first-pass judgment summary")
    latest = _latest_by_id(judgments_dir, "judgment", "judgment_id")
    payload = _plan_payload(
        original_summary=original_summary,
        latest=latest,
        original_summary_sha256=original_summary_sha,
    )
    path = _atomic_write(output_dir, "judge-throttle-recovery-plan", payload)
    return {**payload, "artifact_sha256": sha256_json(payload), "plan_path": str(path)}


def _verify_bindings(
    *,
    task_bank: Mapping[str, Any],
    comparison_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any],
    cost_envelope: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[str, str, str, str, str]:
    task_sha = _verify_artifact(task_bank, "task bank")
    comparison_sha = _verify_artifact(comparison_manifest, "comparison manifest")
    judge_sha = _verify_artifact(judge_manifest, "judge manifest")
    cost_sha = _verify_artifact(cost_envelope, "judge cost envelope")
    plan_sha = _verify_artifact(plan, "judge throttle recovery plan")
    expected = {
        "task_bank_artifact_sha256": task_sha,
        "comparison_manifest_artifact_sha256": comparison_sha,
        "judge_manifest_artifact_sha256": judge_sha,
        "judge_cost_envelope_artifact_sha256": cost_sha,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise JudgmentRecoveryError("recovery plan is bound to another frozen experiment")
    if (
        task_bank.get("synthetic_tasks") != 0
        or comparison_manifest.get("synthetic_comparisons") != 0
        or plan.get("synthetic_judgments") != 0
        or plan.get("preference_outcomes_inspected") is not False
        or plan.get("eligible_error_type") != RECOVERABLE_ERROR_TYPE
        or plan.get("maximum_total_provider_attempts_per_judgment") != 2
    ):
        raise JudgmentRecoveryError("recovery plan violates the zero-synthetic retry boundary")
    protocol = judge_manifest.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("version") != PROTOCOL_VERSION
        or protocol.get("system_prompt_sha256") != JUDGE_SYSTEM_PROMPT_SHA256
        or protocol.get("judgment_schema_sha256") != JUDGMENT_SCHEMA_SHA256
        or protocol.get("max_output_tokens") != MAX_OUTPUT_TOKENS
    ):
        raise JudgmentRecoveryError("judge manifest no longer matches the frozen protocol")
    return task_sha, comparison_sha, judge_sha, cost_sha, plan_sha


def _attempt_exposure(records: Sequence[Mapping[str, Any]]) -> tuple[Decimal, Decimal, int]:
    attributed = Decimal(0)
    reservations = Decimal(0)
    documented_rejections = 0
    for record in records:
        result = record.get("result")
        actual = result.get("estimated_cost_usd") if isinstance(result, Mapping) else None
        if actual is not None:
            attributed += Decimal(str(actual))
        elif _is_documented_throttle_rejection(record):
            documented_rejections += 1
        else:
            reservations += Decimal(str(record.get("reservation_usd") or 0))
    return attributed, reservations, documented_rejections


def _orphaned_recovery_record(
    *,
    item: JudgmentWorkItem,
    prior: Mapping[str, Any],
    event: Mapping[str, Any],
    contracts: Mapping[str, Any],
    plan_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": JUDGMENT_SCHEMA_VERSION,
        "judgment_id": item.judgment_id,
        "comparison_id": item.comparison["comparison_id"],
        "track": item.comparison["track"],
        "task_id": item.comparison["task_id"],
        "task_family": item.comparison["task_family"],
        "judge": dict(item.judge),
        "orientation": item.orientation,
        "status": "failed",
        "delivery_state": "uncertain",
        "error_type": "OrphanedRecoveryRequestEvent",
        "error": "recovery request-start event had no terminal response and was not replayed",
        "reservation_usd": event["reservation_usd"],
        "started_at": event["started_at"],
        "completed_at": _utc_now(),
        "contracts": dict(contracts),
        "result": {
            "request_payload_sha256": event["request_payload_sha256"],
            "prompt_sha256": event["prompt_sha256"],
            "estimated_cost_usd": None,
            "latency_ms": None,
        },
        "attempt_number": 2,
        "supersedes_artifact_sha256": prior["artifact_sha256"],
        "recovery": {
            "protocol_version": RECOVERY_PROTOCOL_VERSION,
            "plan_artifact_sha256": plan_sha,
            "original_error_type": RECOVERABLE_ERROR_TYPE,
        },
        "synthetic": False,
    }


def _final_summary(
    *,
    output_dir: Path,
    work_items: Sequence[JudgmentWorkItem],
    plan: Mapping[str, Any],
    plan_sha: str,
    contracts: Mapping[str, Any],
    run_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    judgment_dir = output_dir / "judgments"
    records = _all_records(judgment_dir)
    latest = _latest_by_id(judgment_dir, "judgment", "judgment_id")
    expected_ids = {item.judgment_id for item in work_items}
    if set(latest) != expected_ids:
        raise JudgmentRecoveryError("final judgment registry is incomplete or mixed")
    success = [row for row in latest.values() if row.get("status") == "success"]
    failed = [row for row in latest.values() if row.get("status") != "success"]
    attributed, reservations, documented_rejections = _attempt_exposure(records)
    recovery_records = [
        row
        for row in records
        if isinstance(row.get("recovery"), Mapping)
        and row["recovery"].get("plan_artifact_sha256") == plan_sha
    ]
    recovered_success = sum(row.get("status") == "success" for row in recovery_records)
    payload: dict[str, Any] = {
        "schema_version": FINAL_SUMMARY_SCHEMA_VERSION,
        "season": "Season 0",
        "status": "collection_complete",
        "synthetic_judgments": 0,
        **dict(contracts),
        "original_collection_summary_artifact_sha256": plan[
            "original_collection_summary_artifact_sha256"
        ],
        "recovery_plan_artifact_sha256": plan_sha,
        "counts": {
            "planned_judgments": len(work_items),
            "terminal_judgments": len(latest),
            "provider_attempt_records": len(records),
            "success": len(success),
            "failed": len(failed),
            "first_pass_documented_throttle_rejections": len(plan["recovery_items"]),
            "planned_recovery_attempts": len(plan["recovery_items"]),
            "recovery_attempts": len(recovery_records),
            "recovered_to_success": recovered_success,
            "recovery_failures": len(recovery_records) - recovered_success,
            "documented_zero_delivery_throttle_rejections": documented_rejections,
            "by_judge": dict(Counter(row["judge"]["judge_id"] for row in success)),
            "by_orientation": dict(Counter(row["orientation"] for row in success)),
            "by_track": dict(Counter(row["track"] for row in success)),
        },
        "estimated_cost_usd": format(attributed + reservations, ".9f"),
        "attributed_estimated_cost_usd": format(attributed, ".9f"),
        "unattributed_conservative_reservations_usd": format(reservations, ".9f"),
        "cost_status": "conservative_exposure_pending_aws_cur_crosscheck",
        "run_configuration": dict(run_configuration),
        "failure_reasons": dict(
            Counter(str(row.get("error_type") or "unspecified") for row in failed)
        ),
        "aws_error_semantics": plan["aws_error_semantics"],
        "judgment_artifact_sha256s": sorted(str(row["artifact_sha256"]) for row in latest.values()),
        "all_attempt_artifact_sha256s": sorted(str(row["artifact_sha256"]) for row in records),
        "implementation_source_sha256": _source_sha256(),
    }
    path = _atomic_write(output_dir, "judgment-final-summary", payload)
    return {**payload, "artifact_sha256": sha256_json(payload), "summary_path": str(path)}


async def recover_throttles(
    *,
    task_bank: Mapping[str, Any],
    arms_dir: Path,
    comparison_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any],
    cost_envelope: Mapping[str, Any],
    plan: Mapping[str, Any],
    output_dir: Path,
    cap_usd: Decimal,
    confirmation: str,
) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise JudgmentRecoveryError("real throttle-recovery confirmation is missing")
    task_sha, comparison_sha, judge_sha, cost_sha, plan_sha = _verify_bindings(
        task_bank=task_bank,
        comparison_manifest=comparison_manifest,
        judge_manifest=judge_manifest,
        cost_envelope=cost_envelope,
        plan=plan,
    )
    if plan.get("implementation_source_sha256") != _source_sha256():
        raise JudgmentRecoveryError("recovery implementation changed after the plan was frozen")
    settings = BedrockLaneSettings.from_environ()
    manifest_cap = Decimal(str(judge_manifest.get("hard_cap_usd") or 0))
    envelope_cap = Decimal(str(cost_envelope.get("hard_cap_usd") or 0))
    if (
        not settings.enabled
        or not settings.live_authorized
        or settings.stage != "season"
        or cap_usd <= 0
        or cap_usd > settings.hard_cap_usd
        or cap_usd > manifest_cap
        or cap_usd > envelope_cap
    ):
        raise JudgmentRecoveryError("Bedrock recovery exceeds the authorized season cap")

    tasks = {
        str(task["task_id"]): task
        for task in task_bank.get("tasks", [])
        if isinstance(task, Mapping)
    }
    arms = _arms_by_id(arms_dir)
    work_items = build_work_items(comparison_manifest, judge_manifest)
    by_id = {item.judgment_id: item for item in work_items}
    judgment_dir = output_dir / "judgments"
    recovery_event_dir = output_dir / "recovery-events"
    latest = _latest_by_id(judgment_dir, "judgment", "judgment_id")
    recovery_events = _latest_by_id(recovery_event_dir, "recovery-event", "judgment_id")
    contracts = {
        "task_bank_artifact_sha256": task_sha,
        "comparison_manifest_artifact_sha256": comparison_sha,
        "judge_manifest_artifact_sha256": judge_sha,
        "judge_cost_envelope_artifact_sha256": cost_sha,
        "protocol_version": PROTOCOL_VERSION,
        "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
        "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
    }
    pending: list[tuple[JudgmentWorkItem, Mapping[str, Any]]] = []
    for row in plan.get("recovery_items", []):
        if not isinstance(row, Mapping):
            raise JudgmentRecoveryError("recovery plan contains a malformed item")
        judgment_id = str(row.get("judgment_id") or "")
        item = by_id.get(judgment_id)
        prior = latest.get(judgment_id)
        if item is None or prior is None:
            raise JudgmentRecoveryError("recovery plan references an unknown judgment")
        recovery = prior.get("recovery")
        if isinstance(recovery, Mapping) and recovery.get("plan_artifact_sha256") == plan_sha:
            continue
        if prior.get("artifact_sha256") != row.get("superseded_artifact_sha256"):
            raise JudgmentRecoveryError("planned throttle record changed before recovery")
        if not _is_documented_throttle_rejection(prior):
            raise JudgmentRecoveryError("planned record no longer meets throttle eligibility")
        event = recovery_events.get(judgment_id)
        if event is not None:
            record = _orphaned_recovery_record(
                item=item,
                prior=prior,
                event=event,
                contracts=contracts,
                plan_sha=plan_sha,
            )
            _atomic_write(judgment_dir, f"judgment-{judgment_id}", record)
            latest[judgment_id] = {**record, "artifact_sha256": sha256_json(record)}
            continue
        pending.append((item, prior))

    records_before = _all_records(judgment_dir)
    attributed, reservations, _ = _attempt_exposure(records_before)
    committed = attributed + reservations
    forecast = committed + sum(
        _reservation(item.judge, _prompt_for(item, tasks=tasks, arms=arms)) for item, _ in pending
    )
    if forecast >= cap_usd * Decimal("0.85"):
        raise JudgmentRecoveryError("recovery reservation forecast exceeds the admission gate")
    budget = JudgmentBudget(cap_usd, committed=committed)
    runtime = create_boto3_clients(settings, client_config=_bedrock_transport_config(4)).runtime
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=4, thread_name_prefix="flavourbench-throttle-recovery")
    )
    global_semaphore = asyncio.Semaphore(4)
    judge_semaphores = {
        str(judge["judge_id"]): asyncio.Semaphore(2) for judge in judge_manifest["judges"]
    }
    next_start: dict[str, float] = Counter()
    throttle_streak: Counter[str] = Counter()
    start_locks = {str(judge["judge_id"]): asyncio.Lock() for judge in judge_manifest["judges"]}
    completed = 0
    completed_lock = asyncio.Lock()

    async def rate_limit(judge_id: str) -> None:
        async with start_locks[judge_id]:
            delay = next_start[judge_id] - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            next_start[judge_id] = time.monotonic() + 1.0

    async def run_one(item: JudgmentWorkItem, prior: Mapping[str, Any]) -> None:
        nonlocal completed
        prompt = _prompt_for(item, tasks=tasks, arms=arms)
        reservation = _reservation(item.judge, prompt)
        request_payload_sha = sha256_json(
            {
                "modelId": item.judge["requested_endpoint_id"],
                "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
                "prompt_sha256": sha256_text(prompt),
                "schema_sha256": JUDGMENT_SCHEMA_SHA256,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "temperature": 0,
            }
        )
        judge_id = str(item.judge["judge_id"])
        async with judge_semaphores[judge_id], global_semaphore:
            await rate_limit(judge_id)
            await budget.reserve(item.judgment_id, reservation)
            started_at = _utc_now()
            event = {
                "schema_version": "flavourbench-season0-judge-recovery-event-v1",
                "event_type": "recovery_request_started",
                "judgment_id": item.judgment_id,
                "comparison_id": item.comparison["comparison_id"],
                "judge_id": judge_id,
                "orientation": item.orientation,
                "attempt_number": 2,
                "supersedes_artifact_sha256": prior["artifact_sha256"],
                "plan_artifact_sha256": plan_sha,
                "request_payload_sha256": request_payload_sha,
                "prompt_sha256": sha256_text(prompt),
                "reservation_usd": format(reservation, ".9f"),
                "started_at": started_at,
            }
            _atomic_write(recovery_event_dir, f"recovery-event-{item.judgment_id}", event)
            started = time.monotonic()
            response: Mapping[str, Any] | None = None
            actual: Decimal | None = None
            status = "failed"
            delivery_state = "uncertain"
            error_type: str | None = None
            error_text: str | None = None
            result: dict[str, Any] = {
                "request_payload_sha256": request_payload_sha,
                "prompt_sha256": sha256_text(prompt),
                "estimated_cost_usd": None,
            }
            try:
                response = await asyncio.to_thread(
                    runtime.converse,
                    modelId=item.judge["requested_endpoint_id"],
                    system=[{"text": JUDGE_SYSTEM_PROMPT}],
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": MAX_OUTPUT_TOKENS, "temperature": 0},
                    outputConfig=structured_output_config(JUDGMENT_SCHEMA),
                    requestMetadata={
                        "flavourbench_phase": "automated_judging_throttle_recovery",
                        "flavourbench_judgment": item.judgment_id[:32],
                    },
                )
                usage = _usage(response)
                actual = _cost(item.judge, usage)
                result.update(
                    {
                        "usage": usage,
                        "estimated_cost_usd": format(actual, ".9f"),
                        "stop_reason": str(response.get("stopReason") or "unknown"),
                        "returned_model_id": item.judge["requested_endpoint_id"],
                        "actual_provider_name": "Amazon Bedrock",
                    }
                )
                delivery_state = "reconciled"
                raw = _response_text(response)
                result["raw_output_sha256"] = sha256_text(raw)
                judgment = validate_judgment(json.loads(raw))
                result["judgment"] = judgment
                result["normalized_choice"] = normalize_choice(
                    str(judgment["choice"]), item.orientation
                )
                status = "success"
            except Exception as error:  # noqa: BLE001 - every real attempt is persisted
                error_type = type(error).__name__
                error_text = str(error)[:500]
                if response is not None:
                    delivery_state = "reconciled"
                elif error_type == RECOVERABLE_ERROR_TYPE:
                    delivery_state = "documented_explicit_rejection"
            if error_type == RECOVERABLE_ERROR_TYPE:
                throttle_streak[judge_id] += 1
                exponent = min(throttle_streak[judge_id] - 1, 4)
                backoff = min(300, 30 * (2**exponent))
                async with start_locks[judge_id]:
                    next_start[judge_id] = max(next_start[judge_id], time.monotonic() + backoff)
            else:
                throttle_streak[judge_id] = 0
            result["latency_ms"] = round((time.monotonic() - started) * 1000)
            if isinstance(response, Mapping):
                metadata = response.get("ResponseMetadata")
                request_id = (
                    str(metadata.get("RequestId") or "") if isinstance(metadata, Mapping) else ""
                )
                result["request_id_sha256"] = sha256_text(request_id) if request_id else None
            await budget.finalize(item.judgment_id, actual)
            record = {
                "schema_version": JUDGMENT_SCHEMA_VERSION,
                "judgment_id": item.judgment_id,
                "comparison_id": item.comparison["comparison_id"],
                "track": item.comparison["track"],
                "task_id": item.comparison["task_id"],
                "task_family": item.comparison["task_family"],
                "judge": dict(item.judge),
                "orientation": item.orientation,
                "status": status,
                "delivery_state": delivery_state,
                "error_type": error_type,
                "error": error_text,
                "reservation_usd": format(reservation, ".9f"),
                "started_at": started_at,
                "completed_at": _utc_now(),
                "contracts": contracts,
                "result": result,
                "attempt_number": 2,
                "supersedes_artifact_sha256": prior["artifact_sha256"],
                "recovery": {
                    "protocol_version": RECOVERY_PROTOCOL_VERSION,
                    "plan_artifact_sha256": plan_sha,
                    "original_error_type": RECOVERABLE_ERROR_TYPE,
                    "aws_error_semantics_url": AWS_ERROR_SEMANTICS_URL,
                },
                "synthetic": False,
            }
            _atomic_write(judgment_dir, f"judgment-{item.judgment_id}", record)
            async with completed_lock:
                completed += 1
                if completed % 50 == 0 or completed == len(pending):
                    print(
                        json.dumps(
                            {
                                "event": "throttle_recovery_progress",
                                "completed": completed,
                                "planned": len(pending),
                            }
                        ),
                        flush=True,
                    )

    await asyncio.gather(*(run_one(item, prior) for item, prior in pending))
    return _final_summary(
        output_dir=output_dir,
        work_items=work_items,
        plan=plan,
        plan_sha=plan_sha,
        contracts={
            "task_bank_artifact_sha256": task_sha,
            "comparison_manifest_artifact_sha256": comparison_sha,
            "judge_manifest_artifact_sha256": judge_sha,
            "judge_cost_envelope_artifact_sha256": cost_sha,
        },
        run_configuration={
            "primary_transport": "see_original_collection_summary",
            "recovery_protocol_version": RECOVERY_PROTOCOL_VERSION,
            "recovery_global_concurrency": 4,
            "recovery_per_judge_concurrency": 2,
            "minimum_start_interval_seconds_per_judge": 1.0,
            "adaptive_throttle_backoff_seconds": [30, 60, 120, 240, 300],
            "bedrock_connect_timeout_seconds": 10,
            "bedrock_read_timeout_seconds": 360,
            "bedrock_total_max_attempts": 1,
            "maximum_total_provider_attempts_per_recovered_judgment": 2,
        },
    )


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--original-summary", type=Path, required=True)
    plan_parser.add_argument("--judgments-dir", type=Path, required=True)
    plan_parser.add_argument("--output-dir", type=Path, required=True)

    recover_parser = subparsers.add_parser("recover")
    recover_parser.add_argument("--task-bank", type=Path, required=True)
    recover_parser.add_argument("--arms-dir", type=Path, required=True)
    recover_parser.add_argument("--comparison-manifest", type=Path, required=True)
    recover_parser.add_argument("--judge-manifest", type=Path, required=True)
    recover_parser.add_argument("--cost-envelope", type=Path, required=True)
    recover_parser.add_argument("--plan", type=Path, required=True)
    recover_parser.add_argument("--output-dir", type=Path, required=True)
    recover_parser.add_argument("--cap-usd", type=Decimal, default=Decimal("5000"))
    recover_parser.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)

    if args.command == "plan":
        result = freeze_recovery_plan(
            original_summary=_load(args.original_summary),
            judgments_dir=args.judgments_dir,
            output_dir=args.output_dir,
        )
    else:
        result = asyncio.run(
            recover_throttles(
                task_bank=_load(args.task_bank),
                arms_dir=args.arms_dir,
                comparison_manifest=_load(args.comparison_manifest),
                judge_manifest=_load(args.judge_manifest),
                cost_envelope=_load(args.cost_envelope),
                plan=_load(args.plan),
                output_dir=args.output_dir,
                cap_usd=args.cap_usd,
                confirmation=args.confirmation,
            )
        )
    compact = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "recovery_items",
            "judgment_artifact_sha256s",
            "all_attempt_artifact_sha256s",
        }
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    run()
