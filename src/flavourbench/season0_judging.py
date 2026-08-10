"""Collect real, blinded, swap-controlled Bedrock judgments for Season 0."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from botocore.config import Config

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients
from .bedrock_provider import structured_output_config
from .real_task_bank import sha256_json, sha256_text
from .season0_judge_manifest import MAX_OUTPUT_TOKENS
from .season0_judge_protocol import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT_SHA256,
    JUDGMENT_SCHEMA,
    JUDGMENT_SCHEMA_SHA256,
    ORIENTATIONS,
    PROTOCOL_VERSION,
    build_judge_prompt,
    normalize_choice,
    validate_judgment,
)

SCHEMA_VERSION = "flavourbench-season0-automated-judgment-v1"
SUMMARY_SCHEMA_VERSION = "flavourbench-season0-automated-judgment-summary-v1"
CONFIRMATION = "RUN_REAL_SEASON0_AUTOMATED_JUDGING_V1"
BEDROCK_CONNECT_TIMEOUT_SECONDS = 10
BEDROCK_READ_TIMEOUT_SECONDS = 360
BEDROCK_TOTAL_MAX_ATTEMPTS = 1
JUDGE_CONCURRENCY_LIMITS = {
    "judge-anthropic-sonnet-4-6": 2,
    "judge-anthropic-haiku-4-5": 2,
    "judge-alibaba-qwen3-next-80b": 16,
    "judge-mistral-devstral-2-123b": 16,
}


class JudgingError(RuntimeError):
    """The official automated-judging run cannot proceed safely."""


@dataclass(frozen=True)
class JudgmentWorkItem:
    judgment_id: str
    comparison: Mapping[str, Any]
    judge: Mapping[str, Any]
    orientation: str


class JudgmentBudget:
    def __init__(self, hard_cap: Decimal, committed: Decimal = Decimal(0)) -> None:
        self.hard_cap = hard_cap
        self.committed = committed
        self.active: dict[str, Decimal] = {}
        self.lock = asyncio.Lock()

    async def reserve(self, judgment_id: str, amount: Decimal) -> None:
        async with self.lock:
            exposure = self.committed + sum(self.active.values(), Decimal(0))
            if exposure + amount >= self.hard_cap * Decimal("0.85"):
                raise JudgingError("Bedrock judge budget admission threshold reached")
            self.active[judgment_id] = amount

    async def finalize(self, judgment_id: str, actual: Decimal | None) -> None:
        async with self.lock:
            reserved = self.active.pop(judgment_id, Decimal(0))
            self.committed += reserved if actual is None else actual


def _bedrock_transport_config(concurrency: int) -> Config:
    """Use one long-lived request per judgment and never retry an inference implicitly."""

    return Config(
        connect_timeout=BEDROCK_CONNECT_TIMEOUT_SECONDS,
        read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
        max_pool_connections=concurrency,
        retries={"total_max_attempts": BEDROCK_TOTAL_MAX_ATTEMPTS, "mode": "standard"},
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise JudgingError(f"expected a JSON object: {path}")
    return value


def _verify_artifact(document: Mapping[str, Any], label: str) -> str:
    claimed = document.get("artifact_sha256")
    if not isinstance(claimed, str):
        raise JudgingError(f"{label} has no artifact hash")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise JudgingError(f"{label} artifact hash mismatch")
    return actual


def _atomic_write(directory: Path, stem: str, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{stem}-{digest}.json"
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(destination)
    return destination


def _latest_by_id(directory: Path, prefix: str, id_field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in directory.glob(f"{prefix}-*.json"):
        document = _load(path)
        identifier = document.get(id_field)
        if isinstance(identifier, str):
            grouped.setdefault(identifier, []).append(document)
    return {
        identifier: sorted(rows, key=lambda row: str(row.get("completed_at") or ""))[-1]
        for identifier, rows in grouped.items()
    }


def _arms_by_id(directory: Path) -> dict[str, dict[str, Any]]:
    return _latest_by_id(directory, "arm", "arm_id")


def build_work_items(
    comparison_manifest: Mapping[str, Any], judge_manifest: Mapping[str, Any]
) -> list[JudgmentWorkItem]:
    comparison_sha = str(comparison_manifest.get("artifact_sha256") or "")
    judge_manifest_sha = str(judge_manifest.get("artifact_sha256") or "")
    comparisons = comparison_manifest.get("comparisons")
    judges = judge_manifest.get("judges")
    if not isinstance(comparisons, list) or not isinstance(judges, list):
        raise JudgingError("comparison or judge manifest is malformed")
    output: list[JudgmentWorkItem] = []
    for comparison in comparisons:
        if not isinstance(comparison, Mapping) or comparison.get("judgable") is not True:
            continue
        for judge in judges:
            if not isinstance(judge, Mapping):
                raise JudgingError("judge manifest contains an invalid judge")
            for orientation in ORIENTATIONS:
                identity = {
                    "schema_version": SCHEMA_VERSION,
                    "season": "Season 0",
                    "comparison_manifest_artifact_sha256": comparison_sha,
                    "judge_manifest_artifact_sha256": judge_manifest_sha,
                    "protocol_version": PROTOCOL_VERSION,
                    "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
                    "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
                    "comparison_id": comparison["comparison_id"],
                    "judge_id": judge["judge_id"],
                    "orientation": orientation,
                }
                output.append(
                    JudgmentWorkItem(
                        judgment_id=sha256_json(identity),
                        comparison=comparison,
                        judge=judge,
                        orientation=orientation,
                    )
                )
    output.sort(key=lambda item: sha256_text(f"judge-order:{item.judgment_id}"))
    if len({item.judgment_id for item in output}) != len(output):
        raise JudgingError("judgment work-item IDs are not unique")
    return output


def _answer(arm: Mapping[str, Any]) -> str:
    result = arm.get("result")
    answer = result.get("answer_markdown") if isinstance(result, Mapping) else None
    if not isinstance(answer, str) or not answer.strip():
        raise JudgingError("comparison references an arm with no answer")
    return answer


def _prompt_for(
    item: JudgmentWorkItem,
    *,
    tasks: Mapping[str, Mapping[str, Any]],
    arms: Mapping[str, Mapping[str, Any]],
) -> str:
    task_id = str(item.comparison["task_id"])
    left_id = str(item.comparison["left_arm_id"])
    right_id = str(item.comparison["right_arm_id"])
    try:
        task = tasks[task_id]
        left = arms[left_id]
        right = arms[right_id]
    except KeyError as error:
        raise JudgingError("comparison references a missing task or arm") from error
    left_answer = _answer(left)
    right_answer = _answer(right)
    if sha256_text(left_answer) != item.comparison["left"]["answer_sha256"]:
        raise JudgingError("left answer hash does not match comparison manifest")
    if sha256_text(right_answer) != item.comparison["right"]["answer_sha256"]:
        raise JudgingError("right answer hash does not match comparison manifest")
    return build_judge_prompt(
        task=task,
        left_answer=left_answer,
        right_answer=right_answer,
        orientation=item.orientation,
    )


def _orphan_terminal_record(
    *,
    item: JudgmentWorkItem,
    event: Mapping[str, Any],
    prompt_sha256: str,
    task_bank_sha256: str,
    comparison_manifest_sha256: str,
    judge_manifest_sha256: str,
    judge_cost_envelope_sha256: str,
    completed_at: str,
) -> dict[str, Any]:
    expected = {
        "judgment_id": item.judgment_id,
        "comparison_id": item.comparison["comparison_id"],
        "judge_id": item.judge["judge_id"],
        "orientation": item.orientation,
    }
    if any(event.get(key) != value for key, value in expected.items()):
        raise JudgingError("orphaned request event identity mismatch")
    reservation = str(event.get("reservation_usd") or "")
    request_payload_sha = event.get("request_payload_sha256")
    if not reservation or not isinstance(request_payload_sha, str):
        raise JudgingError("orphaned request event lacks reservation or request hash")
    return {
        "schema_version": SCHEMA_VERSION,
        "judgment_id": item.judgment_id,
        "comparison_id": item.comparison["comparison_id"],
        "track": item.comparison["track"],
        "task_id": item.comparison["task_id"],
        "task_family": item.comparison["task_family"],
        "judge": dict(item.judge),
        "orientation": item.orientation,
        "status": "failed",
        "delivery_state": "uncertain",
        "error_type": "OrphanedRequestEvent",
        "error": (
            "request-start journal found without a terminal response; the paid request "
            "was not resent"
        ),
        "reservation_usd": reservation,
        "started_at": event.get("started_at"),
        "completed_at": completed_at,
        "contracts": {
            "task_bank_artifact_sha256": task_bank_sha256,
            "comparison_manifest_artifact_sha256": comparison_manifest_sha256,
            "judge_manifest_artifact_sha256": judge_manifest_sha256,
            "judge_cost_envelope_artifact_sha256": judge_cost_envelope_sha256,
            "protocol_version": PROTOCOL_VERSION,
            "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
            "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
        },
        "result": {
            "request_payload_sha256": request_payload_sha,
            "prompt_sha256": prompt_sha256,
            "estimated_cost_usd": None,
            "latency_ms": None,
        },
        "synthetic": False,
    }


def _reservation(judge: Mapping[str, Any], prompt: str) -> Decimal:
    input_tokens = math.ceil((len(JUDGE_SYSTEM_PROMPT) + len(prompt)) / 3)
    return (
        Decimal(input_tokens) * Decimal(str(judge["input_usd_per_million"]))
        + Decimal(MAX_OUTPUT_TOKENS) * Decimal(str(judge["output_usd_per_million"]))
    ) / Decimal(1_000_000)


def _usage(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise JudgingError("Bedrock judge response has no usage")
    return {
        "input_tokens": int(usage.get("inputTokens") or 0),
        "output_tokens": int(usage.get("outputTokens") or 0),
        "total_tokens": int(usage.get("totalTokens") or 0),
        "cache_read_input_tokens": int(usage.get("cacheReadInputTokens") or 0),
        "cache_write_input_tokens": int(usage.get("cacheWriteInputTokens") or 0),
    }


def _cost(judge: Mapping[str, Any], usage: Mapping[str, int]) -> Decimal:
    return (
        Decimal(usage["input_tokens"]) * Decimal(str(judge["input_usd_per_million"]))
        + Decimal(usage["output_tokens"]) * Decimal(str(judge["output_usd_per_million"]))
    ) / Decimal(1_000_000)


def _response_text(response: Mapping[str, Any]) -> str:
    output = response.get("output")
    message = output.get("message") if isinstance(output, Mapping) else None
    blocks = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(blocks, list):
        raise JudgingError("Bedrock judge response has no content")
    text = "".join(str(block.get("text") or "") for block in blocks if isinstance(block, Mapping))
    if not text:
        raise JudgingError("Bedrock judge response is empty")
    return text


async def collect_judgments(
    *,
    task_bank: Mapping[str, Any],
    arms_dir: Path,
    comparison_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any],
    cost_envelope: Mapping[str, Any],
    output_dir: Path,
    cap_usd: Decimal,
    concurrency: int,
    confirmation: str,
) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise JudgingError("official real-judging confirmation is missing")
    task_bank_sha = _verify_artifact(task_bank, "task bank")
    comparison_sha = _verify_artifact(comparison_manifest, "comparison manifest")
    judge_manifest_sha = _verify_artifact(judge_manifest, "judge manifest")
    cost_envelope_sha = _verify_artifact(cost_envelope, "judge cost envelope")
    if task_bank.get("synthetic_tasks") != 0:
        raise JudgingError("official judging requires the real task bank")
    if comparison_manifest.get("synthetic_comparisons") != 0:
        raise JudgingError("official judging refuses synthetic comparisons")
    if comparison_manifest.get("task_bank_artifact_sha256") != task_bank_sha:
        raise JudgingError("comparison manifest is bound to another task bank")
    if judge_manifest.get("task_bank_artifact_sha256") != task_bank_sha:
        raise JudgingError("judge manifest is bound to another task bank")
    if (
        cost_envelope.get("status") != "frozen_for_automated_judge_admission"
        or cost_envelope.get("task_bank_artifact_sha256") != task_bank_sha
        or cost_envelope.get("comparison_manifest_artifact_sha256") != comparison_sha
        or cost_envelope.get("judge_manifest_artifact_sha256") != judge_manifest_sha
    ):
        raise JudgingError("judge cost envelope binding mismatch")
    protocol = judge_manifest.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("version") != PROTOCOL_VERSION
        or protocol.get("system_prompt_sha256") != JUDGE_SYSTEM_PROMPT_SHA256
        or protocol.get("judgment_schema_sha256") != JUDGMENT_SCHEMA_SHA256
        or protocol.get("orientations") != list(ORIENTATIONS)
        or protocol.get("max_output_tokens") != MAX_OUTPUT_TOKENS
    ):
        raise JudgingError("judge manifest is bound to another local protocol")
    settings = BedrockLaneSettings.from_environ()
    if not settings.enabled or not settings.live_authorized or settings.stage != "season":
        raise JudgingError("Bedrock season-stage live authorization is required")
    manifest_cap = Decimal(str(judge_manifest.get("hard_cap_usd") or "0"))
    envelope_cap = Decimal(str(cost_envelope.get("hard_cap_usd") or "0"))
    if (
        cap_usd <= 0
        or cap_usd > settings.hard_cap_usd
        or cap_usd > manifest_cap
        or cap_usd > envelope_cap
    ):
        raise JudgingError("judge cap exceeds a frozen or environment hard cap")
    if concurrency < 1 or concurrency > 64:
        raise JudgingError("judge concurrency must be between 1 and 64")

    tasks = {
        str(task["task_id"]): task
        for task in task_bank.get("tasks", [])
        if isinstance(task, Mapping)
    }
    arms = _arms_by_id(arms_dir)
    work_items = build_work_items(comparison_manifest, judge_manifest)
    workload_rows = []
    for item in work_items:
        prompt = _prompt_for(item, tasks=tasks, arms=arms)
        workload_rows.append(
            {
                "judgment_id": item.judgment_id,
                "prompt_sha256": sha256_text(prompt),
                "reservation_usd": format(_reservation(item.judge, prompt), ".9f"),
            }
        )
    workload_rows.sort(key=lambda row: row["judgment_id"])
    if (
        cost_envelope.get("planned_judgments") != len(work_items)
        or cost_envelope.get("workload_sha256") != sha256_json(workload_rows)
        or Decimal(str(cost_envelope.get("total_reservation_usd") or 0))
        != sum(Decimal(row["reservation_usd"]) for row in workload_rows)
    ):
        raise JudgingError("judge workload does not match the frozen cost envelope")
    judgment_dir = output_dir / "judgments"
    event_dir = output_dir / "events"
    terminal = _latest_by_id(judgment_dir, "judgment", "judgment_id")
    expected_ids = {item.judgment_id for item in work_items}
    if set(terminal) - expected_ids:
        raise JudgingError("judgment directory contains records from another manifest")
    events = _latest_by_id(event_dir, "event", "judgment_id")
    if set(events) - expected_ids:
        raise JudgingError("event directory contains records from another manifest")
    orphaned = set(events) - set(terminal)
    item_by_id = {item.judgment_id: item for item in work_items}
    recovered_orphans = len(orphaned)
    for judgment_id in sorted(orphaned):
        item = item_by_id[judgment_id]
        prompt = _prompt_for(item, tasks=tasks, arms=arms)
        recovered = _orphan_terminal_record(
            item=item,
            event=events[judgment_id],
            prompt_sha256=sha256_text(prompt),
            task_bank_sha256=task_bank_sha,
            comparison_manifest_sha256=comparison_sha,
            judge_manifest_sha256=judge_manifest_sha,
            judge_cost_envelope_sha256=cost_envelope_sha,
            completed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        _atomic_write(judgment_dir, f"judgment-{judgment_id}", recovered)
    if recovered_orphans:
        terminal = _latest_by_id(judgment_dir, "judgment", "judgment_id")
        orphaned = set(events) - set(terminal)
    committed = sum(
        Decimal(str((row.get("result") or {}).get("estimated_cost_usd") or 0))
        if row.get("delivery_state") == "reconciled"
        else Decimal(str(row.get("reservation_usd") or 0))
        for row in terminal.values()
    ) + sum(Decimal(str(events[item].get("reservation_usd") or 0)) for item in orphaned)
    budget = JudgmentBudget(cap_usd, committed=committed)
    pending = [
        item
        for item in work_items
        if item.judgment_id not in terminal and item.judgment_id not in orphaned
    ]
    forecast = committed + sum(
        _reservation(item.judge, _prompt_for(item, tasks=tasks, arms=arms)) for item in pending
    )
    if forecast >= cap_usd * Decimal("0.85"):
        raise JudgingError("full judge reservation forecast exceeds the admission threshold")

    transport_config = _bedrock_transport_config(concurrency)
    runtime = create_boto3_clients(
        settings,
        client_config=transport_config,
    ).runtime
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="flavourbench-bedrock-judge",
        )
    )
    semaphore = asyncio.Semaphore(concurrency)
    judge_ids = [str(judge["judge_id"]) for judge in judge_manifest["judges"]]
    fair_share = max(1, math.ceil(concurrency / len(judge_ids)))
    per_judge_concurrency = {
        judge_id: min(fair_share, JUDGE_CONCURRENCY_LIMITS.get(judge_id, fair_share))
        for judge_id in judge_ids
    }
    judge_semaphores = {
        judge_id: asyncio.Semaphore(per_judge_concurrency[judge_id]) for judge_id in judge_ids
    }
    completed_counter = 0
    counter_lock = asyncio.Lock()

    async def run_one(item: JudgmentWorkItem) -> None:
        nonlocal completed_counter
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
        async with judge_semaphores[judge_id], semaphore:
            await budget.reserve(item.judgment_id, reservation)
            started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            event = {
                "schema_version": "flavourbench-season0-judge-request-event-v1",
                "event_type": "request_started",
                "judgment_id": item.judgment_id,
                "comparison_id": item.comparison["comparison_id"],
                "judge_id": item.judge["judge_id"],
                "orientation": item.orientation,
                "request_payload_sha256": request_payload_sha,
                "reservation_usd": format(reservation, ".9f"),
                "started_at": started_at,
            }
            _atomic_write(event_dir, f"event-{item.judgment_id}-request-started", event)
            started = time.monotonic()
            response: Mapping[str, Any] | None = None
            usage: dict[str, int] | None = None
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
                        "flavourbench_phase": "automated_judging",
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
                parsed = json.loads(raw)
                judgment = validate_judgment(parsed)
                result["judgment"] = judgment
                result["normalized_choice"] = normalize_choice(
                    str(judgment["choice"]), item.orientation
                )
                status = "success"
            except Exception as error:  # noqa: BLE001 - persist every paid failure
                error_type = type(error).__name__
                error_text = str(error)[:500]
                if response is not None:
                    delivery_state = "reconciled"
                result.setdefault("latency_ms", round((time.monotonic() - started) * 1000))
            result["latency_ms"] = round((time.monotonic() - started) * 1000)
            if isinstance(response, Mapping):
                metadata = response.get("ResponseMetadata")
                request_id = (
                    str(metadata.get("RequestId") or "") if isinstance(metadata, Mapping) else ""
                )
                result["request_id_sha256"] = sha256_text(request_id) if request_id else None
            await budget.finalize(item.judgment_id, actual)
            record = {
                "schema_version": SCHEMA_VERSION,
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
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "contracts": {
                    "task_bank_artifact_sha256": task_bank_sha,
                    "comparison_manifest_artifact_sha256": comparison_sha,
                    "judge_manifest_artifact_sha256": judge_manifest_sha,
                    "judge_cost_envelope_artifact_sha256": cost_envelope_sha,
                    "protocol_version": PROTOCOL_VERSION,
                    "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
                    "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
                },
                "result": result,
                "synthetic": False,
            }
            _atomic_write(judgment_dir, f"judgment-{item.judgment_id}", record)
            async with counter_lock:
                completed_counter += 1
                if completed_counter % 100 == 0 or completed_counter == len(pending):
                    print(
                        json.dumps(
                            {
                                "event": "judging_progress",
                                "completed_this_run": completed_counter,
                                "pending_this_run": len(pending),
                            }
                        ),
                        flush=True,
                    )

    await asyncio.gather(*(run_one(item) for item in pending))
    records = _latest_by_id(judgment_dir, "judgment", "judgment_id")
    success = [row for row in records.values() if row.get("status") == "success"]
    failed = [row for row in records.values() if row.get("status") != "success"]
    attributed_cost = sum(
        Decimal(str((row.get("result") or {}).get("estimated_cost_usd") or 0))
        for row in records.values()
    )
    unattributed_reservations = sum(
        Decimal(str(row.get("reservation_usd") or 0))
        for row in records.values()
        if (row.get("result") or {}).get("estimated_cost_usd") is None
    )
    conservative_exposure = attributed_cost + unattributed_reservations
    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "season": "Season 0",
        "status": "collection_complete",
        "synthetic_judgments": 0,
        "task_bank_artifact_sha256": task_bank_sha,
        "comparison_manifest_artifact_sha256": comparison_sha,
        "judge_manifest_artifact_sha256": judge_manifest_sha,
        "judge_cost_envelope_artifact_sha256": cost_envelope_sha,
        "counts": {
            "planned_judgments": len(work_items),
            "terminal_judgments": len(records),
            "success": len(success),
            "failed": len(failed),
            "orphaned_request_events": len(orphaned),
            "recovered_orphaned_request_events": recovered_orphans,
            "by_judge": dict(Counter(row["judge"]["judge_id"] for row in success)),
            "by_orientation": dict(Counter(row["orientation"] for row in success)),
            "by_track": dict(Counter(row["track"] for row in success)),
        },
        "estimated_cost_usd": format(conservative_exposure, ".9f"),
        "attributed_estimated_cost_usd": format(attributed_cost, ".9f"),
        "unattributed_conservative_reservations_usd": format(unattributed_reservations, ".9f"),
        "cost_status": "conservative_exposure_pending_aws_cur_crosscheck",
        "run_configuration": {
            "global_concurrency": concurrency,
            "per_judge_concurrency": per_judge_concurrency,
            "bedrock_http_pool_connections": concurrency,
            "bedrock_worker_threads": concurrency,
            "bedrock_connect_timeout_seconds": BEDROCK_CONNECT_TIMEOUT_SECONDS,
            "bedrock_read_timeout_seconds": BEDROCK_READ_TIMEOUT_SECONDS,
            "bedrock_total_max_attempts": BEDROCK_TOTAL_MAX_ATTEMPTS,
            "scheduler": "fair_per_judge_semaphores_v1",
        },
        "failure_reasons": dict(
            Counter(str(row.get("error_type") or "unspecified") for row in failed)
        ),
        "judgment_artifact_sha256s": sorted(
            str(row["artifact_sha256"]) for row in records.values()
        ),
    }
    summary_path = _atomic_write(output_dir, "judgment-collection-summary", payload)
    return {**payload, "summary_path": str(summary_path)}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--cost-envelope", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("5000"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(
        collect_judgments(
            task_bank=_load(args.task_bank),
            arms_dir=args.arms_dir,
            comparison_manifest=_load(args.comparison_manifest),
            judge_manifest=_load(args.judge_manifest),
            cost_envelope=_load(args.cost_envelope),
            output_dir=args.output_dir,
            cap_usd=args.cap_usd,
            concurrency=args.concurrency,
            confirmation=args.confirmation,
        )
    )
    compact = {**result, "judgment_artifact_sha256s": "omitted_from_console"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    run()
