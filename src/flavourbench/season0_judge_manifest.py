"""Run real Bedrock judge contract smokes and freeze the Season 0 judge panel."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .bedrock_auth import BedrockLaneSettings, create_boto3_clients
from .bedrock_provider import structured_output_config
from .execution_policy import assert_legacy_paid_cli_allowed
from .real_task_bank import sha256_json, sha256_text
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

SCHEMA_VERSION = "flavourbench-season0-judge-manifest-v1"
CONFIRMATION = "RUN_REAL_SEASON0_JUDGE_CONTRACT_SMOKE_V1"
MAX_OUTPUT_TOKENS = 8192

JUDGE_SPECS: tuple[dict[str, str], ...] = (
    {
        "judge_id": "judge-anthropic-sonnet-4-6",
        "display_name": "Anthropic Claude Sonnet 4.6",
        "canonical_model_id": "anthropic/claude-sonnet-4.6",
        "requested_endpoint_id": "global.anthropic.claude-sonnet-4-6",
        "self_season_model_id": "fb-s0-model-01",
        "input_usd_per_million": "3",
        "output_usd_per_million": "15",
    },
    {
        "judge_id": "judge-anthropic-haiku-4-5",
        "display_name": "Anthropic Claude Haiku 4.5",
        "canonical_model_id": "anthropic/claude-haiku-4.5",
        "requested_endpoint_id": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "self_season_model_id": "",
        "input_usd_per_million": "1",
        "output_usd_per_million": "5",
    },
    {
        "judge_id": "judge-alibaba-qwen3-next-80b",
        "display_name": "Alibaba Qwen3 Next 80B A3B",
        "canonical_model_id": "qwen/qwen3-next-80b-a3b",
        "requested_endpoint_id": "qwen.qwen3-next-80b-a3b",
        "self_season_model_id": "fb-s0-model-05",
        "input_usd_per_million": "0.18",
        "output_usd_per_million": "1.41",
    },
    {
        "judge_id": "judge-mistral-devstral-2-123b",
        "display_name": "Mistral Devstral 2 123B",
        "canonical_model_id": "mistral/devstral-2-123b",
        "requested_endpoint_id": "mistral.devstral-2-123b",
        "self_season_model_id": "fb-s0-model-07",
        "input_usd_per_million": "0.48",
        "output_usd_per_million": "2.40",
    },
)


class JudgeManifestError(RuntimeError):
    """The real judge panel cannot be frozen."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise JudgeManifestError(f"expected a JSON object: {path}")
    return value


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


def _smoke_pair(
    *, task_bank: Mapping[str, Any], arms_dir: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    tasks = {
        str(task["task_id"]): task
        for task in task_bank.get("tasks", [])
        if isinstance(task, Mapping)
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for path in arms_dir.glob("arm-*.json"):
        arm = _load(path)
        if (
            arm.get("phase") == "calibration"
            and arm.get("condition") == "epicure_on"
            and arm.get("status") == "success"
            and arm.get("rank_eligible") is True
            and arm.get("model", {}).get("season_model_id")
            not in {spec["self_season_model_id"] for spec in JUDGE_SPECS}
        ):
            grouped[str(arm["task"]["task_id"])].append(arm)
    for task_id in sorted(grouped):
        candidates = sorted(grouped[task_id], key=lambda arm: str(arm["model"]["season_model_id"]))
        if len(candidates) >= 2 and task_id in tasks:
            return tasks[task_id], candidates[0], candidates[1]
    raise JudgeManifestError("calibration arms contain no eligible real judge smoke pair")


def _answer(arm: Mapping[str, Any]) -> str:
    result = arm.get("result")
    answer = result.get("answer_markdown") if isinstance(result, Mapping) else None
    if not isinstance(answer, str) or not answer.strip():
        raise JudgeManifestError("judge smoke arm has no answer")
    return answer


def _response_text(response: Mapping[str, Any]) -> str:
    output = response.get("output")
    message = output.get("message") if isinstance(output, Mapping) else None
    blocks = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(blocks, list):
        raise JudgeManifestError("judge response has no content blocks")
    text = "".join(str(block.get("text") or "") for block in blocks if isinstance(block, Mapping))
    if not text:
        raise JudgeManifestError("judge response has no structured text")
    return text


def _usage(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise JudgeManifestError("judge response has no usage")
    return {
        "input_tokens": int(usage.get("inputTokens") or 0),
        "output_tokens": int(usage.get("outputTokens") or 0),
        "total_tokens": int(usage.get("totalTokens") or 0),
    }


def _cost(spec: Mapping[str, str], usage: Mapping[str, int]) -> Decimal:
    return (
        Decimal(usage["input_tokens"]) * Decimal(spec["input_usd_per_million"])
        + Decimal(usage["output_tokens"]) * Decimal(spec["output_usd_per_million"])
    ) / Decimal(1_000_000)


def _reservation(spec: Mapping[str, str], prompt: str) -> Decimal:
    input_tokens = math.ceil((len(JUDGE_SYSTEM_PROMPT) + len(prompt)) / 3)
    return (
        Decimal(input_tokens) * Decimal(spec["input_usd_per_million"])
        + Decimal(MAX_OUTPUT_TOKENS) * Decimal(spec["output_usd_per_million"])
    ) / Decimal(1_000_000)


async def _run_one(
    *,
    runtime: Any,
    spec: Mapping[str, str],
    task: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    orientation: str,
) -> dict[str, Any]:
    prompt = build_judge_prompt(
        task=task,
        left_answer=_answer(left),
        right_answer=_answer(right),
        orientation=orientation,
    )
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "judge_id": spec["judge_id"],
        "task_id": task["task_id"],
        "left_arm_id": left["arm_id"],
        "right_arm_id": right["arm_id"],
        "orientation": orientation,
    }
    request_sha = sha256_json(
        {
            **identity,
            "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
            "prompt_sha256": sha256_text(prompt),
            "schema_sha256": JUDGMENT_SCHEMA_SHA256,
            "model_id": spec["requested_endpoint_id"],
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
        }
    )
    started = time.monotonic()
    base: dict[str, Any] = {
        **identity,
        "request_sha256": request_sha,
        "prompt_sha256": sha256_text(prompt),
    }
    try:
        response = await asyncio.to_thread(
            runtime.converse,
            modelId=spec["requested_endpoint_id"],
            system=[{"text": JUDGE_SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": MAX_OUTPUT_TOKENS, "temperature": 0},
            outputConfig=structured_output_config(JUDGMENT_SCHEMA),
            requestMetadata={
                "flavourbench_phase": "judge_contract_smoke",
                "flavourbench_request": request_sha[:32],
            },
        )
        raw = _response_text(response)
        usage = _usage(response)
        metadata = response.get("ResponseMetadata")
        request_id = str(metadata.get("RequestId") or "") if isinstance(metadata, Mapping) else ""
        response_fields: dict[str, Any] = {
            "returned_model_id": spec["requested_endpoint_id"],
            "stop_reason": str(response.get("stopReason") or "unknown"),
            "usage": usage,
            "estimated_cost_usd": format(_cost(spec, usage), ".9f"),
            "request_id_sha256": sha256_text(request_id) if request_id else None,
            "raw_output_sha256": sha256_text(raw),
        }
        try:
            parsed = json.loads(raw)
            judgment = validate_judgment(parsed)
        except Exception as error:  # noqa: BLE001 - retain exact smoke incompatibility
            return {
                **base,
                **response_fields,
                "status": "failed",
                "delivery_state": "reconciled",
                "error_type": type(error).__name__,
                "error": str(error)[:500],
                "parsed_unvalidated": parsed if "parsed" in locals() else None,
                "latency_ms": round((time.monotonic() - started) * 1000),
            }
        return {
            **base,
            **response_fields,
            "status": "success",
            "delivery_state": "reconciled",
            "error_type": None,
            "error": None,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "judgment": judgment,
            "normalized_choice": normalize_choice(str(judgment["choice"]), orientation),
        }
    except Exception as error:  # noqa: BLE001 - persist pre-response/provider failures
        return {
            **base,
            "status": "failed",
            "delivery_state": "uncertain",
            "error_type": type(error).__name__,
            "error": str(error)[:500],
            "estimated_cost_usd": None,
            "latency_ms": round((time.monotonic() - started) * 1000),
        }


async def freeze_judge_manifest(
    *,
    task_bank: Mapping[str, Any],
    arms_dir: Path,
    output_dir: Path,
    cap_usd: Decimal,
    confirmation: str,
) -> dict[str, Any]:
    if confirmation != CONFIRMATION:
        raise JudgeManifestError("real judge smoke confirmation is missing")
    if task_bank.get("synthetic_tasks") != 0:
        raise JudgeManifestError("judge smoke requires the real Season 0 task bank")
    settings = BedrockLaneSettings.from_environ()
    if not settings.enabled or not settings.live_authorized or settings.stage != "season":
        raise JudgeManifestError("Bedrock season-stage live authorization is required")
    if cap_usd <= 0 or cap_usd > settings.hard_cap_usd:
        raise JudgeManifestError("judge smoke cap must be positive and within the hard cap")
    task, left, right = _smoke_pair(task_bank=task_bank, arms_dir=arms_dir)
    reservations = []
    for spec in JUDGE_SPECS:
        for orientation in ORIENTATIONS:
            prompt = build_judge_prompt(
                task=task,
                left_answer=_answer(left),
                right_answer=_answer(right),
                orientation=orientation,
            )
            reservations.append(_reservation(spec, prompt))
    if sum(reservations, Decimal(0)) > cap_usd * Decimal("0.85"):
        raise JudgeManifestError("judge smoke forecast exceeds the admission threshold")
    runtime = create_boto3_clients(settings).runtime
    results = await asyncio.gather(
        *(
            _run_one(
                runtime=runtime,
                spec=spec,
                task=task,
                left=left,
                right=right,
                orientation=orientation,
            )
            for spec in JUDGE_SPECS
            for orientation in ORIENTATIONS
        )
    )
    by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_judge[str(result["judge_id"])].append(result)
    swap_consistency = {
        judge_id: len(rows) == 2
        and all(row["status"] == "success" for row in rows)
        and len({row["normalized_choice"] for row in rows}) == 1
        for judge_id, rows in by_judge.items()
    }
    smoke_payload: dict[str, Any] = {
        "schema_version": "flavourbench-season0-judge-contract-smoke-v1",
        "run_class": "paid_real_judge_contract_smoke_excluded_from_scoring",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "synthetic_calls": 0,
        "real_bedrock_calls": len(results),
        "task_bank_artifact_sha256": task_bank["artifact_sha256"],
        "task_id": task["task_id"],
        "arm_artifact_sha256s": [left["artifact_sha256"], right["artifact_sha256"]],
        "protocol_version": PROTOCOL_VERSION,
        "judge_system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
        "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
        "swap_consistency": swap_consistency,
        "estimated_cost_usd": format(
            sum(Decimal(str(row.get("estimated_cost_usd") or 0)) for row in results),
            ".9f",
        ),
        "results": results,
    }
    smoke_path = _atomic_write(output_dir, "judge-contract-smoke", smoke_payload)
    if not all(result["status"] == "success" for result in results):
        raise JudgeManifestError(f"judge contract smoke failed; evidence retained at {smoke_path}")
    if not all(swap_consistency.values()):
        raise JudgeManifestError(
            f"judge swap consistency failed; evidence retained at {smoke_path}"
        )
    smoke_sha = json.loads(smoke_path.read_bytes())["artifact_sha256"]
    manifest_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "season": "Season 0",
        "status": "frozen_for_real_automated_judging",
        "frozen_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "synthetic_compatibility_calls": 0,
        "real_compatibility_calls": len(results),
        "contract_smoke_artifact_sha256": smoke_sha,
        "task_bank_artifact_sha256": task_bank["artifact_sha256"],
        "protocol": {
            "version": PROTOCOL_VERSION,
            "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
            "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
            "orientations": list(ORIENTATIONS),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "provider_structured_output_required": True,
            "accepted_reference_role": "non_binding_orientation_aid",
        },
        "judges": [dict(spec) for spec in JUDGE_SPECS],
        "primary_policy": {
            "orientation_disagreement": "exclude_that_judge_comparison",
            "self_judgment": "exclude_from_primary_and_include_in_sensitivity_only",
            "cohort_pooling": "never_pool_with_public_or_expert_human_votes",
            "consensus": "majority_of_orientation_consistent_non_self_judges",
        },
        "maximum_planned_real_bedrock_calls": 2_160 * len(JUDGE_SPECS) * len(ORIENTATIONS),
        "hard_cap_usd": "5000",
    }
    manifest_path = _atomic_write(output_dir, "season0-judge-manifest", manifest_payload)
    return {
        **manifest_payload,
        "contract_smoke_path": str(smoke_path),
        "manifest_path": str(manifest_path),
    }


def run(argv: Sequence[str] | None = None) -> None:
    assert_legacy_paid_cli_allowed("flavourbench-freeze-season0-judges")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--calibration-arms-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("5"))
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args(argv)
    result = asyncio.run(
        freeze_judge_manifest(
            task_bank=_load(args.task_bank),
            arms_dir=args.calibration_arms_dir,
            output_dir=args.output_dir,
            cap_usd=args.cap_usd,
            confirmation=args.confirmation,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
