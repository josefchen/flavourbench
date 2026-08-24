"""No-call evidence aggregation for the real OpenRouter exploratory dataset.

The live collector intentionally writes many small immutable records.  This
module verifies those records and turns one point-in-time checkpoint into a
content-addressed analytical cube.  It does not instantiate a provider or MCP
adapter and makes no network calls; the records it verifies were produced by
real OpenRouter inference routed through Cloudflare AI Gateway and real Epicure
MCP execution.

The aggregate is operational evidence only.  In particular, response-format
validity, declared constraint acknowledgement, tool execution, latency,
tokens, and cost are never converted into a preference, quality, or uplift
estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .execution_policy import ExecutionPolicy
from .frontier_contract_runner import (
    AUTHORIZED_TOTAL_CAP_USD,
    DEFAULT_ADMISSION_FRACTION,
    IntegrityError,
    load_candidate_manifest,
    select_candidates,
)
from .real_dataset_runner import (
    CONDITIONS,
    CURRENT_DATASET_MANIFEST_SHA256,
    TASK_FAMILIES,
    WorkItem,
    _load_state,
    _validate_state_against_workload,
    _verify_response_artifact,
    build_balanced_work_items,
    select_balanced_tasks,
)
from .run_journal import JournalRecoveryState, scan_recovery_journals
from .validators import VALIDATOR_VERSION, validate_output

SCHEMA_VERSION = "flavourbench-real-exploratory-evidence-v1"
ROUTE_LABEL = "OpenRouter via Cloudflare AI Gateway"
DISPLAY_NAMES = {
    "openai/gpt-5.6-sol-pro": "GPT-5.6 Sol (OR pro)",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "anthropic/claude-fable-5": "Claude Fable 5",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "x-ai/grok-4.5": "Grok 4.5",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek/deepseek-r1-0528": "DeepSeek R1 0528",
    "qwen/qwen3.5-397b-a17b": "Qwen 3.5 397B",
    "z-ai/glm-5.2": "GLM 5.2",
    "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra",
}


class CheckpointChanged(IntegrityError):
    """The append-only collection advanced while a filesystem snapshot was read."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_inventory(artifact_root: Path) -> dict[str, Any]:
    real_root = artifact_root / "real-exploratory"
    ledger = real_root / "ledger.jsonl"

    def names(directory: Path, pattern: str) -> list[str]:
        return sorted(path.name for path in directory.glob(pattern)) if directory.exists() else []

    source_root = real_root / "source-runs"
    return {
        "dataset_ledger_file_sha256": _file_sha256(ledger) if ledger.exists() else None,
        "source_artifact_filenames": names(source_root, "*.json"),
        "response_artifact_filenames": names(real_root / "responses", "*.json"),
        "journal_filenames": sorted(
            [
                *names(source_root, "flavourbench-live-smoke-journal-*.jsonl"),
                *names(source_root, ".flavourbench-live-smoke-journal-*.inprogress.jsonl"),
            ]
        ),
        "summary_filenames": names(real_root / "summaries", "*.json"),
        "correction_filenames": names(real_root / "corrections", "*.json"),
        "incident_resolution_filenames": names(real_root / "resolutions", "*.json"),
    }


def _decimal_text(value: Decimal) -> str:
    rendered = format(value.quantize(Decimal("0.000001")), "f")
    return rendered.rstrip("0").rstrip(".") or "0"


def _percentile(values: Sequence[int], quantile: Decimal) -> int | None:
    """Return a deterministic nearest-rank percentile for small evidence sets."""

    if not values:
        return None
    ordered = sorted(values)
    rank = max(
        1,
        int((quantile * Decimal(len(ordered))).to_integral_value(rounding="ROUND_CEILING")),
    )
    return ordered[rank - 1]


def _distribution(values: Iterable[int]) -> dict[str, int | None]:
    materialized = list(values)
    return {
        "n": len(materialized),
        "minimum": min(materialized) if materialized else None,
        "p50": _percentile(materialized, Decimal("0.50")),
        "p95": _percentile(materialized, Decimal("0.95")),
        "maximum": max(materialized) if materialized else None,
        "mean": round(sum(materialized) / len(materialized)) if materialized else None,
    }


def _verify_content_addressed_summary(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"summary must be a regular, non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError(f"summary is not a JSON object: {path}")
    address = value.get("content_address")
    if not isinstance(address, Mapping):
        raise IntegrityError(f"summary has no content address: {path}")
    digest = address.get("digest")
    unhashed = dict(value)
    unhashed.pop("content_address", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or address.get("algorithm") != "sha256"
        or address.get("uri") != f"sha256:{digest}"
        or _sha256(unhashed) != digest
        or not path.stem.endswith(digest)
    ):
        raise IntegrityError(f"summary content address is invalid: {path}")
    return value


def _structured_and_constraint_labels(
    response: Mapping[str, Any],
    *,
    prompt: str,
    model_id: str,
) -> dict[str, Any]:
    output = response.get("output_json")
    answer = response.get("answer_markdown")
    traces = response.get("tool_trace")
    if not isinstance(output, dict) or not isinstance(answer, str) or not isinstance(traces, list):
        return {
            "structured_valid": False,
            "constraint_status": "not_evaluated_invalid_structure",
            "constraint_applicable": False,
            "scope": "format-only label; no substantive compliance judgment",
        }
    tool_errors = sum(
        isinstance(trace, Mapping) and bool(trace.get("is_error")) for trace in traces
    )
    validations = {
        validation.name: validation
        for validation in validate_output(
            prompt=prompt,
            output=output,
            answer=answer,
            model_name=model_id,
            tool_errors=tool_errors,
            tool_calls=len(traces),
        )
    }
    structured = validations["structured_response"]
    constraint = validations["constraint_acknowledgement"]
    return {
        "structured_valid": structured.status == "pass",
        "constraint_status": constraint.status,
        "constraint_applicable": constraint.status != "not_applicable",
        "scope": "acknowledgement label only; expert review determines substantive compliance",
    }


@dataclass(frozen=True)
class ConditionObservation:
    condition: str
    provider_attempted: bool
    request_count: int
    provider_generation_count: int
    reconciled_generation_count: int
    actual_cost_micros: int
    normalized_response: bool
    outcome_class: str
    normalized_response_cost_micros: int
    structured_valid: bool
    constraint_status: str
    constraint_applicable: bool
    latency_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    accounting_tokens_prompt: int
    accounting_tokens_completion: int
    accounting_native_tokens_prompt: int
    accounting_native_tokens_completion: int
    provider_generation_times_ms: tuple[int, ...]
    provider_upstream_latencies_ms: tuple[int, ...]
    route_http_statuses: tuple[int, ...]
    tool_calls: int
    tool_successes: int
    tool_errors: int
    actual_model_id: str | None
    actual_provider: str | None
    accounting_identities: tuple[tuple[str, str], ...]
    response_artifact_sha256: str | None


@dataclass(frozen=True)
class PairObservation:
    work_item: WorkItem
    pair_status: str
    admitted: bool
    attempted: bool
    finalized: bool
    source_artifact_sha256: str | None
    source_artifact_filename: str | None
    conditions: Mapping[str, ConditionObservation]
    incident_resolution: Mapping[str, Any] | None = None


def _condition_accounting(
    source_artifact: Mapping[str, Any] | None,
    condition: str,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    if source_artifact is None:
        return 0, 0, 0, []
    suffix = f":{condition}"
    events = [
        event
        for event in source_artifact.get("provider_attempt_events", [])
        if isinstance(event, Mapping) and str(event.get("arm_id") or "").endswith(suffix)
    ]
    requests = sum(event.get("event_type") == "request_started" for event in events)
    received = {
        str(event.get("generation_id"))
        for event in events
        if event.get("event_type") == "response_received" and event.get("generation_id")
    }
    reconciled: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") != "accounting_reconciled":
            continue
        generation_id = str(event.get("generation_id") or "")
        metadata = event.get("metadata")
        if not generation_id or not isinstance(metadata, Mapping):
            raise IntegrityError("condition accounting event has no generation metadata")
        if generation_id in reconciled:
            raise IntegrityError(f"generation was reconciled twice: {generation_id}")
        cost_micros = metadata.get("cost_micros")
        if (
            metadata.get("reconciled") is not True
            or not isinstance(cost_micros, int)
            or isinstance(cost_micros, bool)
            or cost_micros < 0
        ):
            raise IntegrityError(f"generation accounting is incomplete: {generation_id}")
        reconciled[generation_id] = dict(metadata)
    if received != set(reconciled):
        raise IntegrityError(
            f"received/reconciled generation mismatch for {condition}: "
            f"{sorted(received)} != {sorted(reconciled)}"
        )
    cost = sum(int(metadata["cost_micros"]) for metadata in reconciled.values())
    return requests, len(received), cost, list(reconciled.values())


def _condition_mcp_events(
    source_artifact: Mapping[str, Any] | None,
    condition: str,
) -> list[dict[str, Any]]:
    if source_artifact is None:
        return []
    suffix = f":{condition}"
    return [
        dict(event)
        for event in source_artifact.get("mcp_trace_events", [])
        if isinstance(event, Mapping) and str(event.get("arm_id") or "").endswith(suffix)
    ]


def _condition_route_http_statuses(
    source_artifact: Mapping[str, Any] | None,
    condition: str,
) -> tuple[int, ...]:
    if source_artifact is None:
        return ()
    suffix = f":{condition}"
    return tuple(
        int(event["http_status"])
        for event in source_artifact.get("provider_attempt_events", [])
        if isinstance(event, Mapping)
        and str(event.get("arm_id") or "").endswith(suffix)
        and event.get("event_type") == "request_rejected"
        and isinstance(event.get("http_status"), int)
    )


def _condition_errors(
    source_artifact: Mapping[str, Any] | None,
    condition: str,
) -> list[str]:
    if source_artifact is None:
        return []
    errors = source_artifact.get("errors")
    if not isinstance(errors, Mapping):
        return []
    return [
        str(value)
        for key, value in errors.items()
        if str(key) == condition or str(key).startswith(f"{condition}_")
    ]


def _outcome_class(
    *,
    normalized: bool,
    source_present: bool,
    finalized: bool,
    errors: Sequence[str],
) -> str:
    if normalized:
        return "normalized_response"
    if not source_present:
        return "not_yet_attempted"
    if not finalized and not errors:
        return "normalization_pending"
    lowered = " ".join(errors).lower()
    if "invalid final json" in lowered:
        return "invalid_final_json"
    if "openrouter returned no final choice" in lowered:
        return "openrouter_http_200_no_choice_without_generation_id"
    if "tool call remained invalid after one repair" in lowered:
        return "invalid_tool_arguments_after_repair"
    if "fan-out exceeded the per-round cap" in lowered:
        return "tool_fanout_cap_exceeded"
    if "openrouter request failed" in lowered or "httpstatuserror" in lowered:
        return "openrouter_http_error"
    if "identity" in lowered or "model mismatch" in lowered or "provider mismatch" in lowered:
        return "identity_mismatch"
    if "mcp" in lowered:
        return "mcp_error"
    if errors:
        return "other_provider_or_normalization_error"
    return "missing_normalized_response"


def _response_payload(path: Path) -> dict[str, Any]:
    verified = _verify_response_artifact(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("artifact_sha256") != verified.artifact_sha256:
        raise IntegrityError(f"response changed after verification: {path}")
    return value


def _pair_status(*, finalized: bool, admitted: bool, responses: int) -> str:
    if finalized:
        return {0: "failed", 1: "partial", 2: "complete"}[responses]
    if admitted:
        return "in_progress"
    return "pending"


def _pair_observations(
    work_items: Sequence[WorkItem],
    *,
    state: Any,
) -> tuple[list[PairObservation], dict[str, dict[str, Any]]]:
    response_payloads = {
        f"{work_item_id}:{condition}": _response_payload(response.path)
        for (work_item_id, condition), response in state.responses.items()
    }
    pairs: list[PairObservation] = []
    generation_metadata: dict[str, dict[str, Any]] = {}
    for work_item in work_items:
        source = state.sources.get(work_item.work_item_id)
        source_artifact = source.artifact if source is not None else None
        admitted = work_item.work_item_id in state.reservations or source is not None
        finalized = work_item.work_item_id in state.finalizations
        response_count = sum(
            (work_item.work_item_id, condition) in state.responses for condition in CONDITIONS
        )
        conditions: dict[str, ConditionObservation] = {}
        source_cost = 0
        for condition in CONDITIONS:
            request_count, generation_count, cost_micros, metadata = _condition_accounting(
                source_artifact,
                condition,
            )
            source_cost += cost_micros
            for generation in metadata:
                generation_id = str(generation.get("generation_id") or "")
                if generation_id in generation_metadata:
                    raise IntegrityError(
                        f"generation appears in multiple source arms: {generation_id}"
                    )
                generation_metadata[generation_id] = generation
            response_record = state.responses.get((work_item.work_item_id, condition))
            payload = response_payloads.get(f"{work_item.work_item_id}:{condition}")
            response = payload.get("response") if payload is not None else None
            if not isinstance(response, Mapping):
                response = None
            labels = (
                _structured_and_constraint_labels(
                    response,
                    prompt=work_item.task.prompt,
                    model_id=work_item.candidate.model_id,
                )
                if response is not None
                else {
                    "structured_valid": False,
                    "constraint_status": "not_evaluated_no_normalized_response",
                    "constraint_applicable": False,
                }
            )
            traces = response.get("tool_trace", []) if response is not None else []
            if not isinstance(traces, list):
                raise IntegrityError("normalized response tool trace is not a list")
            normalized_tool_errors = sum(
                isinstance(trace, Mapping) and bool(trace.get("is_error")) for trace in traces
            )
            mcp_events = _condition_mcp_events(source_artifact, condition)
            mcp_errors = sum(bool(event.get("is_error")) for event in mcp_events)
            if response is not None and (
                len(traces) != len(mcp_events) or normalized_tool_errors != mcp_errors
            ):
                raise IntegrityError(
                    "normalized tool trace does not match the source MCP journal events"
                )
            normalized_cost = (
                int(payload["cost"]["actual_cost_micros"]) if payload is not None else 0
            )
            outcome_class = _outcome_class(
                normalized=response is not None,
                source_present=source is not None,
                finalized=finalized,
                errors=_condition_errors(source_artifact, condition),
            )
            conditions[condition] = ConditionObservation(
                condition=condition,
                provider_attempted=request_count > 0,
                request_count=request_count,
                provider_generation_count=generation_count,
                reconciled_generation_count=generation_count,
                actual_cost_micros=cost_micros,
                normalized_response=response is not None,
                outcome_class=outcome_class,
                normalized_response_cost_micros=normalized_cost,
                structured_valid=bool(labels["structured_valid"]),
                constraint_status=str(labels["constraint_status"]),
                constraint_applicable=bool(labels["constraint_applicable"]),
                latency_ms=(
                    int(response["latency_ms"])
                    if response is not None and isinstance(response.get("latency_ms"), int)
                    else None
                ),
                prompt_tokens=(
                    int(response["prompt_tokens"])
                    if response is not None and isinstance(response.get("prompt_tokens"), int)
                    else None
                ),
                completion_tokens=(
                    int(response["completion_tokens"])
                    if response is not None and isinstance(response.get("completion_tokens"), int)
                    else None
                ),
                reasoning_tokens=(
                    int(response["reasoning_tokens"])
                    if response is not None and isinstance(response.get("reasoning_tokens"), int)
                    else None
                ),
                accounting_tokens_prompt=sum(
                    int(generation.get("tokens_prompt") or 0) for generation in metadata
                ),
                accounting_tokens_completion=sum(
                    int(generation.get("tokens_completion") or 0) for generation in metadata
                ),
                accounting_native_tokens_prompt=sum(
                    int(generation.get("native_tokens_prompt") or 0) for generation in metadata
                ),
                accounting_native_tokens_completion=sum(
                    int(generation.get("native_tokens_completion") or 0) for generation in metadata
                ),
                provider_generation_times_ms=tuple(
                    int(generation["generation_time_ms"])
                    for generation in metadata
                    if isinstance(generation.get("generation_time_ms"), int)
                ),
                provider_upstream_latencies_ms=tuple(
                    int(generation["upstream_latency_ms"])
                    for generation in metadata
                    if isinstance(generation.get("upstream_latency_ms"), int)
                ),
                route_http_statuses=_condition_route_http_statuses(source_artifact, condition),
                tool_calls=len(mcp_events),
                tool_successes=len(mcp_events) - mcp_errors,
                tool_errors=mcp_errors,
                actual_model_id=(
                    str(response.get("actual_model_id"))
                    if response is not None and response.get("actual_model_id")
                    else None
                ),
                actual_provider=(
                    str(response.get("actual_provider"))
                    if response is not None and response.get("actual_provider")
                    else None
                ),
                accounting_identities=tuple(
                    sorted(
                        {
                            (str(generation.get("model")), str(generation.get("provider")))
                            for generation in metadata
                            if generation.get("model") and generation.get("provider")
                        }
                    )
                ),
                response_artifact_sha256=(
                    response_record.artifact_sha256 if response_record is not None else None
                ),
            )
        if source is not None:
            expected_cost = int(source.exposure.actual_cost_usd * Decimal(1_000_000))
            if source_cost != expected_cost:
                raise IntegrityError(
                    f"source condition costs do not reconcile for {source.path}: "
                    f"{source_cost} != {expected_cost}"
                )
        pairs.append(
            PairObservation(
                work_item=work_item,
                pair_status=_pair_status(
                    finalized=finalized,
                    admitted=admitted,
                    responses=response_count,
                ),
                admitted=admitted,
                attempted=source is not None,
                finalized=finalized,
                source_artifact_sha256=(source.artifact_sha256 if source else None),
                source_artifact_filename=(source.path.name if source else None),
                conditions=conditions,
                incident_resolution=(
                    {
                        "filename": resolution.path.name,
                        "artifact_sha256": resolution.artifact_sha256,
                        "ledger_event_sha256": resolution.ledger_event_sha256,
                        "affected_condition": resolution.affected_condition,
                        "normalizable_conditions": list(resolution.normalizable_conditions),
                        "provider_reconciled_actual_cost_usd": _decimal_text(
                            resolution.provider_reconciled_actual_cost_usd
                        ),
                        "conservative_budget_exposure_usd": _decimal_text(
                            resolution.conservative_budget_exposure_usd
                        ),
                        "provider_cost_exact_for_unidentified_response": False,
                        "safe_to_replay": False,
                    }
                    if (resolution := state.incident_resolutions.get(work_item.work_item_id))
                    is not None
                    else None
                ),
            )
        )
    return pairs, generation_metadata


def summarize_slice(
    pairs: Sequence[PairObservation],
    condition: str,
) -> dict[str, Any]:
    """Summarize one explicit model/family/condition slice."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    condition_rows = [pair.conditions[condition] for pair in pairs]
    statuses = Counter(pair.pair_status for pair in pairs)
    normalized = [row for row in condition_rows if row.normalized_response]
    applicable = [row for row in normalized if row.constraint_applicable]
    constraint_counts = Counter(row.constraint_status for row in normalized)
    outcome_counts = Counter(row.outcome_class for row in condition_rows)
    route_http_status_counts = Counter(
        status for row in condition_rows for status in row.route_http_statuses
    )
    latency = [row.latency_ms for row in normalized if row.latency_ms is not None]
    prompt_tokens = [row.prompt_tokens for row in normalized if row.prompt_tokens is not None]
    completion_tokens = [
        row.completion_tokens for row in normalized if row.completion_tokens is not None
    ]
    reasoning_tokens = [
        row.reasoning_tokens for row in normalized if row.reasoning_tokens is not None
    ]
    provider_generation_times = [
        value for row in condition_rows for value in row.provider_generation_times_ms
    ]
    provider_upstream_latencies = [
        value for row in condition_rows for value in row.provider_upstream_latencies_ms
    ]
    cost_micros = sum(row.actual_cost_micros for row in condition_rows)
    normalized_cost_micros = sum(row.normalized_response_cost_micros for row in normalized)
    return {
        "condition": condition,
        "pairs": {
            "expected": len(pairs),
            "admitted": sum(pair.admitted for pair in pairs),
            "attempted": sum(pair.attempted for pair in pairs),
            "finalized": sum(pair.finalized for pair in pairs),
            "complete": statuses["complete"],
            "partial": statuses["partial"],
            "failed": statuses["failed"],
            "in_progress": statuses["in_progress"],
            "pending": statuses["pending"],
        },
        "arms": {
            "expected": len(pairs),
            "provider_attempted": sum(row.provider_attempted for row in condition_rows),
            "request_count": sum(row.request_count for row in condition_rows),
            "normalized_responses": len(normalized),
            "structured_valid": sum(row.structured_valid for row in normalized),
            "structured_invalid_within_normalized": len(normalized)
            - sum(row.structured_valid for row in normalized),
            "attempted_without_normalized_response": sum(
                row.provider_attempted and not row.normalized_response for row in condition_rows
            ),
            "not_yet_provider_attempted": sum(not row.provider_attempted for row in condition_rows),
            "outcome_class_counts": {key: outcome_counts[key] for key in sorted(outcome_counts)},
        },
        "constraint_acknowledgement": {
            "validator_version": VALIDATOR_VERSION,
            "label": "constraint_acknowledgement",
            "scope": "acknowledgement only; not substantive task compliance",
            "applicable_n": len(applicable),
            "pass": constraint_counts["pass"],
            "warn": constraint_counts["warn"],
            "not_applicable": constraint_counts["not_applicable"],
            "not_evaluated": len(pairs) - len(normalized),
        },
        "tools": {
            "attempted_arms_with_tool_use": sum(row.tool_calls > 0 for row in condition_rows),
            "normalized_arms_with_tool_use": sum(row.tool_calls > 0 for row in normalized),
            "calls": sum(row.tool_calls for row in condition_rows),
            "successful_calls": sum(row.tool_successes for row in condition_rows),
            "error_calls": sum(row.tool_errors for row in condition_rows),
            "normalized_trace_calls": sum(row.tool_calls for row in normalized),
            "normalized_trace_successful_calls": sum(row.tool_successes for row in normalized),
            "normalized_trace_error_calls": sum(row.tool_errors for row in normalized),
        },
        "provider_errors": {
            "route_http_status_counts": {
                str(status): route_http_status_counts[status]
                for status in sorted(route_http_status_counts)
            }
        },
        "latency_ms": {
            **_distribution(latency),
            "basis": "normalized_response_end_to_end",
            "provider_generation": _distribution(provider_generation_times),
            "provider_upstream": _distribution(provider_upstream_latencies),
        },
        "tokens": {
            "normalized_response_n": len(normalized),
            "normalized_prompt_n": len(prompt_tokens),
            "normalized_prompt_total": sum(prompt_tokens),
            "normalized_completion_n": len(completion_tokens),
            "normalized_completion_total": sum(completion_tokens),
            "normalized_reasoning_n": len(reasoning_tokens),
            "normalized_reasoning_total": sum(reasoning_tokens),
            "accounting_generation_n": sum(
                row.reconciled_generation_count for row in condition_rows
            ),
            "accounting_openrouter_prompt_total": sum(
                row.accounting_tokens_prompt for row in condition_rows
            ),
            "accounting_openrouter_completion_total": sum(
                row.accounting_tokens_completion for row in condition_rows
            ),
            "accounting_native_prompt_total": sum(
                row.accounting_native_tokens_prompt for row in condition_rows
            ),
            "accounting_native_completion_total": sum(
                row.accounting_native_tokens_completion for row in condition_rows
            ),
        },
        "cost": {
            "provider_generations": sum(row.provider_generation_count for row in condition_rows),
            "reconciled_provider_generations": sum(
                row.reconciled_generation_count for row in condition_rows
            ),
            "actual_cost_micros": cost_micros,
            "actual_cost_usd": _decimal_text(Decimal(cost_micros) / Decimal(1_000_000)),
            "normalized_response_cost_micros": normalized_cost_micros,
            "failed_or_non_normalized_cost_micros": cost_micros - normalized_cost_micros,
        },
    }


def classify_collection_state(
    pair_statuses: Mapping[str, int],
    *,
    expected_pairs: int,
    active_journal_count: int,
) -> str:
    terminal_pair_count = sum(
        int(pair_statuses.get(status, 0)) for status in ("complete", "partial", "failed")
    )
    if terminal_pair_count == expected_pairs and active_journal_count == 0:
        return "complete"
    if active_journal_count or int(pair_statuses.get("in_progress", 0)):
        return "active_checkpoint"
    return "partial_checkpoint"


def _pair_payload(pair: PairObservation) -> dict[str, Any]:
    return {
        "ordinal": pair.work_item.ordinal,
        "work_item_id": pair.work_item.work_item_id,
        "task_id": pair.work_item.task.public_id,
        "task_family": pair.work_item.task.family,
        "prompt_sha256": pair.work_item.task.prompt_sha256,
        "model_id": pair.work_item.candidate.model_id,
        "provider_tag": pair.work_item.candidate.provider_tag,
        "pair_status": pair.pair_status,
        "admitted": pair.admitted,
        "attempted": pair.attempted,
        "finalized": pair.finalized,
        "source_artifact_filename": pair.source_artifact_filename,
        "source_artifact_sha256": pair.source_artifact_sha256,
        "incident_resolution": pair.incident_resolution,
        "conditions": {
            condition: {
                "provider_attempted": observation.provider_attempted,
                "request_count": observation.request_count,
                "provider_generation_count": observation.provider_generation_count,
                "reconciled_generation_count": observation.reconciled_generation_count,
                "actual_cost_micros": observation.actual_cost_micros,
                "normalized_response": observation.normalized_response,
                "outcome_class": observation.outcome_class,
                "normalized_response_cost_micros": (observation.normalized_response_cost_micros),
                "structured_valid": observation.structured_valid,
                "constraint_acknowledgement": observation.constraint_status,
                "latency_ms": observation.latency_ms,
                "prompt_tokens": observation.prompt_tokens,
                "completion_tokens": observation.completion_tokens,
                "reasoning_tokens": observation.reasoning_tokens,
                "accounting_tokens_prompt": observation.accounting_tokens_prompt,
                "accounting_tokens_completion": observation.accounting_tokens_completion,
                "accounting_native_tokens_prompt": (observation.accounting_native_tokens_prompt),
                "accounting_native_tokens_completion": (
                    observation.accounting_native_tokens_completion
                ),
                "provider_generation_times_ms": list(observation.provider_generation_times_ms),
                "provider_upstream_latencies_ms": list(observation.provider_upstream_latencies_ms),
                "route_http_statuses": list(observation.route_http_statuses),
                "tool_calls": observation.tool_calls,
                "tool_successes": observation.tool_successes,
                "tool_errors": observation.tool_errors,
                "actual_model_id": observation.actual_model_id,
                "actual_provider": observation.actual_provider,
                "accounting_identities": [
                    {"model_id": model_id, "provider": provider}
                    for model_id, provider in observation.accounting_identities
                ],
                "response_artifact_sha256": observation.response_artifact_sha256,
            }
            for condition, observation in pair.conditions.items()
        },
    }


def _journal_payload(state: JournalRecoveryState) -> dict[str, Any]:
    return {
        "filename": state.path.name,
        "sha256": state.journal_sha256,
        "head_entry_sha256": state.head_entry_sha256,
        "entry_count": state.entry_count,
        "run_id": state.run_id,
        "finalized": state.finalized,
        "generation_ids": list(state.generation_ids),
        "unreconciled_generation_ids": list(state.unreconciled_generation_ids),
        "uncertain_attempt_ids": list(state.uncertain_attempt_ids),
        "mcp_trace_count": state.mcp_trace_count,
        "recovery_action": state.recovery_action,
        "safe_to_replay": state.safe_to_replay,
    }


def _latest_observed_at(
    ledger: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> str | None:
    values = [
        str(value)
        for value in [
            *(entry.get("recorded_at") for entry in ledger),
            *(summary.get("completed_at") for summary in summaries),
        ]
        if value
    ]
    return max(values) if values else None


def _verify_terminal_runner_summary(
    summary: Mapping[str, Any],
    *,
    state: Any,
    pairs: Sequence[PairObservation],
    overall: Mapping[str, Mapping[str, Any]],
) -> None:
    coverage = summary.get("coverage_and_reliability", {}).get("overall", {})
    expected = len(pairs)
    comparisons = {
        "expected_pairs": expected,
        "expected_arms": expected * len(CONDITIONS),
        "source_attempts": sum(pair.attempted for pair in pairs),
        "finalized_pairs": sum(pair.finalized for pair in pairs),
        "epicure_off_responses": overall["epicure_off"]["arms"]["normalized_responses"],
        "epicure_on_responses": overall["epicure_on"]["arms"]["normalized_responses"],
        "complete_pairs": overall["epicure_off"]["pairs"]["complete"],
        "failed_or_partial_pairs": (
            int(overall["epicure_off"]["pairs"]["failed"])
            + int(overall["epicure_off"]["pairs"]["partial"])
        ),
        "epicure_on_tool_used": overall["epicure_on"]["tools"]["normalized_arms_with_tool_use"],
    }
    for field, expected_value in comparisons.items():
        if int(coverage.get(field, -1)) != int(expected_value):
            raise IntegrityError(
                f"terminal runner summary {field} disagrees with immutable records: "
                f"{coverage.get(field)} != {expected_value}"
            )
    budget = summary.get("budget", {})
    if Decimal(str(budget.get("dataset_actual_cost_usd", "-1"))) != (state.dataset_actual_cost_usd):
        raise IntegrityError("terminal runner summary dataset cost disagrees with artifacts")
    if Decimal(str(budget.get("dataset_source_exposure_usd", "-1"))) != (
        state.dataset_source_exposure_usd
    ):
        raise IntegrityError(
            "terminal runner summary conservative source exposure disagrees with artifacts"
        )
    ledger = summary.get("ledger", {})
    expected_head = state.ledger[-1]["entry_sha256"] if state.ledger else None
    if (
        int(ledger.get("entry_count", -1)) != len(state.ledger)
        or ledger.get("head_entry_sha256") != expected_head
    ):
        raise IntegrityError("terminal runner summary ledger head disagrees with current ledger")


def _validate_slice_invariants(slice_value: Mapping[str, Any]) -> None:
    pairs = slice_value["pairs"]
    arms = slice_value["arms"]
    tools = slice_value["tools"]
    cost = slice_value["cost"]
    expected = int(pairs["expected"])
    finalized = int(pairs["finalized"])
    if sum(int(pairs[key]) for key in ("complete", "partial", "failed")) != finalized:
        raise IntegrityError("complete/partial/failed pair counts do not equal finalized pairs")
    if finalized + int(pairs["in_progress"]) + int(pairs["pending"]) != expected:
        raise IntegrityError("pair status categories do not partition the expected workload")
    attempted = int(arms["provider_attempted"])
    normalized = int(arms["normalized_responses"])
    structured = int(arms["structured_valid"])
    if not 0 <= structured <= normalized <= attempted <= expected:
        raise IntegrityError("arm response denominators are not nested")
    if (
        int(arms["attempted_without_normalized_response"]) != attempted - normalized
        or int(arms["not_yet_provider_attempted"]) != expected - attempted
    ):
        raise IntegrityError("arm missingness categories do not match their denominators")
    outcome_counts = arms["outcome_class_counts"]
    if (
        sum(int(value) for value in outcome_counts.values()) != expected
        or int(outcome_counts.get("normalized_response", 0)) != normalized
    ):
        raise IntegrityError("arm outcome classes do not partition their denominator")
    if int(tools["successful_calls"]) + int(tools["error_calls"]) != int(tools["calls"]):
        raise IntegrityError("tool success/error counts do not equal all tool calls")
    if int(cost["provider_generations"]) != int(cost["reconciled_provider_generations"]):
        raise IntegrityError("provider generation reconciliation is incomplete")
    if int(cost["normalized_response_cost_micros"]) + int(
        cost["failed_or_non_normalized_cost_micros"]
    ) != int(cost["actual_cost_micros"]):
        raise IntegrityError("normalized and non-normalized cost do not sum to actual cost")


def validate_payload_invariants(payload: Mapping[str, Any]) -> None:
    """Recompute high-impact denominators and subtotals before publication."""

    workload = payload["workload"]
    progress = payload["progress"]
    status_counts = progress["pair_status_counts"]
    expected_pairs = int(workload["expected_pairs"])
    if sum(int(value) for value in status_counts.values()) != expected_pairs:
        raise IntegrityError("top-level pair statuses do not partition the workload")
    family_assignments = workload["pair_assignments_by_task_family"]
    family_counts = [int(value) for value in family_assignments.values()]
    if (
        set(family_assignments) != set(TASK_FAMILIES)
        or sum(family_counts) != expected_pairs
        or max(family_counts) - min(family_counts) > 1
    ):
        raise IntegrityError("task-family assignments are not a complete balanced partition")
    if int(status_counts["complete"]) + int(status_counts["partial"]) + int(
        status_counts["failed"]
    ) != int(progress["finalized_pairs"]):
        raise IntegrityError("top-level finalized pair count is inconsistent")
    pairs = payload["pair_records"]
    work_item_ids = [str(pair["work_item_id"]) for pair in pairs]
    if len(pairs) != expected_pairs or len(set(work_item_ids)) != expected_pairs:
        raise IntegrityError("pair records do not have one unique row per work item")

    overall = payload["overall_by_condition"]
    for condition in CONDITIONS:
        _validate_slice_invariants(overall[condition])
        model_slices = [model["conditions"][condition] for model in payload["models"]]
        family_slices = [payload["by_task_family"][family][condition] for family in TASK_FAMILIES]
        for field in ("normalized_responses", "structured_valid"):
            expected_value = int(overall[condition]["arms"][field])
            if sum(int(value["arms"][field]) for value in model_slices) != expected_value:
                raise IntegrityError(f"model {condition} {field} subtotal does not reconcile")
            if sum(int(value["arms"][field]) for value in family_slices) != expected_value:
                raise IntegrityError(f"family {condition} {field} subtotal does not reconcile")
        for field in ("provider_generations", "actual_cost_micros"):
            expected_value = int(overall[condition]["cost"][field])
            if sum(int(value["cost"][field]) for value in model_slices) != expected_value:
                raise IntegrityError(f"model {condition} {field} subtotal does not reconcile")
            if sum(int(value["cost"][field]) for value in family_slices) != expected_value:
                raise IntegrityError(f"family {condition} {field} subtotal does not reconcile")
        for value in [*model_slices, *family_slices]:
            _validate_slice_invariants(value)

    if sum(
        int(overall[condition]["arms"]["normalized_responses"]) for condition in CONDITIONS
    ) != int(progress["normalized_responses"]):
        raise IntegrityError("condition response totals do not equal verified response artifacts")
    if sum(
        int(overall[condition]["cost"]["actual_cost_micros"]) for condition in CONDITIONS
    ) != int(payload["cost"]["dataset_actual_cost_micros"]):
        raise IntegrityError("condition cost totals do not equal dataset actual cost")
    if sum(
        int(overall[condition]["cost"]["provider_generations"]) for condition in CONDITIONS
    ) != int(payload["cost"]["provider_generation_count"]):
        raise IntegrityError("condition generation totals do not equal dataset generations")
    cost = payload["cost"]
    if "dataset_source_budget_exposure_usd" in cost:
        actual = Decimal(str(cost["dataset_actual_cost_usd"]))
        source_exposure = Decimal(str(cost["dataset_source_budget_exposure_usd"]))
        increment = Decimal(str(cost["resolved_no_id_exposure_increment_usd"]))
        if source_exposure < actual or increment < 0:
            raise IntegrityError(
                "conservative source exposure cannot be below provider-reconciled actual cost"
            )
        incidents = int(cost["resolved_no_id_incident_count"])
        if incidents and (
            cost.get("all_provider_attempts_have_generation_ids") is not False
            or cost.get("provider_cost_exact_for_all_attempts") is not False
        ):
            raise IntegrityError(
                "a no-ID incident cannot claim complete generation identity or exact cost"
            )
        resolution_inputs = payload.get("inputs", {}).get("source_incident_resolutions")
        if resolution_inputs is not None and len(resolution_inputs) != incidents:
            raise IntegrityError(
                "source incident resolution inputs do not match the cost incident count"
            )
    expected_cube_size = (
        int(workload["model_count"]) * int(workload["task_family_count"]) * len(CONDITIONS)
    )
    cube = payload["model_task_family_condition_cube"]
    cube_keys = {(row["model_id"], row["task_family"], row["condition"]) for row in cube}
    if len(cube) != expected_cube_size or len(cube_keys) != expected_cube_size:
        raise IntegrityError("model/family/condition cube is incomplete or duplicated")
    if payload["collection_state"] == "complete" and (
        int(status_counts["complete"])
        + int(status_counts["partial"])
        + int(status_counts["failed"])
        != expected_pairs
        or int(progress["active_journals"]) != 0
    ):
        raise IntegrityError("complete collection still has unfinished work")


def build_payload(
    evaluation_root: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Verify and aggregate the current on-disk checkpoint without provider calls."""

    flavourbench_root = evaluation_root / "flavourbench"
    artifact_root = flavourbench_root / "artifacts"
    checkpoint_inventory = _checkpoint_inventory(artifact_root)
    manifest_path = manifest_path or (
        artifact_root
        / "manifests"
        / f"flavourbench-openrouter-unranked-{CURRENT_DATASET_MANIFEST_SHA256}.json"
    )
    manifest = load_candidate_manifest(
        manifest_path,
        expected_digest=CURRENT_DATASET_MANIFEST_SHA256,
    )
    candidates = select_candidates(manifest, [])
    cohort_by_slot = {
        str(model.get("slot", {}).get("slot_id")): model.get("slot", {}).get("cohort")
        for model in manifest.get("models", [])
        if isinstance(model, Mapping)
    }
    selected_tasks, registry_sha256 = select_balanced_tasks(tasks_per_family=3)
    policy = ExecutionPolicy()
    work_items = build_balanced_work_items(
        manifest_sha256=CURRENT_DATASET_MANIFEST_SHA256,
        task_registry_digest=registry_sha256,
        selected_tasks=selected_tasks,
        candidates=candidates,
        execution_policy=policy,
        assignments_per_model=10,
    )
    source_directory = artifact_root / "real-exploratory" / "source-runs"
    response_directory = artifact_root / "real-exploratory" / "responses"
    dataset_ledger_path = artifact_root / "real-exploratory" / "ledger.jsonl"
    state = _load_state(
        prior_artifact_directory=artifact_root / "live-smoke",
        prior_corrections_directory=artifact_root / "corrections",
        prior_reservation_ledger_path=artifact_root / "frontier-contract" / "ledger.jsonl",
        source_directory=source_directory,
        source_corrections_directory=artifact_root / "real-exploratory" / "corrections",
        response_directory=response_directory,
        ledger_path=dataset_ledger_path,
    )
    _validate_state_against_workload(state, work_items)
    pairs, generation_metadata = _pair_observations(work_items, state=state)

    journals = scan_recovery_journals(source_directory)
    linked_journal_hashes = {
        str(source.artifact.get("run_journal", {}).get("sha256"))
        for source in state.sources.values()
    }
    finalized_journals = [journal for journal in journals if journal.finalized]
    unlinked_finalized = [
        journal
        for journal in finalized_journals
        if journal.journal_sha256 not in linked_journal_hashes
    ]
    if len(linked_journal_hashes - {journal.journal_sha256 for journal in journals}) > 0:
        raise IntegrityError("a source artifact refers to a journal outside the verified set")

    summary_directory = artifact_root / "real-exploratory" / "summaries"
    summary_records: list[dict[str, Any]] = []
    for path in sorted(summary_directory.glob("*.json")):
        summary = _verify_content_addressed_summary(path)
        summary_records.append(summary | {"_filename": path.name})

    model_order = [candidate.model_id for candidate in candidates]
    models: list[dict[str, Any]] = []
    for candidate in candidates:
        model_pairs = [
            pair for pair in pairs if pair.work_item.candidate.model_id == candidate.model_id
        ]
        identities = sorted(
            {
                identity
                for pair in model_pairs
                for observation in pair.conditions.values()
                for identity in (
                    *observation.accounting_identities,
                    *(
                        ((observation.actual_model_id, observation.actual_provider),)
                        if observation.actual_model_id and observation.actual_provider
                        else ()
                    ),
                )
            }
        )
        models.append(
            {
                "model_id": candidate.model_id,
                "display_name": DISPLAY_NAMES.get(candidate.model_id, candidate.model_id),
                "slot_id": candidate.slot_id,
                "cohort": cohort_by_slot.get(candidate.slot_id),
                "provider_tag": candidate.provider_tag,
                "actual_identities": [
                    {"model_id": model_id, "provider": provider}
                    for model_id, provider in identities
                ],
                "conditions": {
                    condition: summarize_slice(model_pairs, condition) for condition in CONDITIONS
                },
                "task_families": {
                    family: {
                        condition: summarize_slice(
                            [pair for pair in model_pairs if pair.work_item.task.family == family],
                            condition,
                        )
                        for condition in CONDITIONS
                    }
                    for family in TASK_FAMILIES
                },
            }
        )

    cube = [
        {
            "model_id": model_id,
            "task_family": family,
            **summarize_slice(
                [
                    pair
                    for pair in pairs
                    if pair.work_item.candidate.model_id == model_id
                    and pair.work_item.task.family == family
                ],
                condition,
            ),
        }
        for model_id in model_order
        for family in TASK_FAMILIES
        for condition in CONDITIONS
    ]
    overall = {condition: summarize_slice(pairs, condition) for condition in CONDITIONS}
    by_task_family = {
        family: {
            condition: summarize_slice(
                [pair for pair in pairs if pair.work_item.task.family == family],
                condition,
            )
            for condition in CONDITIONS
        }
        for family in TASK_FAMILIES
    }
    pair_statuses = Counter(pair.pair_status for pair in pairs)
    active_journals = [journal for journal in journals if not journal.finalized]
    collection_state = classify_collection_state(
        pair_statuses,
        expected_pairs=len(pairs),
        active_journal_count=len(active_journals),
    )

    source_records = [
        {
            "filename": source.path.name,
            "sha256": source.artifact_sha256,
            "work_item_id": source.work_item_id,
            "run_id": source.artifact.get("run_id"),
            "journal_sha256": source.artifact.get("run_journal", {}).get("sha256"),
        }
        for source in sorted(state.sources.values(), key=lambda item: item.path.name)
    ]
    response_records = [
        {
            "filename": response.path.name,
            "sha256": response.artifact_sha256,
            "work_item_id": response.work_item_id,
            "condition": response.condition,
        }
        for response in sorted(state.responses.values(), key=lambda item: item.path.name)
    ]
    resolution_records = [
        {
            "filename": resolution.path.name,
            "sha256": resolution.artifact_sha256,
            "work_item_id": resolution.work_item_id,
            "source_artifact_sha256": resolution.source_artifact_sha256,
            "ledger_event_sha256": resolution.ledger_event_sha256,
            "affected_condition": resolution.affected_condition,
            "provider_cost_exact_for_unidentified_response": False,
            "safe_to_replay": False,
        }
        for resolution in sorted(
            state.incident_resolutions.values(), key=lambda item: item.path.name
        )
    ]
    ledger_head = state.ledger[-1]["entry_sha256"] if state.ledger else None
    summary_inputs = [
        {
            "filename": str(summary["_filename"]),
            "sha256": summary["content_address"]["digest"],
            "completed_at": summary.get("completed_at"),
        }
        for summary in summary_records
    ]
    dataset_cost_micros = sum(row["cost"]["actual_cost_micros"] for row in overall.values())
    expected_dataset_cost_micros = int(state.dataset_actual_cost_usd * Decimal(1_000_000))
    if dataset_cost_micros != expected_dataset_cost_micros:
        raise IntegrityError(
            f"aggregate condition costs do not equal verified dataset cost: "
            f"{dataset_cost_micros} != {expected_dataset_cost_micros}"
        )
    resolved_no_id_allowance_usd = sum(
        (
            resolution.conservative_budget_exposure_usd
            for resolution in state.incident_resolutions.values()
        ),
        Decimal(0),
    )
    resolved_no_id_provider_actual_usd = sum(
        (
            resolution.provider_reconciled_actual_cost_usd
            for resolution in state.incident_resolutions.values()
        ),
        Decimal(0),
    )
    resolved_no_id_exposure_increment_usd = (
        resolved_no_id_allowance_usd - resolved_no_id_provider_actual_usd
    )
    if resolved_no_id_exposure_increment_usd < 0:
        raise IntegrityError("resolved no-ID exposure is below provider-reconciled actual cost")

    terminal_summaries = [
        summary
        for summary in summary_records
        if summary.get("mode") == "execute"
        and int(
            summary.get("coverage_and_reliability", {}).get("overall", {}).get("finalized_pairs", 0)
        )
        == len(pairs)
    ]
    terminal_summary = (
        max(terminal_summaries, key=lambda value: str(value.get("completed_at")))
        if terminal_summaries
        else None
    )
    if terminal_summary is not None:
        _verify_terminal_runner_summary(
            terminal_summary,
            state=state,
            pairs=pairs,
            overall=overall,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "real_openrouter_epicure_exploratory_operational",
        "route": ROUTE_LABEL,
        "generation_evidence": {
            "synthetic": False,
            "model_api": "OpenRouter",
            "transport_gateway": "Cloudflare AI Gateway",
            "epicure_condition": "real Epicure MCP execution on the private Compose network",
            "aggregation_network_calls": 0,
            "scope": (
                "the aggregate verifies immutable real-call artifacts; rebuilding the "
                "aggregate itself sends no model or MCP requests"
            ),
        },
        "official": False,
        "rank_eligible": False,
        "research_result": False,
        "collection_state": collection_state,
        "observed_through": _latest_observed_at(state.ledger, summary_records),
        "human_judgments": {
            "public": 0,
            "expert": 0,
            "preference_estimate": None,
            "bradley_terry_rating": None,
            "epicure_uplift_estimate": None,
        },
        "workload": {
            "manifest_sha256": CURRENT_DATASET_MANIFEST_SHA256,
            "task_registry_sha256": registry_sha256,
            "execution_policy_sha256": policy.sha256,
            "model_count": len(candidates),
            "task_family_count": len(TASK_FAMILIES),
            "expected_pairs": len(pairs),
            "expected_arms": len(pairs) * len(CONDITIONS),
            "conditions": list(CONDITIONS),
            "distinct_candidate_tasks": len(selected_tasks),
            "assignments_per_model": 10,
            "task_review_status": "candidate_not_confirmatory",
            "schedule": "model-family diagonal round robin",
            "pair_assignments_by_task_family": {
                family: sum(pair.work_item.task.family == family for pair in pairs)
                for family in TASK_FAMILIES
            },
        },
        "progress": {
            "pair_status_counts": {
                status: pair_statuses[status]
                for status in ("complete", "partial", "failed", "in_progress", "pending")
            },
            "source_attempted_pairs": sum(pair.attempted for pair in pairs),
            "admitted_pairs": sum(pair.admitted for pair in pairs),
            "finalized_pairs": sum(pair.finalized for pair in pairs),
            "normalized_responses": len(state.responses),
            "verified_source_artifacts": len(state.sources),
            "verified_response_artifacts": len(state.responses),
            "verified_finalized_journals": len(finalized_journals),
            "active_journals": len(active_journals),
            "unlinked_finalized_journals_pending_source": len(unlinked_finalized),
        },
        "cost": {
            "currency": "USD",
            "authorized_hard_cap_usd": _decimal_text(AUTHORIZED_TOTAL_CAP_USD),
            "admission_ceiling_usd": _decimal_text(
                AUTHORIZED_TOTAL_CAP_USD * DEFAULT_ADMISSION_FRACTION
            ),
            "dataset_actual_cost_micros": dataset_cost_micros,
            "dataset_actual_cost_usd": _decimal_text(
                Decimal(dataset_cost_micros) / Decimal(1_000_000)
            ),
            "provider_generation_count": len(generation_metadata),
            "reconciled_provider_generation_count": len(generation_metadata),
            "all_provider_generations_reconciled": True,
            "all_identified_provider_generations_reconciled": True,
            "all_provider_attempts_have_generation_ids": not bool(state.incident_resolutions)
            and state.unresolved_dataset_source_reserve_usd == 0,
            "provider_cost_exact_for_all_attempts": not bool(state.incident_resolutions)
            and state.unresolved_dataset_source_reserve_usd == 0,
            "prior_effective_exposure_usd": _decimal_text(state.prior_effective_exposure_usd),
            "active_frontier_reservations_usd": _decimal_text(state.prior_active_reservation_usd),
            "active_dataset_reservations_without_source_usd": _decimal_text(
                state.orphan_reservation_usd
            ),
            "unresolved_dataset_source_reserve_usd": _decimal_text(
                state.unresolved_dataset_source_reserve_usd
            ),
            "dataset_source_budget_exposure_usd": _decimal_text(state.dataset_source_exposure_usd),
            "resolved_no_id_incident_count": len(state.incident_resolutions),
            "resolved_no_id_full_allowance_usd": _decimal_text(resolved_no_id_allowance_usd),
            "resolved_no_id_provider_reconciled_actual_usd": _decimal_text(
                resolved_no_id_provider_actual_usd
            ),
            "resolved_no_id_exposure_increment_usd": _decimal_text(
                resolved_no_id_exposure_increment_usd
            ),
            "actual_verified_exposure_usd": _decimal_text(
                state.prior_effective_exposure_usd + state.dataset_actual_cost_usd
            ),
            "conservative_total_exposure_usd": _decimal_text(state.total_exposure_usd),
            "remaining_hard_cap_usd": _decimal_text(
                AUTHORIZED_TOTAL_CAP_USD - state.total_exposure_usd
            ),
        },
        "overall_by_condition": overall,
        "by_task_family": by_task_family,
        "models": models,
        "model_task_family_condition_cube": cube,
        "pair_records": [_pair_payload(pair) for pair in pairs],
        "verification": {
            "all_checks_passed": True,
            "source_content_addresses_verified": len(state.sources),
            "response_content_addresses_verified": len(state.responses),
            "journal_hash_chains_verified": len(journals),
            "dataset_ledger_hash_chain_verified": True,
            "dataset_ledger_entry_count": len(state.ledger),
            "dataset_ledger_head_sha256": ledger_head,
            "dataset_ledger_file_sha256": (checkpoint_inventory["dataset_ledger_file_sha256"]),
            "checkpoint_inventory_sha256": _sha256(checkpoint_inventory),
            "checkpoint_stable_during_aggregation": True,
            "summary_content_addresses_verified": len(summary_records),
            "source_incident_resolutions_verified": len(state.incident_resolutions),
            "manifest_content_address_verified": True,
            "workload_and_finalization_links_verified": True,
            "generation_costs_reconciled": True,
            "terminal_runner_summary_verified": terminal_summary is not None,
            "terminal_runner_summary_sha256": (
                terminal_summary["content_address"]["digest"]
                if terminal_summary is not None
                else None
            ),
        },
        "data_quality": {
            "assessment": (
                "ready_as_unranked_operational_evidence_with_caveats"
                if collection_state == "complete"
                else "integrity_valid_checkpoint_not_publishable"
            ),
            "intended_grain": (
                "one frozen model-task work item with two separately reported conditions"
            ),
            "primary_keys": {
                "pair": "work_item_id",
                "response": ["work_item_id", "condition"],
                "provider_generation": "generation_id",
            },
            "checks": [
                {
                    "name": "work_item_uniqueness",
                    "status": "pass",
                    "observed": len({pair.work_item.work_item_id for pair in pairs}),
                    "expected": len(pairs),
                },
                {
                    "name": "response_composite_key_uniqueness",
                    "status": "pass",
                    "observed": len(state.responses),
                    "expected": len(state.responses),
                },
                {
                    "name": "generation_id_uniqueness_and_reconciliation",
                    "status": "pass",
                    "observed": len(generation_metadata),
                    "expected": len(generation_metadata),
                },
                {
                    "name": "source_response_journal_ledger_integrity",
                    "status": "pass",
                    "verified_source_count": len(state.sources),
                    "verified_response_count": len(state.responses),
                    "verified_journal_count": len(journals),
                    "ledger_entry_count": len(state.ledger),
                },
                {
                    "name": "condition_cost_subtotals",
                    "status": "pass",
                    "actual_cost_micros": dataset_cost_micros,
                },
                {
                    "name": "human_judgment_availability",
                    "status": "not_available",
                    "public": 0,
                    "expert": 0,
                    "impact": "preference, ranking, and uplift remain undefined",
                },
                {
                    "name": "unidentified_generation_cost_exactness",
                    "status": (
                        "conservative_allowance_retained"
                        if state.incident_resolutions
                        else "not_observed"
                    ),
                    "incident_count": len(state.incident_resolutions),
                    "provider_cost_exact_for_unidentified_responses": False,
                    "provider_reconciled_actual_cost_usd": _decimal_text(
                        resolved_no_id_provider_actual_usd
                    ),
                    "full_allowance_usd": _decimal_text(resolved_no_id_allowance_usd),
                    "exposure_increment_usd": _decimal_text(resolved_no_id_exposure_increment_usd),
                },
            ],
            "analytical_risks": [
                {
                    "name": "cross_model_comparability",
                    "severity": "high",
                    "status": "not_comparable_for_quality",
                    "evidence": (
                        "the diagonal schedule assigns different candidate task subsets to "
                        "models and no preference judgments exist"
                    ),
                    "impact": "model rows cannot be interpreted as a quality ranking",
                },
                {
                    "name": "execution_failure_profile",
                    "severity": "high",
                    "status": "retain_all_failures_in_denominators",
                    "evidence": {
                        "partial_pairs": pair_statuses["partial"],
                        "failed_pairs": pair_statuses["failed"],
                        "invalid_final_json_arms": sum(
                            overall[condition]["arms"]["outcome_class_counts"].get(
                                "invalid_final_json", 0
                            )
                            for condition in CONDITIONS
                        ),
                    },
                    "impact": (
                        "surviving normalized outputs are a selected subset; response-level "
                        "metrics must not omit execution failures"
                    ),
                },
                {
                    "name": "survivor_conditioned_response_metrics",
                    "severity": "high",
                    "status": "separately_labeled",
                    "evidence": (
                        "normalized-answer latency, response tokens, constraint labels, and "
                        "tool traces are reported alongside all-generation accounting"
                    ),
                    "impact": (
                        "normalized response metrics alone cannot compare model reliability "
                        "or culinary quality"
                    ),
                },
                {
                    "name": "unidentified_response_cost",
                    "severity": "high",
                    "status": (
                        "append_only_resolution_with_full_allowance"
                        if state.incident_resolutions
                        else "not_observed"
                    ),
                    "evidence": {
                        "incident_count": len(state.incident_resolutions),
                        "provider_cost_exact": False,
                        "safe_to_replay": False,
                        "full_allowance_usd": _decimal_text(resolved_no_id_allowance_usd),
                    },
                    "impact": (
                        "known-generation actual spend and conservative budget exposure "
                        "must remain separate"
                    ),
                },
            ],
        },
        "inputs": {
            "manifest": {
                "filename": manifest_path.name,
                "sha256": CURRENT_DATASET_MANIFEST_SHA256,
                "file_sha256": _file_sha256(manifest_path),
            },
            "source_artifacts": source_records,
            "response_artifacts": response_records,
            "source_incident_resolutions": resolution_records,
            "journals": [_journal_payload(journal) for journal in journals],
            "summaries": summary_inputs,
        },
        "privacy": {
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_raw_tool_results": False,
            "contains_raw_ip_addresses": False,
            "contains_generation_ids": True,
            "research_release_approved": False,
            "scope": (
                "derived operational metrics and immutable identifiers only; raw source "
                "records remain outside the aggregate"
            ),
        },
        "metric_definitions": {
            "attempted_pair": "an immutable source run artifact exists",
            "complete_pair": "a finalized pair has both normalized off/on response artifacts",
            "partial_pair": "a finalized pair has exactly one normalized response artifact",
            "failed_pair": "a finalized pair has no normalized response artifact",
            "structured_valid": (
                "the normalized object passes the frozen required-field/type validator"
            ),
            "constraint_acknowledgement": (
                "a label from the existing regex/field validator; it checks whether a model "
                "declared addressing an explicit constraint, not whether it complied"
            ),
            "tool_success": "an Epicure MCP trace event with is_error=false",
            "actual_cost": (
                "OpenRouter generation metadata reconciled for every provider generation, "
                "including non-normalized outcomes"
            ),
            "conservative_no_id_exposure": (
                "the complete admitted allowance retained for an HTTP-200 response that "
                "had no generation ID; it is budget exposure, not exact provider cost"
            ),
            "failure_classes": (
                "provider/normalization errors attached to the condition; pending and "
                "not-yet-attempted arms are separate non-failure states"
            ),
        },
        "limitations": [
            "This is a mutable collection checkpoint until collection_state is complete.",
            "The candidate tasks have not passed confirmatory human review.",
            "Operational success, latency, tokens, and cost do not measure culinary quality.",
            "Epicure-on denotes tool access; models may choose not to invoke a tool.",
            "No public or expert preference judgments exist in this dataset.",
            "No preference ranking or Epicure-uplift estimate may be inferred from this aggregate.",
            (
                "An HTTP-200/no-choice response without a generation ID has no exact "
                "provider cost; its append-only resolution retains the complete admitted "
                "allowance and forbids replay."
            ),
            (
                "Normalized-answer latency, response tokens, constraint labels, and traces "
                "describe survivors only; separate provider-accounting fields retain failed "
                "generation tokens, latency, tool events, and cost."
            ),
            (
                "The deterministic diagonal schedule assigns different candidate prompts to "
                "models; per-model operational rates are not quality comparisons."
            ),
        ],
    }
    validate_payload_invariants(payload)
    if _checkpoint_inventory(artifact_root) != checkpoint_inventory:
        raise CheckpointChanged(
            "real exploratory collection advanced during aggregation; retry the no-call scan"
        )
    return payload


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def write_aggregate(
    payload: Mapping[str, Any],
    *,
    aggregate_directory: Path,
) -> tuple[dict[str, Any], Path]:
    value = dict(payload)
    digest = _sha256(value)
    value["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    destination = aggregate_directory / f"real-exploratory-evidence-{digest}.json"
    if destination.exists() and destination.read_bytes() != encoded:
        raise IntegrityError(f"refusing to overwrite conflicting aggregate: {destination}")
    if not destination.exists():
        _atomic_write(destination, encoded)
    return value, destination


def publish_copies(
    aggregate_path: Path,
    payload: Mapping[str, Any],
    *,
    destinations: Sequence[Path],
) -> None:
    if payload.get("collection_state") != "complete":
        raise IntegrityError("refusing to publish a non-final real-dataset checkpoint")
    if payload.get("verification", {}).get("terminal_runner_summary_verified") is not True:
        raise IntegrityError("refusing to publish before the terminal runner summary verifies")
    staged: list[tuple[Path, Path]] = []
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            with aggregate_path.open("rb") as source:
                shutil.copyfileobj(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        if temporary.read_bytes() != aggregate_path.read_bytes():
            raise IntegrityError(f"staged aggregate copy drifted: {destination}")
        staged.append((temporary, destination))
    for temporary, destination in staged:
        os.replace(temporary, destination)
        destination.chmod(0o644)
        if destination.read_bytes() != aggregate_path.read_bytes():
            raise IntegrityError(f"published aggregate copy drifted: {destination}")


def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _tex_arm_cell(slice_value: Mapping[str, Any]) -> str:
    arms = slice_value["arms"]
    attempted = int(arms["provider_attempted"])
    normalized = int(arms["normalized_responses"])
    if attempted == 0:
        return "\\cellcolor{NeutralBG}---"
    if normalized == attempted:
        background, label = "ValidBG", "complete"
    elif normalized:
        background, label = "WarnBG", "mixed"
    else:
        background, label = "FailBG", "none"
    return (
        f"\\cellcolor{{{background}}}\\statuslabel{{{normalized}/{attempted}}}"
        f"\\newline{{\\tiny {label}}}"
    )


def render_tex_table(payload: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for model in payload["models"]:
        off = model["conditions"]["epicure_off"]
        on = model["conditions"]["epicure_on"]
        pairs = off["pairs"]
        tools = on["tools"]
        cost = int(off["cost"]["actual_cost_micros"]) + int(on["cost"]["actual_cost_micros"])
        identities = model.get("actual_identities") or []
        identity = identities[0]["model_id"] if identities else "unresolved"
        pair_label = (
            f"C {pairs['complete']}/{pairs['attempted']}; P {pairs['partial']}; F {pairs['failed']}"
        )
        rows.append(
            f"{_tex_escape(model['display_name'])}\\newline"
            f"{{\\tiny\\nolinkurl{{{identity}}}}} & "
            f"{_tex_escape(pair_label)} & "
            f"{_tex_arm_cell(off)} & "
            f"{_tex_arm_cell(on)} & "
            f"{tools['successful_calls']}/{tools['calls']} & "
            f"\\${cost / 1_000_000:.6f} \\\\"
        )
    return "\n".join(
        [
            "% Generated by flavourbench.evidence_aggregate; do not edit by hand.",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Exploratory paired real-data collection. C/P/F denote complete, "
            "partial, and failed finalized pairs. Off/on cells report normalized responses "
            "over provider-attempted arms; `complete' means response normalization only. "
            "Tool is successful/total real Epicure calls. Cost includes non-normalized "
            "identified provider generations; any conservative no-ID allowance is "
            "reported separately. All rows have zero human judgments and are unranked.}",
            "\\label{tab:real-exploratory}",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabularx}{\\textwidth}{@{}p{0.25\\textwidth}Xccrr@{}}",
            "\\toprule",
            "Model and returned identity & Pair outcomes & Epicure off & Epicure available "
            "& Tool & Actual cost \\\\ ",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabularx}",
            "\\end{table*}",
            "",
        ]
    )


def render_tex_macros(payload: Mapping[str, Any]) -> str:
    progress = payload["progress"]
    statuses = progress["pair_status_counts"]
    cost = payload["cost"]
    off = payload["overall_by_condition"]["epicure_off"]
    on = payload["overall_by_condition"]["epicure_on"]
    values = {
        "realDatasetPairCount": payload["workload"]["expected_pairs"],
        "realDatasetFinalizedPairs": progress["finalized_pairs"],
        "realDatasetCompletePairs": statuses["complete"],
        "realDatasetPartialPairs": statuses["partial"],
        "realDatasetFailedPairs": statuses["failed"],
        "realDatasetNormalizedResponses": progress["normalized_responses"],
        "realDatasetOffAttempted": off["arms"]["provider_attempted"],
        "realDatasetOffNormalized": off["arms"]["normalized_responses"],
        "realDatasetOnAttempted": on["arms"]["provider_attempted"],
        "realDatasetOnNormalized": on["arms"]["normalized_responses"],
        "realDatasetToolCalls": on["tools"]["calls"],
        "realDatasetSuccessfulToolCalls": on["tools"]["successful_calls"],
        "realDatasetToolErrors": on["tools"]["error_calls"],
        "realDatasetInvalidFinalJSON": sum(
            payload["overall_by_condition"][condition]["arms"]["outcome_class_counts"].get(
                "invalid_final_json", 0
            )
            for condition in CONDITIONS
        ),
        "realDatasetProviderGenerations": cost["provider_generation_count"],
        "realDatasetCostUSD": cost["dataset_actual_cost_usd"],
        "realDatasetSourceExposureUSD": cost["dataset_source_budget_exposure_usd"],
        "realDatasetNoIDIncidents": cost["resolved_no_id_incident_count"],
        "realDatasetNoIDAllowanceUSD": cost["resolved_no_id_full_allowance_usd"],
        "realDatasetNoIDExposureIncrementUSD": cost["resolved_no_id_exposure_increment_usd"],
        "realDatasetActualExposureUSD": cost["actual_verified_exposure_usd"],
        "realDatasetConservativeExposureUSD": cost["conservative_total_exposure_usd"],
        "realDatasetAggregateDigest": payload["content_address"]["digest"],
    }
    return "\n".join(
        [
            "% Generated by flavourbench.evidence_aggregate; do not edit by hand.",
            *(f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in values.items()),
            "",
        ]
    )


def render_tex_section() -> str:
    """Render claim-scoped manuscript prose; all quantities come from macros."""

    return "\n".join(
        [
            "% Generated by flavourbench.evidence_aggregate; do not edit by hand.",
            "\\subsection{Exploratory paired real-model execution}",
            "",
            "Beyond endpoint probes, the governed collector executed all "
            "\\realDatasetPairCount{} frozen model--task assignments with concurrent "
            "Epicure-off and Epicure-available arms. These are real OpenRouter "
            "generations routed through Cloudflare AI Gateway and, for enabled arms, the "
            "attested real Epicure MCP runtime. The prompts are deterministic, fixed "
            "candidate tasks. They have not received "
            "confirmatory human review, and the run is permanently unranked.",
            "",
            "All \\realDatasetFinalizedPairs{} pairs reached a terminal ledger state: "
            "\\realDatasetCompletePairs{} yielded both normalized arms, "
            "\\realDatasetPartialPairs{} yielded one, and \\realDatasetFailedPairs{} "
            "yielded neither. Epicure-off normalized \\realDatasetOffNormalized{} of "
            "\\realDatasetOffAttempted{} attempted arms; Epicure-available normalized "
            "\\realDatasetOnNormalized{} of \\realDatasetOnAttempted{}. These fractions "
            "are execution/response-contract evidence, not correctness rates.",
            "",
            "\\input{figures/real_exploratory_matrix}",
            "",
            "The source journals record \\realDatasetSuccessfulToolCalls{} successful and "
            "\\realDatasetToolErrors{} errored Epicure calls among "
            "\\realDatasetToolCalls{} total calls, including calls made by arms whose "
            "final answer did not normalize. Invalid final JSON accounts for "
            "\\realDatasetInvalidFinalJSON{} arm outcomes. All "
            "\\realDatasetProviderGenerations{} identified OpenRouter generation records "
            "and their costs reconcile. One HTTP-200/no-choice response lacked a "
            "generation identifier (\\realDatasetNoIDIncidents{} such incident), so its "
            "exact provider cost is unknowable and no identifier is inferred. The "
            "append-only resolution retains its full USD~\\realDatasetNoIDAllowanceUSD{} "
            "allowance, adding USD~\\realDatasetNoIDExposureIncrementUSD{} above "
            "provider-reconciled actual cost. Known-generation dataset spend is "
            "USD~\\realDatasetCostUSD{}; dataset source budget exposure is "
            "USD~\\realDatasetSourceExposureUSD{}. Actual verified known-generation "
            "exposure including prior traces is USD~\\realDatasetActualExposureUSD{}, "
            "while conservative total budget exposure is "
            "USD~\\realDatasetConservativeExposureUSD{}.",
            "",
            "Attrition is model dependent, so answer-level latency, response tokens, "
            "constraint acknowledgements, and normalized tool traces are survivor "
            "conditioned. The aggregate separately retains all-generation accounting, "
            "MCP events, failures, and costs. Moreover, the diagonal schedule assigns "
            "different candidate prompts to models. Consequently, neither normalized "
            "response rates nor any other operational row supports a frontier-quality "
            "ranking. With zero public and zero expert judgments, preference and Epicure "
            "uplift remain undefined. The derived, raw-text-free aggregate is "
            "\\nolinkurl{sha256:\\realDatasetAggregateDigest}.",
            "",
        ]
    )


def _parser() -> argparse.ArgumentParser:
    evaluation_root = Path(__file__).resolve().parents[3]
    workspace_root = evaluation_root.parents[1] / "epicure"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=evaluation_root)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--aggregate-directory",
        type=Path,
        default=evaluation_root / "flavourbench" / "artifacts" / "aggregates",
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--paper-json",
        type=Path,
        default=(
            evaluation_root / "paper" / "flavourbench" / "data" / "real_exploratory_evidence.json"
        ),
    )
    parser.add_argument(
        "--web-json",
        type=Path,
        default=workspace_root
        / "epicure-webapp"
        / "lib"
        / "flavourbench-real-exploratory-evidence.json",
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=(
            evaluation_root / "paper" / "flavourbench" / "figures" / "real_exploratory_matrix.tex"
        ),
    )
    parser.add_argument(
        "--paper-macros",
        type=Path,
        default=(
            evaluation_root / "paper" / "flavourbench" / "figures" / "real_exploratory_macros.tex"
        ),
    )
    parser.add_argument(
        "--paper-section",
        type=Path,
        default=(
            evaluation_root / "paper" / "flavourbench" / "figures" / "real_exploratory_section.tex"
        ),
    )
    return parser


def run() -> None:
    arguments = _parser().parse_args()
    payload: dict[str, Any] | None = None
    for _attempt in range(5):
        try:
            payload = build_payload(
                arguments.evaluation_root.resolve(),
                manifest_path=arguments.manifest,
            )
            break
        except CheckpointChanged:
            continue
    if payload is None:
        raise CheckpointChanged(
            "collection advanced during five no-call scan attempts; retry after the active pair"
        )
    written, path = write_aggregate(
        payload,
        aggregate_directory=arguments.aggregate_directory,
    )
    if arguments.publish:
        rendered_table = render_tex_table(written).encode("utf-8")
        rendered_macros = render_tex_macros(written).encode("utf-8")
        rendered_section = render_tex_section().encode("utf-8")
        publish_copies(
            path,
            written,
            destinations=(arguments.paper_json, arguments.web_json),
        )
        _atomic_write(arguments.paper_table, rendered_table)
        _atomic_write(arguments.paper_macros, rendered_macros)
        _atomic_write(arguments.paper_section, rendered_section)
    print(
        json.dumps(
            {
                "aggregate": str(path.resolve()),
                "content_address": written["content_address"]["digest"],
                "collection_state": written["collection_state"],
                "source_attempted_pairs": written["progress"]["source_attempted_pairs"],
                "finalized_pairs": written["progress"]["finalized_pairs"],
                "normalized_responses": written["progress"]["normalized_responses"],
                "dataset_actual_cost_usd": written["cost"]["dataset_actual_cost_usd"],
                "dataset_source_budget_exposure_usd": written["cost"][
                    "dataset_source_budget_exposure_usd"
                ],
                "resolved_no_id_incident_count": written["cost"]["resolved_no_id_incident_count"],
                "provider_cost_exact_for_all_attempts": written["cost"][
                    "provider_cost_exact_for_all_attempts"
                ],
                "dataset_generation_route": written["route"],
                "dataset_uses_real_openrouter_records": True,
                "aggregation_provider_calls_made": False,
                "rank_eligible": False,
                "human_judgments": 0,
                "published": bool(arguments.publish),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
