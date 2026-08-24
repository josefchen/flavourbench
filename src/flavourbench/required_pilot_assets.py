"""Verify and render the required-Epicure current-frontier pilot.

The module is intentionally strict. It accepts only the frozen 14-model, four-task
development run in which every retained ``epicure_on`` response contains at least one
successful real Epicure call. It produces descriptive reliability, tool, latency, and
cost assets. It never derives an answer-quality score or a model ordering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from .current_pilot_assets import DISPLAY_NAMES, FAMILY_NAMES, FAMILY_ORDER, MODEL_ORDER
from .engine import is_complete_finish_reason
from .real_task_bank import sha256_json

SUMMARY_SCHEMA_VERSION = "flavourbench-real-exploratory-summary-v1"
RESPONSE_SCHEMA_VERSION = "flavourbench-real-exploratory-response-v1"
SOURCE_SCHEMA_VERSION = "flavourbench-live-smoke-v1"
ASSET_SCHEMA_VERSION = "flavourbench-required-frontier-pilot-assets-v1"
AUDIT_SCHEMA_VERSION = "flavourbench-required-frontier-pilot-audit-v1"

EXPECTED_SUMMARY_SHA256 = (
    "f1a38e30042b9614fa82f5c38b43b98c7c9a18916c4541f21bb12d3bcce8ba70"
)
EXPECTED_MANIFEST_SHA256 = (
    "a9b3f7f711503f15e6226d916dd6e3fdb4b8b4ffc666d596dcbebe35dae84ecc"
)
EXPECTED_POLICY_SHA256 = (
    "40f6800c77236c6b18c804a00f4fec32002f285fb3c875318ec2981be5f2ec80"
)
EXPECTED_EPICURE_RELEASE = "exploratory-unmatched-1790-runtime"
EXPECTED_EPICURE_BUNDLE_SHA256 = (
    "98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1"
)
EXPECTED_EPICURE_APPLICATION_SHA256 = (
    "be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313"
)
EXPECTED_EPICURE_TOOL_SCHEMA_SHA256 = (
    "666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd"
)

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
INK = "#202124"
MID = "#667085"
LIGHT = "#E7EBF0"
PALE = "#F4F6F8"


class RequiredPilotAssetError(RuntimeError):
    """The run graph or a publication claim boundary failed verification."""


@dataclass(frozen=True)
class VerifiedRequiredPilot:
    """Verified raw graph and derived descriptive rows."""

    summary: dict[str, Any]
    model_rows: tuple[dict[str, Any], ...]
    family_rows: tuple[dict[str, Any], ...]
    work_items: tuple[dict[str, Any], ...]
    sources: dict[str, dict[str, Any]]
    responses: dict[tuple[str, str], dict[str, Any]]
    input_hashes: dict[str, Any]
    counts: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RequiredPilotAssetError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise RequiredPilotAssetError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RequiredPilotAssetError(f"{label} must be an object")
    return value


def _require_sequence(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RequiredPilotAssetError(f"{label} must be an array")
    return value


def _verify_summary(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    content_address = _require_mapping(document.get("content_address"), "summary address")
    digest = str(content_address.get("digest") or "")
    unhashed = {key: value for key, value in document.items() if key != "content_address"}
    if digest != sha256_json(unhashed) or digest not in path.name:
        raise RequiredPilotAssetError("summary content address does not verify")
    if digest != EXPECTED_SUMMARY_SHA256:
        raise RequiredPilotAssetError("summary is not the frozen required-Epicure final run")
    if (
        document.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or document.get("mode") != "execute"
        or document.get("official") is not False
        or document.get("rank_eligible") is not False
        or document.get("research_result") is not False
    ):
        raise RequiredPilotAssetError("summary crossed its development-only claim boundary")
    policy = _require_mapping(document.get("execution_policy"), "execution policy")
    if (
        document.get("execution_policy_sha256") != EXPECTED_POLICY_SHA256
        or policy.get("epicure_on_tool_required") is not True
        or policy.get("evidence_protocol") != "matched_evidence_v2"
        or policy.get("matched_planning") is not True
    ):
        raise RequiredPilotAssetError("summary is not bound to the required matched protocol")
    return document


def _verify_hashed_artifact(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    digest = str(document.get("artifact_sha256") or "")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    schema_version = document.get("schema_version")
    if schema_version == SOURCE_SCHEMA_VERSION:
        expected = hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    elif schema_version == RESPONSE_SCHEMA_VERSION:
        expected = sha256_json(unhashed)
    else:
        raise RequiredPilotAssetError(f"unexpected artifact schema: {path.name}")
    if len(digest) != 64 or digest != expected or digest[:12] not in path.name:
        raise RequiredPilotAssetError(f"artifact address does not verify: {path.name}")
    if (
        document.get("official") is not False
        or document.get("rank_eligible") is not False
        or document.get("research_result") is not False
    ):
        raise RequiredPilotAssetError(f"artifact crossed claim boundary: {path.name}")
    return document


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise RequiredPilotAssetError("invalid Wilson interval input")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), percentile, method="linear"))


def _manifest_contracts(
    summary: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    manifest = _require_mapping(summary.get("manifest"), "model manifest")
    if manifest.get("sha256") != EXPECTED_MANIFEST_SHA256:
        raise RequiredPilotAssetError("manifest digest changed")
    models = _require_sequence(manifest.get("models"), "model manifest rows")
    order = tuple(str(row.get("model_id")) for row in models if isinstance(row, dict))
    if len(order) != len(MODEL_ORDER) or set(order) != set(MODEL_ORDER):
        raise RequiredPilotAssetError("frozen 14-model membership changed")
    contracts: dict[str, dict[str, Any]] = {}
    for raw in models:
        row = _require_mapping(raw, "model contract")
        model_id = str(row.get("model_id") or "")
        if model_id in contracts:
            raise RequiredPilotAssetError("duplicate model contract")
        route = _require_mapping(row.get("route_selection"), "route selection")
        if (
            route.get("selection_frozen_before_generation") is not True
            or route.get("generation_time_automatic_fallback") is not False
        ):
            raise RequiredPilotAssetError("model route was not frozen before generation")
        contracts[model_id] = row
    return order, contracts


def _generation_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results = _require_mapping(source.get("results"), "source results")
    for raw in results.values():
        result = _require_mapping(raw, "source result")
        rows.extend(
            _require_mapping(value, "generation metadata")
            for value in _require_sequence(
                result.get("generation_metadata"), "generation metadata"
            )
        )
    rows.extend(
        _require_mapping(value, "incomplete generation metadata")
        for value in _require_sequence(
            source.get("incomplete_generation_metadata"),
            "incomplete generation metadata",
        )
    )
    return rows


def _source_cost_basis(source: Mapping[str, Any]) -> str:
    budget = _require_mapping(source.get("budget"), "source budget")
    if budget.get("all_generation_costs_reconciled") is True:
        return "provider_reconciled_actual"
    if (
        source.get("execution_backend") == "kimi_direct"
        and budget.get("all_generation_usage_rate_card_accounted") is True
        and budget.get("provider_charge_available") is False
    ):
        return "frozen_rate_card_estimate"
    raise RequiredPilotAssetError("source has neither exact nor permitted estimated accounting")


def _response_matches_result(
    response: Mapping[str, Any], source_result: Mapping[str, Any]
) -> bool:
    keys = (
        "actual_model_id",
        "actual_provider",
        "answer_markdown",
        "finish_reason",
        "generation_id",
        "generation_ids",
        "cost_micros",
        "tool_trace",
    )
    return all(response.get(key) == source_result.get(key) for key in keys)


def verify_required_pilot(
    summary_path: Path,
    source_dir: Path,
    response_dir: Path,
) -> VerifiedRequiredPilot:
    """Verify the complete immutable run graph and derive descriptive metrics."""

    summary = _verify_summary(summary_path)
    model_order, contracts = _manifest_contracts(summary)
    workload = _require_mapping(summary.get("workload"), "workload")
    work_items = tuple(
        _require_mapping(value, "work item")
        for value in _require_sequence(workload.get("work_items"), "work items")
    )
    if len(work_items) != 56 or workload.get("expected_response_count") != 112:
        raise RequiredPilotAssetError("workload is not the frozen 14 by 4 design")
    work_by_id: dict[str, dict[str, Any]] = {}
    cells: set[tuple[str, str]] = set()
    for item in work_items:
        work_item_id = str(item.get("work_item_id") or "")
        model_id = str(item.get("model_id") or "")
        family = str(item.get("task_family") or "")
        contract = contracts.get(model_id)
        if (
            not work_item_id
            or work_item_id in work_by_id
            or family not in FAMILY_ORDER
            or contract is None
            or item.get("canonical_model_slug") != contract.get("canonical_model_slug")
            or item.get("execution_backend") != contract.get("execution_backend")
            or item.get("execution_policy_sha256") != EXPECTED_POLICY_SHA256
        ):
            raise RequiredPilotAssetError("invalid workload identity or contract")
        work_by_id[work_item_id] = item
        cells.add((model_id, family))
    expected_cells = {(model_id, family) for model_id in model_order for family in FAMILY_ORDER}
    if cells != expected_cells:
        raise RequiredPilotAssetError("workload does not cover every model-family cell")

    task_selection = _require_mapping(summary.get("task_selection"), "task selection")
    task_source = _require_mapping(task_selection.get("source"), "task source")
    if (
        task_selection.get("selected_task_count") != 4
        or task_source.get("synthetic_tasks") != 0
        or task_source.get("confirmatory_eligible") is not False
        or task_source.get("rank_eligible") is not False
    ):
        raise RequiredPilotAssetError("task selection crossed its development claim boundary")

    source_paths = sorted(source_dir.glob("*.json"))
    response_paths = sorted(response_dir.glob("*.json"))
    if len(source_paths) != 56 or len(response_paths) != 95:
        raise RequiredPilotAssetError("raw file counts changed")

    sources: dict[str, dict[str, Any]] = {}
    source_hashes: list[str] = []
    generation_ids: set[str] = set()
    source_cost_micros = 0
    exact_cost_micros = 0
    estimated_cost_micros = 0
    provider_generations = 0
    mcp_calls = 0
    mcp_successes = 0
    intermediate_ceiling_events = 0
    for path in source_paths:
        source = _verify_hashed_artifact(path)
        work_item_id = str(source.get("dataset_work_item_id") or "")
        item = work_by_id.get(work_item_id)
        if item is None or work_item_id in sources:
            raise RequiredPilotAssetError("unknown or duplicate source work item")
        contract = contracts[str(item["model_id"])]
        observed_backend = source.get("execution_backend")
        if observed_backend is None and contract.get("execution_backend") == "openrouter":
            observed_backend = "openrouter"
        if (
            source.get("dataset_task_id") != item.get("task_id")
            or source.get("category") != item.get("task_family")
            or source.get("requested_model_id") != item.get("model_id")
            or source.get("requested_provider") != contract.get("provider_tag")
            or observed_backend != contract.get("execution_backend")
            or source.get("execution_policy_sha256") != EXPECTED_POLICY_SHA256
        ):
            raise RequiredPilotAssetError("source does not match its workload contract")
        epicure = _require_mapping(source.get("epicure"), "Epicure provenance")
        if (
            epicure.get("release_id") != EXPECTED_EPICURE_RELEASE
            or epicure.get("bundle_sha256") != EXPECTED_EPICURE_BUNDLE_SHA256
            or epicure.get("application_sha256") != EXPECTED_EPICURE_APPLICATION_SHA256
            or source.get("epicure_tool_schema_sha256")
            != EXPECTED_EPICURE_TOOL_SCHEMA_SHA256
        ):
            raise RequiredPilotAssetError("Epicure runtime identity changed")

        cost_basis = _source_cost_basis(source)
        budget = _require_mapping(source.get("budget"), "source budget")
        cost_micros = budget.get("actual_cost_micros")
        if isinstance(cost_micros, bool) or not isinstance(cost_micros, int) or cost_micros < 0:
            raise RequiredPilotAssetError("source cost is invalid")
        metadata = _generation_rows(source)
        if sum(int(row.get("cost_micros") or 0) for row in metadata) != cost_micros:
            raise RequiredPilotAssetError("source generation costs do not reconcile")
        for generation in metadata:
            generation_id = str(generation.get("generation_id") or "")
            if (
                not generation_id
                or generation_id in generation_ids
                or generation.get("model") != contract.get("canonical_model_slug")
            ):
                raise RequiredPilotAssetError("provider generation identity is invalid")
            if (
                cost_basis == "provider_reconciled_actual"
                and generation.get("reconciled") is not True
            ):
                raise RequiredPilotAssetError("exact provider generation was not reconciled")
            generation_ids.add(generation_id)
        provider_generations += len(metadata)
        incomplete = _require_sequence(
            source.get("incomplete_generation_metadata"), "incomplete metadata"
        )
        intermediate_ceiling_events += sum(
            int(row.get("native_tokens_completion") or row.get("tokens_completion") or 0)
            == 2048
            for row in incomplete
            if isinstance(row, Mapping)
        )
        trace = _require_sequence(source.get("mcp_trace_events"), "MCP trace")
        for raw_event in trace:
            event = _require_mapping(raw_event, "MCP event")
            result = event.get("result")
            if not isinstance(result, str):
                raise RequiredPilotAssetError("MCP result is not text")
            if hashlib.sha256(result.encode()).hexdigest() != event.get("result_sha256"):
                raise RequiredPilotAssetError("MCP result digest does not verify")
            mcp_successes += int(not bool(event.get("is_error")))
        mcp_calls += len(trace)
        provider_events = _require_sequence(
            source.get("provider_attempt_events"), "provider attempt events"
        )
        event_counts = Counter(str(event.get("event_type")) for event in provider_events)
        if (
            event_counts["request_started"] != len(metadata)
            or event_counts["response_received"] != len(metadata)
            or event_counts["accounting_reconciled"] != len(metadata)
            or event_counts["mcp_call_completed"] != len(trace)
        ):
            raise RequiredPilotAssetError("provider journal does not reconcile")
        source_cost_micros += cost_micros
        if cost_basis == "provider_reconciled_actual":
            exact_cost_micros += cost_micros
        else:
            estimated_cost_micros += cost_micros
        sources[work_item_id] = source
        source_hashes.append(str(source["artifact_sha256"]))
    if set(sources) != set(work_by_id):
        raise RequiredPilotAssetError("sources do not cover the full workload")

    responses: dict[tuple[str, str], dict[str, Any]] = {}
    response_hashes: list[str] = []
    for path in response_paths:
        document = _verify_hashed_artifact(path)
        if document.get("research_release_eligible") is not False:
            raise RequiredPilotAssetError("response crossed its release boundary")
        work_item_id = str(document.get("work_item_id") or "")
        condition = str(document.get("condition") or "")
        key = (work_item_id, condition)
        item = work_by_id.get(work_item_id)
        if key in responses or condition not in {"epicure_off", "epicure_on"} or item is None:
            raise RequiredPilotAssetError("invalid or duplicate response identity")
        source = sources[work_item_id]
        source_result = _require_mapping(
            _require_mapping(source.get("results"), "source results").get(condition),
            "linked source result",
        )
        model = _require_mapping(document.get("model"), "response model")
        response = _require_mapping(document.get("response"), "response payload")
        task = _require_mapping(document.get("task"), "response task")
        provenance = _require_mapping(document.get("provenance"), "response provenance")
        contract = contracts[str(item["model_id"])]
        if (
            _require_mapping(document.get("source"), "response source").get("artifact_sha256")
            != source.get("artifact_sha256")
            or document.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
            or document.get("execution_policy_sha256") != EXPECTED_POLICY_SHA256
            or model.get("requested_model_id") != item.get("model_id")
            or model.get("canonical_model_slug") != contract.get("canonical_model_slug")
            or model.get("actual_model_id") != contract.get("canonical_model_slug")
            or response.get("actual_model_id") != contract.get("canonical_model_slug")
            or task.get("public_id") != item.get("task_id")
            or task.get("family") != item.get("task_family")
            or task.get("review_status") != "candidate"
            or bool(provenance.get("epicure_access")) != (condition == "epicure_on")
            or not _response_matches_result(response, source_result)
        ):
            raise RequiredPilotAssetError("response provenance does not reconcile")
        if not is_complete_finish_reason(str(response.get("finish_reason") or "")):
            raise RequiredPilotAssetError("normalized response has an incomplete final finish")
        answer = response.get("answer_markdown")
        if not isinstance(answer, str) or not answer.strip():
            raise RequiredPilotAssetError("normalized response has no final answer")
        trace = _require_sequence(response.get("tool_trace"), "response tool trace")
        if condition == "epicure_off" and trace:
            raise RequiredPilotAssetError("control response contains an Epicure call")
        if condition == "epicure_on" and (
            not trace or not any(not bool(event.get("is_error")) for event in trace)
        ):
            raise RequiredPilotAssetError("retained treatment lacks a successful Epicure call")
        cost_basis = _source_cost_basis(source)
        if cost_basis == "provider_reconciled_actual":
            if response.get("cost_reconciled") is not True:
                raise RequiredPilotAssetError("exact response cost was not reconciled")
        elif (
            response.get("cost_reconciled") is not False
            or response.get("cost_accounting_basis")
            != "frozen_rate_card_times_kimi_returned_usage"
        ):
            raise RequiredPilotAssetError("estimated response cost lacks its frozen basis")
        responses[key] = document
        response_hashes.append(str(document["artifact_sha256"]))

    overall = Counter()
    model_accumulators: dict[str, dict[str, Any]] = {}
    family_accumulators: dict[str, Counter[str]] = {
        family: Counter() for family in FAMILY_ORDER
    }
    for model_id in model_order:
        model_accumulators[model_id] = {
            "scheduled_pairs": 0,
            "complete_pairs": 0,
            "partial_pairs": 0,
            "failed_pairs": 0,
            "off_responses": 0,
            "on_responses": 0,
            "latencies_s": [],
            "cost_micros": 0,
            "mcp_calls": 0,
            "mcp_successes": 0,
            "provider_generations": 0,
            "intermediate_ceiling_events": 0,
        }
    for work_item_id, item in work_by_id.items():
        model_id = str(item["model_id"])
        family = str(item["task_family"])
        source = sources[work_item_id]
        off = responses.get((work_item_id, "epicure_off"))
        on = responses.get((work_item_id, "epicure_on"))
        arm_count = int(off is not None) + int(on is not None)
        complete = int(arm_count == 2)
        partial = int(arm_count == 1)
        failed = int(arm_count == 0)
        metrics = {
            "scheduled_pairs": 1,
            "complete_pairs": complete,
            "partial_pairs": partial,
            "failed_pairs": failed,
            "off_responses": int(off is not None),
            "on_responses": int(on is not None),
        }
        overall.update(metrics)
        family_accumulators[family].update(metrics)
        accumulator = model_accumulators[model_id]
        for key, value in metrics.items():
            accumulator[key] += value
        source_cost = int(_require_mapping(source.get("budget"), "budget")["actual_cost_micros"])
        accumulator["cost_micros"] += source_cost
        metadata = _generation_rows(source)
        accumulator["provider_generations"] += len(metadata)
        accumulator["intermediate_ceiling_events"] += sum(
            int(row.get("native_tokens_completion") or row.get("tokens_completion") or 0)
            == 2048
            for row in _require_sequence(
                source.get("incomplete_generation_metadata"), "incomplete metadata"
            )
            if isinstance(row, Mapping)
        )
        trace = _require_sequence(source.get("mcp_trace_events"), "MCP trace")
        accumulator["mcp_calls"] += len(trace)
        accumulator["mcp_successes"] += sum(
            not bool(event.get("is_error"))
            for event in trace
            if isinstance(event, Mapping)
        )
        for arm in (off, on):
            if arm is not None:
                accumulator["latencies_s"].append(
                    float(_require_mapping(arm["response"], "response")["latency_ms"])
                    / 1000
                )

    reported = _require_mapping(summary.get("coverage_and_reliability"), "coverage")
    reported_overall = _require_mapping(reported.get("overall"), "overall coverage")
    expected_overall = {
        "expected_pairs": 56,
        "expected_arms": 112,
        "source_attempts": 56,
        "finalized_pairs": 56,
        "epicure_off_responses": overall["off_responses"],
        "epicure_on_responses": overall["on_responses"],
        "complete_pairs": overall["complete_pairs"],
        "failed_or_partial_pairs": overall["partial_pairs"] + overall["failed_pairs"],
        "epicure_on_tool_used": overall["on_responses"],
    }
    if reported_overall != expected_overall:
        raise RequiredPilotAssetError("summary coverage does not reconcile")

    model_rows: list[dict[str, Any]] = []
    for ordinal, model_id in enumerate(model_order, start=1):
        contract = contracts[model_id]
        acc = model_accumulators[model_id]
        complete_lower, complete_upper = _wilson(acc["complete_pairs"], 4)
        call_lower, call_upper = (
            _wilson(acc["mcp_successes"], acc["mcp_calls"])
            if acc["mcp_calls"]
            else (float("nan"), float("nan"))
        )
        route = _require_mapping(contract.get("route_selection"), "route")
        cost_basis = (
            "frozen_rate_card_estimate"
            if contract.get("execution_backend") == "kimi_direct"
            else "provider_reconciled_actual"
        )
        model_rows.append(
            {
                "manifest_ordinal": ordinal,
                "display_name": DISPLAY_NAMES[model_id],
                "model_id": model_id,
                "canonical_model_slug": contract["canonical_model_slug"],
                "execution_backend": contract["execution_backend"],
                "provider_tag": contract["provider_tag"],
                "route_selection_reason": route.get("selection_reason"),
                "fallback_used": bool(route.get("fallback_used")),
                "scheduled_pairs": 4,
                "complete_pairs": acc["complete_pairs"],
                "partial_pairs": acc["partial_pairs"],
                "failed_pairs": acc["failed_pairs"],
                "pair_completion_rate": acc["complete_pairs"] / 4,
                "pair_completion_wilson_lower_95": complete_lower,
                "pair_completion_wilson_upper_95": complete_upper,
                "completed_off_arms": acc["off_responses"],
                "completed_on_arms": acc["on_responses"],
                "normalized_arms": acc["off_responses"] + acc["on_responses"],
                "latency_median_s": statistics.median(acc["latencies_s"])
                if acc["latencies_s"]
                else float("nan"),
                "latency_q1_s": _percentile(acc["latencies_s"], 25),
                "latency_q3_s": _percentile(acc["latencies_s"], 75),
                "provider_generations": acc["provider_generations"],
                "intermediate_ceiling_events": acc["intermediate_ceiling_events"],
                "epicure_calls": acc["mcp_calls"],
                "epicure_successful_calls": acc["mcp_successes"],
                "epicure_error_calls": acc["mcp_calls"] - acc["mcp_successes"],
                "epicure_call_success_rate": acc["mcp_successes"] / acc["mcp_calls"]
                if acc["mcp_calls"]
                else float("nan"),
                "epicure_call_success_wilson_lower_95": call_lower,
                "epicure_call_success_wilson_upper_95": call_upper,
                "total_cost_usd": acc["cost_micros"] / 1_000_000,
                "mean_cost_per_scheduled_pair_usd": acc["cost_micros"] / 4_000_000,
                "cost_basis": cost_basis,
                "quality_judgments": 0,
                "rank_eligible": False,
            }
        )

    family_rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        acc = family_accumulators[family]
        lower, upper = _wilson(acc["complete_pairs"], 14)
        family_rows.append(
            {
                "family": family,
                "display_name": FAMILY_NAMES[family],
                "scheduled_pairs": 14,
                "complete_pairs": acc["complete_pairs"],
                "partial_pairs": acc["partial_pairs"],
                "failed_pairs": acc["failed_pairs"],
                "pair_completion_rate": acc["complete_pairs"] / 14,
                "pair_completion_wilson_lower_95": lower,
                "pair_completion_wilson_upper_95": upper,
                "completed_off_arms": acc["off_responses"],
                "completed_on_arms": acc["on_responses"],
            }
        )

    exact_budget = Decimal(str(summary["budget"]["dataset_provider_reconciled_actual_cost_usd"]))
    estimated_budget = Decimal(str(summary["budget"]["dataset_rate_card_estimated_cost_usd"]))
    total_budget = Decimal(str(summary["budget"]["dataset_actual_cost_usd"]))
    if (
        source_cost_micros != 4_719_197
        or exact_cost_micros != 4_381_703
        or estimated_cost_micros != 337_494
        or total_budget != Decimal(source_cost_micros) / Decimal(1_000_000)
        or exact_budget != Decimal(exact_cost_micros) / Decimal(1_000_000)
        or estimated_budget != Decimal(estimated_cost_micros) / Decimal(1_000_000)
    ):
        raise RequiredPilotAssetError("cost partitions do not reconcile")
    if (
        overall["complete_pairs"] != 43
        or overall["partial_pairs"] != 9
        or overall["failed_pairs"] != 4
        or overall["off_responses"] != 51
        or overall["on_responses"] != 44
        or provider_generations != 334
        or mcp_calls != 194
        or mcp_successes != 87
        or intermediate_ceiling_events != 10
    ):
        raise RequiredPilotAssetError("frozen final-run totals changed")

    route_counts = Counter(
        str(contract.get("execution_backend")) for contract in contracts.values()
    )
    counts = {
        "models": 14,
        "tasks": 4,
        "scheduled_pairs": 56,
        "scheduled_arms": 112,
        "normalized_arms": 95,
        "complete_pairs": 43,
        "partial_pairs": 9,
        "failed_pairs": 4,
        "failed_or_partial_pairs": 13,
        "off_responses": 51,
        "on_responses": 44,
        "provider_generations": 334,
        "epicure_calls": 194,
        "epicure_successful_calls": 87,
        "epicure_error_calls": 107,
        "intermediate_ceiling_events": 10,
        "final_length_responses": 0,
        "exact_cost_usd": float(exact_budget),
        "estimated_cost_usd": float(estimated_budget),
        "combined_measured_cost_usd": float(total_budget),
        "kimi_direct_models": route_counts["kimi_direct"],
        "openrouter_models": route_counts["openrouter"],
        "bedrock_models": route_counts["bedrock"],
        "quality_judgments": 0,
        "synthetic_tasks": 0,
        "synthetic_arms": 0,
    }
    if (counts["kimi_direct_models"], counts["openrouter_models"], counts["bedrock_models"]) != (
        1,
        13,
        0,
    ):
        raise RequiredPilotAssetError("executed provider route counts changed")

    input_hashes = {
        "summary_file_sha256": _file_sha256(summary_path),
        "summary_content_sha256": EXPECTED_SUMMARY_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "execution_policy_sha256": EXPECTED_POLICY_SHA256,
        "source_artifact_sha256s": sorted(source_hashes),
        "response_artifact_sha256s": sorted(response_hashes),
        "source_set_sha256": sha256_json(sorted(source_hashes)),
        "response_set_sha256": sha256_json(sorted(response_hashes)),
    }
    return VerifiedRequiredPilot(
        summary=summary,
        model_rows=tuple(model_rows),
        family_rows=tuple(family_rows),
        work_items=work_items,
        sources=sources,
        responses=responses,
        input_hashes=input_hashes,
        counts=counts,
    )


def build_audit_document(pilot: VerifiedRequiredPilot) -> dict[str, Any]:
    """Build a content-addressed, claim-bounded audit document."""

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source": pilot.input_hashes,
        "run": {
            "started_at": pilot.summary["started_at"],
            "completed_at": pilot.summary["completed_at"],
            "runner_run_id": pilot.summary["runner_run_id"],
            "mode": "real_provider_execution",
        },
        "observed": pilot.counts,
        "models": list(pilot.model_rows),
        "families": list(pilot.family_rows),
        "epicure": {
            "release_id": EXPECTED_EPICURE_RELEASE,
            "bundle_sha256": EXPECTED_EPICURE_BUNDLE_SHA256,
            "application_sha256": EXPECTED_EPICURE_APPLICATION_SHA256,
            "tool_schema_sha256": EXPECTED_EPICURE_TOOL_SCHEMA_SHA256,
            "lineage_status": "unmatched_exploratory_runtime",
        },
        "accounting": {
            "provider_reconciled_actual_usd": pilot.counts["exact_cost_usd"],
            "direct_kimi_rate_card_estimate_usd": pilot.counts["estimated_cost_usd"],
            "combined_measured_usd": pilot.counts["combined_measured_cost_usd"],
            "active_reservations_usd": 0,
            "unreconciled_source_reserve_usd": 0,
        },
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "quality_judgments": 0,
            "quality_leaderboard_permitted": False,
            "permitted_claims": [
                "real endpoint execution",
                "required successful Epicure treatment",
                "answer-arm and pair completion",
                "tool-call reliability",
                "observed latency",
                "provider accounting with cost-basis separation",
            ],
            "prohibited_claims": [
                "model quality ranking",
                "Epicure answer-quality uplift",
                "general culinary competence",
                "official Season 0 result",
            ],
            "blocking_conditions": [
                "no submitted blinded human judgments",
                "only one development task per family",
                "Epicure runtime has unmatched public lineage",
            ],
        },
    }
    digest = sha256_json(payload)
    return {
        **payload,
        "content_address": {
            "algorithm": "sha256",
            "digest": digest,
            "uri": f"sha256:{digest}",
        },
    }


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: Any, pdf_path: Path) -> list[Path]:
    png_path = pdf_path.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    fig.savefig(png_path, dpi=220, facecolor="white")
    return [pdf_path, png_path]


def _reliability_figure(pilot: VerifiedRequiredPilot, path: Path) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    rows = list(pilot.model_rows)
    labels = [row["display_name"] for row in rows]
    y = np.arange(len(rows))
    complete = np.asarray([row["complete_pairs"] for row in rows])
    partial = np.asarray([row["partial_pairs"] for row in rows])
    failed = np.asarray([row["failed_pairs"] for row in rows])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 5.25),
        gridspec_kw={"width_ratios": [1.65, 1]},
    )
    ax = axes[0]
    ax.barh(y, complete, color=BLUE, height=0.62, label="Complete pair")
    ax.barh(y, partial, left=complete, color=SKY, height=0.62, label="One arm")
    ax.barh(y, failed, left=complete + partial, color=LIGHT, height=0.62, label="No arm")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.42)
    ax.set_xticks(range(5))
    ax.set_xlabel("Scheduled same-model pairs")
    ax.set_title("a  Pair disposition by endpoint", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for i, row in enumerate(rows):
        ax.text(4.08, i, f'{row["complete_pairs"]}/4', va="center", fontsize=7.3)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    family_rows = list(pilot.family_rows)
    family_labels = [
        "Evidence" if row["family"] == "evidence" else row["display_name"]
        for row in family_rows
    ]
    fy = np.arange(len(family_rows))
    rates = np.asarray([row["pair_completion_rate"] for row in family_rows])
    lower = np.asarray([row["pair_completion_wilson_lower_95"] for row in family_rows])
    upper = np.asarray([row["pair_completion_wilson_upper_95"] for row in family_rows])
    ax = axes[1]
    ax.errorbar(
        rates,
        fy,
        xerr=np.vstack((rates - lower, upper - rates)),
        fmt="o",
        color=INK,
        markerfacecolor=BLUE,
        markeredgecolor="white",
        markersize=6,
        capsize=2,
        linewidth=1,
    )
    ax.set_yticks(fy, family_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.13)
    ax.set_xticks(np.linspace(0, 1, 6), ["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel("Complete pairs (%)")
    ax.set_title("b  Fixed-task family cells", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    for i, row in enumerate(family_rows):
        ax.text(1.015, i, f'{row["complete_pairs"]}/14', va="center", fontsize=7.3)
    fig.subplots_adjust(wspace=0.42)
    return _save_figure(fig, path)


def _tool_figure(pilot: VerifiedRequiredPilot, path: Path) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    rows = list(pilot.model_rows)
    labels = [row["display_name"] for row in rows]
    y = np.arange(len(rows))
    success = np.asarray([row["epicure_successful_calls"] for row in rows])
    errors = np.asarray([row["epicure_error_calls"] for row in rows])
    rates = np.asarray(
        [row["epicure_call_success_rate"] * 100 for row in rows], dtype=float
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 5.25),
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    ax = axes[0]
    ax.barh(y, success, color=GREEN, height=0.62, label="Successful result")
    ax.barh(y, errors, left=success, color=ORANGE, height=0.62, label="Error result")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Live Epicure calls")
    ax.set_title("a  Executed MCP results", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    for i, row in enumerate(rows):
        total = row["epicure_calls"]
        ax.text(total + 0.45, i, str(total), va="center", fontsize=7.2)

    ax = axes[1]
    finite = np.isfinite(rates)
    ax.scatter(rates[finite], y[finite], s=30, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Successful calls (%)")
    ax.set_title("b  Call-level success", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    for i, row in enumerate(rows):
        if row["epicure_calls"]:
            ax.text(
                min(102, rates[i] + 3),
                i,
                f'{row["epicure_successful_calls"]}/{row["epicure_calls"]}',
                va="center",
                fontsize=7.1,
            )
    fig.subplots_adjust(wspace=0.23)
    return _save_figure(fig, path)


def _efficiency_figure(pilot: VerifiedRequiredPilot, path: Path) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    rows = list(pilot.model_rows)
    labels = [row["display_name"] for row in rows]
    y = np.arange(len(rows))
    medians = np.asarray([row["latency_median_s"] for row in rows])
    q1 = np.asarray([row["latency_q1_s"] for row in rows])
    q3 = np.asarray([row["latency_q3_s"] for row in rows])
    costs = np.asarray([row["total_cost_usd"] for row in rows])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 5.25),
        gridspec_kw={"width_ratios": [1.45, 1]},
    )
    ax = axes[0]
    ax.errorbar(
        medians,
        y,
        xerr=np.vstack((medians - q1, q3 - medians)),
        fmt="o",
        color=INK,
        markerfacecolor=BLUE,
        markeredgecolor="white",
        markersize=5.5,
        capsize=2,
        linewidth=1,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Completed-arm latency (seconds, log scale)")
    ax.set_title("a  Median and interquartile range", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT, linewidth=0.6, which="both")
    ax.set_axisbelow(True)

    ax = axes[1]
    colors = [SKY if row["cost_basis"] == "frozen_rate_card_estimate" else BLUE for row in rows]
    edges = [INK if row["cost_basis"] == "frozen_rate_card_estimate" else "white" for row in rows]
    ax.barh(y, costs, color=colors, edgecolor=edges, linewidth=0.8, height=0.62)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("All-source cost (USD, log scale)")
    ax.set_title("b  Four scheduled pairs per endpoint", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT, linewidth=0.6, which="both")
    ax.set_axisbelow(True)
    for i, row in enumerate(rows):
        prefix = "~" if row["cost_basis"] == "frozen_rate_card_estimate" else "$"
        label = f'{prefix}{row["total_cost_usd"]:.3f}'
        ax.text(costs[i] * 1.08, i, label, va="center", fontsize=7.0)
    fig.text(
        0.69,
        0.025,
        "~ direct Kimi rate-card estimate; all other bars use provider generation metadata",
        ha="center",
        fontsize=7.0,
        color=MID,
    )
    fig.subplots_adjust(wspace=0.2, bottom=0.13)
    return _save_figure(fig, path)


def _attrition_figure(pilot: VerifiedRequiredPilot, path: Path) -> list[Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    stages = [
        ("Scheduled", 112, "56 matched pairs"),
        ("Normalized answers", 95, "51 control + 44 treatment"),
        ("Complete pairs", 86, "43 pairs, 86 arms"),
        ("Human judgments", 0, "not yet collected"),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 2.45))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    xs = [0.2, 2.75, 5.3, 7.85]
    widths = [2.0, 2.0, 2.0, 2.0]
    fills = [PALE, "#EAF4FA", "#DDEEF7", "white"]
    edges = [MID, BLUE, BLUE, MID]
    for index, ((title, count, subtitle), x, width, fill, edge) in enumerate(
        zip(stages, xs, widths, fills, edges, strict=True)
    ):
        box = FancyBboxPatch(
            (x, 0.55),
            width,
            1.35,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, 1.62, title, ha="center", va="center", fontweight="bold")
        ax.text(x + width / 2, 1.18, str(count), ha="center", va="center", fontsize=18, color=INK)
        ax.text(x + width / 2, 0.78, subtitle, ha="center", va="center", fontsize=7.4, color=MID)
        if index < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[index + 1] - 0.1, 1.22),
                xytext=(x + width + 0.1, 1.22),
                arrowprops={"arrowstyle": "-|>", "color": MID, "lw": 0.9},
            )
    ax.text(
        0.2,
        0.22,
        "Treatment answers are retained only after at least one successful live Epicure call. "
        "Failures remain in reliability denominators; no preference is inferred without a ballot.",
        fontsize=7.4,
        color=INK,
    )
    return _save_figure(fig, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RequiredPilotAssetError("cannot write an empty table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _write_table(pilot: VerifiedRequiredPilot, path: Path) -> None:
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}p{0.135\textwidth} X p{0.105\textwidth} r r r r@{}}",
        r"\toprule",
        r"Endpoint & Returned identity & Route & Pairs & Arms & MCP ok/err & Cost \\",
        r"\midrule",
    ]
    for row in pilot.model_rows:
        route = "Kimi direct" if row["execution_backend"] == "kimi_direct" else row["provider_tag"]
        cost_prefix = r"$\sim$" if row["cost_basis"] == "frozen_rate_card_estimate" else r"\$"
        lines.append(
            f'{_latex_escape(row["display_name"])} & '
            f'\\path{{{row["canonical_model_slug"]}}} & '
            f'\\path{{{_latex_escape(route)}}} & '
            f'{row["complete_pairs"]}/4 & '
            f'{row["normalized_arms"]}/8 & '
            f'{row["epicure_successful_calls"]}/{row["epicure_error_calls"]} & '
            f'{cost_prefix}{row["total_cost_usd"]:.3f} \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_macros(pilot: VerifiedRequiredPilot, path: Path) -> None:
    counts = pilot.counts
    macros = {
        "CurrentPilotModelCount": counts["models"],
        "CurrentPilotScheduledPairs": counts["scheduled_pairs"],
        "CurrentPilotScheduledArms": counts["scheduled_arms"],
        "CurrentPilotCompletedResponseArms": counts["normalized_arms"],
        "CurrentPilotCompletePairs": counts["complete_pairs"],
        "CurrentPilotPartialPairs": counts["partial_pairs"],
        "CurrentPilotZeroArmPairs": counts["failed_pairs"],
        "CurrentPilotFailedPairs": counts["failed_or_partial_pairs"],
        "CurrentPilotOffResponses": counts["off_responses"],
        "CurrentPilotOnResponses": counts["on_responses"],
        "CurrentPilotPairCompletionRate": (
            f'{100 * counts["complete_pairs"] / counts["scheduled_pairs"]:.1f}\\%'
        ),
        "CurrentPilotProviderGenerations": counts["provider_generations"],
        "CurrentPilotEpicureCalls": counts["epicure_calls"],
        "CurrentPilotEpicureSuccessfulCalls": counts["epicure_successful_calls"],
        "CurrentPilotEpicureErrorCalls": counts["epicure_error_calls"],
        "CurrentPilotToolActiveArms": counts["on_responses"],
        "CurrentPilotCompletedOnArms": counts["on_responses"],
        "CurrentPilotIntermediateCeilingEvents": counts["intermediate_ceiling_events"],
        "CurrentPilotFinalLengthResponses": counts["final_length_responses"],
        "CurrentPilotActualCost": f'\\${counts["combined_measured_cost_usd"]:.6f}',
        "CurrentPilotExactCost": f'\\${counts["exact_cost_usd"]:.6f}',
        "CurrentPilotEstimatedCost": f'\\${counts["estimated_cost_usd"]:.6f}',
        "CurrentPilotKimiDirectModels": counts["kimi_direct_models"],
        "CurrentPilotOpenRouterModels": counts["openrouter_models"],
        "CurrentPilotBedrockModels": counts["bedrock_models"],
        "CurrentPilotQualityJudgments": 0,
        "CurrentPilotSummaryHash": EXPECTED_SUMMARY_SHA256,
        "CurrentPilotManifestHash": EXPECTED_MANIFEST_SHA256,
        "CurrentPilotExecutionPolicyHash": EXPECTED_POLICY_SHA256,
    }
    path.write_text(
        "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items())
        + "\n",
        encoding="utf-8",
    )


def _write_audit_markdown(audit: Mapping[str, Any], path: Path) -> None:
    observed = _require_mapping(audit.get("observed"), "audit observations")
    lines = [
        "# Required-Epicure frontier pilot audit",
        "",
        f"Content address: `{audit['content_address']['digest']}`",
        "",
        "## Verified observations",
        "",
        (
            "- 14 exact model routes, 4 real human-authored development tasks, "
            f"{observed['scheduled_pairs']} scheduled pairs."
        ),
        (
            f"- {observed['normalized_arms']} normalized real answer arms and "
            f"{observed['complete_pairs']} complete same-model pairs."
        ),
        (
            f"- {observed['provider_generations']} real provider generations and "
            f"{observed['epicure_calls']} live Epicure calls."
        ),
        (
            f"- {observed['epicure_successful_calls']} successful and "
            f"{observed['epicure_error_calls']} error-marked Epicure results."
        ),
        (
            f"- Provider-accounted cost: ${observed['exact_cost_usd']:.6f} exact plus "
            f"${observed['estimated_cost_usd']:.6f} direct-Kimi rate-card estimate."
        ),
        "- Zero synthetic tasks and zero synthetic answer arms.",
        "",
        "## Claim boundary",
        "",
        (
            "This audit supports execution, reliability, tool-trace, latency, and cost "
            "claims only. It does not support a model-quality ranking or an Epicure "
            "quality-uplift estimate. Human judgments have not yet been submitted, the "
            "pilot has one task per family, and the Epicure runtime lineage remains "
            "unmatched to a public release."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def render_assets(
    pilot: VerifiedRequiredPilot,
    output_dir: Path,
    audit_dir: Path | None = None,
) -> dict[str, Any]:
    """Write static academic assets and a content-addressed audit receipt."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model_csv = output_dir / "current-frontier-pilot-model-metrics.csv"
    family_csv = output_dir / "current-frontier-pilot-family-metrics.csv"
    table_tex = output_dir / "current-frontier-pilot-table.tex"
    macros_tex = output_dir / "current-frontier-pilot-macros.tex"
    _write_csv(model_csv, pilot.model_rows)
    _write_csv(family_csv, pilot.family_rows)
    _write_table(pilot, table_tex)
    _write_macros(pilot, macros_tex)
    figure_paths: list[Path] = []
    figure_paths.extend(
        _reliability_figure(pilot, output_dir / "current-frontier-pilot-reliability.pdf")
    )
    figure_paths.extend(_tool_figure(pilot, output_dir / "current-frontier-pilot-tools.pdf"))
    figure_paths.extend(
        _efficiency_figure(pilot, output_dir / "current-frontier-pilot-efficiency.pdf")
    )
    figure_paths.extend(
        _attrition_figure(pilot, output_dir / "current-frontier-pilot-attrition.pdf")
    )

    audit = build_audit_document(pilot)
    audit_digest = str(audit["content_address"]["digest"])
    target_audit_dir = audit_dir or output_dir
    target_audit_dir.mkdir(parents=True, exist_ok=True)
    audit_json = target_audit_dir / f"required-frontier-pilot-audit-{audit_digest}.json"
    audit_markdown = target_audit_dir / f"required-frontier-pilot-audit-{audit_digest}.md"
    audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_audit_markdown(audit, audit_markdown)

    generated = [model_csv, family_csv, table_tex, macros_tex, *figure_paths]
    asset_hashes = {
        path.name: _file_sha256(path)
        for path in sorted(generated, key=lambda candidate: candidate.name)
    }
    provenance_payload = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "source": pilot.input_hashes,
        "audit_content_sha256": audit_digest,
        "counts": pilot.counts,
        "generated_asset_sha256s": asset_hashes,
        "plot_contract": {
            "background": "white",
            "static_library": "matplotlib",
            "model_order": "frozen manifest order",
            "quality_ordering": False,
            "error_intervals": "Wilson 95 percent descriptive reference intervals",
        },
        "claim_boundary": audit["claim_boundary"],
    }
    provenance_digest = sha256_json(provenance_payload)
    provenance = {
        **provenance_payload,
        "content_address": {
            "algorithm": "sha256",
            "digest": provenance_digest,
            "uri": f"sha256:{provenance_digest}",
        },
    }
    provenance_path = output_dir / "current-frontier-pilot-provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "auditPath": str(audit_json),
        "auditMarkdownPath": str(audit_markdown),
        "auditSha256": audit_digest,
        "provenancePath": str(provenance_path),
        "provenanceSha256": provenance_digest,
        "counts": pilot.counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and render the frozen required-Epicure frontier pilot."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    pilot = verify_required_pilot(
        arguments.summary,
        arguments.source_dir,
        arguments.response_dir,
    )
    result = render_assets(pilot, arguments.output_dir, arguments.audit_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run()
