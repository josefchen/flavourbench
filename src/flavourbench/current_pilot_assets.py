"""Verify and render the dated real frontier development pilot.

This module produces operational evidence only. It refuses official or rank-eligible
inputs, verifies every linked source and response artifact, preserves manifest order,
and never infers a model-quality ordering from reliability, cost, latency, or tool use.
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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib as mpl

from .real_task_bank import sha256_json

SUMMARY_SCHEMA_VERSION = "flavourbench-real-exploratory-summary-v1"
RESPONSE_SCHEMA_VERSION = "flavourbench-real-exploratory-response-v1"
SOURCE_SCHEMA_VERSION = "flavourbench-live-smoke-v1"
ASSET_SCHEMA_VERSION = "flavourbench-current-frontier-pilot-assets-v1"

MODEL_ORDER = (
    "openai/gpt-5.6-sol-pro",
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.6-flash",
    "x-ai/grok-4.5",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash-0731",
    "minimax/minimax-m3",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "mistralai/mistral-medium-3-5",
)

COHERE_MODEL_ORDER = (
    "cohere/command-a-plus-05-2026",
    "cohere/command-a-reasoning-08-2025",
)
EXTENDED_MODEL_ORDER = MODEL_ORDER + COHERE_MODEL_ORDER

DISPLAY_NAMES = {
    "openai/gpt-5.6-sol-pro": "GPT-5.6 Sol (OR pro)",
    "anthropic/claude-fable-5": "Claude Fable 5",
    "anthropic/claude-opus-5": "Claude Opus 5",
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.6-flash": "Gemini 3.6 Flash",
    "x-ai/grok-4.5": "Grok 4.5",
    "moonshotai/kimi-k3": "Kimi K3",
    "z-ai/glm-5.2": "GLM 5.2",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash",
    "minimax/minimax-m3": "MiniMax M3",
    "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra",
    "mistralai/mistral-medium-3-5": "Mistral Medium 3.5",
    "cohere/command-a-plus-05-2026": "Command A Plus",
    "cohere/command-a-reasoning-08-2025": "Command A Reasoning",
}

FAMILY_ORDER = ("substitution", "composition", "cookability", "evidence")
FAMILY_NAMES = {
    "substitution": "Substitution",
    "composition": "Composition",
    "cookability": "Cookability",
    "evidence": "Evidence interpretation",
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
SKY = "#56B4E9"
NAVY = "#184E77"
INK = "#202124"
MID = "#667085"
LIGHT = "#E7EBF0"


class CurrentPilotAssetError(RuntimeError):
    """A publication boundary or artifact integrity check failed."""


@dataclass(frozen=True)
class VerifiedPilot:
    """Verified input documents and publication-ready descriptive rows."""

    summary: dict[str, Any]
    model_rows: tuple[dict[str, Any], ...]
    family_rows: tuple[dict[str, Any], ...]
    input_hashes: dict[str, Any]
    counts: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CurrentPilotAssetError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise CurrentPilotAssetError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_summary(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    content_address = document.get("content_address")
    if not isinstance(content_address, Mapping):
        raise CurrentPilotAssetError("summary has no content address")
    digest = str(content_address.get("digest") or "")
    unhashed = {key: value for key, value in document.items() if key != "content_address"}
    if digest != sha256_json(unhashed) or digest not in path.name:
        raise CurrentPilotAssetError("summary content address does not verify")
    if (
        document.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or document.get("mode") != "execute"
        or document.get("official") is not False
        or document.get("rank_eligible") is not False
        or document.get("research_result") is not False
    ):
        raise CurrentPilotAssetError("summary crossed its development-only claim boundary")
    return document


def _verify_hashed_artifact(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    digest = str(document.get("artifact_sha256") or "")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    schema_version = document.get("schema_version")
    if schema_version == SOURCE_SCHEMA_VERSION:
        expected = hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    elif schema_version == RESPONSE_SCHEMA_VERSION:
        expected = sha256_json(unhashed)
    else:
        raise CurrentPilotAssetError(f"unexpected artifact schema: {path.name}")
    if len(digest) != 64 or digest != expected or digest[:12] not in path.name:
        raise CurrentPilotAssetError(f"artifact content address does not verify: {path.name}")
    return document


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise CurrentPilotAssetError("invalid Wilson interval input")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise CurrentPilotAssetError("cannot summarize an empty metric")
    return float(np.percentile(np.asarray(values, dtype=float), percentile, method="linear"))


def _coverage_record(
    *,
    expected_pairs: int,
    source_attempts: int,
    finalized_pairs: int,
    off_responses: int,
    on_responses: int,
    complete_pairs: int,
    failed_pairs: int,
    tool_used: int,
) -> dict[str, int]:
    return {
        "expected_pairs": expected_pairs,
        "expected_arms": 2 * expected_pairs,
        "source_attempts": source_attempts,
        "finalized_pairs": finalized_pairs,
        "epicure_off_responses": off_responses,
        "epicure_on_responses": on_responses,
        "complete_pairs": complete_pairs,
        "failed_or_partial_pairs": failed_pairs,
        "epicure_on_tool_used": tool_used,
    }


def _require_status_boundary(document: Mapping[str, Any], label: str) -> None:
    if (
        document.get("official") is not False
        or document.get("rank_eligible") is not False
        or document.get("research_result") is not False
    ):
        raise CurrentPilotAssetError(f"{label} crossed its development-only claim boundary")


def _model_contracts(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = summary.get("manifest")
    if not isinstance(manifest, Mapping):
        raise CurrentPilotAssetError("summary has no model manifest")
    models = manifest.get("models")
    if not isinstance(models, list) or tuple(row.get("model_id") for row in models) != MODEL_ORDER:
        raise CurrentPilotAssetError("the frozen 14-model order or membership changed")
    if int(manifest.get("selected_model_count") or 0) != len(MODEL_ORDER):
        raise CurrentPilotAssetError("manifest model count does not reconcile")
    contracts: dict[str, dict[str, Any]] = {}
    for row in models:
        if not isinstance(row, dict):
            raise CurrentPilotAssetError("invalid model manifest row")
        model_id = str(row["model_id"])
        if model_id in contracts:
            raise CurrentPilotAssetError("duplicate model manifest identity")
        contracts[model_id] = row
    return contracts


def verify_pilot(
    summary_path: Path,
    source_dir: Path,
    response_dir: Path,
) -> VerifiedPilot:
    """Verify the complete pilot graph and calculate descriptive metrics."""

    summary = _verify_summary(summary_path)
    contracts = _model_contracts(summary)
    task_selection = summary.get("task_selection")
    if not isinstance(task_selection, Mapping):
        raise CurrentPilotAssetError("summary has no task selection record")
    task_source = task_selection.get("source")
    if (
        not isinstance(task_source, Mapping)
        or task_source.get("synthetic_tasks") != 0
        or task_source.get("rank_eligible") is not False
        or task_source.get("confirmatory_eligible") is not False
    ):
        raise CurrentPilotAssetError("task source is not a zero-synthetic development set")

    workload = summary.get("workload")
    work_items = workload.get("work_items") if isinstance(workload, Mapping) else None
    if not isinstance(work_items, list) or len(work_items) != len(MODEL_ORDER) * len(FAMILY_ORDER):
        raise CurrentPilotAssetError("workload is not the complete 14 by 4 design")
    work_by_id: dict[str, dict[str, Any]] = {}
    model_family_cells: set[tuple[str, str]] = set()
    for item in work_items:
        if not isinstance(item, dict):
            raise CurrentPilotAssetError("invalid workload row")
        work_item_id = str(item.get("work_item_id") or "")
        model_id = str(item.get("model_id") or "")
        family = str(item.get("task_family") or "")
        contract = contracts.get(model_id)
        if (
            not work_item_id
            or work_item_id in work_by_id
            or family not in FAMILY_ORDER
            or contract is None
            or item.get("canonical_model_slug") != contract["canonical_model_slug"]
            or item.get("provider_tag") != contract["provider_tag"]
        ):
            raise CurrentPilotAssetError("invalid or duplicate workload identity")
        work_by_id[work_item_id] = item
        model_family_cells.add((model_id, family))
    expected_cells = {(model_id, family) for model_id in MODEL_ORDER for family in FAMILY_ORDER}
    if model_family_cells != expected_cells:
        raise CurrentPilotAssetError("workload does not cover every model and task family")

    source_paths = sorted(source_dir.glob("*.json"))
    response_paths = sorted(response_dir.glob("*.json"))
    if len(source_paths) != 56 or len(response_paths) != 103:
        raise CurrentPilotAssetError("pilot file counts do not match the frozen final run")

    sources: dict[str, dict[str, Any]] = {}
    generation_ids: set[str] = set()
    generation_count = 0
    source_cost_micros = 0
    for path in source_paths:
        document = _verify_hashed_artifact(path)
        _require_status_boundary(document, "source artifact")
        work_item_id = str(document.get("dataset_work_item_id") or "")
        item = work_by_id.get(work_item_id)
        if item is None or work_item_id in sources:
            raise CurrentPilotAssetError("source artifact has an unknown or duplicate work item")
        contract = contracts[str(item["model_id"])]
        if (
            document.get("requested_model_id") != item["model_id"]
            or document.get("requested_provider") != contract["provider_tag"]
            or document.get("category") != item["task_family"]
            or document.get("dataset_task_id") != item["task_id"]
        ):
            raise CurrentPilotAssetError("source artifact does not match its workload row")
        budget = document.get("budget")
        if (
            not isinstance(budget, Mapping)
            or budget.get("all_generation_costs_reconciled") is not True
        ):
            raise CurrentPilotAssetError("source generation costs are not fully reconciled")
        actual_cost = budget.get("actual_cost_micros")
        if not isinstance(actual_cost, int) or isinstance(actual_cost, bool) or actual_cost < 0:
            raise CurrentPilotAssetError("source artifact has an invalid actual cost")
        source_cost_micros += actual_cost
        metadata: list[Mapping[str, Any]] = []
        results = document.get("results")
        if not isinstance(results, Mapping) or set(results) - {"epicure_off", "epicure_on"}:
            raise CurrentPilotAssetError("source artifact has an invalid result set")
        for result in results.values():
            if not isinstance(result, Mapping):
                raise CurrentPilotAssetError("source result is not an object")
            metadata.extend(result.get("generation_metadata") or [])
        metadata.extend(document.get("incomplete_generation_metadata") or [])
        if sum(int(row["cost_micros"]) for row in metadata) != actual_cost:
            raise CurrentPilotAssetError("source generation costs do not sum to actual cost")
        for generation in metadata:
            generation_id = str(generation.get("generation_id") or "")
            if (
                not generation_id
                or generation_id in generation_ids
                or generation.get("reconciled") is not True
                or generation.get("model") != contract["canonical_model_slug"]
            ):
                raise CurrentPilotAssetError("provider generation identity does not reconcile")
            generation_ids.add(generation_id)
            generation_count += 1
        sources[work_item_id] = document
    if set(sources) != set(work_by_id):
        raise CurrentPilotAssetError("source artifacts do not cover the workload")

    responses: dict[tuple[str, str], dict[str, Any]] = {}
    response_hashes: dict[str, Path] = {}
    for path in response_paths:
        document = _verify_hashed_artifact(path)
        _require_status_boundary(document, "response artifact")
        if document.get("schema_version") != RESPONSE_SCHEMA_VERSION:
            raise CurrentPilotAssetError("response schema version changed")
        work_item_id = str(document.get("work_item_id") or "")
        condition = str(document.get("condition") or "")
        key = (work_item_id, condition)
        item = work_by_id.get(work_item_id)
        contract = contracts.get(str(item.get("model_id"))) if item else None
        response = document.get("response")
        model = document.get("model")
        source = document.get("source")
        if (
            key in responses
            or condition not in {"epicure_off", "epicure_on"}
            or item is None
            or contract is None
            or not isinstance(response, Mapping)
            or not isinstance(model, Mapping)
            or not isinstance(source, Mapping)
        ):
            raise CurrentPilotAssetError("response has an invalid or duplicate identity")
        source_document = sources[work_item_id]
        if (
            source.get("artifact_sha256") != source_document["artifact_sha256"]
            or model.get("requested_model_id") != item["model_id"]
            or model.get("canonical_model_slug") != contract["canonical_model_slug"]
            or model.get("actual_model_id") != contract["canonical_model_slug"]
            or response.get("actual_model_id") != contract["canonical_model_slug"]
            or response.get("cost_reconciled") is not True
            or document.get("research_release_eligible") is not False
        ):
            raise CurrentPilotAssetError("response provenance or provider identity mismatch")
        task = document.get("task")
        if (
            not isinstance(task, Mapping)
            or task.get("public_id") != item["task_id"]
            or task.get("family") != item["task_family"]
            or task.get("review_status") != "candidate"
        ):
            raise CurrentPilotAssetError("response task provenance mismatch")
        digest = str(document["artifact_sha256"])
        responses[key] = document
        response_hashes[digest] = path
    if len(response_hashes) != len(response_paths):
        raise CurrentPilotAssetError("duplicate response artifact digest")

    outcomes = summary.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(work_by_id):
        raise CurrentPilotAssetError("summary outcomes do not cover the workload")
    outcome_by_id: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise CurrentPilotAssetError("invalid summary outcome")
        work_item_id = str(outcome.get("work_item_id") or "")
        source = sources.get(work_item_id)
        if source is None or work_item_id in outcome_by_id:
            raise CurrentPilotAssetError("summary outcome has an unknown or duplicate work item")
        expected_hashes = {
            document["artifact_sha256"]
            for (candidate_id, _), document in responses.items()
            if candidate_id == work_item_id
        }
        if (
            outcome.get("source_artifact_sha256") != source["artifact_sha256"]
            or set(outcome.get("response_artifact_sha256s") or []) != expected_hashes
            or Decimal(str(outcome.get("source_actual_cost_usd")))
            != Decimal(source["budget"]["actual_cost_micros"]) / Decimal(1_000_000)
        ):
            raise CurrentPilotAssetError("summary outcome does not reconcile to raw artifacts")
        outcome_by_id[work_item_id] = outcome

    summary_cost = Decimal(str(summary["budget"]["dataset_actual_cost_usd"]))
    if summary_cost != Decimal(source_cost_micros) / Decimal(1_000_000):
        raise CurrentPilotAssetError("summary dataset cost does not reconcile")

    model_accumulator: dict[str, dict[str, Any]] = {}
    family_accumulator: dict[str, Counter[str]] = {family: Counter() for family in FAMILY_ORDER}
    overall = Counter()
    for model_id in MODEL_ORDER:
        model_accumulator[model_id] = {
            "coverage": Counter(),
            "latencies": [],
            "pair_costs": [],
            "tool_calls": 0,
            "tool_errors": 0,
            "tool_active_arms": 0,
            "completed_on_arms": 0,
        }

    for work_item_id, item in work_by_id.items():
        model_id = str(item["model_id"])
        family = str(item["task_family"])
        off = responses.get((work_item_id, "epicure_off"))
        on = responses.get((work_item_id, "epicure_on"))
        complete = off is not None and on is not None
        failed = not complete
        tool_used = bool(on and on["response"].get("tool_trace"))
        metrics = _coverage_record(
            expected_pairs=1,
            source_attempts=1,
            finalized_pairs=1,
            off_responses=int(off is not None),
            on_responses=int(on is not None),
            complete_pairs=int(complete),
            failed_pairs=int(failed),
            tool_used=int(tool_used),
        )
        overall.update(metrics)
        family_accumulator[family].update(metrics)
        accumulator = model_accumulator[model_id]
        accumulator["coverage"].update(metrics)
        accumulator["pair_costs"].append(
            float(Decimal(str(outcome_by_id[work_item_id]["source_actual_cost_usd"])))
        )
        for arm in (off, on):
            if arm is not None:
                accumulator["latencies"].append(float(arm["response"]["latency_ms"]) / 1000)
        if on is not None:
            accumulator["completed_on_arms"] += 1
            trace = on["response"].get("tool_trace") or []
            accumulator["tool_active_arms"] += int(bool(trace))
            accumulator["tool_calls"] += len(trace)
            accumulator["tool_errors"] += sum(bool(event.get("is_error")) for event in trace)

    reported = summary.get("coverage_and_reliability")
    if not isinstance(reported, Mapping):
        raise CurrentPilotAssetError("summary has no coverage reconciliation")
    recomputed_by_model = {
        model_id: dict(model_accumulator[model_id]["coverage"]) for model_id in MODEL_ORDER
    }
    recomputed_by_family = {
        family: dict(family_accumulator[family]) for family in FAMILY_ORDER
    }
    if (
        dict(overall) != reported.get("overall")
        or recomputed_by_model != reported.get("by_model")
        or recomputed_by_family != reported.get("by_task_family")
    ):
        raise CurrentPilotAssetError("coverage metrics do not reproduce from response artifacts")

    model_rows: list[dict[str, Any]] = []
    for ordinal, model_id in enumerate(MODEL_ORDER, start=1):
        accumulator = model_accumulator[model_id]
        coverage = accumulator["coverage"]
        complete_pairs = int(coverage["complete_pairs"])
        expected_pairs = int(coverage["expected_pairs"])
        completion_lower, completion_upper = _wilson(complete_pairs, expected_pairs)
        on_arms = int(accumulator["completed_on_arms"])
        tool_active = int(accumulator["tool_active_arms"])
        tool_lower, tool_upper = _wilson(tool_active, on_arms)
        latencies = list(accumulator["latencies"])
        pair_costs = list(accumulator["pair_costs"])
        tool_calls = int(accumulator["tool_calls"])
        tool_errors = int(accumulator["tool_errors"])
        contract = contracts[model_id]
        model_rows.append(
            {
                "manifest_ordinal": ordinal,
                "display_name": DISPLAY_NAMES[model_id],
                "model_id": model_id,
                "canonical_model_slug": contract["canonical_model_slug"],
                "provider_tag": contract["provider_tag"],
                "complete_pairs": complete_pairs,
                "scheduled_pairs": expected_pairs,
                "pair_completion_rate": complete_pairs / expected_pairs,
                "pair_completion_wilson_lower_95": completion_lower,
                "pair_completion_wilson_upper_95": completion_upper,
                "completed_off_arms": int(coverage["epicure_off_responses"]),
                "completed_on_arms": on_arms,
                "completed_arms_for_latency": len(latencies),
                "latency_median_s": statistics.median(latencies),
                "latency_q1_s": _percentile(latencies, 25),
                "latency_q3_s": _percentile(latencies, 75),
                "tool_active_on_arms": tool_active,
                "tool_adoption_rate_among_completed_on": tool_active / on_arms,
                "tool_adoption_wilson_lower_95": tool_lower,
                "tool_adoption_wilson_upper_95": tool_upper,
                "epicure_calls": tool_calls,
                "epicure_successful_calls": tool_calls - tool_errors,
                "epicure_semantic_error_calls": tool_errors,
                "mean_actual_cost_per_scheduled_pair_usd": statistics.mean(pair_costs),
                "min_actual_pair_cost_usd": min(pair_costs),
                "max_actual_pair_cost_usd": max(pair_costs),
                "total_actual_cost_usd": sum(pair_costs),
                "quality_judgments": 0,
                "rank_eligible": False,
            }
        )

    family_rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        coverage = family_accumulator[family]
        complete_pairs = int(coverage["complete_pairs"])
        expected_pairs = int(coverage["expected_pairs"])
        lower, upper = _wilson(complete_pairs, expected_pairs)
        family_rows.append(
            {
                "family": family,
                "display_name": FAMILY_NAMES[family],
                "complete_pairs": complete_pairs,
                "scheduled_pairs": expected_pairs,
                "pair_completion_rate": complete_pairs / expected_pairs,
                "pair_completion_wilson_lower_95": lower,
                "pair_completion_wilson_upper_95": upper,
                "failed_or_partial_pairs": int(coverage["failed_or_partial_pairs"]),
                "completed_on_arms": int(coverage["epicure_on_responses"]),
                "tool_active_on_arms": int(coverage["epicure_on_tool_used"]),
            }
        )

    tool_calls = sum(int(row["epicure_calls"]) for row in model_rows)
    tool_errors = sum(int(row["epicure_semantic_error_calls"]) for row in model_rows)
    counts = {
        "models": len(MODEL_ORDER),
        "task_families": len(FAMILY_ORDER),
        "scheduled_pairs": int(overall["expected_pairs"]),
        "scheduled_arms": int(overall["expected_arms"]),
        "completed_response_arms": len(responses),
        "complete_pairs": int(overall["complete_pairs"]),
        "failed_or_partial_pairs": int(overall["failed_or_partial_pairs"]),
        "provider_generations": generation_count,
        "epicure_calls": tool_calls,
        "epicure_successful_calls": tool_calls - tool_errors,
        "epicure_semantic_error_calls": tool_errors,
        "tool_active_on_arms": int(overall["epicure_on_tool_used"]),
        "completed_on_arms": int(overall["epicure_on_responses"]),
        "actual_cost_usd": float(summary_cost),
        "quality_judgments": 0,
    }
    if counts != {
        "models": 14,
        "task_families": 4,
        "scheduled_pairs": 56,
        "scheduled_arms": 112,
        "completed_response_arms": 103,
        "complete_pairs": 47,
        "failed_or_partial_pairs": 9,
        "provider_generations": 166,
        "epicure_calls": 67,
        "epicure_successful_calls": 25,
        "epicure_semantic_error_calls": 42,
        "tool_active_on_arms": 9,
        "completed_on_arms": 47,
        "actual_cost_usd": 3.329832,
        "quality_judgments": 0,
    }:
        raise CurrentPilotAssetError("final pilot totals changed")

    input_hashes = {
        "summary": {
            "filename": summary_path.name,
            "sha256": _file_sha256(summary_path),
            "content_address": summary["content_address"]["digest"],
        },
        "sources": {path.name: _file_sha256(path) for path in source_paths},
        "responses": {path.name: _file_sha256(path) for path in response_paths},
    }
    return VerifiedPilot(
        summary=summary,
        model_rows=tuple(model_rows),
        family_rows=tuple(family_rows),
        input_hashes=input_hashes,
        counts=counts,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise CurrentPilotAssetError(f"refusing empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _latex(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _render_table(rows: Sequence[Mapping[str, Any]]) -> str:
    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            " & ".join(
                [
                    _latex(row["display_name"]),
                    rf"\path{{{row['canonical_model_slug']}}}",
                    rf"\path{{{row['provider_tag']}}}",
                    f"{row['complete_pairs']}/{row['scheduled_pairs']}",
                    f"{row['tool_active_on_arms']}/{row['completed_on_arms']}",
                    (
                        f"{row['epicure_successful_calls']}/"
                        f"{row['epicure_semantic_error_calls']}"
                    ),
                    f"{row['latency_median_s']:.1f}",
                    f"{row['mean_actual_cost_per_scheduled_pair_usd']:.4f}",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{tabularx}{\textwidth}{@{}p{0.13\textwidth} X p{0.12\textwidth} r r r r r@{}}",
            r"\toprule",
            (
                r"Endpoint & Returned canonical identity & Provider route & Pairs & Tool arms & "
                r"MCP ok/err & Med. s & \$/pair \\"
            ),
            r"\midrule",
            *rendered_rows,
            r"\bottomrule",
            r"\end{tabularx}",
            "",
        ]
    )


def _render_macros(pilot: VerifiedPilot) -> str:
    counts = pilot.counts
    summary_hash = pilot.summary["content_address"]["digest"]
    completion_rate = 100 * counts["complete_pairs"] / counts["scheduled_pairs"]
    adoption_rate = 100 * counts["tool_active_on_arms"] / counts["completed_on_arms"]
    return "\n".join(
        [
            rf"\newcommand{{\CurrentPilotModelCount}}{{{counts['models']}}}",
            rf"\newcommand{{\CurrentPilotScheduledPairs}}{{{counts['scheduled_pairs']}}}",
            rf"\newcommand{{\CurrentPilotScheduledArms}}{{{counts['scheduled_arms']}}}",
            (
                rf"\newcommand{{\CurrentPilotCompletedResponseArms}}"
                rf"{{{counts['completed_response_arms']}}}"
            ),
            rf"\newcommand{{\CurrentPilotCompletePairs}}{{{counts['complete_pairs']}}}",
            rf"\newcommand{{\CurrentPilotFailedPairs}}{{{counts['failed_or_partial_pairs']}}}",
            rf"\newcommand{{\CurrentPilotPairCompletionRate}}{{{completion_rate:.1f}\%}}",
            (
                rf"\newcommand{{\CurrentPilotProviderGenerations}}"
                rf"{{{counts['provider_generations']}}}"
            ),
            rf"\newcommand{{\CurrentPilotEpicureCalls}}{{{counts['epicure_calls']}}}",
            (
                rf"\newcommand{{\CurrentPilotEpicureSuccessfulCalls}}"
                rf"{{{counts['epicure_successful_calls']}}}"
            ),
            (
                rf"\newcommand{{\CurrentPilotEpicureErrorCalls}}"
                rf"{{{counts['epicure_semantic_error_calls']}}}"
            ),
            (
                rf"\newcommand{{\CurrentPilotToolActiveArms}}"
                rf"{{{counts['tool_active_on_arms']}}}"
            ),
            rf"\newcommand{{\CurrentPilotCompletedOnArms}}{{{counts['completed_on_arms']}}}",
            rf"\newcommand{{\CurrentPilotToolAdoptionRate}}{{{adoption_rate:.1f}\%}}",
            rf"\newcommand{{\CurrentPilotActualCost}}{{\${counts['actual_cost_usd']:.6f}}}",
            rf"\newcommand{{\CurrentPilotQualityJudgments}}{{{counts['quality_judgments']}}}",
            rf"\newcommand{{\CurrentPilotSummaryHash}}{{{summary_hash}}}",
            "",
        ]
    )


def _configure_figures() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "axes.edgecolor": MID,
            "axes.linewidth": 0.65,
            "axes.labelcolor": INK,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _clean_axis(axis: mpl.axes.Axes, *, grid: str | None = "x") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid:
        axis.grid(axis=grid, color=LIGHT, linewidth=0.65, zorder=0)
    axis.set_axisbelow(True)


def _save_figure(figure: mpl.figure.Figure, pdf_path: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = pdf_path.with_suffix(".png")
    metadata = {
        "Title": pdf_path.stem,
        "Author": "Josef Chen",
        "Creator": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(
        pdf_path,
        format="pdf",
        facecolor="white",
        transparent=False,
        metadata=metadata,
    )
    figure.savefig(png_path, dpi=220, facecolor="white", transparent=False)
    plt.close(figure)
    return pdf_path, png_path


def _completion_figure(pilot: VerifiedPilot, output: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rows = list(pilot.model_rows)
    families = list(pilot.family_rows)
    figure, (model_axis, family_axis) = plt.subplots(
        1,
        2,
        figsize=(7.15, 4.45),
        gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.64},
    )
    y = np.arange(len(rows))[::-1]
    rates = np.asarray([100 * float(row["pair_completion_rate"]) for row in rows])
    lower = np.asarray([100 * float(row["pair_completion_wilson_lower_95"]) for row in rows])
    upper = np.asarray([100 * float(row["pair_completion_wilson_upper_95"]) for row in rows])
    model_axis.errorbar(
        rates,
        y,
        xerr=np.maximum(0, np.vstack((rates - lower, upper - rates))),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        markeredgecolor=INK,
        markeredgewidth=0.35,
        markersize=4.5,
        linewidth=1.0,
        capsize=1.8,
        zorder=3,
    )
    model_axis.set_yticks(y, [str(row["display_name"]) for row in rows])
    model_axis.set_xlim(0, 119)
    model_axis.set_xticks([0, 25, 50, 75, 100])
    model_axis.set_xlabel("Complete matched pairs (%)")
    model_axis.set_title("a  Endpoint completion in frozen manifest order", loc="left", pad=8)
    for yi, row in zip(y, rows, strict=True):
        model_axis.text(
            104,
            yi,
            f"{row['complete_pairs']}/{row['scheduled_pairs']}",
            va="center",
            fontsize=7.7,
            color=MID,
        )
    _clean_axis(model_axis)

    family_y = np.arange(len(families))[::-1] + 8
    family_rates = np.asarray(
        [100 * float(row["pair_completion_rate"]) for row in families]
    )
    family_lower = np.asarray(
        [100 * float(row["pair_completion_wilson_lower_95"]) for row in families]
    )
    family_upper = np.asarray(
        [100 * float(row["pair_completion_wilson_upper_95"]) for row in families]
    )
    family_axis.errorbar(
        family_rates,
        family_y,
        xerr=np.maximum(
            0,
            np.vstack((family_rates - family_lower, family_upper - family_rates)),
        ),
        fmt="s",
        color=ORANGE,
        ecolor=ORANGE,
        markeredgecolor=INK,
        markeredgewidth=0.35,
        markersize=4.5,
        linewidth=1.0,
        capsize=1.8,
        zorder=3,
    )
    family_axis.set_yticks(family_y, [str(row["display_name"]) for row in families])
    family_axis.set_xlim(0, 122)
    family_axis.set_xticks([0, 25, 50, 75, 100])
    family_axis.set_xlabel("Complete matched pairs (%)")
    family_axis.set_title("b  Completion by task family", loc="left", pad=8)
    for yi, row in zip(family_y, families, strict=True):
        family_axis.text(
            105,
            yi,
            f"{row['complete_pairs']}/{row['scheduled_pairs']}",
            va="center",
            fontsize=7.7,
            color=MID,
        )
    family_axis.set_ylim(-1.6, len(families) + 8.3)
    family_axis.text(
        0,
        5.7,
        (
            "Every missing pair was an Epicure-on\n"
            "tool-selection failure. Epicure-off: 56/56.\n\n"
            "These are execution outcomes, not\n"
            "culinary quality scores."
        ),
        fontsize=8.0,
        color=MID,
        va="top",
    )
    _clean_axis(family_axis)
    figure.suptitle(
        "Real 14-endpoint development pilot: paired answer completion",
        x=0.02,
        y=1.015,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
    )
    pdf, png = _save_figure(figure, output)
    return {
        "figure": pdf.name,
        "preview": png.name,
        "chart_type": "manifest-ordered Wilson interval forest plots",
        "question": "Which scheduled pairs yielded two normalized answers?",
        "interval": "Wilson 95% interval",
        "ordering": "frozen manifest order, not estimated quality",
        "denominators": {"per_model": 4, "per_family": 14},
    }


def _efficiency_figure(pilot: VerifiedPilot, output: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rows = list(pilot.model_rows)
    figure, (latency_axis, cost_axis) = plt.subplots(
        1,
        2,
        figsize=(7.15, 4.45),
        sharey=True,
        gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.17},
    )
    y = np.arange(len(rows))[::-1]
    medians = np.asarray([float(row["latency_median_s"]) for row in rows])
    q1 = np.asarray([float(row["latency_q1_s"]) for row in rows])
    q3 = np.asarray([float(row["latency_q3_s"]) for row in rows])
    latency_axis.errorbar(
        medians,
        y,
        xerr=np.vstack((medians - q1, q3 - medians)),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        markeredgecolor=INK,
        markeredgewidth=0.35,
        markersize=4.4,
        linewidth=1.0,
        capsize=1.8,
        zorder=3,
    )
    latency_axis.set_xscale("log")
    latency_axis.set_yticks(y, [str(row["display_name"]) for row in rows])
    latency_axis.set_xlabel("Completed-arm latency (s, log scale)")
    latency_axis.set_title("a  Median and interquartile range", loc="left", pad=8)
    _clean_axis(latency_axis)

    mean_cost = np.asarray(
        [float(row["mean_actual_cost_per_scheduled_pair_usd"]) for row in rows]
    )
    min_cost = np.asarray([float(row["min_actual_pair_cost_usd"]) for row in rows])
    max_cost = np.asarray([float(row["max_actual_pair_cost_usd"]) for row in rows])
    cost_axis.errorbar(
        mean_cost,
        y,
        xerr=np.vstack((mean_cost - min_cost, max_cost - mean_cost)),
        fmt="s",
        color=ORANGE,
        ecolor=ORANGE,
        markeredgecolor=INK,
        markeredgewidth=0.35,
        markersize=4.3,
        linewidth=1.0,
        capsize=1.8,
        zorder=3,
    )
    cost_axis.set_xscale("log")
    cost_axis.set_xlabel("Actual cost per scheduled pair (USD, log scale)")
    cost_axis.set_title("b  Mean and observed range, n=4", loc="left", pad=8)
    cost_axis.tick_params(axis="y", labelleft=False)
    _clean_axis(cost_axis)
    figure.suptitle(
        "Provider efficiency in the real development pilot",
        x=0.02,
        y=1.015,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        (
            "Latency uses completed arms only (6 to 8 per endpoint). Cost retains all four "
            "scheduled pairs, including failed generations."
        ),
        ha="center",
        color=MID,
        fontsize=7.8,
    )
    pdf, png = _save_figure(figure, output)
    return {
        "figure": pdf.name,
        "preview": png.name,
        "chart_type": "manifest-ordered log-scale interval plots",
        "question": "What latency and reconciled provider cost did the run incur?",
        "latency_denominator": "completed response arms per endpoint",
        "cost_denominator": "four scheduled pairs per endpoint",
        "ordering": "frozen manifest order, not estimated quality",
    }


def _tool_figure(pilot: VerifiedPilot, output: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    rows = list(pilot.model_rows)
    figure, (adoption_axis, calls_axis) = plt.subplots(
        1,
        2,
        figsize=(7.15, 4.45),
        sharey=True,
        gridspec_kw={"width_ratios": [1.25, 1.05], "wspace": 0.14},
    )
    y = np.arange(len(rows))[::-1]
    rates = np.asarray(
        [100 * float(row["tool_adoption_rate_among_completed_on"]) for row in rows]
    )
    lower = np.asarray([100 * float(row["tool_adoption_wilson_lower_95"]) for row in rows])
    upper = np.asarray([100 * float(row["tool_adoption_wilson_upper_95"]) for row in rows])
    adoption_axis.errorbar(
        rates,
        y,
        xerr=np.maximum(0, np.vstack((rates - lower, upper - rates))),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        markeredgecolor=INK,
        markeredgewidth=0.35,
        markersize=4.4,
        linewidth=1.0,
        capsize=1.8,
        zorder=3,
    )
    adoption_axis.set_yticks(y, [str(row["display_name"]) for row in rows])
    adoption_axis.set_xlim(0, 121)
    adoption_axis.set_xticks([0, 25, 50, 75, 100])
    adoption_axis.set_xlabel("Completed Epicure-on arms using a tool (%)")
    adoption_axis.set_title("a  Selective tool adoption", loc="left", pad=8)
    for yi, row in zip(y, rows, strict=True):
        adoption_axis.text(
            104,
            yi,
            f"{row['tool_active_on_arms']}/{row['completed_on_arms']}",
            va="center",
            fontsize=7.7,
            color=MID,
        )
    _clean_axis(adoption_axis)

    successful = np.asarray([int(row["epicure_successful_calls"]) for row in rows])
    errors = np.asarray([int(row["epicure_semantic_error_calls"]) for row in rows])
    calls_axis.barh(y, successful, height=0.56, color=SKY, label="Successful result", zorder=2)
    calls_axis.barh(
        y,
        errors,
        left=successful,
        height=0.56,
        color=ORANGE,
        label="Semantic error result",
        zorder=2,
    )
    calls_axis.set_xlabel("Recorded Epicure calls")
    calls_axis.set_title("b  Tool-result outcomes", loc="left", pad=8)
    calls_axis.tick_params(axis="y", labelleft=False)
    calls_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=7.6,
        handlelength=1.3,
        columnspacing=0.9,
    )
    for yi, ok, error in zip(y, successful, errors, strict=True):
        calls_axis.text(
            ok + error + 0.45,
            yi,
            str(ok + error),
            va="center",
            fontsize=7.6,
            color=MID,
        )
    calls_axis.set_xlim(0, max(successful + errors) + 4)
    calls_axis.set_ylim(-0.9, 15.0)
    _clean_axis(calls_axis)
    figure.suptitle(
        "Epicure adoption and tool-result reliability",
        x=0.02,
        y=1.015,
        ha="left",
        fontsize=10.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        (
            "Tool use was optional. Semantic errors remain in the trace and are not treated "
            "as culinary evidence or model-quality outcomes."
        ),
        ha="center",
        color=MID,
        fontsize=7.8,
    )
    pdf, png = _save_figure(figure, output)
    return {
        "figure": pdf.name,
        "preview": png.name,
        "chart_type": "Wilson interval forest plot and stacked call counts",
        "question": "Which completed Epicure-on arms called tools, and what did the calls return?",
        "adoption_denominator": "completed Epicure-on response arms per endpoint",
        "call_total": pilot.counts["epicure_calls"],
        "ordering": "frozen manifest order, not estimated quality",
    }


def render_assets(pilot: VerifiedPilot, output_dir: Path) -> Path:
    """Write data, TeX, vector figures, previews, and content-addressed provenance."""

    import matplotlib as mpl

    output_dir.mkdir(parents=True, exist_ok=True)
    model_csv = output_dir / "current-frontier-pilot-model-metrics.csv"
    family_csv = output_dir / "current-frontier-pilot-family-metrics.csv"
    table_tex = output_dir / "current-frontier-pilot-table.tex"
    macros_tex = output_dir / "current-frontier-pilot-macros.tex"
    _write_csv(model_csv, pilot.model_rows)
    _write_csv(family_csv, pilot.family_rows)
    table_tex.write_text(_render_table(pilot.model_rows), encoding="utf-8")
    macros_tex.write_text(_render_macros(pilot), encoding="utf-8")

    _configure_figures()
    figure_specs = [
        _completion_figure(pilot, output_dir / "current-frontier-pilot-reliability.pdf"),
        _efficiency_figure(pilot, output_dir / "current-frontier-pilot-efficiency.pdf"),
        _tool_figure(pilot, output_dir / "current-frontier-pilot-tools.pdf"),
    ]
    outputs = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "current-frontier-pilot-provenance.json"
    )
    payload: dict[str, Any] = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "permitted_claims": [
                "real provider execution",
                "exact returned model identity",
                "response reliability",
                "optional Epicure tool adoption and call outcomes",
                "reconciled cost and observed latency",
            ],
            "prohibited_claims": [
                "model quality ranking",
                "Epicure quality uplift",
                "confirmatory benchmark result",
            ],
        },
        "counts": pilot.counts,
        "renderer": {
            "library": "matplotlib",
            "version": mpl.__version__,
            "background": "white",
            "paper_format": "vector-pdf",
            "preview_format": "png",
            "font_embedding": "TrueType",
        },
        "figures": figure_specs,
        "inputs": pilot.input_hashes,
        "outputs": {path.name: _file_sha256(path) for path in outputs},
    }
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    provenance = output_dir / "current-frontier-pilot-provenance.json"
    provenance.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return provenance


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--response-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    pilot = verify_pilot(arguments.summary, arguments.source_dir, arguments.response_dir)
    provenance = render_assets(pilot, arguments.output_dir)
    print(
        json.dumps(
            {
                "provenance": str(provenance),
                "models": pilot.counts["models"],
                "scheduled_pairs": pilot.counts["scheduled_pairs"],
                "complete_pairs": pilot.counts["complete_pairs"],
                "quality_judgments": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
