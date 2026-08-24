"""Verify and render a multi-run exact-frontier development pilot.

The output of this module is deliberately operational.  It combines immutable
real-call runs only when they share an execution-policy hash, verifies every
source and normalized response, and reports reliability, tool use, latency, and
cost without inferring a model-quality ordering.  Human judgments are a separate
input and are never synthesized here.
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

from .current_pilot_assets import (
    COHERE_MODEL_ORDER,
    DISPLAY_NAMES,
    EXTENDED_MODEL_ORDER,
    FAMILY_NAMES,
    FAMILY_ORDER,
    MODEL_ORDER,
)
from .engine import is_complete_finish_reason
from .frontier_contract_runner import ArtifactExposure, scan_live_smoke_artifacts
from .real_task_bank import sha256_json

SUMMARY_SCHEMA_VERSION = "flavourbench-real-exploratory-summary-v1"
SOURCE_SCHEMA_VERSION = "flavourbench-live-smoke-v1"
RESPONSE_SCHEMA_VERSION = "flavourbench-real-exploratory-response-v1"
ASSET_SCHEMA_VERSION = "flavourbench-frontier-multirun-assets-v1"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#8A93A0"
LIGHT = "#E6EAF0"
INK = "#202124"


class FrontierMultirunAssetError(RuntimeError):
    """An integrity check or publication boundary failed."""


@dataclass(frozen=True)
class RunInput:
    """Paths belonging to one immutable runner summary."""

    summary: Path
    sources: Path
    responses: Path


@dataclass(frozen=True)
class VerifiedFrontierPilot:
    """Fully checked rows and their content-addressed aggregate."""

    aggregate: dict[str, Any]
    model_rows: tuple[dict[str, Any], ...]
    family_rows: tuple[dict[str, Any], ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrontierMultirunAssetError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise FrontierMultirunAssetError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_summary(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    content_address = document.get("content_address")
    if not isinstance(content_address, Mapping):
        raise FrontierMultirunAssetError(f"summary has no content address: {path.name}")
    digest = str(content_address.get("digest") or "")
    payload = {key: value for key, value in document.items() if key != "content_address"}
    if digest != sha256_json(payload) or digest not in path.name:
        raise FrontierMultirunAssetError(f"summary hash does not verify: {path.name}")
    if (
        document.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or document.get("official") is not False
        or document.get("rank_eligible") is not False
        or document.get("research_result") is not False
    ):
        raise FrontierMultirunAssetError("an input crossed the development-only boundary")
    if document.get("mode") != "execute":
        raise FrontierMultirunAssetError("only completed real-call execute summaries may be pooled")
    source = document.get("task_selection", {}).get("source", {})
    if (
        not isinstance(source, Mapping)
        or source.get("synthetic_tasks") != 0
        or source.get("rank_eligible") is not False
        or source.get("confirmatory_eligible") is not False
        or source.get("source_class") != "licensed_real_human_authored_public_questions"
    ):
        raise FrontierMultirunAssetError("task source is not the frozen zero-synthetic bank")
    return document


def _verify_artifact(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    digest = str(document.get("artifact_sha256") or "")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if document.get("schema_version") == SOURCE_SCHEMA_VERSION:
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    elif document.get("schema_version") == RESPONSE_SCHEMA_VERSION:
        expected = sha256_json(payload)
    else:
        raise FrontierMultirunAssetError(f"unexpected artifact schema: {path.name}")
    if digest != expected or digest[:12] not in path.name:
        raise FrontierMultirunAssetError(f"artifact hash does not verify: {path.name}")
    if (
        document.get("official") is not False
        or document.get("rank_eligible") is not False
        or document.get("research_result") is not False
    ):
        raise FrontierMultirunAssetError("a raw artifact crossed its claim boundary")
    return document


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise FrontierMultirunAssetError("invalid Wilson interval input")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile, method="linear"))


def _manifest_contracts(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = summary.get("manifest")
    models = manifest.get("models") if isinstance(manifest, Mapping) else None
    if not isinstance(models, list) or not models or len(models) > len(EXTENDED_MODEL_ORDER):
        raise FrontierMultirunAssetError(
            "summary does not contain a non-empty frozen-panel manifest"
        )
    contracts: dict[str, dict[str, Any]] = {}
    for row in models:
        if not isinstance(row, dict):
            raise FrontierMultirunAssetError("invalid model manifest row")
        model_id = str(row.get("model_id") or "")
        if model_id in contracts:
            raise FrontierMultirunAssetError("duplicate model identity in manifest")
        contracts[model_id] = row
    if not set(contracts) <= set(EXTENDED_MODEL_ORDER):
        raise FrontierMultirunAssetError("targeted run contains a model outside the panel")
    return contracts


def _tool_trace_metrics(response: Mapping[str, Any], *, condition: str) -> tuple[int, int]:
    trace = response.get("tool_trace")
    if not isinstance(trace, list):
        raise FrontierMultirunAssetError("response tool trace is not an array")
    successes = 0
    for event in trace:
        if not isinstance(event, Mapping):
            raise FrontierMultirunAssetError("tool trace event is not an object")
        result = event.get("result")
        digest = event.get("result_sha256")
        if (
            not isinstance(event.get("arguments"), Mapping)
            or not isinstance(result, str)
            or digest != hashlib.sha256(result.encode()).hexdigest()
        ):
            raise FrontierMultirunAssetError("tool trace content address does not verify")
        successes += int(event.get("is_error") is False)
    if condition == "epicure_off" and trace:
        raise FrontierMultirunAssetError("Epicure-off arm contains a tool call")
    if condition == "epicure_on" and successes == 0:
        raise FrontierMultirunAssetError(
            "normalized Epicure-on arm has no successful real tool call"
        )
    return len(trace), successes


def _source_costs(exposure: ArtifactExposure) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """Return reconciled, rate-card, unresolved reserve, and total exposure."""

    basis = exposure.exposure_basis
    reconciled = (
        exposure.actual_cost_usd
        if basis
        in {
            "fully_reconciled_actual",
            "failed_but_all_attempts_cost_reconciled_actual",
        }
        else Decimal(0)
    )
    estimated = exposure.actual_cost_usd if "rate_card_estimated" in basis else Decimal(0)
    unresolved = (
        exposure.exposure_usd
        if basis
        in {
            "failed_or_unreconciled_full_admitted_allowance",
            "resolved_conservative_full_admitted_allowance",
        }
        else Decimal(0)
    )
    return reconciled, estimated, unresolved, exposure.exposure_usd


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "flavourbench-v1",
        }
    )


def verify_runs(inputs: Sequence[RunInput]) -> VerifiedFrontierPilot:
    """Verify real-call runs and combine one common frozen protocol stratum."""

    if not inputs:
        raise FrontierMultirunAssetError("at least one run is required")
    summaries = [_verify_summary(item.summary) for item in inputs]
    policy_hashes = {str(summary.get("execution_policy_sha256") or "") for summary in summaries}
    if len(policy_hashes) != 1 or "" in policy_hashes:
        raise FrontierMultirunAssetError("runs with different execution policies cannot be pooled")
    contract_by_model: dict[str, dict[str, Any]] = {}
    task_prompts: dict[str, str] = {}
    experimental_units: set[tuple[str, str]] = set()
    work_item_ids: set[str] = set()
    generation_ids: set[str] = set()
    response_hashes: set[str] = set()
    all_source_hashes: set[str] = set()
    model_acc: dict[str, dict[str, Any]] = {
        model_id: {
            "coverage": Counter(),
            "latencies": [],
            "tool_calls": 0,
            "tool_successes": 0,
            "reconciled_cost": Decimal(0),
            "rate_card_cost": Decimal(0),
            "unresolved_reserve": Decimal(0),
            "exposure": Decimal(0),
        }
        for model_id in EXTENDED_MODEL_ORDER
    }
    family_acc = {family: Counter() for family in FAMILY_ORDER}
    overall = Counter()
    input_records: list[dict[str, Any]] = []

    for run_number, (run_input, summary) in enumerate(zip(inputs, summaries, strict=True), start=1):
        contracts = _manifest_contracts(summary)
        for model_id, contract in contracts.items():
            prior = contract_by_model.setdefault(model_id, contract)
            stable_fields = (
                "canonical_model_slug",
                "provider_tag",
                "execution_backend",
                "endpoint_execution_sha256",
            )
            if any(contract.get(field) != prior.get(field) for field in stable_fields):
                raise FrontierMultirunAssetError(
                    f"route or canonical identity changed across runs: {model_id}"
                )
        workload = summary.get("workload")
        items = workload.get("work_items") if isinstance(workload, Mapping) else None
        if not isinstance(items, list):
            raise FrontierMultirunAssetError("summary workload is missing")
        work_by_id: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                raise FrontierMultirunAssetError("invalid work item")
            work_item_id = str(item.get("work_item_id") or "")
            model_id = str(item.get("model_id") or "")
            task_id = str(item.get("task_id") or "")
            prompt_sha256 = str(item.get("prompt_sha256") or "")
            family = str(item.get("task_family") or "")
            contract = contracts.get(model_id)
            unit = (model_id, task_id)
            if (
                not work_item_id
                or not task_id
                or not prompt_sha256
                or work_item_id in work_item_ids
                or work_item_id in work_by_id
                or unit in experimental_units
                or family not in FAMILY_ORDER
                or contract is None
                or item.get("canonical_model_slug") != contract.get("canonical_model_slug")
                or item.get("provider_tag") != contract.get("provider_tag")
            ):
                raise FrontierMultirunAssetError("duplicate or invalid experimental unit")
            prior_prompt = task_prompts.setdefault(task_id, prompt_sha256)
            if prior_prompt != prompt_sha256:
                raise FrontierMultirunAssetError("task ID was reused with changed content")
            work_by_id[work_item_id] = item
            work_item_ids.add(work_item_id)
            experimental_units.add(unit)
        expected_pairs = int(workload.get("expected_pair_count") or 0)
        if len(items) != expected_pairs:
            raise FrontierMultirunAssetError("workload pair count does not reconcile")

        sources: dict[str, dict[str, Any]] = {}
        source_paths = sorted(run_input.sources.glob("*.json"))
        scanned = scan_live_smoke_artifacts(run_input.sources)
        exposure_by_sha = {item.artifact_sha256: item for item in scanned.artifacts}
        if len(exposure_by_sha) != len(source_paths):
            raise FrontierMultirunAssetError("source exposure scan does not cover every artifact")
        for path in source_paths:
            source = _verify_artifact(path)
            work_item_id = str(source.get("dataset_work_item_id") or "")
            item = work_by_id.get(work_item_id)
            if item is None or work_item_id in sources:
                raise FrontierMultirunAssetError("source has unknown or duplicate work item")
            model_id = str(item["model_id"])
            contract = contracts[model_id]
            if (
                source.get("requested_model_id") != model_id
                or source.get("requested_provider") != contract.get("provider_tag")
                or source.get("category") != item.get("task_family")
                or source.get("dataset_task_id") != item.get("task_id")
            ):
                raise FrontierMultirunAssetError("source provenance does not match workload")
            source_hash = str(source["artifact_sha256"])
            if source_hash in all_source_hashes:
                raise FrontierMultirunAssetError("duplicate source artifact across runs")
            all_source_hashes.add(source_hash)
            sources[work_item_id] = source
            reconciled, estimated, unresolved, exposure = _source_costs(
                exposure_by_sha[source_hash]
            )
            accumulator = model_acc[model_id]
            accumulator["reconciled_cost"] += reconciled
            accumulator["rate_card_cost"] += estimated
            accumulator["unresolved_reserve"] += unresolved
            accumulator["exposure"] += exposure
            metadata: list[Mapping[str, Any]] = []
            results = source.get("results")
            if not isinstance(results, Mapping) or set(results) - {"epicure_off", "epicure_on"}:
                raise FrontierMultirunAssetError("source result set is invalid")
            for result in results.values():
                if not isinstance(result, Mapping):
                    raise FrontierMultirunAssetError("source result is not an object")
                metadata.extend(result.get("generation_metadata") or [])
            metadata.extend(source.get("incomplete_generation_metadata") or [])
            for generation in metadata:
                generation_id = str(generation.get("generation_id") or "")
                if (
                    not generation_id
                    or generation_id in generation_ids
                    or generation.get("model") != contract.get("canonical_model_slug")
                ):
                    raise FrontierMultirunAssetError(
                        "provider generation identity does not reconcile"
                    )
                generation_ids.add(generation_id)

        responses: dict[tuple[str, str], dict[str, Any]] = {}
        response_paths = sorted(run_input.responses.glob("*.json"))
        for path in response_paths:
            document = _verify_artifact(path)
            if document.get("schema_version") != RESPONSE_SCHEMA_VERSION:
                raise FrontierMultirunAssetError("unexpected response schema")
            work_item_id = str(document.get("work_item_id") or "")
            condition = str(document.get("condition") or "")
            key = (work_item_id, condition)
            item = work_by_id.get(work_item_id)
            source = sources.get(work_item_id)
            if (
                item is None
                or source is None
                or key in responses
                or condition
                not in {
                    "epicure_off",
                    "epicure_on",
                }
            ):
                raise FrontierMultirunAssetError("response has invalid or duplicate identity")
            model_id = str(item["model_id"])
            contract = contracts[model_id]
            model = document.get("model")
            response = document.get("response")
            source_link = document.get("source")
            task = document.get("task")
            if not all(
                isinstance(value, Mapping) for value in (model, response, source_link, task)
            ):
                raise FrontierMultirunAssetError("response provenance block is missing")
            assert isinstance(model, Mapping)
            assert isinstance(response, Mapping)
            assert isinstance(source_link, Mapping)
            assert isinstance(task, Mapping)
            if (
                model.get("requested_model_id") != model_id
                or model.get("canonical_model_slug") != contract.get("canonical_model_slug")
                or model.get("actual_model_id") != contract.get("canonical_model_slug")
                or response.get("actual_model_id") != contract.get("canonical_model_slug")
                or source_link.get("artifact_sha256") != source.get("artifact_sha256")
                or task.get("public_id") != item.get("task_id")
                or task.get("family") != item.get("task_family")
                or task.get("prompt_sha256") != item.get("prompt_sha256")
                or task.get("review_status") != "candidate"
                or document.get("research_release_eligible") is not False
                or not is_complete_finish_reason(str(response.get("finish_reason") or ""))
            ):
                raise FrontierMultirunAssetError(
                    "response model, task, or source identity mismatch"
                )
            prompt_hash = str(task["prompt_sha256"])
            if task_prompts.get(str(task["public_id"])) != prompt_hash:
                raise FrontierMultirunAssetError("task ID was reused with changed content")
            tool_calls, tool_successes = _tool_trace_metrics(response, condition=condition)
            accumulator = model_acc[model_id]
            accumulator["tool_calls"] += tool_calls
            accumulator["tool_successes"] += tool_successes
            latency = response.get("latency_ms")
            if not isinstance(latency, int | float) or isinstance(latency, bool) or latency < 0:
                raise FrontierMultirunAssetError("response latency is invalid")
            accumulator["latencies"].append(float(latency) / 1000)
            digest = str(document["artifact_sha256"])
            if digest in response_hashes:
                raise FrontierMultirunAssetError("duplicate response artifact")
            response_hashes.add(digest)
            responses[key] = document

        # Replenishment runs may freeze a strict subset of the panel.  Their
        # per-run reliability block therefore contains only scheduled models;
        # the union-of-runs check below still requires the complete panel.
        run_model_counts = {model_id: Counter() for model_id in contracts}
        run_family_counts = {family: Counter() for family in FAMILY_ORDER}
        run_overall = Counter()
        for work_item_id, item in work_by_id.items():
            model_id = str(item["model_id"])
            family = str(item["task_family"])
            off = responses.get((work_item_id, "epicure_off"))
            on = responses.get((work_item_id, "epicure_on"))
            finalized = work_item_id in sources
            complete = off is not None and on is not None
            metrics = Counter(
                {
                    "expected_pairs": 1,
                    "expected_arms": 2,
                    "source_attempts": int(finalized),
                    "finalized_pairs": int(finalized),
                    "epicure_off_responses": int(off is not None),
                    "epicure_on_responses": int(on is not None),
                    "complete_pairs": int(complete),
                    "failed_or_partial_pairs": int(finalized and not complete),
                    "epicure_on_tool_used": int(on is not None),
                }
            )
            run_model_counts[model_id].update(metrics)
            run_family_counts[family].update(metrics)
            run_overall.update(metrics)
            model_acc[model_id]["coverage"].update(metrics)
            family_acc[family].update(metrics)
            overall.update(metrics)
        reported = summary.get("coverage_and_reliability")
        if not isinstance(reported, Mapping):
            raise FrontierMultirunAssetError("summary coverage block is missing")
        if (
            {key: dict(value) for key, value in run_model_counts.items()}
            != reported.get("by_model")
            or {key: dict(value) for key, value in run_family_counts.items()}
            != reported.get("by_task_family")
            or dict(run_overall) != reported.get("overall")
        ):
            raise FrontierMultirunAssetError(
                "summary coverage does not reproduce from raw artifacts"
            )
        input_records.append(
            {
                "run_number": run_number,
                "summary_filename": run_input.summary.name,
                "summary_sha256": _file_sha256(run_input.summary),
                "summary_content_address": summary["content_address"]["digest"],
                "manifest_sha256": summary["manifest"]["sha256"],
                "source_files": len(source_paths),
                "response_files": len(response_paths),
                "source_directory_sha256": sha256_json(
                    {path.name: _file_sha256(path) for path in source_paths}
                ),
                "response_directory_sha256": sha256_json(
                    {path.name: _file_sha256(path) for path in response_paths}
                ),
            }
        )

    if not set(MODEL_ORDER) <= set(contract_by_model):
        missing = sorted(set(MODEL_ORDER) - set(contract_by_model))
        raise FrontierMultirunAssetError(
            "pooled runs do not cover the complete frozen model panel: "
            + ", ".join(missing)
        )
    optional_models = frozenset(set(contract_by_model) - set(MODEL_ORDER))
    if optional_models not in {frozenset(), frozenset(COHERE_MODEL_ORDER)}:
        raise FrontierMultirunAssetError(
            "optional Cohere extension must contain both frozen direct endpoints"
        )
    active_model_order = tuple(
        model_id for model_id in EXTENDED_MODEL_ORDER if model_id in contract_by_model
    )
    model_rows: list[dict[str, Any]] = []
    for ordinal, model_id in enumerate(active_model_order, start=1):
        accumulator = model_acc[model_id]
        coverage = accumulator["coverage"]
        scheduled = int(coverage["expected_pairs"])
        complete = int(coverage["complete_pairs"])
        lower, upper = _wilson(complete, scheduled)
        calls = int(accumulator["tool_calls"])
        successes = int(accumulator["tool_successes"])
        latencies = list(accumulator["latencies"])
        contract = contract_by_model[model_id]
        provider_charge_available = contract["execution_backend"] != "cohere_direct"
        conservative_exposure = (
            float(accumulator["exposure"]) if provider_charge_available else None
        )
        row = {
            "manifest_ordinal": ordinal,
            "display_name": DISPLAY_NAMES[model_id],
            "model_id": model_id,
            "canonical_model_slug": contract["canonical_model_slug"],
            "provider_tag": contract["provider_tag"],
            "execution_backend": contract["execution_backend"],
            "scheduled_pairs": scheduled,
            "finalized_pairs": int(coverage["finalized_pairs"]),
            "complete_pairs": complete,
            "failed_or_partial_pairs": int(coverage["failed_or_partial_pairs"]),
            "pair_completion_rate": complete / scheduled,
            "pair_completion_wilson_lower_95": lower,
            "pair_completion_wilson_upper_95": upper,
            "completed_off_arms": int(coverage["epicure_off_responses"]),
            "completed_on_arms": int(coverage["epicure_on_responses"]),
            "tool_active_on_arms": int(coverage["epicure_on_tool_used"]),
            "epicure_calls": calls,
            "epicure_successful_calls": successes,
            "epicure_semantic_error_calls": calls - successes,
            "tool_call_success_rate": successes / calls if calls else None,
            "completed_arms_for_latency": len(latencies),
            "latency_median_s": statistics.median(latencies) if latencies else None,
            "latency_q1_s": _percentile(latencies, 25),
            "latency_q3_s": _percentile(latencies, 75),
            "provider_reconciled_actual_cost_usd": float(accumulator["reconciled_cost"]),
            "direct_rate_card_estimated_cost_usd": float(accumulator["rate_card_cost"]),
            "unresolved_full_allowance_usd": float(accumulator["unresolved_reserve"]),
            "conservative_cost_exposure_usd": conservative_exposure,
            "provider_charge_available": provider_charge_available,
            "cost_display_status": (
                "known_usd_exposure" if provider_charge_available else "provider_charge_unavailable"
            ),
            "minimum_eight_complete_pairs": complete >= 8,
            "quality_judgments": 0,
            "quality_status": "awaiting_blinded_human_judgment" if complete else "not_estimable",
            "rank_eligible": False,
        }
        model_rows.append(row)

    family_rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        counts = family_acc[family]
        scheduled = int(counts["expected_pairs"])
        complete = int(counts["complete_pairs"])
        lower, upper = _wilson(complete, scheduled)
        family_rows.append(
            {
                "family": family,
                "display_name": FAMILY_NAMES[family],
                "scheduled_pairs": scheduled,
                "finalized_pairs": int(counts["finalized_pairs"]),
                "complete_pairs": complete,
                "failed_or_partial_pairs": int(counts["failed_or_partial_pairs"]),
                "pair_completion_rate": complete / scheduled,
                "pair_completion_wilson_lower_95": lower,
                "pair_completion_wilson_upper_95": upper,
            }
        )

    totals = {
        "runs": len(inputs),
        "models": len(active_model_order),
        "task_families": len(FAMILY_ORDER),
        "distinct_tasks": len(task_prompts),
        "scheduled_pairs": int(overall["expected_pairs"]),
        "finalized_pairs": int(overall["finalized_pairs"]),
        "complete_pairs": int(overall["complete_pairs"]),
        "failed_or_partial_pairs": int(overall["failed_or_partial_pairs"]),
        "completed_response_arms": len(response_hashes),
        "provider_generation_ids": len(generation_ids),
        "epicure_calls": sum(int(row["epicure_calls"]) for row in model_rows),
        "epicure_successful_calls": sum(int(row["epicure_successful_calls"]) for row in model_rows),
        "models_with_at_least_eight_complete_pairs": sum(
            bool(row["minimum_eight_complete_pairs"]) for row in model_rows
        ),
        "synthetic_tasks": 0,
        "quality_judgments": 0,
    }
    unpriced_models = [
        str(row["model_id"]) for row in model_rows if not row["provider_charge_available"]
    ]
    cost = {
        "currency": "USD",
        "provider_reconciled_actual_cost_usd": sum(
            float(row["provider_reconciled_actual_cost_usd"]) for row in model_rows
        ),
        "direct_rate_card_estimated_cost_usd": sum(
            float(row["direct_rate_card_estimated_cost_usd"]) for row in model_rows
        ),
        "unresolved_full_allowance_usd": sum(
            float(row["unresolved_full_allowance_usd"]) for row in model_rows
        ),
        "known_conservative_exposure_subtotal_usd": sum(
            float(row["conservative_cost_exposure_usd"])
            for row in model_rows
            if row["conservative_cost_exposure_usd"] is not None
        ),
        "provider_charge_complete": not unpriced_models,
        "unpriced_model_ids": unpriced_models,
    }
    task_records = [
        {"task_id": task_id, "prompt_sha256": task_prompts[task_id]}
        for task_id in sorted(task_prompts)
    ]
    aggregate: dict[str, Any] = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "status": "verified_real_development_pilot",
        "official": False,
        "rank_eligible": False,
        "quality_ranking": False,
        "execution_policy_sha256": next(iter(policy_hashes)),
        "task_set_sha256": sha256_json(task_records),
        "tasks": task_records,
        "task_source": "licensed_real_human_authored_public_questions",
        "synthetic_tasks": 0,
        "inputs": input_records,
        "totals": totals,
        "cost": cost,
        "model_rows": model_rows,
        "family_rows": family_rows,
        "limitations": [
            "Operational completion is not a model-quality score.",
            "No preference or rubric judgment is present in this artifact.",
            "Direct Kimi charges are frozen-rate-card estimates when provider charges "
            "are unavailable.",
            "Direct Cohere token usage is retained, but no USD provider charge was returned. "
            "Cohere endpoints are unpriced in tables and figures.",
            "Ambiguous deliveries retain their full admitted allowance and are never "
            "replayed blindly.",
        ],
    }
    aggregate["artifact_sha256"] = sha256_json(aggregate)
    return VerifiedFrontierPilot(aggregate, tuple(model_rows), tuple(family_rows))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_reliability_figure(
    rows: Sequence[Mapping[str, Any]], output: Path, *, stratum_label: str
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    labels = [str(row["display_name"]) for row in rows][::-1]
    rates = np.asarray([float(row["pair_completion_rate"]) for row in rows][::-1])
    lower = np.asarray([float(row["pair_completion_wilson_lower_95"]) for row in rows][::-1])
    upper = np.asarray([float(row["pair_completion_wilson_upper_95"]) for row in rows][::-1])
    y = np.arange(len(rows))
    figure_height = max(5.15, 0.32 * len(rows) + 0.8)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.15, figure_height), gridspec_kw={"wspace": 0.48}
    )
    ax = axes[0]
    ax.barh(y, 100 * rates, height=0.58, color=BLUE, alpha=0.88)
    ax.errorbar(
        100 * rates,
        y,
        # Wilson endpoints can differ from an exact boundary by a few ULPs
        # (for example, a zero-success row may have a lower bound of 2.8e-17).
        # Matplotlib rejects those harmless round-off residues as negative
        # error lengths, so clamp only the plotted distance at zero.
        xerr=np.vstack(
            (
                100 * np.maximum(0.0, rates - lower),
                100 * np.maximum(0.0, upper - rates),
            )
        ),
        fmt="none",
        ecolor=INK,
        elinewidth=0.7,
        capsize=2,
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 120)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Complete matched pairs (%)")
    ax.set_title(
        f"a  {stratum_label.capitalize()} pair completion",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axvline(100, color=INK, linewidth=0.55)
    for index, row in enumerate(rows[::-1]):
        ax.text(
            110,
            index,
            f"{row['complete_pairs']}/{row['scheduled_pairs']}",
            ha="center",
            va="center",
            fontsize=6.8,
            color=INK,
        )

    ax = axes[1]
    success = [
        100 * float(row["tool_call_success_rate"])
        if row["tool_call_success_rate"] is not None
        else 0
        for row in rows[::-1]
    ]
    errors = [
        100 - value if int(row["epicure_calls"]) else 0
        for value, row in zip(success, rows[::-1], strict=True)
    ]
    ax.barh(y, success, height=0.58, color=GREEN, label="successful")
    ax.barh(y, errors, left=success, height=0.58, color=ORANGE, label="semantic error")
    ax.set_yticks(y, ["" for _ in labels])
    ax.set_xlim(0, 120)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Recorded Epicure calls (%)")
    ax.set_title("b  Tool-call outcomes", loc="left", fontweight="bold")
    ax.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axvline(100, color=INK, linewidth=0.55)
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    for index, row in enumerate(rows[::-1]):
        calls = int(row["epicure_calls"])
        label = f"n={calls}" if calls else "n=0"
        ax.text(110, index, label, ha="center", va="center", fontsize=6.8, color=INK)

    fig.suptitle(
        "Exact-frontier development pilot: execution reliability, not model quality",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.14)
    fig.text(
        0.06,
        0.012,
        (
            f"Bars show real matched Epicure-off/on runs in the frozen {stratum_label} "
            "stratum; whiskers are 95% Wilson intervals."
        ),
        fontsize=6.7,
        color="#59636E",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _render_cost_latency_figure(
    rows: Sequence[Mapping[str, Any]], output: Path, *, stratum_label: str
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    has_unpriced = any(
        row["latency_median_s"] is not None
        and row["conservative_cost_exposure_usd"] is None
        for row in rows
    )
    if has_unpriced:
        fig, (ax, unpriced_ax) = plt.subplots(
            2,
            1,
            figsize=(7.15, 5.25),
            sharex=True,
            gridspec_kw={"height_ratios": (10, 1.15), "hspace": 0.13},
        )
    else:
        fig, ax = plt.subplots(figsize=(7.15, 4.9))
        unpriced_ax = None
    label_offsets: dict[str, tuple[float, float, str]] = {
        "openai/gpt-5.6-sol-pro": (8, 18, "left"),
        "anthropic/claude-fable-5": (8, -18, "left"),
        "anthropic/claude-opus-5": (5, -6, "left"),
        "anthropic/claude-sonnet-5": (9, -16, "left"),
        "google/gemini-3.1-pro-preview": (-8, 12, "right"),
        "x-ai/grok-4.5": (9, -14, "left"),
        "moonshotai/kimi-k3": (-8, -16, "right"),
        "nvidia/nemotron-3-ultra-550b-a55b": (-8, -13, "right"),
        "minimax/minimax-m3": (5, -10, "left"),
        "mistralai/mistral-medium-3-5": (-8, 16, "right"),
    }
    unpriced_label_offsets: dict[str, tuple[float, float, str]] = {
        "cohere/command-a-plus-05-2026": (-7, 6, "right"),
        "cohere/command-a-reasoning-08-2025": (7, 6, "left"),
    }
    unpriced_plot_names = {
        "cohere/command-a-plus-05-2026": "Cohere A+",
        "cohere/command-a-reasoning-08-2025": "Cohere Reasoning",
    }
    for row in rows:
        latency = row["latency_median_s"]
        if latency is None:
            continue
        scheduled = int(row["scheduled_pairs"])
        exposure = row["conservative_cost_exposure_usd"]
        if exposure is None:
            if unpriced_ax is None:
                continue
            unpriced_ax.scatter(
                float(latency),
                0.5,
                s=22 + 7 * int(row["complete_pairs"]),
                facecolor="white",
                edgecolor=ORANGE,
                linewidth=1.0,
                marker="s",
                zorder=3,
            )
            x_offset, y_offset, horizontal_alignment = unpriced_label_offsets.get(
                str(row["model_id"]), (5, -3, "left")
            )
            unpriced_ax.annotate(
                unpriced_plot_names.get(str(row["model_id"]), str(row["display_name"])),
                (float(latency), 0.5),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=6.6,
                ha=horizontal_alignment,
                va="bottom",
            )
            continue
        cents = 100 * float(exposure) / scheduled
        backend = str(row["execution_backend"])
        color = (
            GREEN
            if backend == "kimi_direct"
            else ORANGE
            if backend == "cohere_direct"
            else BLUE
        )
        marker = (
            "^"
            if backend == "kimi_direct"
            else "s"
            if backend == "cohere_direct"
            else "o"
        )
        ax.scatter(
            float(latency),
            cents,
            s=22 + 7 * int(row["complete_pairs"]),
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.88,
            zorder=3,
        )
        x_offset, y_offset, horizontal_alignment = label_offsets.get(
            str(row["model_id"]), (5, 4, "left")
        )
        ax.annotate(
            str(row["display_name"]),
            (float(latency), cents),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=6.6,
            ha=horizontal_alignment,
            annotation_clip=False,
        )
    ax.set_xscale("log")
    observed_latencies = [
        float(row["latency_median_s"])
        for row in rows
        if row["latency_median_s"] is not None
    ]
    ax.set_xlim(min(observed_latencies) / 1.18, max(observed_latencies) * 1.12)
    ax.set_yscale("log")
    positive_cents = [
        100 * float(row["conservative_cost_exposure_usd"]) / int(row["scheduled_pairs"])
        for row in rows
        if int(row["scheduled_pairs"]) > 0
        and row["conservative_cost_exposure_usd"] is not None
        and float(row["conservative_cost_exposure_usd"]) > 0
    ]
    ax.set_ylim(min(positive_cents, default=0.1) / 2.4, max(positive_cents, default=100.0) * 2.4)
    if unpriced_ax is None:
        ax.set_xlabel("Median normalized-arm latency (seconds, log scale)", labelpad=8)
    else:
        ax.tick_params(axis="x", which="both", labelbottom=False)
        unpriced_ax.set_xscale("log")
        unpriced_ax.set_xlim(ax.get_xlim())
        unpriced_ax.set_ylim(0, 1)
        unpriced_ax.set_yticks([0.5])
        unpriced_ax.set_yticklabels(["Charge not returned"], fontsize=7)
        unpriced_ax.tick_params(axis="y", length=0, pad=7)
        unpriced_ax.set_xlabel(
            "Median normalized-arm latency (seconds, log scale)", labelpad=8
        )
        unpriced_ax.grid(axis="x", which="both", color=LIGHT, linewidth=0.6)
        unpriced_ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            unpriced_ax.spines[side].set_visible(False)
    ax.set_ylabel("Known exposure per scheduled pair (US cents, log scale)")
    ax.set_title(
        f"Cost–latency profile under the {stratum_label}",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    ax.grid(which="both", color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.06,
        0.035,
        "Marker area scales with complete pairs. Triangle: direct Kimi; circles: frozen routed "
        "endpoints.\nOpen squares: direct Cohere. The categorical strip reports latency only; "
        "the provider returned no USD charge.",
        fontsize=6.7,
        color="#59636E",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_assets(
    pilot: VerifiedFrontierPilot,
    output_dir: Path,
    *,
    stratum_label: str = "strict protocol",
) -> dict[str, Path]:
    """Write content-addressed data, tables, macros, and vector figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    digest = str(pilot.aggregate["artifact_sha256"])
    aggregate_path = output_dir / f"frontier-multirun-{digest}.json"
    aggregate_path.write_text(
        json.dumps(pilot.aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance_path = output_dir / "frontier-multirun-provenance.json"
    provenance_path.write_text(
        json.dumps(pilot.aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model_csv = output_dir / "frontier-multirun-models.csv"
    family_csv = output_dir / "frontier-multirun-families.csv"
    _write_csv(model_csv, pilot.model_rows)
    _write_csv(family_csv, pilot.family_rows)
    totals = pilot.aggregate["totals"]
    cost = pilot.aggregate["cost"]
    macros = output_dir / "frontier-multirun-macros.tex"
    macros.write_text(
        "\n".join(
            [
                rf"\newcommand{{\FrontierPilotRunCount}}{{{totals['runs']}}}",
                rf"\newcommand{{\FrontierPilotModelCount}}{{{totals['models']}}}",
                rf"\newcommand{{\FrontierPilotTaskCount}}{{{totals['distinct_tasks']}}}",
                rf"\newcommand{{\FrontierPilotScheduledPairs}}{{{totals['scheduled_pairs']}}}",
                rf"\newcommand{{\FrontierPilotCompletePairs}}{{{totals['complete_pairs']}}}",
                rf"\newcommand{{\FrontierPilotResponseArms}}{{{totals['completed_response_arms']}}}",
                rf"\newcommand{{\FrontierPilotEpicureCalls}}{{{totals['epicure_calls']}}}",
                rf"\newcommand{{\FrontierPilotEightModelCount}}{{{totals['models_with_at_least_eight_complete_pairs']}}}",
                rf"\newcommand{{\FrontierPilotExposure}}{{\${cost['known_conservative_exposure_subtotal_usd']:.3f}}}",
                rf"\newcommand{{\FrontierPilotUnpricedModelCount}}{{{len(cost['unpriced_model_ids'])}}}",
                rf"\newcommand{{\FrontierPilotAssetHash}}{{{pilot.aggregate['artifact_sha256']}}}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    reliability = output_dir / "frontier-multirun-reliability.pdf"
    cost_latency = output_dir / "frontier-multirun-cost-latency.pdf"
    _render_reliability_figure(pilot.model_rows, reliability, stratum_label=stratum_label)
    _render_cost_latency_figure(pilot.model_rows, cost_latency, stratum_label=stratum_label)
    return {
        "aggregate": aggregate_path,
        "provenance": provenance_path,
        "model_csv": model_csv,
        "family_csv": family_csv,
        "macros": macros,
        "reliability_figure": reliability,
        "cost_latency_figure": cost_latency,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("SUMMARY", "SOURCE_DIR", "RESPONSE_DIR"),
        required=True,
        help="Repeat once per immutable run in the common protocol stratum.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stratum-label",
        default="strict protocol",
        help="Short, publication-facing name for the frozen execution stratum.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    inputs = [
        RunInput(Path(summary), Path(source), Path(response))
        for summary, source, response in arguments.run
    ]
    pilot = verify_runs(inputs)
    paths = write_assets(pilot, arguments.output_dir, stratum_label=arguments.stratum_label)
    print(
        json.dumps(
            {
                "status": "verified",
                "artifact_sha256": pilot.aggregate["artifact_sha256"],
                "totals": pilot.aggregate["totals"],
                "cost": pilot.aggregate["cost"],
                "outputs": {key: str(value.resolve()) for key, value in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
