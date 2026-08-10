"""Reproduce Season 0 automated consensus, rankings, uplift, and diagnostics."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from .ranking import (
    _fit_bradley_terry,
    _fit_local_bradley_terry,
    _paired_tie_aware_profile,
)
from .real_task_bank import sha256_json, sha256_text
from .season0_arm_corrections import validate_arm_interpretation_correction
from .season0_collection import ARM_SCHEMA, build_work_items
from .season0_completion_corrections import (
    apply_completion_interpretation,
    validate_completion_interpretation_correction,
)
from .season0_costs import reconcile_costs
from .season0_judge_protocol import (
    CHOICES,
    DIMENSIONS,
    JUDGE_SYSTEM_PROMPT_SHA256,
    JUDGMENT_SCHEMA_SHA256,
    ORIENTATIONS,
    PROTOCOL_VERSION,
    normalize_choice,
    validate_judgment,
)
from .season0_judging import SCHEMA_VERSION as JUDGMENT_RECORD_SCHEMA
from .season0_pairs import freeze_comparisons, identity_leak_tags

SCHEMA_VERSION = "flavourbench-season0-automated-analysis-v1"
FAMILIES = ("substitution", "composition", "cookability", "evidence")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.IGNORECASE)


class Season0AnalysisError(RuntimeError):
    """Official judgment artifacts are incomplete, mixed, or invalid."""


def _implementation_manifest() -> dict[str, Any]:
    source_dir = Path(__file__).resolve().parent
    filenames = (
        "season0_analysis.py",
        "season0_arm_corrections.py",
        "season0_completion_corrections.py",
        "ranking.py",
        "season0_judge_protocol.py",
        "season0_pairs.py",
        "season0_costs.py",
        "season0_judging.py",
        "season0_judgment_recovery.py",
        "season0_collection.py",
    )
    return {
        "source_sha256": {
            filename: hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
            for filename in filenames
        },
        "dependencies": {
            "arena-rank": importlib.metadata.version("arena-rank"),
            "numpy": importlib.metadata.version("numpy"),
        },
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise Season0AnalysisError(f"expected a JSON object: {path}")
    return value


def _verify_artifact(document: Mapping[str, Any], label: str) -> str:
    claimed = document.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise Season0AnalysisError(f"{label} artifact hash mismatch")
    return str(actual)


def _latest(directory: Path, prefix: str, id_field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Release archives may use short deterministic filenames to satisfy portable
    # tar path limits. Record identity comes from the immutable JSON payload, not
    # from a presentation filename. The dedicated input directories contain only
    # one record class, so content-driven discovery is both safer and portable.
    for path in directory.glob("*.json"):
        value = _load(path)
        identifier = value.get(id_field)
        if isinstance(identifier, str):
            grouped[identifier].append(value)
    return {
        identifier: sorted(rows, key=lambda row: str(row.get("completed_at") or ""))[-1]
        for identifier, rows in grouped.items()
    }


def _validate_real_arms(
    *,
    arms: Mapping[str, Mapping[str, Any]],
    task_bank: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    task_sha: str,
    model_sha: str,
) -> dict[str, Any]:
    """Re-derive the dense arm registry and verify every immutable real record."""

    tasks = [task for task in task_bank.get("tasks", []) if isinstance(task, Mapping)]
    family_counts = Counter(str(task.get("family") or "") for task in tasks)
    if len(tasks) != 120 or family_counts != Counter({family: 30 for family in FAMILIES}):
        raise Season0AnalysisError("analysis requires exactly 30 real tasks per family")
    if task_bank.get("synthetic_tasks") != 0:
        raise Season0AnalysisError("analysis refuses a task bank containing synthetic tasks")
    work_items = build_work_items(
        task_bank,
        model_manifest,
        phase="scored",
        per_family=30,
    )
    expected = {item.arm_id: item for item in work_items}
    if set(arms) != set(expected):
        missing = len(set(expected) - set(arms))
        unexpected = len(set(arms) - set(expected))
        raise Season0AnalysisError(
            f"scored arm registry mismatch: {missing} missing, {unexpected} unexpected"
        )

    expected_contracts = {
        "task_bank_artifact_sha256": task_sha,
        "model_manifest_artifact_sha256": model_sha,
        "task_set_sha256": task_bank.get("task_set_sha256"),
        "model_set_sha256": model_manifest.get("model_set_sha256"),
        "epicure_intervention_artifact_sha256": model_manifest.get(
            "epicure_intervention_artifact_sha256"
        ),
        "execution_contract_sha256": sha256_json(
            model_manifest.get("execution_contract") or {}
        ),
        "normalization_mode": "lossless_client_text_wrapper_v1",
    }
    collector_versions: set[str] = set()
    collector_source_hashes: set[str] = set()
    successful = 0
    failed = 0
    real_provider_calls = 0
    real_epicure_calls = 0
    for arm_id, item in expected.items():
        record = arms[arm_id]
        _verify_artifact(record, f"arm {arm_id}")
        if (
            record.get("schema_version") != ARM_SCHEMA
            or record.get("arm_id") != arm_id
            or record.get("phase") != "scored"
            or record.get("condition") != item.condition
            or record.get("synthetic") is not False
        ):
            raise Season0AnalysisError(f"arm {arm_id} violates the real scored-arm identity")
        task = record.get("task")
        model = record.get("model")
        contracts = record.get("contracts")
        if not isinstance(task, Mapping) or not isinstance(model, Mapping):
            raise Season0AnalysisError(f"arm {arm_id} has malformed task or model provenance")
        if any(
            task.get(key) != item.task.get(key)
            for key in ("task_id", "family", "task_sha256", "prompt_sha256")
        ) or any(
            model.get(key) != item.model.get(key)
            for key in (
                "season_model_id",
                "display_name",
                "canonical_model_id",
                "provider",
                "requested_endpoint_id",
                "compatibility_artifact_sha256",
            )
        ):
            raise Season0AnalysisError(f"arm {arm_id} provenance differs from its frozen inputs")
        if not isinstance(contracts, Mapping) or any(
            contracts.get(key) != value for key, value in expected_contracts.items()
        ):
            raise Season0AnalysisError(f"arm {arm_id} contract binding mismatch")
        collector_version = contracts.get("collector_version")
        collector_source = contracts.get("collector_source_sha256")
        if not isinstance(collector_version, str) or not collector_version:
            raise Season0AnalysisError(f"arm {arm_id} has no collector version")
        if not isinstance(collector_source, str) or not re.fullmatch(
            r"[0-9a-f]{64}", collector_source
        ):
            raise Season0AnalysisError(f"arm {arm_id} has no collector source hash")
        collector_versions.add(collector_version)
        collector_source_hashes.add(collector_source)

        status = record.get("status")
        delivery = record.get("delivery_state")
        rank_eligible = record.get("rank_eligible")
        if status not in {"success", "failed", "not_admitted"}:
            raise Season0AnalysisError(f"arm {arm_id} has an unknown terminal status")
        if rank_eligible is not (status == "success" and delivery == "reconciled"):
            raise Season0AnalysisError(f"arm {arm_id} has inconsistent rank eligibility")
        result = record.get("result")
        if status == "success":
            successful += 1
            if not isinstance(result, Mapping) or not str(
                result.get("answer_markdown") or ""
            ).strip():
                raise Season0AnalysisError(f"successful arm {arm_id} has no final answer")
        else:
            failed += 1
        if result is None:
            continue
        if not isinstance(result, Mapping):
            raise Season0AnalysisError(f"arm {arm_id} has a malformed provider result")
        provider_calls = result.get("provider_calls")
        trace = result.get("tool_trace")
        declared_epicure_calls = result.get("real_epicure_calls")
        if (
            not isinstance(provider_calls, int)
            or isinstance(provider_calls, bool)
            or provider_calls < 1
        ):
            raise Season0AnalysisError(f"arm {arm_id} has an invalid real provider-call count")
        payload_hashes = result.get("request_payload_sha256s")
        if (
            not isinstance(payload_hashes, list)
            or len(payload_hashes) != provider_calls
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in payload_hashes
            )
        ):
            raise Season0AnalysisError(f"arm {arm_id} has incomplete request provenance")
        if status == "success" and item.model["provider"] == "openrouter":
            returned_models = result.get("returned_model_ids")
            returned_providers = result.get("actual_provider_names")
            if (
                result.get("model_identity_verified") is not True
                or result.get("provider_identity_verified") is not True
                or not isinstance(returned_models, list)
                or not returned_models
                or set(returned_models) != {item.model["canonical_model_id"]}
                or not isinstance(returned_providers, list)
                or not returned_providers
                or set(returned_providers) != {item.model["provider_name"]}
            ):
                raise Season0AnalysisError(f"arm {arm_id} failed exact OpenRouter identity")
        if status == "success" and item.model["provider"] == "bedrock":
            request_ids = result.get("request_id_sha256s")
            if (
                result.get("actual_provider_name") != "Amazon Bedrock"
                or not isinstance(request_ids, list)
                or len(request_ids) != provider_calls
                or any(
                    not isinstance(value, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in request_ids
                )
            ):
                raise Season0AnalysisError(f"arm {arm_id} has invalid Bedrock provenance")
        if not isinstance(trace, list) or not isinstance(declared_epicure_calls, int):
            raise Season0AnalysisError(f"arm {arm_id} has an invalid Epicure trace")
        if declared_epicure_calls != len(trace):
            raise Season0AnalysisError(f"arm {arm_id} Epicure call count does not match its trace")
        if item.condition == "epicure_off" and trace:
            raise Season0AnalysisError(f"Epicure-off arm {arm_id} contains a tool call")
        real_provider_calls += provider_calls
        real_epicure_calls += declared_epicure_calls
        for call in trace:
            if not isinstance(call, Mapping):
                raise Season0AnalysisError(f"arm {arm_id} contains a malformed Epicure call")
            arguments = call.get("arguments")
            result_value = call.get("result")
            visible_text = call.get("model_visible_result")
            if (
                not isinstance(arguments, Mapping)
                or not isinstance(visible_text, str)
                or call.get("arguments_sha256") != sha256_json(arguments)
                or call.get("result_sha256") != sha256_json(result_value)
                or call.get("model_visible_result_sha256") != sha256_text(visible_text)
                or not isinstance(call.get("is_error"), bool)
                or not isinstance(call.get("round_index"), int)
                or not isinstance(call.get("latency_ms"), int)
            ):
                raise Season0AnalysisError(f"arm {arm_id} contains an invalid Epicure trace entry")

    if len(collector_versions) != 1 or len(collector_source_hashes) != 1:
        raise Season0AnalysisError("scored arms mix collector versions or source hashes")
    return {
        "expected_arms": len(expected),
        "successful_arms": successful,
        "failed_arms": failed,
        "recorded_provider_calls_including_partial": real_provider_calls,
        "recorded_epicure_calls_including_partial": real_epicure_calls,
        "collector_version": next(iter(collector_versions)),
        "collector_source_sha256": next(iter(collector_source_hashes)),
    }


def _validate_comparison_reproduction(
    *,
    arms_dir: Path,
    task_bank: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    comparison_sha: str,
) -> None:
    """Rebuild the deterministic comparison schedule and require byte-semantic identity."""

    with tempfile.TemporaryDirectory(prefix="flavourbench-comparison-audit-") as temporary:
        regenerated = freeze_comparisons(
            arms_dir=arms_dir,
            task_bank=task_bank,
            model_manifest=model_manifest,
            output_dir=Path(temporary),
        )
    regenerated.pop("summary_path", None)
    if sha256_json(regenerated) != comparison_sha:
        raise Season0AnalysisError("comparison manifest is not reproducible from scored arms")


def _apply_completion_policy_to_comparisons(
    comparisons: Sequence[Mapping[str, Any]],
    corrected_arm_ids: frozenset[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exclude comparisons containing a final response that did not end normally."""

    output: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for source in comparisons:
        row = copy.deepcopy(dict(source))
        if row.get("judgable") is True:
            counters["source_judgable"] += 1
            arm_ids = {
                str((row.get(side) or {}).get("arm_id") or "")
                for side in ("left", "right")
            }
            if arm_ids & corrected_arm_ids:
                row["source_judgable"] = True
                row["judgable"] = False
                row["exclusion_reason"] = "incomplete_final_response"
                counters["excluded_incomplete_final_response"] += 1
                counters[f"excluded_{row.get('track')}"] += 1
            else:
                counters["effective_judgable"] += 1
                counters[f"effective_{row.get('track')}"] += 1
        output.append(row)
    return output, dict(sorted(counters.items()))


def _expected_judgment_ids(
    *,
    comparisons: Sequence[Mapping[str, Any]],
    judges: Sequence[Mapping[str, Any]],
    comparison_sha: str,
    judge_sha: str,
) -> set[str]:
    expected: set[str] = set()
    for comparison in comparisons:
        if comparison.get("judgable") is not True:
            continue
        for judge in judges:
            for orientation in ORIENTATIONS:
                identity = {
                    "schema_version": JUDGMENT_RECORD_SCHEMA,
                    "season": "Season 0",
                    "comparison_manifest_artifact_sha256": comparison_sha,
                    "judge_manifest_artifact_sha256": judge_sha,
                    "protocol_version": PROTOCOL_VERSION,
                    "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
                    "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
                    "comparison_id": comparison["comparison_id"],
                    "judge_id": judge["judge_id"],
                    "orientation": orientation,
                }
                expected.add(sha256_json(identity))
    return expected


def _validate_target_cost_audit(
    *,
    target_cost_audit: Mapping[str, Any],
    rate_card: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    arm_interpretation_correction_sha256: str | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Bind per-model operational cost to the complete conservative target audit."""

    audit_sha = _verify_artifact(target_cost_audit, "target cost audit")
    if (
        target_cost_audit.get("synthetic_arms") != 0
        or target_cost_audit.get("complete_exposure_accounting") is not True
        or target_cost_audit.get("complete_openrouter_request_level_attribution") is not True
        or target_cost_audit.get("rate_card_sha256") != sha256_json(rate_card)
        or target_cost_audit.get("arm_interpretation_correction_artifact_sha256")
        != arm_interpretation_correction_sha256
    ):
        raise Season0AnalysisError("target cost audit is incomplete, synthetic, or misbound")
    counts = target_cost_audit.get("counts")
    audit_models = target_cost_audit.get("models")
    unattributed = target_cost_audit.get("unattributed")
    totals = target_cost_audit.get("cost_usd")
    if (
        not isinstance(counts, Mapping)
        or counts.get("arms") != len(arms)
        or not isinstance(audit_models, Mapping)
        or not isinstance(unattributed, list)
        or not isinstance(totals, Mapping)
    ):
        raise Season0AnalysisError("target cost audit has an invalid population")

    model_ids = {str(model["season_model_id"]) for model in models}
    if set(audit_models) != model_ids:
        raise Season0AnalysisError("target cost audit model population mismatch")
    unattributed_by_model: dict[str, Decimal] = defaultdict(Decimal)
    unattributed_counts: Counter[str] = Counter()
    seen_unattributed: set[str] = set()
    for row in unattributed:
        if not isinstance(row, Mapping):
            raise Season0AnalysisError("target cost audit has a malformed unattributed row")
        arm_id = str(row.get("arm_id") or "")
        arm = arms.get(arm_id)
        if arm is None or arm_id in seen_unattributed:
            raise Season0AnalysisError("target cost audit has an unknown or duplicate arm")
        seen_unattributed.add(arm_id)
        model_id = str((arm.get("model") or {}).get("season_model_id") or "")
        provider = str((arm.get("model") or {}).get("provider") or "")
        reservation = Decimal(str(row.get("conservative_reservation_usd")))
        if (
            row.get("provider") != provider
            or reservation != Decimal(str(arm.get("reservation_usd")))
            or reservation < 0
        ):
            raise Season0AnalysisError("unattributed cost reservation does not match its arm")
        unattributed_by_model[model_id] += reservation
        unattributed_counts[model_id] += 1
    if (
        counts.get("unattributed_arms") != len(unattributed)
        or counts.get("attributed_arms") != len(arms) - len(unattributed)
    ):
        raise Season0AnalysisError("target cost audit attribution counts do not reconcile")

    output: dict[str, dict[str, Any]] = {}
    attributed_total = Decimal(0)
    conservative_total = Decimal(0)
    for model in models:
        model_id = str(model["season_model_id"])
        row = audit_models[model_id]
        if not isinstance(row, Mapping):
            raise Season0AnalysisError("target cost audit has a malformed model row")
        arms_count = int(row.get("arms") or 0)
        attributed_arms = int(row.get("attributed_arms") or 0)
        attributed_cost = Decimal(str(row.get("cost_usd")))
        reservation = unattributed_by_model[model_id]
        if (
            arms_count != 240
            or attributed_arms + unattributed_counts[model_id] != arms_count
            or row.get("provider") != model.get("provider")
            or row.get("display_name") != model.get("display_name")
            or attributed_cost < 0
        ):
            raise Season0AnalysisError(f"target cost audit does not reconcile {model_id}")
        conservative = attributed_cost + reservation
        attributed_total += attributed_cost
        conservative_total += conservative
        output[model_id] = {
            "attributed_cost_usd": float(attributed_cost),
            "unattributed_cost_reservation_usd": float(reservation),
            "conservative_cost_usd": float(conservative),
            "mean_arm_cost_usd": float(conservative / arms_count),
            "cost_attributed_arms": attributed_arms,
            "cost_unattributed_arms": unattributed_counts[model_id],
            "cost_accounting_basis": (
                "generation metadata or published Bedrock rate for attributed arms; "
                "frozen reservation for unattributed possible-delivery arms"
            ),
        }
    if (
        attributed_total != Decimal(str(totals.get("combined_attributed")))
        or conservative_total
        != Decimal(str(totals.get("combined_conservative_exposure")))
        or conservative_total - attributed_total
        != Decimal(str(totals.get("unattributed_conservative_reservations")))
    ):
        raise Season0AnalysisError("target cost audit totals do not reconcile by model")
    return audit_sha, output


def _validate_cost_reproduction(
    *,
    arms_dir: Path,
    rate_card: Mapping[str, Any],
    target_cost_sha: str,
    corrections_dir: Path | None = None,
    arm_interpretation_correction: dict[str, Any] | None = None,
) -> None:
    if corrections_dir is None:
        corrections_dir = arms_dir.parent / "cost-corrections"
    with tempfile.TemporaryDirectory(prefix="flavourbench-cost-audit-") as temporary:
        regenerated = reconcile_costs(
            arms_dir=arms_dir,
            rate_card=rate_card,
            output_dir=Path(temporary),
            corrections_dir=corrections_dir if corrections_dir.is_dir() else None,
            arm_interpretation_correction=arm_interpretation_correction,
        )
    regenerated.pop("summary_path", None)
    if sha256_json(regenerated) != target_cost_sha:
        raise Season0AnalysisError("target cost audit is not reproducible from scored arms")


def _side_from_orientation(record: Mapping[str, Any], original_side: str) -> Mapping[str, Any]:
    result = record.get("result")
    judgment = result.get("judgment") if isinstance(result, Mapping) else None
    if not isinstance(judgment, Mapping):
        raise Season0AnalysisError("successful judgment has no parsed judgment")
    orientation = record.get("orientation")
    side = original_side
    if orientation == "swapped":
        side = "right" if original_side == "left" else "left"
    value = judgment.get(side)
    if not isinstance(value, Mapping):
        raise Season0AnalysisError("judgment has no normalized side scores")
    return value


def _mean_side(records: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    oriented = [_side_from_orientation(record, side) for record in records]
    return {
        "scores": {
            dimension: statistics.mean(
                float(value["scores"][dimension]) for value in oriented
            )
            for dimension in DIMENSIONS
        },
        "fatal_failure_rate": statistics.mean(
            float(bool(value["fatal_failure"])) for value in oriented
        ),
    }


def _self_models(comparison: Mapping[str, Any]) -> set[str]:
    if comparison.get("track") == "model_arena":
        return {
            str(comparison["left"]["season_model_id"]),
            str(comparison["right"]["season_model_id"]),
        }
    return {str(comparison["season_model_id"])}


def _majority(choices: Sequence[str]) -> str | None:
    if len(choices) < 2:
        return None
    counts = Counter(choices)
    choice, count = counts.most_common(1)[0]
    return choice if count >= len(choices) // 2 + 1 else None


def _mean_ci(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "lower": None, "upper": None, "n": 0}
    mean = statistics.mean(values)
    if len(values) == 1:
        return {"mean": mean, "lower": None, "upper": None, "n": 1}
    error = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "lower": max(1.0, mean - error),
        "upper": min(5.0, mean + error),
        "n": len(values),
    }


def _percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _comparison_graph_diagnostics(
    comparisons: Sequence[tuple[str, str, float]], model_ids: Sequence[str]
) -> dict[str, Any]:
    adjacency = {model_id: set() for model_id in model_ids}
    unique_edges: set[tuple[str, str]] = set()
    for first, second, _ in comparisons:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
        unique_edges.add(tuple(sorted((first, second))))
    remaining = set(adjacency)
    components: list[list[str]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        remaining -= component
        components.append(sorted(component))
    components.sort(key=lambda values: (-len(values), values))
    return {
        "connected": len(components) == 1,
        "components": components,
        "component_count": len(components),
        "unique_edge_count": len(unique_edges),
        "comparison_count": len(comparisons),
        "degree_by_model": {
            model_id: len(adjacency[model_id]) for model_id in sorted(adjacency)
        },
    }


def _failure_class(row: Mapping[str, Any]) -> str:
    if row.get("status") == "success":
        return "success"
    if str(row.get("error_type") or "") in {
        "ConnectionClosedError",
        "ReadTimeoutError",
        "ResponseStreamingError",
    }:
        return "uncertain_delivery"
    delivery = str(row.get("delivery_state") or "")
    if delivery == "reconciled":
        return "model_behavior_failure"
    if delivery == "safe_pre_inference":
        return "provider_pre_inference_failure"
    if delivery in {"uncertain", "uncertain_delivery"}:
        return "uncertain_delivery"
    if row.get("status") == "not_admitted":
        return "not_admitted"
    return "other_failure"


def _word_tokens(text: str) -> list[str]:
    return [match.group(0).lower().replace("’", "'") for match in TOKEN_PATTERN.finditer(text)]


def _ngrams(tokens: Sequence[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _reference_overlap_audit(
    *,
    arms: Mapping[str, Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    model_names: Mapping[str, str],
) -> tuple[dict[str, Any], set[str]]:
    reference_by_task: dict[str, dict[int, set[tuple[str, ...]]]] = {}
    for task in tasks:
        reference = task.get("human_reference")
        reference_text = reference.get("text") if isinstance(reference, Mapping) else None
        prompt = task.get("prompt")
        if not isinstance(reference_text, str) or not isinstance(prompt, str):
            continue
        prompt_tokens = _word_tokens(prompt)
        reference_tokens = _word_tokens(reference_text)
        reference_by_task[str(task["task_id"])] = {
            size: _ngrams(reference_tokens, size) - _ngrams(prompt_tokens, size)
            for size in (8, 12)
        }
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    flagged_arm_ids: set[str] = set()
    for arm_id, arm in arms.items():
        if arm.get("status") != "success":
            continue
        result = arm.get("result")
        answer = result.get("answer_markdown") if isinstance(result, Mapping) else None
        task_id = str((arm.get("task") or {}).get("task_id") or "")
        references = reference_by_task.get(task_id)
        if not isinstance(answer, str) or not references:
            continue
        model_id = str((arm.get("model") or {}).get("season_model_id") or "")
        answer_tokens = _word_tokens(answer)
        counts = {size: len(_ngrams(answer_tokens, size) & references[size]) for size in (8, 12)}
        by_model[model_id]["answers"] += 1
        overall["answers"] += 1
        for size in (8, 12):
            by_model[model_id][f"matched_{size}grams"] += counts[size]
            overall[f"matched_{size}grams"] += counts[size]
            if counts[size]:
                by_model[model_id][f"answers_with_{size}gram_match"] += 1
                overall[f"answers_with_{size}gram_match"] += 1
        if counts[12]:
            flagged_arm_ids.add(arm_id)

    def summary(counter: Counter[str]) -> dict[str, Any]:
        answers = counter["answers"]
        return {
            "answers": answers,
            "answers_with_novel_reference_8gram_match": counter[
                "answers_with_8gram_match"
            ],
            "answers_with_novel_reference_12gram_match": counter[
                "answers_with_12gram_match"
            ],
            "novel_reference_8gram_match_rate": (
                counter["answers_with_8gram_match"] / answers if answers else None
            ),
            "novel_reference_12gram_match_rate": (
                counter["answers_with_12gram_match"] / answers if answers else None
            ),
            "matched_novel_reference_8grams": counter["matched_8grams"],
            "matched_novel_reference_12grams": counter["matched_12grams"],
        }

    return (
        {
            "method": (
                "exact normalized token n-gram overlap with hidden accepted human reference "
                "after removing n-grams already present in the prompt; overlap is a flag, not "
                "proof of training contamination"
            ),
            "overall": summary(overall),
            "by_model": {
                model_id: {
                    "season_model_id": model_id,
                    "display_name": model_names[model_id],
                    **summary(by_model[model_id]),
                }
                for model_id in model_names
            },
            "sensitivity_exclusion": "exclude comparisons containing a 12-token flagged arm",
        },
        flagged_arm_ids,
    )


def _verbosity_diagnostics(
    consensus_rows: Sequence[Mapping[str, Any]],
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    word_counts: dict[str, int] = {}
    for arm_id, arm in arms.items():
        result = arm.get("result")
        answer = result.get("answer_markdown") if isinstance(result, Mapping) else None
        if isinstance(answer, str):
            word_counts[arm_id] = len(_word_tokens(answer))
    directional = 0
    unequal = 0
    preferred_longer = 0
    absolute_differences: list[int] = []
    by_track: dict[str, Counter[str]] = defaultdict(Counter)
    for row in consensus_rows:
        choice = row.get("primary_consensus_choice")
        if choice not in {"left", "right"}:
            continue
        left_count = word_counts.get(str(row["left"]["arm_id"]))
        right_count = word_counts.get(str(row["right"]["arm_id"]))
        if left_count is None or right_count is None:
            continue
        directional += 1
        track = str(row["track"])
        by_track[track]["directional_preferences"] += 1
        absolute_differences.append(abs(left_count - right_count))
        if left_count == right_count:
            continue
        unequal += 1
        by_track[track]["unequal_length_preferences"] += 1
        chosen_count = left_count if choice == "left" else right_count
        other_count = right_count if choice == "left" else left_count
        if chosen_count > other_count:
            preferred_longer += 1
            by_track[track]["preferred_longer"] += 1

    def track_summary(counter: Counter[str]) -> dict[str, Any]:
        denominator = counter["unequal_length_preferences"]
        return {
            **dict(counter),
            "preferred_longer_rate_among_unequal": (
                counter["preferred_longer"] / denominator if denominator else None
            ),
        }

    return {
        "directional_preferences": directional,
        "unequal_length_preferences": unequal,
        "preferred_longer": preferred_longer,
        "preferred_longer_rate_among_unequal": (
            preferred_longer / unequal if unequal else None
        ),
        "median_absolute_word_difference": (
            statistics.median(absolute_differences) if absolute_differences else None
        ),
        "by_track": {
            track: track_summary(counter) for track, counter in sorted(by_track.items())
        },
    }


def _cohen_kappa(first: Sequence[str], second: Sequence[str]) -> dict[str, float | int | None]:
    if len(first) != len(second) or not first:
        return {"n": len(first), "agreement": None, "kappa": None}
    labels = list(CHOICES)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        first_counts[label] * second_counts[label] / (len(first) ** 2) for label in labels
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"n": len(first), "agreement": observed, "kappa": kappa}


def aggregate_consensus(
    *,
    comparisons: Sequence[Mapping[str, Any]],
    judges: Sequence[Mapping[str, Any]],
    judgments: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in judgments.values():
        by_pair[(str(record["comparison_id"]), str(record["judge"]["judge_id"]))].append(
            record
        )
    judge_stats: dict[str, Counter[str]] = {
        str(judge["judge_id"]): Counter() for judge in judges
    }
    judge_choices: dict[tuple[str, str], str] = {}
    consensus_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        if comparison.get("judgable") is not True:
            continue
        comparison_id = str(comparison["comparison_id"])
        self_models = _self_models(comparison)
        votes: list[dict[str, Any]] = []
        for judge in judges:
            judge_id = str(judge["judge_id"])
            stats = judge_stats[judge_id]
            stats["planned_comparisons"] += 1
            records = by_pair.get((comparison_id, judge_id), [])
            successful = [row for row in records if row.get("status") == "success"]
            stats["successful_orientations"] += len(successful)
            if len(successful) != 2 or {row.get("orientation") for row in successful} != {
                "original",
                "swapped",
            }:
                stats["incomplete_comparisons"] += 1
                continue
            normalized = [str(row["result"]["normalized_choice"]) for row in successful]
            if len(set(normalized)) != 1:
                stats["orientation_disagreement"] += 1
                continue
            stats["orientation_consistent"] += 1
            choice = normalized[0]
            is_self = str(judge.get("self_season_model_id") or "") in self_models
            if is_self:
                stats["self_judgments"] += 1
            vote = {
                "judge_id": judge_id,
                "choice": choice,
                "self_judgment": is_self,
                "left": _mean_side(successful, "left"),
                "right": _mean_side(successful, "right"),
            }
            votes.append(vote)
            judge_choices[(comparison_id, judge_id)] = choice
        eligible = [vote for vote in votes if not vote["self_judgment"]]
        consensus = _majority([str(vote["choice"]) for vote in eligible])
        score_eligible = len(eligible) >= 2
        side_scores: dict[str, Any] = {}
        if score_eligible:
            for side in ("left", "right"):
                side_scores[side] = {
                    "scores": {
                        dimension: statistics.mean(
                            float(vote[side]["scores"][dimension]) for vote in eligible
                        )
                        for dimension in DIMENSIONS
                    },
                    "fatal_failure_rate": statistics.mean(
                        float(vote[side]["fatal_failure_rate"]) for vote in eligible
                    ),
                }
        consensus_rows.append(
            {
                "comparison_id": comparison_id,
                "track": comparison["track"],
                "task_id": comparison["task_id"],
                "task_family": comparison["task_family"],
                "left": comparison["left"],
                "right": comparison["right"],
                "season_model_id": comparison.get("season_model_id"),
                "consistent_judge_votes": votes,
                "primary_nonself_vote_count": len(eligible),
                "primary_consensus_choice": consensus,
                "primary_consensus_available": consensus is not None,
                "primary_scores_available": score_eligible,
                "primary_side_scores": side_scores,
            }
        )

    pairwise_agreement: dict[str, Any] = {}
    for first_index, first_judge in enumerate(judges):
        for second_judge in judges[first_index + 1 :]:
            first_id = str(first_judge["judge_id"])
            second_id = str(second_judge["judge_id"])
            first_values: list[str] = []
            second_values: list[str] = []
            for comparison in comparisons:
                comparison_id = str(comparison["comparison_id"])
                first_choice = judge_choices.get((comparison_id, first_id))
                second_choice = judge_choices.get((comparison_id, second_id))
                if first_choice is not None and second_choice is not None:
                    first_values.append(first_choice)
                    second_values.append(second_choice)
            pairwise_agreement[f"{first_id}__{second_id}"] = _cohen_kappa(
                first_values, second_values
            )
    diagnostics = {
        "judges": {
            judge_id: {
                **dict(stats),
                "orientation_consistency_rate": (
                    stats["orientation_consistent"] / stats["planned_comparisons"]
                    if stats["planned_comparisons"]
                    else None
                ),
            }
            for judge_id, stats in judge_stats.items()
        },
        "pairwise_agreement": pairwise_agreement,
        "primary_consensus_coverage": (
            sum(row["primary_consensus_available"] for row in consensus_rows)
            / len(consensus_rows)
            if consensus_rows
            else None
        ),
    }
    return consensus_rows, diagnostics


def _judge_family(judge: Mapping[str, Any]) -> str:
    canonical = str(judge.get("canonical_model_id") or "").lower()
    owner = canonical.replace("/", ".").split(".", 1)[0]
    if not owner:
        raise Season0AnalysisError("judge manifest has no canonical model family")
    return owner


def family_balanced_consensus(
    consensus_rows: Sequence[Mapping[str, Any]],
    judges: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Give each judge-model lineage at most one internally consistent vote."""

    judge_families = {
        str(judge["judge_id"]): _judge_family(judge) for judge in judges
    }
    output: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    family_vote_counts: Counter[str] = Counter()
    for row in consensus_rows:
        by_family: dict[str, list[str]] = defaultdict(list)
        for vote in row.get("consistent_judge_votes", []):
            if not isinstance(vote, Mapping) or vote.get("self_judgment") is True:
                continue
            judge_id = str(vote.get("judge_id") or "")
            family = judge_families.get(judge_id)
            choice = str(vote.get("choice") or "")
            if family is None or choice not in CHOICES:
                raise Season0AnalysisError("consensus row has an unknown judge-family vote")
            by_family[family].append(choice)
        family_votes: list[dict[str, str]] = []
        for family, choices in sorted(by_family.items()):
            if len(set(choices)) != 1:
                counters["within_family_disagreement"] += 1
                continue
            family_votes.append({"family": family, "choice": choices[0]})
            family_vote_counts[family] += 1
        choice = _majority([vote["choice"] for vote in family_votes])
        if choice is None:
            counters["no_consensus"] += 1
        else:
            counters["consensus_available"] += 1
        output.append(
            {
                **dict(row),
                "primary_consensus_choice": choice,
                "primary_consensus_available": choice is not None,
                "primary_scores_available": False,
                "primary_side_scores": {},
                "family_balanced_votes": family_votes,
            }
        )
    counters["rows"] = len(output)
    return output, {
        **dict(counters),
        "coverage": (
            counters["consensus_available"] / len(output) if output else None
        ),
        "judge_families": judge_families,
        "family_vote_counts": dict(sorted(family_vote_counts.items())),
        "rule": (
            "orientation-consistent non-self votes are collapsed within canonical model "
            "family only when unanimous; strict majority is then applied across families"
        ),
    }


def _arena_rows(
    consensus_rows: Sequence[Mapping[str, Any]],
    model_names: Mapping[str, str],
    family: str | None,
) -> list[dict[str, Any]]:
    comparisons: list[tuple[str, str, float]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in consensus_rows:
        if row["track"] != "model_arena" or (family and row["task_family"] != family):
            continue
        choice = row.get("primary_consensus_choice")
        if choice is None:
            continue
        left = str(row["left"]["season_model_id"])
        right = str(row["right"]["season_model_id"])
        counts[left]["judgments"] += 1
        counts[right]["judgments"] += 1
        if choice == "both_bad":
            counts[left]["both_bad"] += 1
            counts[right]["both_bad"] += 1
            continue
        counts[left]["comparisons"] += 1
        counts[right]["comparisons"] += 1
        if choice == "left":
            counts[left]["wins"] += 1
            counts[right]["losses"] += 1
            outcome = 1.0
        elif choice == "right":
            counts[right]["wins"] += 1
            counts[left]["losses"] += 1
            outcome = 0.0
        else:
            counts[left]["ties"] += 1
            counts[right]["ties"] += 1
            outcome = 0.5
        comparisons.append((left, right, outcome))
    graph = _comparison_graph_diagnostics(comparisons, list(model_names))
    ratings = (
        _fit_bradley_terry(comparisons, require_arena_rank=True)
        if graph["connected"]
        else {}
    )
    output = []
    for model_id in model_names:
        rating = ratings.get(model_id)
        stats = counts[model_id]
        output.append(
            {
                "season_model_id": model_id,
                "display_name": model_names[model_id],
                "rating": rating[0] if rating else None,
                "rating_lower": rating[1] if rating else None,
                "rating_upper": rating[2] if rating else None,
                "comparisons": stats["comparisons"],
                "judgments": stats["judgments"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "both_bad": stats["both_bad"],
                "sample_size_below_100": stats["comparisons"] < 100 if family is None else None,
            }
        )
    output.sort(
        key=lambda row: float(row["rating"]) if row["rating"] is not None else -math.inf,
        reverse=True,
    )
    return output


def _uplift_rows(
    consensus_rows: Sequence[Mapping[str, Any]],
    model_names: Mapping[str, str],
    family: str | None,
) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in consensus_rows:
        if row["track"] != "epicure_uplift" or (family and row["task_family"] != family):
            continue
        choice = row.get("primary_consensus_choice")
        if choice is None:
            continue
        model_id = str(row["season_model_id"])
        record = counts[model_id]
        record["judgments"] += 1
        if choice == "both_bad":
            record["both_bad"] += 1
            continue
        record["comparisons"] += 1
        if choice == "tie":
            record["ties"] += 1
        else:
            winning_condition = row[choice]["condition"]
            record["epicure_wins" if winning_condition == "epicure_on" else "unaided_wins"] += 1
    output = []
    for model_id in model_names:
        record = counts[model_id]
        estimate, low, high = _paired_tie_aware_profile(
            record["epicure_wins"], record["ties"], record["unaided_wins"]
        )
        output.append(
            {
                "season_model_id": model_id,
                "display_name": model_names[model_id],
                "epicure_win_share": estimate,
                "interval_lower": low,
                "interval_upper": high,
                "comparisons": record["comparisons"],
                "judgments": record["judgments"],
                "epicure_wins": record["epicure_wins"],
                "unaided_wins": record["unaided_wins"],
                "ties": record["ties"],
                "both_bad": record["both_bad"],
                "sample_size_below_50": record["comparisons"] < 50 if family is None else None,
            }
        )
    output.sort(key=lambda row: float(row["epicure_win_share"]), reverse=True)
    return output


def _panel_uplift_summary(
    consensus_rows: Sequence[Mapping[str, Any]], family: str | None
) -> dict[str, Any]:
    scores_by_task: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in consensus_rows:
        if row["track"] != "epicure_uplift" or (family and row["task_family"] != family):
            continue
        choice = row.get("primary_consensus_choice")
        if choice is None:
            counts["no_consensus"] += 1
            continue
        if choice == "both_bad":
            counts["both_bad"] += 1
            continue
        if choice == "tie":
            score = 0.5
            counts["ties"] += 1
        else:
            winning_condition = str(row[choice]["condition"])
            if winning_condition == "epicure_on":
                score = 1.0
                counts["epicure_wins"] += 1
            else:
                score = 0.0
                counts["unaided_wins"] += 1
        scores_by_task[str(row["task_id"])].append(score)
    scores = [score for task_scores in scores_by_task.values() for score in task_scores]
    profile, profile_low, profile_high = _paired_tie_aware_profile(
        counts["epicure_wins"], counts["ties"], counts["unaided_wins"]
    )
    cluster_estimate = statistics.mean(scores) if scores else 0.5
    cluster_low, cluster_high = 0.0, 1.0
    task_ids = sorted(scores_by_task)
    if task_ids:
        seed_by_family = {
            None: 20260716,
            "substitution": 20260717,
            "composition": 20260718,
            "cookability": 20260719,
            "evidence": 20260720,
        }
        generator = np.random.default_rng(seed_by_family[family])
        bootstraps = np.empty(5000, dtype=float)
        for index in range(len(bootstraps)):
            sampled = generator.choice(task_ids, size=len(task_ids), replace=True)
            sampled_scores = [
                score for task_id in sampled for score in scores_by_task[str(task_id)]
            ]
            bootstraps[index] = statistics.mean(sampled_scores)
        cluster_low, cluster_high = (
            float(np.quantile(bootstraps, 0.025)),
            float(np.quantile(bootstraps, 0.975)),
        )
    return {
        "family": family or "all",
        "valid_comparisons": len(scores),
        "task_clusters": len(task_ids),
        "epicure_wins": counts["epicure_wins"],
        "ties": counts["ties"],
        "unaided_wins": counts["unaided_wins"],
        "both_bad": counts["both_bad"],
        "no_consensus": counts["no_consensus"],
        "tie_aware_profile_win_share": profile,
        "tie_aware_profile_interval_lower": profile_low,
        "tie_aware_profile_interval_upper": profile_high,
        "task_cluster_win_share": cluster_estimate,
        "task_cluster_interval_lower": cluster_low,
        "task_cluster_interval_upper": cluster_high,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": seed_by_family[family] if task_ids else None,
        "estimand": (
            "complete-case mean over observed frozen model-task contrasts; ties score 0.5; "
            "uncertainty resamples tasks while retaining all observed model contrasts"
        ),
    }


def _arena_task_cluster_bootstrap(
    consensus_rows: Sequence[Mapping[str, Any]],
    model_names: Mapping[str, str],
    *,
    replicates: int = 1000,
) -> dict[str, Any]:
    rows_by_task: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for row in consensus_rows:
        if row["track"] != "model_arena":
            continue
        choice = row.get("primary_consensus_choice")
        if choice in {None, "both_bad"}:
            continue
        left = str(row["left"]["season_model_id"])
        right = str(row["right"]["season_model_id"])
        outcome = 1.0 if choice == "left" else 0.0 if choice == "right" else 0.5
        rows_by_task[str(row["task_id"])].append((left, right, outcome))
    task_ids = sorted(rows_by_task)
    generator = np.random.default_rng(20260721)
    ratings: dict[str, list[float]] = defaultdict(list)
    ranks: dict[str, list[int]] = defaultdict(list)
    successful = 0
    disconnected = 0
    for _ in range(replicates):
        sampled = generator.choice(task_ids, size=len(task_ids), replace=True)
        comparisons = [
            comparison
            for task_id in sampled
            for comparison in rows_by_task[str(task_id)]
        ]
        graph = _comparison_graph_diagnostics(comparisons, list(model_names))
        if not graph["connected"]:
            disconnected += 1
            continue
        fitted = _fit_local_bradley_terry(comparisons)
        if set(fitted) != set(model_names):
            disconnected += 1
            continue
        ordered = sorted(
            model_names, key=lambda model_id: (-fitted[model_id][0], model_id)
        )
        for rank, model_id in enumerate(ordered, 1):
            ratings[model_id].append(fitted[model_id][0])
            ranks[model_id].append(rank)
        successful += 1
    models = {}
    for model_id, display_name in model_names.items():
        values = ratings[model_id]
        model_ranks = ranks[model_id]
        models[model_id] = {
            "season_model_id": model_id,
            "display_name": display_name,
            "successful_replicates": len(values),
            "rating_median": float(np.median(values)) if values else None,
            "rating_interval_lower": float(np.quantile(values, 0.025)) if values else None,
            "rating_interval_upper": float(np.quantile(values, 0.975)) if values else None,
            "rank_median": float(np.median(model_ranks)) if model_ranks else None,
            "rank_one_probability": (
                sum(rank == 1 for rank in model_ranks) / len(model_ranks)
                if model_ranks
                else None
            ),
        }
    return {
        "method": (
            "task-cluster nonparametric bootstrap; local Bradley-Terry refit used only "
            "for sensitivity, with the pinned arena-rank fit retained as primary"
        ),
        "seed": 20260721,
        "requested_replicates": replicates,
        "successful_replicates": successful,
        "disconnected_replicates": disconnected,
        "task_clusters": len(task_ids),
        "models": models,
    }


def _operational_rows(
    arms: Mapping[str, Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    cost_by_model: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for model in models:
        model_id = str(model["season_model_id"])
        rows = [row for row in arms.values() if row["model"]["season_model_id"] == model_id]
        if len(rows) != 240:
            raise Season0AnalysisError(f"{model_id} does not have 240 scored arms")
        latencies: list[int] = []
        answer_word_counts: list[int] = []
        answer_word_counts_by_condition: dict[str, list[int]] = defaultdict(list)
        identity_leak_arms = 0
        tool_calls = 0
        tool_successes = 0
        epicure_on = [row for row in rows if row["condition"] == "epicure_on"]
        epicure_on_with_calls = 0
        for row in rows:
            result = row.get("result")
            if not isinstance(result, Mapping):
                continue
            latency = result.get("wall_clock_latency_ms")
            if isinstance(latency, int):
                latencies.append(latency)
            answer = result.get("answer_markdown")
            if row.get("status") == "success" and isinstance(answer, str) and answer:
                count = len(_word_tokens(answer))
                answer_word_counts.append(count)
                answer_word_counts_by_condition[str(row["condition"])].append(count)
                if identity_leak_tags(answer):
                    identity_leak_arms += 1
            trace = result.get("tool_trace")
            if isinstance(trace, list):
                tool_calls += len(trace)
                tool_successes += sum(
                    isinstance(call, Mapping) and call.get("is_error") is False for call in trace
                )
                if row["condition"] == "epicure_on" and trace:
                    epicure_on_with_calls += 1
        failure_classes = Counter(_failure_class(row) for row in rows)
        failures = len(rows) - failure_classes["success"]
        model_failures = failure_classes["model_behavior_failure"]
        route_failures = (
            failure_classes["provider_pre_inference_failure"]
            + failure_classes["uncertain_delivery"]
        )
        model_cost = cost_by_model.get(model_id)
        if not isinstance(model_cost, Mapping):
            raise Season0AnalysisError(f"cost audit has no operational row for {model_id}")
        output[model_id] = {
            "season_model_id": model_id,
            "display_name": model["display_name"],
            "provider": model["provider"],
            "arms": len(rows),
            "success": failure_classes["success"],
            "failed": failures,
            "model_behavior_failures": model_failures,
            "incomplete_final_response_failures": sum(
                row.get("error_type") == "IncompleteFinalResponse" for row in rows
            ),
            "provider_route_failures": route_failures,
            "failure_breakdown": dict(sorted(failure_classes.items())),
            "invalid_response_rate": model_failures / len(rows),
            "end_to_end_failure_rate": failures / len(rows),
            "provider_route_failure_rate": route_failures / len(rows),
            "invalid_rate_epicure_off": sum(
                _failure_class(row) == "model_behavior_failure"
                and row["condition"] == "epicure_off"
                for row in rows
            )
            / 120,
            "invalid_rate_epicure_on": sum(
                _failure_class(row) == "model_behavior_failure"
                and row["condition"] == "epicure_on"
                for row in rows
            )
            / 120,
            "end_to_end_failure_rate_epicure_off": sum(
                row.get("status") != "success" and row["condition"] == "epicure_off"
                for row in rows
            )
            / 120,
            "end_to_end_failure_rate_epicure_on": sum(
                row.get("status") != "success" and row["condition"] == "epicure_on"
                for row in rows
            )
            / 120,
            "total_cost_usd": model_cost["conservative_cost_usd"],
            "attributed_cost_usd": model_cost["attributed_cost_usd"],
            "unattributed_cost_reservation_usd": model_cost[
                "unattributed_cost_reservation_usd"
            ],
            "mean_arm_cost_usd": model_cost["mean_arm_cost_usd"],
            "cost_attributed_arms": model_cost["cost_attributed_arms"],
            "cost_unattributed_arms": model_cost["cost_unattributed_arms"],
            "cost_accounting_basis": model_cost["cost_accounting_basis"],
            "latency_median_ms": statistics.median(latencies) if latencies else None,
            "latency_p95_ms": _percentile(latencies, 0.95),
            "answer_words_median": (
                statistics.median(answer_word_counts) if answer_word_counts else None
            ),
            "answer_words_mean_epicure_off": (
                statistics.mean(answer_word_counts_by_condition["epicure_off"])
                if answer_word_counts_by_condition["epicure_off"]
                else None
            ),
            "answer_words_mean_epicure_on": (
                statistics.mean(answer_word_counts_by_condition["epicure_on"])
                if answer_word_counts_by_condition["epicure_on"]
                else None
            ),
            "identity_leak_arms": identity_leak_arms,
            "identity_leak_rate": identity_leak_arms / len(rows),
            "tool_calls": tool_calls,
            "tool_success_rate": tool_successes / tool_calls if tool_calls else None,
            "epicure_on_tool_use_rate": epicure_on_with_calls / len(epicure_on),
        }
    return output


def _dimension_rows(
    consensus_rows: Sequence[Mapping[str, Any]], model_names: Mapping[str, str]
) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in consensus_rows:
        if row["track"] != "model_arena" or not row["primary_scores_available"]:
            continue
        for side in ("left", "right"):
            model_id = str(row[side]["season_model_id"])
            for dimension in DIMENSIONS:
                values[(model_id, dimension)].append(
                    float(row["primary_side_scores"][side]["scores"][dimension])
                )
    output = []
    for model_id, display_name in model_names.items():
        for dimension in DIMENSIONS:
            output.append(
                {
                    "season_model_id": model_id,
                    "display_name": display_name,
                    "dimension": dimension,
                    **_mean_ci(values[(model_id, dimension)]),
                }
            )
    return output


def _uplift_dimension_rows(
    consensus_rows: Sequence[Mapping[str, Any]], model_names: Mapping[str, str]
) -> list[dict[str, Any]]:
    deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in consensus_rows:
        if row["track"] != "epicure_uplift" or not row["primary_scores_available"]:
            continue
        model_id = str(row["season_model_id"])
        side_by_condition = {
            str(row[side]["condition"]): side for side in ("left", "right")
        }
        for dimension in DIMENSIONS:
            on = float(
                row["primary_side_scores"][side_by_condition["epicure_on"]]["scores"][
                    dimension
                ]
            )
            off = float(
                row["primary_side_scores"][side_by_condition["epicure_off"]]["scores"][
                    dimension
                ]
            )
            deltas[(model_id, dimension)].append(on - off)
    output = []
    for model_id, display_name in model_names.items():
        for dimension in DIMENSIONS:
            values = deltas[(model_id, dimension)]
            mean = statistics.mean(values) if values else None
            if len(values) > 1:
                error = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
                low = mean - error if mean is not None else None
                high = mean + error if mean is not None else None
            else:
                low = high = None
            output.append(
                {
                    "season_model_id": model_id,
                    "display_name": display_name,
                    "dimension": dimension,
                    "mean_delta": mean,
                    "lower": low,
                    "upper": high,
                    "n": len(values),
                }
            )
    return output


def _panel_uplift_dimension_rows(
    consensus_rows: Sequence[Mapping[str, Any]], *, replicates: int = 5000
) -> list[dict[str, Any]]:
    """Estimate panel-level rubric changes while resampling tasks as clusters."""

    values: dict[str, dict[str, list[float]]] = {
        dimension: defaultdict(list) for dimension in DIMENSIONS
    }
    for row in consensus_rows:
        if row["track"] != "epicure_uplift" or not row["primary_scores_available"]:
            continue
        side_by_condition = {
            str(row[side]["condition"]): side for side in ("left", "right")
        }
        task_id = str(row["task_id"])
        for dimension in DIMENSIONS:
            on = float(
                row["primary_side_scores"][side_by_condition["epicure_on"]]["scores"][
                    dimension
                ]
            )
            off = float(
                row["primary_side_scores"][side_by_condition["epicure_off"]]["scores"][
                    dimension
                ]
            )
            values[dimension][task_id].append(on - off)

    output = []
    for dimension_index, dimension in enumerate(DIMENSIONS):
        by_task = values[dimension]
        task_ids = sorted(by_task)
        observed = [value for task_id in task_ids for value in by_task[task_id]]
        mean = statistics.mean(observed) if observed else None
        lower = upper = None
        seed = 20260722 + dimension_index
        if task_ids:
            generator = np.random.default_rng(seed)
            bootstraps = np.empty(replicates, dtype=float)
            for index in range(replicates):
                sampled = generator.choice(task_ids, size=len(task_ids), replace=True)
                sampled_values = [
                    value for task_id in sampled for value in by_task[str(task_id)]
                ]
                bootstraps[index] = statistics.mean(sampled_values)
            lower = float(np.quantile(bootstraps, 0.025))
            upper = float(np.quantile(bootstraps, 0.975))
        output.append(
            {
                "dimension": dimension,
                "mean_delta": mean,
                "lower": lower,
                "upper": upper,
                "comparisons": len(observed),
                "task_clusters": len(task_ids),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed if task_ids else None,
                "estimand": (
                    "Epicure-on minus Epicure-off automated rubric score over the "
                    "frozen model panel and task corpus"
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _atomic_analysis(directory: Path, payload: Mapping[str, Any]) -> Path:
    digest = sha256_json(payload)
    document = {**payload, "artifact_sha256": digest}
    data = (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"season0-automated-analysis-{digest}.json"
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
    temporary.replace(destination)
    return destination


def analyze(
    *,
    task_bank: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    comparison_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any],
    arms_dir: Path,
    judgments_dir: Path,
    rate_card: Mapping[str, Any],
    target_cost_audit: Mapping[str, Any],
    output_dir: Path,
    target_cost_corrections_dir: Path | None = None,
    arm_interpretation_correction: dict[str, Any] | None = None,
    completion_interpretation_correction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_sha = _verify_artifact(task_bank, "task bank")
    model_sha = _verify_artifact(model_manifest, "model manifest")
    comparison_sha = _verify_artifact(comparison_manifest, "comparison manifest")
    judge_sha = _verify_artifact(judge_manifest, "judge manifest")
    if comparison_manifest.get("task_bank_artifact_sha256") != task_sha:
        raise Season0AnalysisError("comparison manifest task binding mismatch")
    if comparison_manifest.get("model_manifest_artifact_sha256") != model_sha:
        raise Season0AnalysisError("comparison manifest model binding mismatch")
    if (
        model_manifest.get("task_bank_artifact_sha256") != task_sha
        or judge_manifest.get("task_bank_artifact_sha256") != task_sha
        or judge_manifest.get("status") != "frozen_for_real_automated_judging"
        or judge_manifest.get("synthetic_compatibility_calls") != 0
    ):
        raise Season0AnalysisError("model or judge manifest task/real-run binding mismatch")
    protocol = judge_manifest.get("protocol")
    if not isinstance(protocol, Mapping) or any(
        protocol.get(key) != value
        for key, value in {
            "version": PROTOCOL_VERSION,
            "orientations": list(ORIENTATIONS),
            "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
            "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
        }.items()
    ):
        raise Season0AnalysisError("judge manifest protocol does not match the analysis")
    models = [model for model in model_manifest["models"] if isinstance(model, Mapping)]
    if len(models) != 12 or len({str(model["season_model_id"]) for model in models}) != 12:
        raise Season0AnalysisError("analysis requires 12 unique frozen model endpoints")
    model_names = {str(model["season_model_id"]): str(model["display_name"]) for model in models}
    comparisons = [
        row for row in comparison_manifest["comparisons"] if isinstance(row, Mapping)
    ]
    judges = [row for row in judge_manifest["judges"] if isinstance(row, Mapping)]
    if len(judges) != 4 or len({str(judge["judge_id"]) for judge in judges}) != 4:
        raise Season0AnalysisError("analysis requires four unique frozen judges")
    arms = _latest(arms_dir, "arm", "arm_id")
    judgments = _latest(judgments_dir, "judgment", "judgment_id")
    interpretation = validate_arm_interpretation_correction(
        correction=arm_interpretation_correction,
        arms_dir=arms_dir,
    )
    interpretation_sha = (
        interpretation.artifact_sha256 if interpretation is not None else None
    )
    arm_validation = _validate_real_arms(
        arms=arms,
        task_bank=task_bank,
        model_manifest=model_manifest,
        task_sha=task_sha,
        model_sha=model_sha,
    )
    if completion_interpretation_correction is None:
        raise Season0AnalysisError(
            "the historical collector requires an explicit final-completion correction"
        )
    completion_interpretation = validate_completion_interpretation_correction(
        correction=completion_interpretation_correction,
        arms_dir=arms_dir,
    )
    completion_interpretation_sha = completion_interpretation.artifact_sha256
    effective_arms = apply_completion_interpretation(
        arms,
        completion_interpretation,
    )
    arm_validation.update(
        {
            "collector_accepted_arms": arm_validation.pop("successful_arms"),
            "collector_rejected_arms": arm_validation.pop("failed_arms"),
            "normal_completion_arms": sum(
                arm.get("status") == "success" for arm in effective_arms.values()
            ),
            "effective_failed_arms": sum(
                arm.get("status") != "success" for arm in effective_arms.values()
            ),
            "completion_interpretation_correction_count": len(
                completion_interpretation.arm_ids
            ),
        }
    )
    _validate_comparison_reproduction(
        arms_dir=arms_dir,
        task_bank=task_bank,
        model_manifest=model_manifest,
        comparison_sha=comparison_sha,
    )
    target_cost_sha, cost_by_model = _validate_target_cost_audit(
        target_cost_audit=target_cost_audit,
        rate_card=rate_card,
        arms=arms,
        models=models,
        arm_interpretation_correction_sha256=interpretation_sha,
    )
    _validate_cost_reproduction(
        arms_dir=arms_dir,
        rate_card=rate_card,
        target_cost_sha=target_cost_sha,
        corrections_dir=target_cost_corrections_dir,
        arm_interpretation_correction=arm_interpretation_correction,
    )
    expected_judgment_ids = _expected_judgment_ids(
        comparisons=comparisons,
        judges=judges,
        comparison_sha=comparison_sha,
        judge_sha=judge_sha,
    )
    if set(judgments) != expected_judgment_ids:
        missing = len(expected_judgment_ids - set(judgments))
        unexpected = len(set(judgments) - expected_judgment_ids)
        raise Season0AnalysisError(
            f"judgment registry mismatch: {missing} missing, {unexpected} unexpected"
        )
    judge_by_id = {str(row["judge_id"]): row for row in judges}
    for record in judgments.values():
        _verify_artifact(record, f"judgment {record.get('judgment_id')}")
        contracts = record.get("contracts")
        if (
            record.get("schema_version") != JUDGMENT_RECORD_SCHEMA
            or record.get("synthetic") is not False
            or not isinstance(contracts, Mapping)
            or contracts.get("comparison_manifest_artifact_sha256") != comparison_sha
            or contracts.get("judge_manifest_artifact_sha256") != judge_sha
            or contracts.get("task_bank_artifact_sha256") != task_sha
            or contracts.get("protocol_version") != PROTOCOL_VERSION
            or contracts.get("system_prompt_sha256") != JUDGE_SYSTEM_PROMPT_SHA256
            or contracts.get("judgment_schema_sha256") != JUDGMENT_SCHEMA_SHA256
        ):
            raise Season0AnalysisError("judgment eligibility contract mismatch")
        identity = {
            "schema_version": JUDGMENT_RECORD_SCHEMA,
            "season": "Season 0",
            "comparison_manifest_artifact_sha256": comparison_sha,
            "judge_manifest_artifact_sha256": judge_sha,
            "protocol_version": PROTOCOL_VERSION,
            "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
            "judgment_schema_sha256": JUDGMENT_SCHEMA_SHA256,
            "comparison_id": record.get("comparison_id"),
            "judge_id": (record.get("judge") or {}).get("judge_id"),
            "orientation": record.get("orientation"),
        }
        if sha256_json(identity) != record.get("judgment_id"):
            raise Season0AnalysisError("judgment identity hash mismatch")
        judge_id = str((record.get("judge") or {}).get("judge_id") or "")
        if record.get("judge") != judge_by_id.get(judge_id):
            raise Season0AnalysisError("judgment judge provenance differs from the manifest")
        status = record.get("status")
        if status not in {"success", "failed"}:
            raise Season0AnalysisError("judgment has an unknown terminal status")
        if status == "success":
            result = record.get("result")
            if not isinstance(result, Mapping):
                raise Season0AnalysisError("successful judgment has no result")
            validated = validate_judgment(result.get("judgment"))
            if result.get("judgment") != validated or result.get(
                "normalized_choice"
            ) != normalize_choice(validated["choice"], str(record.get("orientation"))):
                raise Season0AnalysisError("successful judgment was not protocol-normalized")
    effective_comparisons, completion_comparison_diagnostics = (
        _apply_completion_policy_to_comparisons(
            comparisons,
            frozenset(completion_interpretation.arm_ids),
        )
    )
    consensus, judge_diagnostics = aggregate_consensus(
        comparisons=effective_comparisons, judges=judges, judgments=judgments
    )
    family_balanced_rows, family_balanced_diagnostics = family_balanced_consensus(
        consensus, judges
    )
    arena_global = _arena_rows(consensus, model_names, None)
    uplift_global = _uplift_rows(consensus, model_names, None)
    arena_by_family = {
        family: _arena_rows(consensus, model_names, family) for family in FAMILIES
    }
    uplift_by_family = {
        family: _uplift_rows(consensus, model_names, family) for family in FAMILIES
    }
    panel_uplift = _panel_uplift_summary(consensus, None)
    panel_uplift_by_family = {
        family: _panel_uplift_summary(consensus, family) for family in FAMILIES
    }
    arena_bootstrap = _arena_task_cluster_bootstrap(consensus, model_names)
    family_balanced_arena = _arena_rows(family_balanced_rows, model_names, None)
    family_balanced_uplift = _uplift_rows(family_balanced_rows, model_names, None)
    family_balanced_graph = _comparison_graph_diagnostics(
        [
            (
                str(row["left"]["season_model_id"]),
                str(row["right"]["season_model_id"]),
                0.5,
            )
            for row in family_balanced_rows
            if row["track"] == "model_arena"
            and row.get("primary_consensus_choice") not in {None, "both_bad"}
        ],
        list(model_names),
    )
    operational = _operational_rows(effective_arms, models, cost_by_model)
    for row in arena_global:
        row.update(operational[row["season_model_id"]])
    dimension_rows = _dimension_rows(consensus, model_names)
    uplift_dimension_rows = _uplift_dimension_rows(consensus, model_names)
    panel_uplift_dimension_rows = _panel_uplift_dimension_rows(consensus)
    reference_overlap, overlap_flagged_arms = _reference_overlap_audit(
        arms=effective_arms,
        tasks=[task for task in task_bank["tasks"] if isinstance(task, Mapping)],
        model_names=model_names,
    )
    overlap_sensitivity_rows = [
        row
        for row in consensus
        if str(row["left"]["arm_id"]) not in overlap_flagged_arms
        and str(row["right"]["arm_id"]) not in overlap_flagged_arms
    ]
    arena_graphs = {
        "global": _comparison_graph_diagnostics(
            [
                (
                    str(row["left"]["season_model_id"]),
                    str(row["right"]["season_model_id"]),
                    0.5,
                )
                for row in consensus
                if row["track"] == "model_arena"
                and row.get("primary_consensus_choice") not in {None, "both_bad"}
            ],
            list(model_names),
        ),
        "by_family": {
            family: _comparison_graph_diagnostics(
                [
                    (
                        str(row["left"]["season_model_id"]),
                        str(row["right"]["season_model_id"]),
                        0.5,
                    )
                    for row in consensus
                    if row["track"] == "model_arena"
                    and row["task_family"] == family
                    and row.get("primary_consensus_choice") not in {None, "both_bad"}
                ],
                list(model_names),
            )
            for family in FAMILIES
        },
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "season": "Season 0",
        "status": "automated_cohort_analysis_complete",
        "synthetic_arms": 0,
        "synthetic_judgments": 0,
        "cohort": "automated_four_judge_three_family_swap_controlled_nonself_consensus",
        "task_bank_artifact_sha256": task_sha,
        "model_manifest_artifact_sha256": model_sha,
        "comparison_manifest_artifact_sha256": comparison_sha,
        "judge_manifest_artifact_sha256": judge_sha,
        "target_cost_audit_artifact_sha256": target_cost_sha,
        "arm_interpretation_correction_artifact_sha256": interpretation_sha,
        "arm_interpretation_correction_count": (
            len(interpretation.arm_ids) if interpretation is not None else 0
        ),
        "completion_interpretation_correction_artifact_sha256": (
            completion_interpretation_sha
        ),
        "completion_interpretation_correction_count": len(
            completion_interpretation.arm_ids
        ),
        "implementation": _implementation_manifest(),
        "methods": {
            "model_arena": "arena-rank 0.1.1 Bradley-Terry; ties are half-wins; both_bad excluded",
            "model_arena_sensitivity": (
                "1,000-replicate task-cluster bootstrap with rank-one probability"
            ),
            "epicure_uplift": (
                "observed tie-half preference with a multinomial "
                "profile-likelihood interval"
            ),
            "epicure_treatment": (
                "offer of access to the frozen Epicure MCP; enabled arms remain treated "
                "regardless of observed tool invocation"
            ),
            "panel_epicure_uplift": (
                "tie-as-half complete-case preference mean with 5,000-replicate "
                "task-cluster bootstrap; a pooled tie-aware profile estimate is also released; "
                "the frozen model panel is treated as fixed"
            ),
            "panel_epicure_uplift_dimensions": (
                "Epicure-on minus Epicure-off rubric deltas with 5,000-replicate "
                "task-cluster bootstrap intervals; the frozen model panel is fixed"
            ),
            "judge_consensus": (
                "majority after original/swapped consistency and self-judge exclusion"
            ),
            "judge_family_balanced_sensitivity": (
                "within-family unanimous collapse followed by strict majority across canonical "
                "judge-model families; no rubric-score sensitivity is inferred"
            ),
            "disconnected_graph_policy": (
                "withhold Bradley-Terry ratings when all frozen models are not connected"
            ),
            "failure_attribution": (
                "reconciled model-behavior failures are separated from provider-route failures"
            ),
            "final_completion_policy": (
                "only stop, end_turn, stop_sequence, or completed final responses are "
                "rank eligible; content_filter, length, and max_tokens are failures"
            ),
            "cost_accounting": (
                "per-model mean uses all 240 attempted arms, with frozen reservations for "
                "unattributed possible-delivery requests"
            ),
            "score_intervals": "normal 95% intervals over task-level comparison means",
            "public_and_expert_pooling": "none",
        },
        "counts": {
            "scored_arms": len(arms),
            "comparison_manifest_rows": len(comparisons),
            "judgment_records": len(judgments),
            "consensus_rows": len(consensus),
            "consensus_available": sum(
                row["primary_consensus_available"] for row in consensus
            ),
            "source_judgable_comparisons": completion_comparison_diagnostics.get(
                "source_judgable", 0
            ),
            "effective_judgable_comparisons": completion_comparison_diagnostics.get(
                "effective_judgable", 0
            ),
            "incomplete_final_response_comparison_exclusions": (
                completion_comparison_diagnostics.get(
                    "excluded_incomplete_final_response", 0
                )
            ),
            "recorded_provider_calls_including_partial": arm_validation[
                "recorded_provider_calls_including_partial"
            ],
            "recorded_epicure_calls_including_partial": arm_validation[
                "recorded_epicure_calls_including_partial"
            ],
        },
        "arm_validation": arm_validation,
        "completion_comparison_diagnostics": completion_comparison_diagnostics,
        "judge_diagnostics": judge_diagnostics,
        "judge_family_balanced_sensitivity": {
            "diagnostics": family_balanced_diagnostics,
            "arena_graph": family_balanced_graph,
            "model_leaderboard": family_balanced_arena,
            "uplift_leaderboard": family_balanced_uplift,
            "panel_uplift": _panel_uplift_summary(family_balanced_rows, None),
        },
        "arena_graph_diagnostics": arena_graphs,
        "arena_task_cluster_bootstrap": arena_bootstrap,
        "reference_overlap_audit": reference_overlap,
        "verbosity_diagnostics": _verbosity_diagnostics(consensus, effective_arms),
        "reference_overlap_sensitivity": {
            "excluded_flagged_arms": len(overlap_flagged_arms),
            "retained_consensus_rows": len(overlap_sensitivity_rows),
            "model_leaderboard": _arena_rows(
                overlap_sensitivity_rows, model_names, None
            ),
            "uplift_leaderboard": _uplift_rows(
                overlap_sensitivity_rows, model_names, None
            ),
        },
        "model_leaderboard": arena_global,
        "model_leaderboard_by_family": arena_by_family,
        "uplift_leaderboard": uplift_global,
        "uplift_leaderboard_by_family": uplift_by_family,
        "panel_uplift": panel_uplift,
        "panel_uplift_by_family": panel_uplift_by_family,
        "operational_metrics": operational,
        "dimension_scores": dimension_rows,
        "uplift_dimension_deltas": uplift_dimension_rows,
        "panel_uplift_dimensions": panel_uplift_dimension_rows,
        "comparison_consensus": consensus,
    }
    analysis_path = _atomic_analysis(output_dir, payload)
    _write_csv(output_dir / "model-leaderboard.csv", arena_global)
    _write_csv(output_dir / "uplift-leaderboard.csv", uplift_global)
    _write_csv(output_dir / "dimension-scores.csv", dimension_rows)
    _write_csv(output_dir / "uplift-dimension-deltas.csv", uplift_dimension_rows)
    _write_csv(
        output_dir / "panel-uplift-dimension-deltas.csv",
        panel_uplift_dimension_rows,
    )
    return {**payload, "analysis_path": str(analysis_path)}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--judgments-dir", type=Path, required=True)
    parser.add_argument("--rate-card", type=Path, required=True)
    parser.add_argument("--target-cost-audit", type=Path, required=True)
    parser.add_argument("--target-cost-corrections-dir", type=Path)
    parser.add_argument("--arm-interpretation-correction", type=Path)
    parser.add_argument("--completion-interpretation-correction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(
        task_bank=_load(args.task_bank),
        model_manifest=_load(args.model_manifest),
        comparison_manifest=_load(args.comparison_manifest),
        judge_manifest=_load(args.judge_manifest),
        arms_dir=args.arms_dir,
        judgments_dir=args.judgments_dir,
        rate_card=_load(args.rate_card),
        target_cost_audit=_load(args.target_cost_audit),
        output_dir=args.output_dir,
        target_cost_corrections_dir=args.target_cost_corrections_dir,
        arm_interpretation_correction=(
            _load(args.arm_interpretation_correction)
            if args.arm_interpretation_correction is not None
            else None
        ),
        completion_interpretation_correction=_load(
            args.completion_interpretation_correction
        ),
    )
    compact = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "comparison_consensus",
            "dimension_scores",
            "uplift_dimension_deltas",
            "panel_uplift_dimensions",
            "model_leaderboard_by_family",
            "uplift_leaderboard_by_family",
        }
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    run()
