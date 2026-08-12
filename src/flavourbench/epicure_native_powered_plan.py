"""Freeze the powered FlavourBench execution and analysis plan.

The plan is written before any powered-panel response is requested.  It binds
the exact model routes, hidden task set, repeat panel, decoding contract,
budget ceiling, missing-data policy, inferential tests, and ranking rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_native_taskset_v2 import CHOICE_LABELS, FAMILIES, verify_taskset
from .execution_policy import ExecutionPolicy
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-powered-analysis-plan-v1"
REPEAT_SCHEMA_VERSION = "flavourbench-powered-repeat-panel-v1"
PLAN_VERSION = "flavourbench-powered-20x640-v1"
PRIMARY_TASK_COUNT = 640
REPEAT_TASK_COUNT = 64
MODEL_COUNT = 20
PAIRWISE_COMPARISON_COUNT = MODEL_COUNT * (MODEL_COUNT - 1) // 2
PRIMARY_CALL_COUNT = PRIMARY_TASK_COUNT * MODEL_COUNT
REPEAT_CALL_COUNT = REPEAT_TASK_COUNT * MODEL_COUNT
TOTAL_CALL_COUNT = PRIMARY_CALL_COUNT + REPEAT_CALL_COUNT
FAMILY_COUNT = 4
TASKS_PER_FAMILY = PRIMARY_TASK_COUNT // FAMILY_COUNT
REPEATS_PER_FAMILY = REPEAT_TASK_COUNT // FAMILY_COUNT
GLOBAL_BUDGET_CAP_USD = "85"
EPICURE_TOOL_SCHEMA_SHA256 = "666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd"


class PoweredPlanError(RuntimeError):
    """The powered execution plan failed an integrity or design check."""


def powered_execution_policy() -> ExecutionPolicy:
    """Return the exact model-only generation policy bound by the plan."""

    return ExecutionPolicy(
        max_output_tokens=2048,
        max_tool_rounds=1,
        max_tool_result_bytes=4096,
        max_cumulative_tool_result_bytes=4096,
        max_tool_calls_per_round=1,
        max_tool_calls_total=1,
        max_provider_attempts=2,
        decoding_temperature=0.0,
        decoding_top_p=1.0,
        decoding_seed=20260810,
        tool_argument_repair_turns=1,
        approximate_non_user_prompt_bytes=2000,
        conservative_bytes_per_token=3,
        pair_arm_scheduling="sequential",
        final_response_mode="plain_text",
        max_intermediate_tokens=1024,
        required_tool_contract_max_intermediate_tokens=2048,
        matched_planning=False,
        evidence_protocol="portable_text_tool_v1",
        intermediate_reasoning_effort=None,
        final_reasoning_effort=None,
        tool_catalog_bytes_bound=16000,
        epicure_on_tool_required=False,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PoweredPlanError(f"{label} must be a regular, non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PoweredPlanError(f"could not load {label}") from error
    if not isinstance(value, dict):
        raise PoweredPlanError(f"{label} must be a JSON object")
    return value


def _binomial_probabilities(n: int, probability: float) -> list[float]:
    if n == 0:
        return [1.0]
    if probability == 0:
        return [1.0, *([0.0] * n)]
    if probability == 1:
        return [*([0.0] * n), 1.0]
    probabilities = [0.0] * (n + 1)
    mode = min(n, int((n + 1) * probability))
    probabilities[mode] = 1.0
    odds = probability / (1 - probability)
    for successes in range(mode, n):
        probabilities[successes + 1] = (
            probabilities[successes] * (n - successes) / (successes + 1) * odds
        )
    for successes in range(mode, 0, -1):
        probabilities[successes - 1] = (
            probabilities[successes] * successes / (n - successes + 1) / odds
        )
    scale = sum(probabilities)
    return [value / scale for value in probabilities]


def exact_mcnemar_power(
    *,
    pairs: int,
    discordance_probability: float,
    absolute_accuracy_difference: float,
    familywise_alpha: float,
    comparisons: int,
) -> float:
    """Exact unconditional power for a two-sided conditional McNemar test.

    The least-favourable Bonferroni threshold is used.  The data-generating
    alternative fixes total discordance and the marginal accuracy difference.
    """

    q = discordance_probability
    delta = absolute_accuracy_difference
    if not 0 < delta < q < 1 or pairs <= 0 or comparisons <= 0:
        raise PoweredPlanError("invalid McNemar power inputs")
    conditional_better = (q + delta) / (2 * q)
    tail_alpha = familywise_alpha / comparisons / 2
    discordance_counts = _binomial_probabilities(pairs, q)
    power = 0.0
    for discordant, probability_discordant in enumerate(discordance_counts):
        null = _binomial_probabilities(discordant, 0.5)
        tail = 0.0
        critical = discordant + 1
        for better in range(discordant, -1, -1):
            tail += null[better]
            if tail <= tail_alpha:
                critical = better
        alternative = _binomial_probabilities(discordant, conditional_better)
        conditional_power = sum(alternative[critical:]) if critical <= discordant else 0.0
        power += probability_discordant * conditional_power
    return power


def _repeat_selection_key(task_id: str) -> str:
    return hashlib.sha256(
        ("flavourbench-powered-repeat-panel-20260811\0" + task_id).encode()
    ).hexdigest()


def _repeat_prompt(question_prefix: str, choices: Mapping[str, str]) -> str:
    rendered = "\n".join(f"{label}. {choices[label]}" for label in CHOICE_LABELS)
    return (
        f"{question_prefix}\n\nChoices:\n{rendered}\n\n"
        "Return exactly one line: `FINAL_CHOICE: X`, replacing X with A, B, C, or D."
    )


def build_repeat_panel(taskset: Mapping[str, Any]) -> dict[str, Any]:
    if not verify_taskset(taskset):
        raise PoweredPlanError("repeat panel requires the exact valid powered task set")
    tasks = taskset["tasks"]
    selected: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_tasks = [task for task in tasks if task["family"] == family]
        chosen = sorted(family_tasks, key=lambda task: _repeat_selection_key(task["task_id"]))[
            :REPEATS_PER_FAMILY
        ]
        for repeat_index, task in enumerate(chosen):
            shift = 1 + (int(_repeat_selection_key(task["task_id"]), 16) % 3)
            original_choices = dict(task["choices"])
            permuted_choices = {
                label: original_choices[CHOICE_LABELS[(index - shift) % 4]]
                for index, label in enumerate(CHOICE_LABELS)
            }
            expected_value = original_choices[task["expected_choice"]]
            expected_choice = next(
                label for label, value in permuted_choices.items() if value == expected_value
            )
            prompt_prefix = str(task["prompt"]).split("\n\nChoices:\n", 1)[0]
            prompt = _repeat_prompt(prompt_prefix, permuted_choices)
            selected.append(
                {
                    "task_id": f"repeat-{family}-{repeat_index + 1:02d}",
                    "original_task_id": task["task_id"],
                    "family": family,
                    "original_prompt_sha256": task["prompt_sha256"],
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "choices": permuted_choices,
                    "permutation_shift": shift,
                    "original_expected_choice": task["expected_choice"],
                    "expected_choice": expected_choice,
                    "expected_choice_value_sha256": hashlib.sha256(
                        expected_value.encode()
                    ).hexdigest(),
                }
            )
    document: dict[str, Any] = {
        "schema_version": REPEAT_SCHEMA_VERSION,
        "status": "frozen_before_powered_model_execution",
        "source_taskset_artifact_sha256": taskset["artifact_sha256"],
        "source_task_set_sha256": taskset["task_set_sha256"],
        "selection_seed": "flavourbench-powered-repeat-panel-20260811",
        "counts": {
            "tasks": REPEAT_TASK_COUNT,
            "tasks_per_family": REPEATS_PER_FAMILY,
            "models": MODEL_COUNT,
            "provider_calls": REPEAT_CALL_COUNT,
        },
        "purpose": {
            "primary": "choice-content repeatability under answer-position permutation",
            "secondary": "answer-position sensitivity",
            "not_used_for": ["primary FlavourBench Score", "model selection", "item pruning"],
        },
        "tasks": selected,
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_repeat_panel(document, source_taskset=taskset):
        raise PoweredPlanError("constructed repeat panel failed verification")
    return document


def verify_repeat_panel(
    document: Mapping[str, Any], *, source_taskset: Mapping[str, Any] | None = None
) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    tasks = document.get("tasks")
    if (
        document.get("schema_version") != REPEAT_SCHEMA_VERSION
        or recorded != _sha256(payload)
        or not isinstance(tasks, list)
        or len(tasks) != REPEAT_TASK_COUNT
        or len({task.get("original_task_id") for task in tasks}) != REPEAT_TASK_COUNT
        or Counter(task.get("family") for task in tasks)
        != Counter({family: REPEATS_PER_FAMILY for family in FAMILIES})
    ):
        return False
    for task in tasks:
        if (
            not isinstance(task, Mapping)
            or task.get("permutation_shift") not in {1, 2, 3}
            or set(task.get("choices") or {}) != set(CHOICE_LABELS)
            or task.get("expected_choice") not in CHOICE_LABELS
            or hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
            != task.get("prompt_sha256")
        ):
            return False
    if source_taskset is not None:
        if not verify_taskset(source_taskset):
            return False
        originals = {task["task_id"]: task for task in source_taskset["tasks"]}
        for repeat in tasks:
            original = originals.get(repeat["original_task_id"])
            if not original:
                return False
            expected_value = original["choices"][original["expected_choice"]]
            if repeat["choices"][repeat["expected_choice"]] != expected_value:
                return False
    return True


def _semantic_digest(document: Mapping[str, Any], field: str) -> str:
    payload = dict(document)
    recorded = str(payload.pop(field, ""))
    if recorded != _sha256(payload):
        raise PoweredPlanError(f"source {field} is invalid")
    return recorded


def build_plan(
    *,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    taskset: Mapping[str, Any],
    taskset_physical_sha256: str,
    repeat_panel: Mapping[str, Any],
    repeat_panel_physical_sha256: str,
    predecessor_release: Mapping[str, Any],
    predecessor_release_physical_sha256: str,
) -> dict[str, Any]:
    if not verify_taskset(taskset) or not verify_repeat_panel(repeat_panel, source_taskset=taskset):
        raise PoweredPlanError("plan inputs are not valid")
    manifest_digest = str((manifest.get("content_address") or {}).get("digest") or "")
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT:
        raise PoweredPlanError("powered plan requires exactly 20 unique model routes")
    predecessor_digest = _semantic_digest(predecessor_release, "artifact_sha256")
    taskset_digest = str(taskset["artifact_sha256"])
    repeat_digest = str(repeat_panel["artifact_sha256"])
    pilot_task_id = min(
        (str(task["task_id"]) for task in taskset["tasks"]),
        key=lambda task_id: hashlib.sha256(
            ("flavourbench-powered-pilot-v1\0" + task_id).encode()
        ).hexdigest(),
    )
    route_rows = [
        {
            "slot_id": candidate.slot_id,
            "model_id": candidate.model_id,
            "model_name": candidate.model_name,
            "canonical_model_slug": candidate.canonical_model_slug,
            "execution_backend": candidate.execution_backend,
            "provider_tag": candidate.provider_tag,
            "provider_name": candidate.provider_name,
            "endpoint_sha256": candidate.endpoint_sha256,
            "endpoint_execution_sha256": candidate.endpoint_execution_sha256,
            "backend_contract_sha256": candidate.backend_contract_sha256,
            "cost_accounting_policy": candidate.cost_accounting_policy,
        }
        for candidate in candidates
    ]
    power_19 = exact_mcnemar_power(
        pairs=PRIMARY_TASK_COUNT,
        discordance_probability=0.30,
        absolute_accuracy_difference=0.10,
        familywise_alpha=0.05,
        comparisons=MODEL_COUNT - 1,
    )
    power_190 = exact_mcnemar_power(
        pairs=PRIMARY_TASK_COUNT,
        discordance_probability=0.30,
        absolute_accuracy_difference=0.10,
        familywise_alpha=0.05,
        comparisons=PAIRWISE_COMPARISON_COUNT,
    )
    power_190_125 = exact_mcnemar_power(
        pairs=PRIMARY_TASK_COUNT,
        discordance_probability=0.30,
        absolute_accuracy_difference=0.125,
        familywise_alpha=0.05,
        comparisons=PAIRWISE_COMPARISON_COUNT,
    )
    execution_policy = powered_execution_policy()
    document: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_version": PLAN_VERSION,
        "status": "preregistered_before_any_powered_provider_response",
        "frozen_date": "2026-08-11",
        "inputs": {
            "route_manifest": {
                "semantic_sha256": manifest_digest,
                "physical_sha256": manifest_physical_sha256,
            },
            "hidden_taskset": {
                "semantic_sha256": taskset_digest,
                "task_set_sha256": taskset["task_set_sha256"],
                "physical_sha256": taskset_physical_sha256,
            },
            "repeat_panel": {
                "semantic_sha256": repeat_digest,
                "physical_sha256": repeat_panel_physical_sha256,
            },
            "predecessor_release": {
                "semantic_sha256": predecessor_digest,
                "physical_sha256": predecessor_release_physical_sha256,
                "role": "development_predecessor_not_reused_as_powered_primary_data",
            },
        },
        "roster": {
            "model_count": MODEL_COUNT,
            "route_count": MODEL_COUNT,
            "models": route_rows,
            "identity_rule": "exact requested route and returned model/provider contract",
            "fallbacks": "disabled",
        },
        "design": {
            "primary_tasks": PRIMARY_TASK_COUNT,
            "tasks_per_family": TASKS_PER_FAMILY,
            "families": list(FAMILIES),
            "difficulty_bands_per_family": 4,
            "tasks_per_difficulty_band_per_family": 40,
            "answer_positions_per_label_per_family": 40,
            "unique_anchor_ingredients": PRIMARY_TASK_COUNT,
            "primary_provider_calls": PRIMARY_CALL_COUNT,
            "repeat_tasks": REPEAT_TASK_COUNT,
            "repeat_provider_calls": REPEAT_CALL_COUNT,
            "total_provider_calls": TOTAL_CALL_COUNT,
            "single_response_per_model_task_cell": True,
            "result_adaptive_sampling": False,
            "posthoc_item_exclusion": False,
        },
        "execution": {
            "condition": "model_only_no_epicure_runtime_access",
            "final_response_mode": "plain_text",
            "evidence_protocol": "portable_text_tool_v1",
            "max_output_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260810,
            "maximum_provider_attempts": 2,
            "ambiguous_delivery_retried": False,
            "external_browsing": False,
            "pilot": {
                "task_id": pilot_task_id,
                "cells": MODEL_COUNT,
                "responses_are_primary_cells_and_are_reused_in_the_full_run": True,
            },
            "schedule": (
                "pilot task across all models, then deterministic per-model SHA-256 task order, "
                "then the frozen repeat panel"
            ),
            "epicure_release_id": taskset["epicure_provenance"]["release_id"],
            "epicure_bundle_sha256": taskset["epicure_provenance"]["bundle_sha256"],
            "epicure_application_sha256": taskset["epicure_provenance"]["application_sha256"],
            "epicure_tool_schema_sha256": EPICURE_TOOL_SCHEMA_SHA256,
            "execution_policy_sha256": execution_policy.sha256,
            "execution_policy": execution_policy.document(),
        },
        "outcomes": {
            "primary": {
                "name": "FlavourBench Score",
                "definition": "100 times equal-family macro exact accuracy",
                "task_weight": "equal within family",
                "family_weight": "0.25 each",
                "chance_score": 25.0,
            },
            "family_scores": "exact accuracy within each of four prespecified families",
            "answer_parser": "last exact FINAL_CHOICE A-D line",
            "missing_unparseable_or_failed": "zero points in intention-to-evaluate score",
            "availability": "reported separately from capability score",
            "epicure_assisted_predecessor": (
                "separate 32-task execution-ceiling diagnostic; excluded from primary ranking"
            ),
        },
        "eligibility": {
            "minimum_completed_fraction": 0.95,
            "minimum_completed_tasks": 608,
            "below_threshold_label": "DNF",
            "dnf_not_interpreted_as_worst_capability": True,
            "all_scheduled_cells_remain_in_denominator": True,
        },
        "inference": {
            "familywise_alpha": 0.05,
            "score_intervals": (
                "Wilson intervals with Bonferroni simultaneous coverage over 20 models"
            ),
            "family_intervals": (
                "Wilson intervals with Bonferroni simultaneous coverage over 80 model-family cells"
            ),
            "chance_tests": "one-sided exact binomial tests, Holm correction across 20 models",
            "paired_model_tests": (
                "two-sided exact conditional McNemar tests on all 190 model pairs, Holm correction"
            ),
            "effect_size": "paired accuracy difference in percentage points",
            "rank_display": "statistical rank groups; no forced ordering inside unresolved groups",
            "definitive_top_rule": (
                "eligible model has positive accuracy difference and Holm-adjusted p<0.05 "
                "against every other eligible model, with repeatability>=0.80"
            ),
            "family_and_difficulty_analyses": "prespecified secondary analyses",
            "task_difficulty_and_discrimination": "descriptive item analysis only",
        },
        "power": {
            "method": "exact unconditional power for two-sided conditional McNemar test",
            "pairs": PRIMARY_TASK_COUNT,
            "assumed_total_discordance": 0.30,
            "target_absolute_difference": 0.10,
            "bonferroni_19_power": round(power_19, 12),
            "bonferroni_190_power": round(power_190, 12),
            "bonferroni_190_power_at_0_125_difference": round(power_190_125, 12),
            "minimum_target": 0.80,
            "all_190_at_10pp_meets_target": power_190 >= 0.80,
            "interpretation": ("least-favourable Bonferroni power bound; Holm is no less powerful"),
        },
        "repeatability": {
            "tasks_per_model": REPEAT_TASK_COUNT,
            "choice_positions_permuted": True,
            "primary_statistic": "selected choice-content agreement with original response",
            "secondary_statistic": "accuracy change and label-position transition matrix",
            "acceptance_floor": 0.80,
            "excluded_from_primary_score": True,
        },
        "budget": {
            "currency": "USD",
            "hard_cap": GLOBAL_BUDGET_CAP_USD,
            "forecast_from_predecessor_mean": "31.6",
            "high_envelope_from_predecessor_observed_maxima": "79.8",
            "admission": "stop before scheduling any cell whose reserved exposure exceeds cap",
            "credentials_in_artifacts": False,
        },
        "release_rule": {
            "publish_if": (
                "integrity checks pass and at least two eligible endpoints complete; "
                "lack of a unique top model does not block publication"
            ),
            "claims": (
                "finite-panel culinary exact-choice performance under this task distribution; "
                "not universal language-model quality"
            ),
            "oracle_role": (
                "Epicure deterministically constructs answer keys; model outputs are scored "
                "without Epicure access in the primary condition"
            ),
        },
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise PoweredPlanError("constructed powered plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        models = document["roster"]["models"]
        design = document["design"]
        power = document["power"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and recorded == _sha256(payload)
        and isinstance(models, list)
        and len(models) == MODEL_COUNT
        and len({model.get("model_id") for model in models}) == MODEL_COUNT
        and design.get("primary_tasks") == PRIMARY_TASK_COUNT
        and design.get("total_provider_calls") == TOTAL_CALL_COUNT
        and power.get("all_190_at_10pp_meets_target") is True
        and float(power.get("bonferroni_190_power") or 0) >= 0.80
    )


def _write(document: Mapping[str, Any], *, output_directory: Path, prefix: str) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    digest = str(document["artifact_sha256"])
    destination = output_directory / f"{prefix}-{digest}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise PoweredPlanError(f"content-addressed {prefix} conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_directory, delete=False
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-semantic-sha256", required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--predecessor-release", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    taskset = _load_json(args.taskset, label="powered task set")
    predecessor = _load_json(args.predecessor_release, label="predecessor release")
    repeat_panel = build_repeat_panel(taskset)
    repeat_path = _write(
        repeat_panel,
        output_directory=args.output_directory,
        prefix="epicure-native-powered-repeat-panel",
    )
    plan = build_plan(
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        taskset=taskset,
        taskset_physical_sha256=_sha256_file(args.taskset),
        repeat_panel=repeat_panel,
        repeat_panel_physical_sha256=_sha256_file(repeat_path),
        predecessor_release=predecessor,
        predecessor_release_physical_sha256=_sha256_file(args.predecessor_release),
    )
    plan_path = _write(
        plan,
        output_directory=args.output_directory,
        prefix="epicure-native-powered-analysis-plan",
    )
    print(plan_path)


if __name__ == "__main__":
    run()
