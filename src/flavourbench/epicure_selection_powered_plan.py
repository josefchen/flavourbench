"""Freeze the powered analysis and repeat plan for Epicure-scored selections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .epicure_native_powered_plan import EPICURE_TOOL_SCHEMA_SHA256, powered_execution_policy
from .epicure_selection_taskset_v1 import (
    ALL_SELECTION_KEYS,
    FAMILIES,
    LABELS,
    SELECTION_SIZE,
    TASK_COUNT,
    selection_parser_contract,
    selection_parser_sha256,
    verify_taskset,
)
from .execution_policy import SELECTION_TEXT_PROTOCOL_V1
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-selection-powered-analysis-plan-v14"
REPEAT_SCHEMA_VERSION = "flavourbench-selection-powered-repeat-panel-v1"
PLAN_VERSION = "flavourbench-selection-20x640-v14"
MODEL_COUNT = 20
REPEAT_TASK_COUNT = 64
REPEATS_PER_FAMILY = 16
PRIMARY_CALL_COUNT = MODEL_COUNT * TASK_COUNT
REPEAT_CALL_COUNT = MODEL_COUNT * REPEAT_TASK_COUNT
TOTAL_CALL_COUNT = PRIMARY_CALL_COUNT + REPEAT_CALL_COUNT
PAIRWISE_COMPARISONS = MODEL_COUNT * (MODEL_COUNT - 1) // 2
PROGRAM_BUDGET_CAP_USD = "87"
SUCCESSOR_RUN_CAP_USD = "83"
MODEL_FINAL_REASONING_EFFORT = {"qwen/qwen3.8-max": "minimal"}


class SelectionPoweredPlanError(RuntimeError):
    """The Epicure selection analysis plan failed verification."""


def selection_execution_policy():
    """Use a plain-text selection protocol without the legacy MCQ suffix."""
    return replace(
        powered_execution_policy(),
        max_output_tokens=8_192,
        evidence_protocol=SELECTION_TEXT_PROTOCOL_V1,
        tool_catalog_bytes_bound=0,
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


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SelectionPoweredPlanError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionPoweredPlanError("plan input is not a JSON object")
    return value


def _semantic_valid(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == _sha256(payload))


def _selection_key(*parts: str) -> str:
    return hashlib.sha256(
        ("flavourbench-selection-repeat-20260811\0" + "\0".join(parts)).encode()
    ).hexdigest()


def _render_repeat_prompt(prefix: str, choices: Mapping[str, str]) -> str:
    rendered = "\n".join(f"{label}. {choices[label].replace('_', ' ')}" for label in LABELS)
    return (
        f"{prefix}\n\nCandidates:\n{rendered}\n\nChoose exactly {SELECTION_SIZE} distinct labels. "
        "Return exactly one line in alphabetical label order: `FINAL_SELECTION: A,C,F`."
    )


def _permuted_score_map(
    *,
    original_choices: Mapping[str, str],
    new_choices: Mapping[str, str],
    scores: Mapping[str, int],
) -> dict[str, int]:
    original_label = {ingredient: label for label, ingredient in original_choices.items()}
    output: dict[str, int] = {}
    for key in ALL_SELECTION_KEYS:
        ingredients = [new_choices[label] for label in key]
        original_key = "".join(sorted(original_label[value] for value in ingredients))
        output[key] = int(scores[original_key])
    return output


def build_repeat_panel(taskset: Mapping[str, Any]) -> dict[str, Any]:
    if not verify_taskset(taskset):
        raise SelectionPoweredPlanError("repeat panel requires a valid selection task set")
    repeats: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_tasks = [task for task in taskset["tasks"] if task["family"] == family]
        selected = sorted(
            family_tasks,
            key=lambda task: _selection_key(family, str(task["task_id"])),
        )[:REPEATS_PER_FAMILY]
        for index, task in enumerate(selected, start=1):
            shift = 1 + int(_selection_key("shift", str(task["task_id"])), 16) % 7
            original = dict(task["choices"])
            permuted = {
                label: original[LABELS[(position - shift) % len(LABELS)]]
                for position, label in enumerate(LABELS)
            }
            scores = _permuted_score_map(
                original_choices=original,
                new_choices=permuted,
                scores=task["selection_scores_bps"],
            )
            optimum = max(scores, key=scores.__getitem__)
            prefix = str(task["prompt"]).split("\n\nCandidates:\n", 1)[0]
            prompt = _render_repeat_prompt(prefix, permuted)
            repeats.append(
                {
                    "task_id": f"repeat-{family}-{index:02d}",
                    "original_task_id": task["task_id"],
                    "family": family,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "choices": permuted,
                    "selection_size": SELECTION_SIZE,
                    "selection_scores_bps": scores,
                    "optimal_selection": optimum,
                    "original_optimal_selection": task["optimal_selection"],
                    "permutation_shift": shift,
                    "oracle_reference_sha256": task["oracle_reference_sha256"],
                }
            )
    document: dict[str, Any] = {
        "schema_version": REPEAT_SCHEMA_VERSION,
        "status": "frozen_before_model_execution",
        "source_taskset_artifact_sha256": taskset["artifact_sha256"],
        "source_task_set_sha256": taskset["task_set_sha256"],
        "selection_seed": "flavourbench-selection-repeat-20260811",
        "counts": {
            "tasks": REPEAT_TASK_COUNT,
            "tasks_per_family": REPEATS_PER_FAMILY,
            "provider_calls": REPEAT_CALL_COUNT,
        },
        "purpose": {
            "primary": "ingredient-set agreement after answer-label permutation",
            "secondary": "score stability and label-position sensitivity",
            "excluded_from_primary_score": True,
        },
        "tasks": repeats,
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_repeat_panel(document, taskset=taskset):
        raise SelectionPoweredPlanError("repeat panel failed verification")
    return document


def verify_repeat_panel(
    document: Mapping[str, Any], *, taskset: Mapping[str, Any] | None = None
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
    if not all(
        task.get("permutation_shift") in set(range(1, 8))
        and set(task.get("choices") or {}) == set(LABELS)
        and set(task.get("selection_scores_bps") or {}) == set(ALL_SELECTION_KEYS)
        and sum(value == 10_000 for value in task["selection_scores_bps"].values()) == 1
        and task["selection_scores_bps"].get(task.get("optimal_selection")) == 10_000
        and hashlib.sha256(str(task.get("prompt") or "").encode()).hexdigest()
        == task.get("prompt_sha256")
        for task in tasks
    ):
        return False
    if taskset is None:
        return True
    if not verify_taskset(taskset):
        return False
    originals = {task["task_id"]: task for task in taskset["tasks"]}
    for repeat in tasks:
        original = originals[repeat["original_task_id"]]
        original_label = {value: label for label, value in original["choices"].items()}
        for key, score in repeat["selection_scores_bps"].items():
            ingredients = [repeat["choices"][label] for label in key]
            source_key = "".join(sorted(original_label[value] for value in ingredients))
            if score != original["selection_scores_bps"][source_key]:
                return False
    return True


def run_commitment(run_directory: Path, *, expected_responses: int) -> dict[str, Any]:
    paths = sorted(run_directory.glob("responses/primary/*/response-*.json"))
    artifacts: list[str] = []
    spend_micros = 0
    for path in paths:
        document = _load(path)
        if not _semantic_valid(document):
            raise SelectionPoweredPlanError("calibration response failed semantic verification")
        artifacts.append(str(document["artifact_sha256"]))
        spend_micros += int((document.get("generation") or {}).get("cost_micros") or 0)
    journal = run_directory / "attempts/provider-attempts.jsonl"
    if len(artifacts) != expected_responses or not journal.is_file() or journal.is_symlink():
        raise SelectionPoweredPlanError("calibration predecessor is incomplete")
    return {
        "response_count": len(artifacts),
        "response_artifact_set_sha256": _sha256(sorted(artifacts)),
        "attempt_journal_physical_sha256": _sha256_file(journal),
        "spend_micros": spend_micros,
        "used_as_primary_data": False,
    }


def normal_approximate_paired_power(
    *, tasks: int, mean_difference_points: float, paired_sd_points: float, comparisons: int
) -> float:
    alpha = 0.05 / comparisons
    critical = NormalDist().inv_cdf(1 - alpha / 2)
    noncentral = mean_difference_points / (paired_sd_points / tasks**0.5)
    normal = NormalDist()
    return (1 - normal.cdf(critical - noncentral)) + normal.cdf(-critical - noncentral)


def build_plan(
    *,
    manifest: Mapping[str, Any],
    manifest_physical_sha256: str,
    taskset: Mapping[str, Any],
    taskset_physical_sha256: str,
    repeat: Mapping[str, Any],
    repeat_physical_sha256: str,
    predecessor_release: Mapping[str, Any],
    predecessor_release_physical_sha256: str,
    calibration_v2: Mapping[str, Any],
    calibration_v3: Mapping[str, Any],
    calibration_v4: Mapping[str, Any],
    calibration_v5: Mapping[str, Any],
    calibration_v6: Mapping[str, Any],
    calibration_v7: Mapping[str, Any],
    calibration_v8: Mapping[str, Any],
    calibration_v9: Mapping[str, Any],
    calibration_v10: Mapping[str, Any],
    calibration_v11: Mapping[str, Any],
    calibration_v12: Mapping[str, Any],
    calibration_v13: Mapping[str, Any],
    calibration_v14: Mapping[str, Any],
    calibration_v15: Mapping[str, Any],
    calibration_v16: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_taskset(taskset) or not verify_repeat_panel(repeat, taskset=taskset):
        raise SelectionPoweredPlanError("invalid task or repeat input")
    if not _semantic_valid(predecessor_release):
        raise SelectionPoweredPlanError("predecessor release failed verification")
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT:
        raise SelectionPoweredPlanError("plan requires exactly 20 routes")
    pilot_ids: list[str] = []
    for family in FAMILIES:
        values = [task for task in taskset["tasks"] if task["family"] == family]
        median_chance = statistics.median(task["chance_score_bps"] for task in values)
        pilot_ids.append(
            min(
                values,
                key=lambda task: (
                    abs(task["chance_score_bps"] - median_chance),
                    _selection_key("pilot", family, task["task_id"]),
                ),
            )["task_id"]
        )
    calibration_spend = sum(
        int(calibration["spend_micros"])
        for calibration in (
            calibration_v2,
            calibration_v3,
            calibration_v4,
            calibration_v5,
            calibration_v6,
            calibration_v7,
            calibration_v8,
            calibration_v9,
            calibration_v10,
            calibration_v11,
            calibration_v12,
            calibration_v13,
            calibration_v14,
            calibration_v15,
            calibration_v16,
        )
    )
    power_5 = normal_approximate_paired_power(
        tasks=TASK_COUNT,
        mean_difference_points=5,
        paired_sd_points=20,
        comparisons=PAIRWISE_COMPARISONS,
    )
    power_3 = normal_approximate_paired_power(
        tasks=TASK_COUNT,
        mean_difference_points=3,
        paired_sd_points=20,
        comparisons=PAIRWISE_COMPARISONS,
    )
    policy = selection_execution_policy()
    document: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_version": PLAN_VERSION,
        "status": "preregistered_after_interface_calibration_before_primary_responses",
        "frozen_date": "2026-08-11",
        "inputs": {
            "route_manifest": {
                "semantic_sha256": manifest["content_address"]["digest"],
                "physical_sha256": manifest_physical_sha256,
            },
            "taskset": {
                "semantic_sha256": taskset["artifact_sha256"],
                "task_set_sha256": taskset["task_set_sha256"],
                "physical_sha256": taskset_physical_sha256,
            },
            "repeat_panel": {
                "semantic_sha256": repeat["artifact_sha256"],
                "physical_sha256": repeat_physical_sha256,
            },
            "development_predecessor": {
                "semantic_sha256": predecessor_release["artifact_sha256"],
                "physical_sha256": predecessor_release_physical_sha256,
            },
            "calibration_v2": dict(calibration_v2),
            "calibration_v3": dict(calibration_v3),
            "calibration_v4": {
                **dict(calibration_v4),
                "interface_finding": (
                    "portable MCQ suffix overrode the three-selection response contract"
                ),
                "superseded_protocol": "portable_text_tool_v1",
            },
            "calibration_v5": {
                **dict(calibration_v5),
                "interface_finding": (
                    "generic Markdown final turn permitted avoidable length terminations"
                ),
                "superseded_protocol": "legacy_v6",
            },
            "calibration_v6": {
                **dict(calibration_v6),
                "interface_finding": (
                    "dedicated selection protocol succeeded outside one Cohere schema keyword"
                ),
                "cohere_rejected_before_generation": True,
                "successor_change": ("explicit 56-value Cohere enum"),
            },
            "calibration_v7": {
                **dict(calibration_v7),
                "interface_finding": (
                    "reasoning field support did not imply acceptance of effort none"
                ),
                "superseded_request_rule": "route-adaptive reasoning effort none",
            },
            "calibration_v8": {
                **dict(calibration_v8),
                "interface_finding": (
                    "the explicit 56-value Cohere enum completed and parsed all four live "
                    "family-stratified pilot cells"
                ),
                "successor_change": (
                    "raise the final-token ceiling from 2048 to 4096 to reduce avoidable "
                    "length terminations on long-reasoning routes"
                ),
            },
            "calibration_v9": {
                **dict(calibration_v9),
                "interface_finding": (
                    "the 4096-token selection contract removed every observed Qwen and "
                    "DeepSeek length termination; Claude Fable retained one route-level "
                    "content-filter refusal"
                ),
                "successor_change": (
                    "replace only Claude Fable's Amazon Bedrock route with the exact healthy "
                    "Azure endpoint while preserving all other routes"
                ),
            },
            "calibration_v10": {
                **dict(calibration_v10),
                "interface_finding": (
                    "Claude Fable reproduced the same single cultural-item refusal on Azure, "
                    "showing a prompt-model interaction rather than an endpoint outage"
                ),
                "successor_change": (
                    "replace normative cultural-coherence wording on all cultural items with "
                    "a neutral target-cuisine-label instruction; score maps remain unchanged"
                ),
            },
            "calibration_v11": {
                **dict(calibration_v11),
                "interface_finding": (
                    "the neutral cultural prompt produced 79 of 80 completed parseable pilot "
                    "cells; the sole failure was DeepSeek Flash exhausting 4096 tokens"
                ),
                "successor_change": (
                    "raise the final-token ceiling to the preregistered runner maximum of 8192; "
                    "the response parser and scoring rule remain unchanged"
                ),
            },
            "calibration_v12": {
                **dict(calibration_v12),
                "interface_finding": (
                    "four concurrent lanes per model caused construct-irrelevant transport "
                    "failures, concentrated on direct Cohere and Kimi routes"
                ),
                "successor_change": (
                    "freeze one in-flight request per model and twenty globally for the primary "
                    "collection; every captured v12 response remains calibration-only"
                ),
                "interrupted_after_detecting_transport_noise": True,
            },
            "calibration_v13": {
                **dict(calibration_v13),
                "interface_finding": (
                    "one request per model removed connection pressure, but the two Cohere "
                    "models still exceeded their shared direct-account per-minute limit"
                ),
                "successor_change": (
                    "pace all direct Cohere request starts 6.5 seconds apart across both models; "
                    "remove the residual cultural-composition display label"
                ),
                "interrupted_after_detecting_rate_limit": True,
            },
            "calibration_v14": {
                **dict(calibration_v14),
                "interface_finding": (
                    "Claude Fable refused four of eight focused cells after both endpoint and "
                    "prompt repairs, including ordinary pairing items"
                ),
                "successor_change": (
                    "replace the refusal-prone Fable slot with the exact DigitalOcean route for "
                    "Meta Llama 4 Maverick before collecting any primary response"
                ),
                "captured_responses_remain_calibration_only": True,
            },
            "calibration_v15": {
                **dict(calibration_v15),
                "interface_findings": [
                    "Google Vertex Flex returned two HTTP-200 responses with finish_reason error",
                    "the pinned Together Nemotron route became unhealthy and failed two cells",
                    "three models returned valid three-label sets in non-alphabetical order",
                ],
                "successor_changes": [
                    "pin Gemini 3.6 Flash to Google Vertex US",
                    "pin Nemotron 3 Ultra to BaseTen FP4",
                    "normalize three distinct answer labels as an unordered set before lookup",
                ],
                "interrupted_after_decisive_findings": True,
            },
            "calibration_v16": {
                **dict(calibration_v16),
                "interface_finding": (
                    "all repaired routes completed, but one of four Qwen cells exhausted the "
                    "8192-token cap on a one-line selection task"
                ),
                "successor_change": (
                    "freeze minimal hidden reasoning for Qwen; permit four concurrent requests "
                    "per OpenRouter model while retaining one per direct Kimi or Cohere model"
                ),
                "captured_responses_remain_calibration_only": True,
            },
        },
        "roster": {
            "model_count": MODEL_COUNT,
            "fallbacks": "disabled",
            "models": [
                {
                    "slot_id": value.slot_id,
                    "model_id": value.model_id,
                    "model_name": value.model_name,
                    "canonical_model_slug": value.canonical_model_slug,
                    "execution_backend": value.execution_backend,
                    "provider_tag": value.provider_tag,
                    "provider_name": value.provider_name,
                    "endpoint_sha256": value.endpoint_sha256,
                    "endpoint_execution_sha256": value.endpoint_execution_sha256,
                    "backend_contract_sha256": value.backend_contract_sha256,
                    "final_reasoning_effort": MODEL_FINAL_REASONING_EFFORT.get(
                        value.model_id, "provider_fixed"
                    ),
                }
                for value in candidates
            ],
        },
        "design": {
            "primary_tasks": TASK_COUNT,
            "families": list(FAMILIES),
            "tasks_per_family": 160,
            "choices_per_task": 8,
            "selection_size": 3,
            "prefrozen_scored_selections_per_task": 56,
            "primary_provider_calls": PRIMARY_CALL_COUNT,
            "repeat_tasks": REPEAT_TASK_COUNT,
            "repeat_provider_calls": REPEAT_CALL_COUNT,
            "total_provider_calls": TOTAL_CALL_COUNT,
            "posthoc_item_exclusion": False,
            "result_adaptive_sampling": False,
        },
        "execution": {
            "condition": "model_only_with_epicure_frozen_as_the_scoring_environment",
            "response_contract": "FINAL_SELECTION with three distinct A-H labels",
            "response_parser": {
                **selection_parser_contract(),
                "sha256": selection_parser_sha256(),
            },
            "reasoning_control": (
                "minimal hidden reasoning for Qwen; provider-fixed for other OpenRouter and "
                "Kimi routes; Cohere selection adapters use their frozen bounded modes"
            ),
            "execution_policy": policy.document(),
            "execution_policy_sha256": policy.sha256,
            "epicure_release_id": taskset["epicure_provenance"]["release_id"],
            "epicure_bundle_sha256": taskset["epicure_provenance"]["bundle_sha256"],
            "epicure_application_sha256": taskset["epicure_provenance"]["application_sha256"],
            "epicure_tool_schema_sha256": EPICURE_TOOL_SCHEMA_SHA256,
            "pilot": {
                "task_ids": pilot_ids,
                "cells": MODEL_COUNT * len(pilot_ids),
                "one_task_per_family_per_model": True,
                "responses_reused_in_primary": True,
            },
            "collection_concurrency": {
                "global": 24,
                "per_model_default": 4,
                "per_model_by_backend": {
                    "openrouter": 4,
                    "kimi_direct": 1,
                    "cohere_direct": 1,
                },
                "reason": (
                    "parallelize independent OpenRouter cells while retaining the stable "
                    "single-flight contracts for direct Kimi and Cohere"
                ),
            },
            "minimum_request_interval_seconds_by_backend": {
                "cohere_direct": 6.5,
            },
        },
        "outcomes": {
            "primary_name": "FlavourBench Score",
            "primary_definition": "equal-family macro mean Epicure selection score from 0 to 100",
            "task_score": "prefrozen continuous lookup over all three-of-eight selections",
            "chance": "exact task-specific mean across all 56 possible selections",
            "failed_unparseable_or_invalid": "zero in intention-to-evaluate denominator",
            "availability_reported_separately": True,
            "epicure_is_judge_not_a_competing_model": True,
        },
        "eligibility": {
            "minimum_completed_fraction": 0.95,
            "minimum_completed_tasks": 608,
            "below_threshold": "DNF, not worst capability",
        },
        "inference": {
            "familywise_alpha": 0.05,
            "score_intervals": "family-stratified task bootstrap with simultaneous max-t bands",
            "paired_tests": (
                "all 190 two-sided paired sign-flip permutation tests with Holm correction"
            ),
            "effect_sizes": (
                "paired mean difference, bootstrap interval, and paired standardized effect"
            ),
            "chance_tests": (
                "paired taskwise difference from exact chance baseline with Holm correction"
            ),
            "rank_display": "statistical rank groups; no forced ordering inside unresolved groups",
            "definitive_top": (
                "positive paired mean difference and Holm p<0.05 versus every other eligible "
                "model, plus repeat set agreement at least 0.80"
            ),
            "bootstrap_resamples": 50_000,
            "permutation_resamples": 100_000,
            "seed": 20260811,
            "no_result_dependent_test_selection": True,
        },
        "power": {
            "method": "preregistered normal approximation for paired continuous task scores",
            "tasks": TASK_COUNT,
            "assumed_paired_sd_points": 20,
            "familywise_comparisons": PAIRWISE_COMPARISONS,
            "five_point_difference_power": round(power_5, 12),
            "three_point_difference_power": round(power_3, 12),
            "target": 0.80,
            "primary_target_difference_points": 5,
            "primary_target_meets_power": power_5 >= 0.80,
            "simulation_required_for_final_observed_precision": True,
        },
        "repeatability": {
            "tasks_per_model": REPEAT_TASK_COUNT,
            "answer_labels_permuted": True,
            "primary": "selected-ingredient-set Jaccard agreement",
            "secondary": "absolute task-score difference",
            "acceptance_floor": 0.80,
            "excluded_from_primary_score": True,
        },
        "budget": {
            "currency": "USD",
            "program_cap": PROGRAM_BUDGET_CAP_USD,
            "calibration_spend_micros": calibration_spend,
            "hard_cap": SUCCESSOR_RUN_CAP_USD,
            "pilot_extrapolated_mean_usd": "33.2",
            "two_x_pilot_contingency_usd": "66.4",
            "admission": "actual spend plus concurrent price reservations must remain within cap",
        },
        "claim_boundary": (
            "finite-panel Epicure alignment and executable culinary decision quality under the "
            "frozen task distribution; not universal language-model quality"
        ),
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise SelectionPoweredPlanError("constructed plan failed verification")
    return document


def verify_plan(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    try:
        roster = document["roster"]["models"]
        design = document["design"]
        power = document["power"]
    except (KeyError, TypeError):
        return False
    return bool(
        document.get("schema_version") == PLAN_SCHEMA_VERSION
        and recorded == _sha256(payload)
        and len(roster) == MODEL_COUNT
        and len({row["model_id"] for row in roster}) == MODEL_COUNT
        and design["primary_tasks"] == TASK_COUNT
        and design["total_provider_calls"] == TOTAL_CALL_COUNT
        and power["primary_target_meets_power"] is True
        and float(power["five_point_difference_power"]) >= 0.80
    )


def _write(document: Mapping[str, Any], directory: Path, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise SelectionPoweredPlanError("content-addressed artifact conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as handle:
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
    parser.add_argument("--calibration-v2-directory", type=Path, required=True)
    parser.add_argument("--calibration-v3-directory", type=Path, required=True)
    parser.add_argument("--calibration-v4-directory", type=Path, required=True)
    parser.add_argument("--calibration-v5-directory", type=Path, required=True)
    parser.add_argument("--calibration-v6-directory", type=Path, required=True)
    parser.add_argument("--calibration-v7-directory", type=Path, required=True)
    parser.add_argument("--calibration-v8-directory", type=Path, required=True)
    parser.add_argument("--calibration-v9-directory", type=Path, required=True)
    parser.add_argument("--calibration-v10-directory", type=Path, required=True)
    parser.add_argument("--calibration-v11-directory", type=Path, required=True)
    parser.add_argument("--calibration-v12-directory", type=Path, required=True)
    parser.add_argument("--calibration-v13-directory", type=Path, required=True)
    parser.add_argument("--calibration-v14-directory", type=Path, required=True)
    parser.add_argument("--calibration-v15-directory", type=Path, required=True)
    parser.add_argument("--calibration-v16-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    taskset = _load(args.taskset)
    predecessor = _load(args.predecessor_release)
    repeat = build_repeat_panel(taskset)
    repeat_path = _write(repeat, args.output_directory, "epicure-selection-repeat-panel")
    plan = build_plan(
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        taskset=taskset,
        taskset_physical_sha256=_sha256_file(args.taskset),
        repeat=repeat,
        repeat_physical_sha256=_sha256_file(repeat_path),
        predecessor_release=predecessor,
        predecessor_release_physical_sha256=_sha256_file(args.predecessor_release),
        calibration_v2=run_commitment(args.calibration_v2_directory, expected_responses=20),
        calibration_v3=run_commitment(args.calibration_v3_directory, expected_responses=80),
        calibration_v4=run_commitment(args.calibration_v4_directory, expected_responses=80),
        calibration_v5=run_commitment(args.calibration_v5_directory, expected_responses=80),
        calibration_v6=run_commitment(args.calibration_v6_directory, expected_responses=80),
        calibration_v7=run_commitment(args.calibration_v7_directory, expected_responses=80),
        calibration_v8=run_commitment(args.calibration_v8_directory, expected_responses=4),
        calibration_v9=run_commitment(args.calibration_v9_directory, expected_responses=20),
        calibration_v10=run_commitment(args.calibration_v10_directory, expected_responses=4),
        calibration_v11=run_commitment(args.calibration_v11_directory, expected_responses=80),
        calibration_v12=run_commitment(args.calibration_v12_directory, expected_responses=207),
        calibration_v13=run_commitment(args.calibration_v13_directory, expected_responses=136),
        calibration_v14=run_commitment(args.calibration_v14_directory, expected_responses=8),
        calibration_v15=run_commitment(args.calibration_v15_directory, expected_responses=103),
        calibration_v16=run_commitment(args.calibration_v16_directory, expected_responses=80),
    )
    print(_write(plan, args.output_directory, "epicure-selection-analysis-plan"))


if __name__ == "__main__":
    run()
