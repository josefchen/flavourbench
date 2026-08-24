"""Freeze the construct-validated powered FlavourBench analysis plan."""

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

from .epicure_native_powered_plan import (
    EPICURE_TOOL_SCHEMA_SHA256,
    exact_mcnemar_power,
    powered_execution_policy,
)
from .epicure_native_taskset_v3 import (
    CHOICE_LABELS,
    FAMILIES,
    TASK_COUNT,
    verify_taskset,
)
from .frontier_contract_runner import load_candidate_manifest, select_candidates

PLAN_SCHEMA_VERSION = "flavourbench-powered-analysis-plan-v2"
REPEAT_SCHEMA_VERSION = "flavourbench-powered-repeat-panel-v2"
PLAN_VERSION = "flavourbench-construct-validated-20x640-v2"
MODEL_COUNT = 20
REPEAT_TASK_COUNT = 64
REPEATS_PER_FAMILY = 16
PRIMARY_CALL_COUNT = MODEL_COUNT * TASK_COUNT
REPEAT_CALL_COUNT = MODEL_COUNT * REPEAT_TASK_COUNT
TOTAL_CALL_COUNT = PRIMARY_CALL_COUNT + REPEAT_CALL_COUNT
PAIRWISE_COMPARISONS = MODEL_COUNT * (MODEL_COUNT - 1) // 2


class PoweredPlanV2Error(RuntimeError):
    """The construct-validated plan failed verification."""


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
        raise PoweredPlanV2Error(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PoweredPlanV2Error("plan input is not a JSON object")
    return value


def _valid_semantic(document: Mapping[str, Any]) -> bool:
    payload = dict(document)
    recorded = str(payload.pop("artifact_sha256", ""))
    return bool(recorded and recorded == _sha256(payload))


def _selection_key(*parts: str) -> str:
    return hashlib.sha256(
        ("flavourbench-validated-repeat-panel-20260811\0" + "\0".join(parts)).encode()
    ).hexdigest()


def _render_repeat_prompt(prefix: str, choices: Mapping[str, str]) -> str:
    rendered = "\n".join(f"{label}. {choices[label].replace('_', ' ')}" for label in CHOICE_LABELS)
    return (
        f"{prefix}\n\nChoices:\n{rendered}\n\n"
        "Return exactly one line: `FINAL_CHOICE: X`, replacing X with A, B, C, or D."
    )


def build_repeat_panel(taskset: Mapping[str, Any]) -> dict[str, Any]:
    if not verify_taskset(taskset):
        raise PoweredPlanV2Error("repeat panel requires a valid successor taskset")
    repeats: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_tasks = [task for task in taskset["tasks"] if task["family"] == family]
        selected = sorted(
            family_tasks,
            key=lambda task: _selection_key(family, str(task["task_id"])),
        )[:REPEATS_PER_FAMILY]
        for index, task in enumerate(selected, start=1):
            shift = 1 + int(_selection_key("shift", str(task["task_id"])), 16) % 3
            original = dict(task["choices"])
            permuted = {
                label: original[CHOICE_LABELS[(position - shift) % 4]]
                for position, label in enumerate(CHOICE_LABELS)
            }
            expected_value = original[task["expected_choice"]]
            expected = next(label for label, value in permuted.items() if value == expected_value)
            prefix = str(task["prompt"]).split("\n\nChoices:\n", 1)[0]
            prompt = _render_repeat_prompt(prefix, permuted)
            repeats.append(
                {
                    "task_id": f"repeat-{family}-{index:02d}",
                    "original_task_id": task["task_id"],
                    "family": family,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "choices": permuted,
                    "expected_choice": expected,
                    "original_expected_choice": task["expected_choice"],
                    "permutation_shift": shift,
                    "oracle_reference_sha256": task["oracle_reference_sha256"],
                }
            )
    document: dict[str, Any] = {
        "schema_version": REPEAT_SCHEMA_VERSION,
        "status": "frozen_before_successor_model_execution",
        "source_taskset_artifact_sha256": taskset["artifact_sha256"],
        "source_task_set_sha256": taskset["task_set_sha256"],
        "selection_seed": "flavourbench-validated-repeat-panel-20260811",
        "counts": {
            "tasks": REPEAT_TASK_COUNT,
            "tasks_per_family": REPEATS_PER_FAMILY,
            "provider_calls": REPEAT_CALL_COUNT,
        },
        "purpose": {
            "primary": "choice-content agreement after answer-position permutation",
            "secondary": "position sensitivity and repeated accuracy",
            "excluded_from_primary_score": True,
        },
        "tasks": repeats,
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_repeat_panel(document, taskset=taskset):
        raise PoweredPlanV2Error("repeat panel failed verification")
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
        task.get("permutation_shift") in {1, 2, 3}
        and set(task.get("choices") or {}) == set(CHOICE_LABELS)
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
    return all(
        originals[repeat["original_task_id"]]["choices"][
            originals[repeat["original_task_id"]]["expected_choice"]
        ]
        == repeat["choices"][repeat["expected_choice"]]
        for repeat in tasks
    )


def calibration_commitment(run_directory: Path) -> dict[str, Any]:
    response_paths = sorted(run_directory.glob("responses/primary/*/response-*.json"))
    artifacts: list[str] = []
    for path in response_paths:
        document = _load(path)
        if not _valid_semantic(document):
            raise PoweredPlanV2Error("calibration response failed semantic verification")
        artifacts.append(str(document["artifact_sha256"]))
    journal = run_directory / "attempts/provider-attempts.jsonl"
    if len(artifacts) != MODEL_COUNT or not journal.is_file() or journal.is_symlink():
        raise PoweredPlanV2Error("calibration predecessor is incomplete")
    return {
        "status": "superseded_construct_validity_calibration",
        "response_count": len(artifacts),
        "response_artifact_set_sha256": _sha256(sorted(artifacts)),
        "attempt_journal_physical_sha256": _sha256_file(journal),
        "used_as_primary_data": False,
    }


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
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    if not verify_taskset(taskset) or not verify_repeat_panel(repeat_panel, taskset=taskset):
        raise PoweredPlanV2Error("invalid plan inputs")
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT:
        raise PoweredPlanV2Error("plan requires exactly 20 routes")
    manifest_sha256 = str(manifest["content_address"]["digest"])
    if not _valid_semantic(predecessor_release):
        raise PoweredPlanV2Error("predecessor release failed verification")
    pilots = [
        min(
            (str(task["task_id"]) for task in taskset["tasks"] if task["family"] == family),
            key=lambda task_id: _selection_key("pilot", family, task_id),
        )
        for family in FAMILIES
    ]
    policy = powered_execution_policy()
    power_all = exact_mcnemar_power(
        pairs=TASK_COUNT,
        discordance_probability=0.30,
        absolute_accuracy_difference=0.10,
        familywise_alpha=0.05,
        comparisons=PAIRWISE_COMPARISONS,
    )
    power_leader = exact_mcnemar_power(
        pairs=TASK_COUNT,
        discordance_probability=0.30,
        absolute_accuracy_difference=0.10,
        familywise_alpha=0.05,
        comparisons=MODEL_COUNT - 1,
    )
    document: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_version": PLAN_VERSION,
        "status": "preregistered_before_construct_validated_provider_responses",
        "frozen_date": "2026-08-11",
        "inputs": {
            "route_manifest": {
                "semantic_sha256": manifest_sha256,
                "physical_sha256": manifest_physical_sha256,
            },
            "taskset": {
                "semantic_sha256": taskset["artifact_sha256"],
                "task_set_sha256": taskset["task_set_sha256"],
                "physical_sha256": taskset_physical_sha256,
            },
            "repeat_panel": {
                "semantic_sha256": repeat_panel["artifact_sha256"],
                "physical_sha256": repeat_panel_physical_sha256,
            },
            "predecessor_release": {
                "semantic_sha256": predecessor_release["artifact_sha256"],
                "physical_sha256": predecessor_release_physical_sha256,
                "role": "development_predecessor_only",
            },
            "calibration_predecessor": dict(calibration),
        },
        "roster": {
            "model_count": MODEL_COUNT,
            "fallbacks": "disabled",
            "models": [
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
                }
                for candidate in candidates
            ],
        },
        "design": {
            "primary_tasks": TASK_COUNT,
            "families": list(FAMILIES),
            "tasks_per_family": 160,
            "validation_strata_per_family": 4,
            "tasks_per_stratum_per_family": 40,
            "unique_anchor_ingredients": TASK_COUNT,
            "primary_provider_calls": PRIMARY_CALL_COUNT,
            "repeat_tasks": REPEAT_TASK_COUNT,
            "repeat_provider_calls": REPEAT_CALL_COUNT,
            "total_provider_calls": TOTAL_CALL_COUNT,
            "posthoc_item_exclusion": False,
            "result_adaptive_sampling": False,
        },
        "construct_validity": dict(taskset["construct_validity_contract"]),
        "execution": {
            "condition": "model_only_no_epicure_runtime_access",
            "execution_policy": policy.document(),
            "execution_policy_sha256": policy.sha256,
            "epicure_release_id": taskset["epicure_provenance"]["release_id"],
            "epicure_bundle_sha256": taskset["epicure_provenance"]["bundle_sha256"],
            "epicure_application_sha256": taskset["epicure_provenance"]["application_sha256"],
            "epicure_tool_schema_sha256": EPICURE_TOOL_SCHEMA_SHA256,
            "pilot": {
                "task_ids": pilots,
                "cells": MODEL_COUNT * len(pilots),
                "one_task_per_family_per_model": True,
                "responses_reused_in_primary": True,
            },
            "schedule": (
                "four-family pilot across all models, deterministic per-model primary order, "
                "then frozen repeats"
            ),
        },
        "outcomes": {
            "primary_name": "FlavourBench Score",
            "primary_definition": "100 times equal-family macro exact accuracy",
            "chance_score": 25.0,
            "missing_unparseable_or_failed": "zero in intention-to-evaluate denominator",
            "availability_reported_separately": True,
            "assisted_diagnostic_excluded": True,
        },
        "eligibility": {
            "minimum_completed_fraction": 0.95,
            "minimum_completed_tasks": 608,
            "below_threshold": "DNF, not worst capability",
        },
        "inference": {
            "familywise_alpha": 0.05,
            "score_intervals": "Bonferroni-simultaneous Wilson over 20 models",
            "family_intervals": "Bonferroni-simultaneous Wilson over 80 model-family cells",
            "chance_tests": "one-sided exact binomial with Holm correction across 20 models",
            "paired_tests": "all 190 two-sided exact McNemar tests with Holm correction",
            "rank_display": "statistical rank groups, not forced total order",
            "definitive_top": (
                "positive paired difference and Holm p<0.05 versus every other eligible model, "
                "plus repeatability at least 0.80"
            ),
            "no_result_dependent_test_selection": True,
        },
        "power": {
            "method": "exact unconditional two-sided conditional McNemar",
            "pairs": TASK_COUNT,
            "assumed_discordance": 0.30,
            "target_accuracy_difference": 0.10,
            "bonferroni_all_190": round(power_all, 12),
            "bonferroni_leader_19": round(power_leader, 12),
            "target": 0.80,
            "all_190_meets_target": power_all >= 0.80,
        },
        "repeatability": {
            "tasks_per_model": REPEAT_TASK_COUNT,
            "choice_positions_permuted": True,
            "primary": "selected choice-content agreement",
            "acceptance_floor": 0.80,
            "excluded_from_primary_score": True,
        },
        "budget": {
            "currency": "USD",
            "hard_cap": "85",
            "high_envelope": "79.8",
            "admission": "actual spend plus in-flight price envelope must remain within cap",
        },
        "claim_boundary": (
            "finite-panel exact culinary reasoning under the frozen task distribution; "
            "not universal language-model quality"
        ),
    }
    document["artifact_sha256"] = _sha256(document)
    if not verify_plan(document):
        raise PoweredPlanV2Error("constructed plan failed verification")
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
        and power["all_190_meets_target"] is True
        and float(power["bonferroni_all_190"]) >= 0.80
    )


def _write(document: Mapping[str, Any], directory: Path, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{prefix}-{document['artifact_sha256']}.json"
    data = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != data:
            raise PoweredPlanV2Error("content-addressed artifact conflict")
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
    parser.add_argument("--calibration-run-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = load_candidate_manifest(args.manifest, expected_digest=args.manifest_semantic_sha256)
    taskset = _load(args.taskset)
    predecessor = _load(args.predecessor_release)
    repeat = build_repeat_panel(taskset)
    repeat_path = _write(repeat, args.output_directory, "epicure-native-validated-repeat-panel")
    plan = build_plan(
        manifest=manifest,
        manifest_physical_sha256=_sha256_file(args.manifest),
        taskset=taskset,
        taskset_physical_sha256=_sha256_file(args.taskset),
        repeat_panel=repeat,
        repeat_panel_physical_sha256=_sha256_file(repeat_path),
        predecessor_release=predecessor,
        predecessor_release_physical_sha256=_sha256_file(args.predecessor_release),
        calibration=calibration_commitment(args.calibration_run_directory),
    )
    print(_write(plan, args.output_directory, "epicure-native-validated-analysis-plan"))


if __name__ == "__main__":
    run()
