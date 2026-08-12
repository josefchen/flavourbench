from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from flavourbench.epicure_native_powered_plan import (
    MODEL_COUNT,
    REPEAT_TASK_COUNT,
    TOTAL_CALL_COUNT,
    build_plan,
    build_repeat_panel,
    exact_mcnemar_power,
    verify_plan,
    verify_repeat_panel,
)

ROOT = Path(__file__).resolve().parents[1]
TASKSET_PATH = next((ROOT / "benchmark/powered-v2/taskset").glob("*.json"))
MANIFEST_PATH = Path(
    ROOT / "benchmark/powered-v2/manifest/"
    "flavourbench-frontier-refresh-20-"
    "44220f0a6c26798871f830f6ddd62ed99d8872ad5dcb7516491b90bfc71887fc.json"
)
PREDECESSOR_PATH = ROOT / "paper/generated/epicure-native/epicure-native-release.json"


def _physical(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_exact_power_clears_frozen_all_pair_target() -> None:
    power = exact_mcnemar_power(
        pairs=640,
        discordance_probability=0.30,
        absolute_accuracy_difference=0.10,
        familywise_alpha=0.05,
        comparisons=190,
    )
    assert 0.83 < power < 0.84
    assert (
        exact_mcnemar_power(
            pairs=320,
            discordance_probability=0.30,
            absolute_accuracy_difference=0.10,
            familywise_alpha=0.05,
            comparisons=190,
        )
        < 0.40
    )


def test_repeat_panel_is_balanced_permuted_and_content_stable() -> None:
    taskset = _json(TASKSET_PATH)
    first = build_repeat_panel(taskset)
    second = build_repeat_panel(taskset)
    assert first == second
    assert verify_repeat_panel(first, source_taskset=taskset)
    assert len(first["tasks"]) == REPEAT_TASK_COUNT
    assert all(task["permutation_shift"] in {1, 2, 3} for task in first["tasks"])
    assert all(
        task["expected_choice"] != task["original_expected_choice"]
        or task["permutation_shift"] != 0
        for task in first["tasks"]
    )
    altered = deepcopy(first)
    altered["tasks"][0]["expected_choice"] = "Z"
    assert not verify_repeat_panel(altered, source_taskset=taskset)


def test_plan_binds_exact_inputs_roster_and_call_count() -> None:
    from flavourbench.frontier_contract_runner import load_candidate_manifest

    manifest = load_candidate_manifest(
        MANIFEST_PATH,
        expected_digest="44220f0a6c26798871f830f6ddd62ed99d8872ad5dcb7516491b90bfc71887fc",
    )
    taskset = _json(TASKSET_PATH)
    repeat = build_repeat_panel(taskset)
    plan = build_plan(
        manifest=manifest,
        manifest_physical_sha256=_physical(MANIFEST_PATH),
        taskset=taskset,
        taskset_physical_sha256=_physical(TASKSET_PATH),
        repeat_panel=repeat,
        repeat_panel_physical_sha256="a" * 64,
        predecessor_release=_json(PREDECESSOR_PATH),
        predecessor_release_physical_sha256=_physical(PREDECESSOR_PATH),
    )
    assert verify_plan(plan)
    assert plan["design"]["total_provider_calls"] == TOTAL_CALL_COUNT
    assert len(plan["roster"]["models"]) == MODEL_COUNT
    assert {row["model_id"] for row in plan["roster"]["models"]} >= {
        "qwen/qwen3.8-max",
        "moonshotai/kimi-k3",
        "cohere/command-a-plus-05-2026",
        "cohere/command-a-reasoning-08-2025",
    }
    assert plan["power"]["all_190_at_10pp_meets_target"] is True
    altered = deepcopy(plan)
    altered["inference"]["familywise_alpha"] = 0.10
    assert not verify_plan(altered)
