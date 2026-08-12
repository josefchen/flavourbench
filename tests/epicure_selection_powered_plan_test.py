from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_selection_powered_plan import (
    build_repeat_panel,
    verify_plan,
    verify_repeat_panel,
)
from flavourbench.epicure_selection_taskset_v1 import verify_taskset

ROOT = Path(__file__).resolve().parents[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v17/plan").glob("*analysis-plan*.json"))


def test_materialized_plan_and_repeat_panel() -> None:
    taskset = json.loads(TASKSET.read_text())
    repeat = json.loads(REPEAT.read_text())
    plan = json.loads(PLAN.read_text())
    assert verify_taskset(taskset)
    assert verify_repeat_panel(repeat, taskset=taskset)
    assert verify_plan(plan)
    assert plan["execution"]["pilot"]["cells"] == 80
    assert plan["design"]["total_provider_calls"] == 14_080
    assert plan["power"]["five_point_difference_power"] > 0.99
    assert plan["execution"]["execution_policy"]["evidence_protocol"] == "selection_text_v1"
    assert plan["execution"]["execution_policy"]["limits"]["max_output_tokens"] == 8_192


def test_repeat_panel_is_deterministic_and_score_invariant() -> None:
    taskset = json.loads(TASKSET.read_text())
    repeat = build_repeat_panel(taskset)
    assert repeat == build_repeat_panel(taskset)
    assert len(repeat["tasks"]) == 64
    assert all(task["permutation_shift"] in set(range(1, 8)) for task in repeat["tasks"])


def test_calibration_spend_is_bound_but_not_reused() -> None:
    plan = json.loads(PLAN.read_text())
    assert plan["inputs"]["calibration_v2"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v3"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v4"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v5"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v6"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v7"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v8"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v9"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v10"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v11"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v12"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v13"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v14"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v15"]["used_as_primary_data"] is False
    assert plan["inputs"]["calibration_v16"]["used_as_primary_data"] is False
    assert plan["execution"]["collection_concurrency"] == {
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
    }
    assert plan["execution"]["minimum_request_interval_seconds_by_backend"] == {
        "cohere_direct": 6.5
    }
    assert plan["budget"]["calibration_spend_micros"] == 3_818_631
