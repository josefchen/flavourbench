from __future__ import annotations

from pathlib import Path

from flavourbench.epicure_native_powered_runner import (
    _task_reference_payload,
    build_generation_spec,
)
from flavourbench.epicure_selection_powered_plan import selection_execution_policy
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.provider import system_prompt_text

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v16/manifest").glob("*.json"))
MANIFEST_SHA = MANIFEST.stem.rsplit("-", 1)[1]
TASKSET = next((ROOT / "benchmark/powered-v16/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v17/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v17/plan").glob("*analysis-plan*.json"))
PREDECESSOR = ROOT / "paper/generated/epicure-native/epicure-native-release.json"


def _validated():
    return validate_inputs(
        manifest_path=MANIFEST,
        manifest_sha256=MANIFEST_SHA,
        taskset_path=TASKSET,
        repeat_panel_path=REPEAT,
        plan_path=PLAN,
        predecessor_release_path=PREDECESSOR,
    )


def test_exact_selection_inputs_validate() -> None:
    manifest, taskset, repeat, plan, predecessor, candidates = _validated()
    assert manifest["content_address"]["digest"] == MANIFEST_SHA
    assert taskset["artifact_sha256"] == plan["inputs"]["taskset"]["semantic_sha256"]
    assert repeat["artifact_sha256"] == plan["inputs"]["repeat_panel"]["semantic_sha256"]
    assert (
        predecessor["artifact_sha256"]
        == plan["inputs"]["development_predecessor"]["semantic_sha256"]
    )
    assert len(candidates) == 20


def test_schedule_counts_and_pilot_reuse() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    schedules = {
        phase: build_cells(
            plan=plan,
            taskset=taskset,
            repeat_panel=repeat,
            candidates=candidates,
            phase=phase,
        )
        for phase in ("pilot", "primary", "repeat", "all")
    }
    assert len(schedules["pilot"]) == 80
    assert len(schedules["primary"]) == 12_800
    assert len(schedules["repeat"]) == 1_280
    assert len(schedules["all"]) == 14_080
    assert {cell.cell_id for cell in schedules["pilot"]} <= {
        cell.cell_id for cell in schedules["primary"]
    }
    assert len({cell.cell_id for cell in schedules["all"]}) == 14_080


def test_selection_task_serializes_its_scoring_reference() -> None:
    _, taskset, _, _, _, _ = _validated()
    task = taskset["tasks"][0]
    assert "expected_choice" not in task
    assert _task_reference_payload(task) == {"optimal_selection": task["optimal_selection"]}


def test_selection_generation_contract_has_no_mcq_override() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    cell = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase="pilot",
    )[0]
    policy = selection_execution_policy()
    spec, _ = build_generation_spec(
        cell=cell,
        plan=plan,
        manifest_sha256=MANIFEST_SHA,
        taskset=taskset,
        reserve_micros=1000,
        execution_policy=policy,
    )
    assert spec.evidence_protocol == "selection_text_v1"
    assert spec.final_reasoning_effort is None
    assert plan["execution"]["execution_policy_sha256"] == policy.sha256
    assert "FINAL_SELECTION" in spec.prompt
    assert "FINAL_CHOICE" not in system_prompt_text(
        spec.condition,
        spec.final_response_mode,
        spec.evidence_protocol,
    )


def test_qwen_uses_its_frozen_minimal_reasoning_override() -> None:
    _, taskset, repeat, plan, _, candidates = _validated()
    qwen = next(value for value in candidates if value.model_id == "qwen/qwen3.8-max")
    cell = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=[qwen],
        phase="pilot",
    )[0]
    spec, bundle = build_generation_spec(
        cell=cell,
        plan=plan,
        manifest_sha256=MANIFEST_SHA,
        taskset=taskset,
        reserve_micros=100_000,
        execution_policy=selection_execution_policy(),
    )
    assert spec.final_reasoning_effort == "minimal"
    assert bundle["run_binding"]["final_reasoning_effort"] == "minimal"
