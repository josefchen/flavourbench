from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

from flavourbench.epicure_native_powered_plan import verify_plan, verify_repeat_panel
from flavourbench.epicure_native_powered_runner import (
    RequestPacer,
    _reserve_micros,
    build_cells,
    validate_inputs,
)
from flavourbench.epicure_native_taskset_v2 import verify_taskset

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(
    ROOT / "benchmark/powered-v2/manifest/"
    "flavourbench-frontier-refresh-20-"
    "44220f0a6c26798871f830f6ddd62ed99d8872ad5dcb7516491b90bfc71887fc.json"
)
MANIFEST_SHA = "44220f0a6c26798871f830f6ddd62ed99d8872ad5dcb7516491b90bfc71887fc"
TASKSET = next((ROOT / "benchmark/powered-v2/taskset").glob("*.json"))
REPEAT = next((ROOT / "benchmark/powered-v2/plan").glob("*repeat-panel*.json"))
PLAN = next((ROOT / "benchmark/powered-v2/plan").glob("*analysis-plan*.json"))
PREDECESSOR = ROOT / "paper/generated/epicure-native/epicure-native-release.json"


def test_exact_frozen_inputs_validate() -> None:
    manifest, taskset, repeat, plan, predecessor, candidates = validate_inputs(
        manifest_path=MANIFEST,
        manifest_sha256=MANIFEST_SHA,
        taskset_path=TASKSET,
        repeat_panel_path=REPEAT,
        plan_path=PLAN,
        predecessor_release_path=PREDECESSOR,
    )
    assert manifest["content_address"]["digest"] == MANIFEST_SHA
    assert verify_taskset(taskset)
    assert verify_repeat_panel(repeat, source_taskset=taskset)
    assert verify_plan(plan)
    assert (
        predecessor["artifact_sha256"] == plan["inputs"]["predecessor_release"]["semantic_sha256"]
    )
    assert len(candidates) == 20


def test_schedule_is_complete_unique_and_pilot_is_reused() -> None:
    _, taskset, repeat, plan, _, candidates = validate_inputs(
        manifest_path=MANIFEST,
        manifest_sha256=MANIFEST_SHA,
        taskset_path=TASKSET,
        repeat_panel_path=REPEAT,
        plan_path=PLAN,
        predecessor_release_path=PREDECESSOR,
    )
    pilot = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase="pilot",
    )
    primary = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase="primary",
    )
    all_cells = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase="all",
    )
    assert len(pilot) == 20
    assert len(primary) == 12_800
    assert len(all_cells) == 14_080
    assert {cell.cell_id for cell in pilot} <= {cell.cell_id for cell in primary}
    assert len({cell.cell_id for cell in all_cells}) == 14_080
    assert len({cell.task["task_id"] for cell in pilot}) == 1


def test_reservations_are_positive_and_bound_historical_maxima() -> None:
    _, _, _, _, predecessor, candidates = validate_inputs(
        manifest_path=MANIFEST,
        manifest_sha256=MANIFEST_SHA,
        taskset_path=TASKSET,
        repeat_panel_path=REPEAT,
        plan_path=PLAN,
        predecessor_release_path=PREDECESSOR,
    )
    maxima = {
        candidate.model_id: _reserve_micros(candidate, predecessor) for candidate in candidates
    }
    assert all(value >= 1000 for value in maxima.values())
    assert maxima["openai/gpt-5.6-sol-pro"] >= 28_235
    assert maxima["moonshotai/kimi-k3"] >= 26_997
    assert maxima["nvidia/nemotron-3-ultra-550b-a55b"] > 1000


def test_plan_and_repeat_physical_hashes_are_bound() -> None:
    plan = json.loads(PLAN.read_text())
    repeat = json.loads(REPEAT.read_text())
    assert (
        hashlib.sha256(REPEAT.read_bytes()).hexdigest()
        == plan["inputs"]["repeat_panel"]["physical_sha256"]
    )
    assert repeat["artifact_sha256"] == plan["inputs"]["repeat_panel"]["semantic_sha256"]


def test_shared_request_pacer_enforces_start_interval() -> None:
    async def exercise() -> float:
        pacer = RequestPacer(0.02)
        started = time.monotonic()
        await asyncio.gather(pacer.wait(), pacer.wait())
        return time.monotonic() - started

    assert asyncio.run(exercise()) >= 0.015
