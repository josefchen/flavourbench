"""Run the construct-validated powered FlavourBench panel.

This additive runner binds the validated v3 task set and v2 analysis plan while
reusing the already tested provider, accounting, response, and journaling
machinery from the first powered runner.  Its default action is a zero-call
preflight.  Live execution requires an explicit version-specific confirmation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .epicure_native_powered_plan_v2 import (
    MODEL_COUNT,
    PLAN_SCHEMA_VERSION,
    REPEAT_SCHEMA_VERSION,
    TASK_COUNT,
    verify_plan,
    verify_repeat_panel,
)
from .epicure_native_powered_runner import (
    SECRET_KEYS,
    PlannedCell,
    PoweredRun,
    PoweredRunnerError,
    _cell_id,
    _load_json,
    _sha256,
    _sha256_file,
    configure_live_environment,
)
from .epicure_native_taskset_v3 import verify_taskset
from .frontier_contract_runner import (
    ContractCandidate,
    load_candidate_manifest,
    select_candidates,
)

RUNNER_SCHEMA_VERSION = "flavourbench-powered-runner-v2"
PREFLIGHT_SCHEMA_VERSION = "flavourbench-powered-runner-preflight-v2"
CONFIRMATION = "RUN_CONSTRUCT_VALIDATED_FLAVOURBENCH_V2"


def _primary_order_key(
    *,
    plan_sha256: str,
    candidate: ContractCandidate,
    task: Mapping[str, Any],
    pilot_task_ids: frozenset[str],
) -> tuple[int, str]:
    task_id = str(task["task_id"])
    return (
        0 if task_id in pilot_task_ids else 1,
        hashlib.sha256(
            (plan_sha256 + "\0primary\0" + candidate.model_id + "\0" + task_id).encode()
        ).hexdigest(),
    )


def build_cells(
    *,
    plan: Mapping[str, Any],
    taskset: Mapping[str, Any],
    repeat_panel: Mapping[str, Any],
    candidates: Sequence[ContractCandidate],
    phase: str,
) -> list[PlannedCell]:
    """Build a complete deterministic schedule with the pilot reused in primary."""

    if phase not in {"pilot", "primary", "repeat", "all"}:
        raise PoweredRunnerError("unsupported run phase")
    plan_sha256 = str(plan["artifact_sha256"])
    pilot_task_ids = frozenset(str(value) for value in plan["execution"]["pilot"]["task_ids"])
    primary_by_id = {str(task["task_id"]): task for task in taskset["tasks"]}
    if len(pilot_task_ids) != 4 or not pilot_task_ids <= primary_by_id.keys():
        raise PoweredRunnerError("frozen four-family pilot is absent or incomplete")
    if {primary_by_id[task_id]["family"] for task_id in pilot_task_ids} != {
        "substitution",
        "pairing",
        "constraint",
        "provenance",
    }:
        raise PoweredRunnerError("pilot must contain exactly one task from every family")

    cells: list[PlannedCell] = []
    for candidate in candidates:
        primary_tasks = sorted(
            taskset["tasks"],
            key=lambda task: _primary_order_key(
                plan_sha256=plan_sha256,
                candidate=candidate,
                task=task,
                pilot_task_ids=pilot_task_ids,
            ),
        )
        if phase == "pilot":
            primary_tasks = [primary_by_id[task_id] for task_id in sorted(pilot_task_ids)]
        elif phase == "repeat":
            primary_tasks = []
        for task in primary_tasks:
            cells.append(
                PlannedCell(
                    cell_id=_cell_id(
                        plan_sha256=plan_sha256,
                        panel="primary",
                        candidate=candidate,
                        task=task,
                    ),
                    panel="primary",
                    candidate=candidate,
                    task=task,
                )
            )
        if phase in {"repeat", "all"}:
            repeated = sorted(
                repeat_panel["tasks"],
                key=lambda task: hashlib.sha256(
                    (
                        plan_sha256
                        + "\0repeat\0"
                        + candidate.model_id
                        + "\0"
                        + str(task["task_id"])
                    ).encode()
                ).hexdigest(),
            )
            for task in repeated:
                cells.append(
                    PlannedCell(
                        cell_id=_cell_id(
                            plan_sha256=plan_sha256,
                            panel="repeat",
                            candidate=candidate,
                            task=task,
                        ),
                        panel="repeat",
                        candidate=candidate,
                        task=task,
                    )
                )

    expected = {
        "pilot": 4 * len(candidates),
        "primary": TASK_COUNT * len(candidates),
        "repeat": 64 * len(candidates),
        "all": (TASK_COUNT + 64) * len(candidates),
    }[phase]
    if len(cells) != expected or len({cell.cell_id for cell in cells}) != expected:
        raise PoweredRunnerError("powered successor schedule is incomplete or duplicated")
    return cells


def validate_inputs(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    taskset_path: Path,
    repeat_panel_path: Path,
    plan_path: Path,
    predecessor_release_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[ContractCandidate],
]:
    manifest = load_candidate_manifest(manifest_path, expected_digest=manifest_sha256)
    taskset = _load_json(taskset_path, label="validated powered task set")
    repeat = _load_json(repeat_panel_path, label="validated powered repeat panel")
    plan = _load_json(plan_path, label="validated powered analysis plan")
    predecessor = _load_json(predecessor_release_path, label="development predecessor release")
    if (
        not verify_taskset(taskset)
        or not verify_repeat_panel(repeat, taskset=taskset)
        or not verify_plan(plan)
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or repeat.get("schema_version") != REPEAT_SCHEMA_VERSION
    ):
        raise PoweredRunnerError("construct-validated powered input verification failed")
    inputs = plan["inputs"]
    exact = {
        "manifest": (
            manifest_sha256,
            _sha256_file(manifest_path),
            inputs["route_manifest"],
        ),
        "taskset": (
            taskset["artifact_sha256"],
            _sha256_file(taskset_path),
            inputs["taskset"],
        ),
        "repeat": (
            repeat["artifact_sha256"],
            _sha256_file(repeat_panel_path),
            inputs["repeat_panel"],
        ),
        "predecessor": (
            predecessor["artifact_sha256"],
            _sha256_file(predecessor_release_path),
            inputs["predecessor_release"],
        ),
    }
    for label, (semantic, physical, recorded) in exact.items():
        if recorded["semantic_sha256"] != semantic or recorded["physical_sha256"] != physical:
            raise PoweredRunnerError(f"plan {label} pin differs from exact input")
    candidates = select_candidates(manifest)
    if len(candidates) != MODEL_COUNT:
        raise PoweredRunnerError("powered manifest does not contain exactly 20 candidates")
    roster = [(row["slot_id"], row["model_id"]) for row in plan["roster"]["models"]]
    selected = [(candidate.slot_id, candidate.model_id) for candidate in candidates]
    if roster != selected:
        raise PoweredRunnerError("plan roster differs from the exact selected routes")
    return manifest, taskset, repeat, plan, predecessor, candidates


async def _async_run(args: argparse.Namespace) -> None:
    manifest, taskset, repeat, plan, predecessor, candidates = validate_inputs(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_semantic_sha256,
        taskset_path=args.taskset,
        repeat_panel_path=args.repeat_panel,
        plan_path=args.plan,
        predecessor_release_path=args.predecessor_release,
    )
    configure_live_environment(secret_file=args.secrets_env_file, plan=plan)
    cells = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase=args.phase,
    )
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    if args.preflight_only:
        document: dict[str, Any] = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "status": "preflight_passed_no_provider_calls",
            "plan_sha256": plan["artifact_sha256"],
            "manifest_sha256": args.manifest_semantic_sha256,
            "taskset_sha256": taskset["artifact_sha256"],
            "repeat_panel_sha256": repeat["artifact_sha256"],
            "models": len(candidates),
            "scheduled_cells": len(cells),
            "pilot_cells": int(plan["execution"]["pilot"]["cells"]),
            "required_credentials_present": sorted(SECRET_KEYS),
            "provider_clients_constructed": False,
            "provider_calls_made": 0,
        }
        document["artifact_sha256"] = _sha256(document)
        print(json.dumps(document, sort_keys=True))
        return
    if args.confirm != CONFIRMATION:
        raise PoweredRunnerError(f"live execution requires --confirm {CONFIRMATION}")
    runner = PoweredRun(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        manifest_sha256=args.manifest_semantic_sha256,
        predecessor_release=predecessor,
        output_directory=args.output_directory,
        global_concurrency=args.global_concurrency,
        per_model_concurrency=args.per_model_concurrency,
    )
    result = await runner.execute(cells)
    result["runner_schema_version"] = RUNNER_SCHEMA_VERSION
    print(json.dumps(result, sort_keys=True))


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-semantic-sha256", required=True)
    parser.add_argument("--taskset", type=Path, required=True)
    parser.add_argument("--repeat-panel", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predecessor-release", type=Path, required=True)
    parser.add_argument("--secrets-env-file", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "primary", "repeat", "all"), default="pilot")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--global-concurrency", type=int, default=40)
    parser.add_argument("--per-model-concurrency", type=int, default=3)
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.global_concurrency <= 80:
        raise PoweredRunnerError("global concurrency must be in [1, 80]")
    if not 1 <= args.per_model_concurrency <= 8:
        raise PoweredRunnerError("per-model concurrency must be in [1, 8]")
    if args.max_cells is not None and args.max_cells <= 0:
        raise PoweredRunnerError("max cells must be positive")
    asyncio.run(_async_run(args))


if __name__ == "__main__":
    run()
