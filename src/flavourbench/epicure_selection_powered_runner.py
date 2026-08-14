"""Execute the powered Epicure-scored FlavourBench selection panel."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
from .epicure_selection_powered_plan import (
    MODEL_COUNT,
    REPEAT_SCHEMA_VERSION,
    TASK_COUNT,
    selection_execution_policy,
    verify_repeat_panel,
)
from .epicure_selection_powered_plan import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V17
from .epicure_selection_powered_plan import verify_plan as verify_plan_v17
from .epicure_selection_powered_plan_v18 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V18
from .epicure_selection_powered_plan_v18 import verify_plan as verify_plan_v18
from .epicure_selection_powered_plan_v19 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V19
from .epicure_selection_powered_plan_v19 import verify_plan as verify_plan_v19
from .epicure_selection_powered_plan_v20 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V20
from .epicure_selection_powered_plan_v20 import verify_plan as verify_plan_v20
from .epicure_selection_powered_plan_v21 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V21
from .epicure_selection_powered_plan_v21 import verify_plan as verify_plan_v21
from .epicure_selection_powered_plan_v22 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V22
from .epicure_selection_powered_plan_v22 import verify_plan as verify_plan_v22
from .epicure_selection_powered_plan_v23 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V23
from .epicure_selection_powered_plan_v23 import verify_plan as verify_plan_v23
from .epicure_selection_powered_plan_v24 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V24
from .epicure_selection_powered_plan_v24 import verify_plan as verify_plan_v24
from .epicure_selection_powered_plan_v25 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V25
from .epicure_selection_powered_plan_v25 import verify_plan as verify_plan_v25
from .epicure_selection_powered_plan_v26 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V26
from .epicure_selection_powered_plan_v26 import verify_plan as verify_plan_v26
from .epicure_selection_powered_plan_v27 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V27
from .epicure_selection_powered_plan_v27 import verify_plan as verify_plan_v27
from .epicure_selection_powered_plan_v28 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V28
from .epicure_selection_powered_plan_v28 import verify_plan as verify_plan_v28
from .epicure_selection_powered_plan_v29 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V29
from .epicure_selection_powered_plan_v29 import verify_plan as verify_plan_v29
from .epicure_selection_powered_plan_v30 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V30
from .epicure_selection_powered_plan_v30 import verify_plan as verify_plan_v30
from .epicure_selection_powered_plan_v31 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V31
from .epicure_selection_powered_plan_v31 import selection_execution_policy_v31
from .epicure_selection_powered_plan_v31 import verify_plan as verify_plan_v31
from .epicure_selection_powered_plan_v32 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V32
from .epicure_selection_powered_plan_v32 import verify_plan as verify_plan_v32
from .epicure_selection_powered_plan_v33 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V33
from .epicure_selection_powered_plan_v33 import verify_plan as verify_plan_v33
from .epicure_selection_powered_plan_v34 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V34
from .epicure_selection_powered_plan_v34 import verify_plan as verify_plan_v34
from .epicure_selection_powered_plan_v35 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V35
from .epicure_selection_powered_plan_v35 import verify_plan as verify_plan_v35
from .epicure_selection_powered_plan_v36 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V36
from .epicure_selection_powered_plan_v36 import verify_plan as verify_plan_v36
from .epicure_selection_powered_plan_v37 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V37
from .epicure_selection_powered_plan_v37 import verify_plan as verify_plan_v37
from .epicure_selection_powered_plan_v38 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V38
from .epicure_selection_powered_plan_v38 import verify_plan as verify_plan_v38
from .epicure_selection_powered_plan_v39 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V39
from .epicure_selection_powered_plan_v39 import verify_plan as verify_plan_v39
from .epicure_selection_powered_plan_v40 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V40
from .epicure_selection_powered_plan_v40 import verify_plan as verify_plan_v40
from .epicure_selection_powered_plan_v41 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V41
from .epicure_selection_powered_plan_v41 import verify_plan as verify_plan_v41
from .epicure_selection_powered_plan_v42 import PLAN_SCHEMA_VERSION as PLAN_SCHEMA_VERSION_V42
from .epicure_selection_powered_plan_v42 import verify_plan as verify_plan_v42
from .epicure_selection_taskset_v1 import FAMILIES, score_answer, verify_taskset
from .frontier_contract_runner import (
    ContractCandidate,
    load_candidate_manifest,
    select_candidates,
)

RUNNER_SCHEMA_VERSION = "flavourbench-selection-powered-runner-v29"
PREFLIGHT_SCHEMA_VERSION = "flavourbench-selection-powered-preflight-v29"
CONFIRMATION = "RUN_EPICURE_SELECTION_FLAVOURBENCH_V28"


def _order_key(
    *,
    plan_sha256: str,
    candidate: ContractCandidate,
    task: Mapping[str, Any],
    pilot_ids: frozenset[str],
) -> tuple[int, str]:
    task_id = str(task["task_id"])
    return (
        0 if task_id in pilot_ids else 1,
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
    if phase not in {"pilot", "primary", "repeat", "all"}:
        raise PoweredRunnerError("unsupported run phase")
    plan_sha256 = str(plan["artifact_sha256"])
    pilot_ids = frozenset(str(value) for value in plan["execution"]["pilot"]["task_ids"])
    primary_by_id = {str(task["task_id"]): task for task in taskset["tasks"]}
    if len(pilot_ids) != len(FAMILIES) or not pilot_ids <= primary_by_id.keys():
        raise PoweredRunnerError("frozen pilot is absent or incomplete")
    if {primary_by_id[value]["family"] for value in pilot_ids} != set(FAMILIES):
        raise PoweredRunnerError("pilot must contain exactly one task from every family")
    cells: list[PlannedCell] = []
    for candidate in candidates:
        primary = sorted(
            taskset["tasks"],
            key=lambda task: _order_key(
                plan_sha256=plan_sha256,
                candidate=candidate,
                task=task,
                pilot_ids=pilot_ids,
            ),
        )
        if phase == "pilot":
            primary = [primary_by_id[value] for value in sorted(pilot_ids)]
        elif phase == "repeat":
            primary = []
        for task in primary:
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
        "pilot": len(FAMILIES) * len(candidates),
        "primary": TASK_COUNT * len(candidates),
        "repeat": 64 * len(candidates),
        "all": (TASK_COUNT + 64) * len(candidates),
    }[phase]
    if len(cells) != expected or len({cell.cell_id for cell in cells}) != expected:
        raise PoweredRunnerError("selection schedule is incomplete or duplicated")
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
    taskset = _load_json(taskset_path, label="Epicure selection task set")
    repeat = _load_json(repeat_panel_path, label="Epicure selection repeat panel")
    plan = _load_json(plan_path, label="Epicure selection analysis plan")
    predecessor = _load_json(predecessor_release_path, label="development predecessor release")
    plan_schema = plan.get("schema_version")
    plan_verifiers = {
        PLAN_SCHEMA_VERSION_V17: verify_plan_v17,
        PLAN_SCHEMA_VERSION_V18: verify_plan_v18,
        PLAN_SCHEMA_VERSION_V19: verify_plan_v19,
        PLAN_SCHEMA_VERSION_V20: verify_plan_v20,
        PLAN_SCHEMA_VERSION_V21: verify_plan_v21,
        PLAN_SCHEMA_VERSION_V22: verify_plan_v22,
        PLAN_SCHEMA_VERSION_V23: verify_plan_v23,
        PLAN_SCHEMA_VERSION_V24: verify_plan_v24,
        PLAN_SCHEMA_VERSION_V25: verify_plan_v25,
        PLAN_SCHEMA_VERSION_V26: verify_plan_v26,
        PLAN_SCHEMA_VERSION_V27: verify_plan_v27,
        PLAN_SCHEMA_VERSION_V28: verify_plan_v28,
        PLAN_SCHEMA_VERSION_V29: verify_plan_v29,
        PLAN_SCHEMA_VERSION_V30: verify_plan_v30,
        PLAN_SCHEMA_VERSION_V31: verify_plan_v31,
        PLAN_SCHEMA_VERSION_V32: verify_plan_v32,
        PLAN_SCHEMA_VERSION_V33: verify_plan_v33,
        PLAN_SCHEMA_VERSION_V34: verify_plan_v34,
        PLAN_SCHEMA_VERSION_V35: verify_plan_v35,
        PLAN_SCHEMA_VERSION_V36: verify_plan_v36,
        PLAN_SCHEMA_VERSION_V37: verify_plan_v37,
        PLAN_SCHEMA_VERSION_V38: verify_plan_v38,
        PLAN_SCHEMA_VERSION_V39: verify_plan_v39,
        PLAN_SCHEMA_VERSION_V40: verify_plan_v40,
        PLAN_SCHEMA_VERSION_V41: verify_plan_v41,
        PLAN_SCHEMA_VERSION_V42: verify_plan_v42,
    }
    plan_valid = plan_schema in plan_verifiers and plan_verifiers[plan_schema](plan)
    if (
        not verify_taskset(taskset)
        or not verify_repeat_panel(repeat, taskset=taskset)
        or not plan_valid
        or repeat.get("schema_version") != REPEAT_SCHEMA_VERSION
    ):
        raise PoweredRunnerError("Epicure selection input verification failed")
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
            inputs["development_predecessor"],
        ),
    }
    for label, (semantic, physical, recorded) in exact.items():
        if recorded["semantic_sha256"] != semantic or recorded["physical_sha256"] != physical:
            raise PoweredRunnerError(f"plan {label} pin differs from exact input")
    candidates = select_candidates(manifest)
    expected_model_count = int(plan.get("roster", {}).get("model_count", MODEL_COUNT))
    if len(candidates) != expected_model_count:
        raise PoweredRunnerError(
            "selection manifest does not contain the plan's exact candidate count"
        )
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
    policy = (
        selection_execution_policy_v31()
        if plan.get("schema_version")
        in {
            PLAN_SCHEMA_VERSION_V31,
            PLAN_SCHEMA_VERSION_V32,
            PLAN_SCHEMA_VERSION_V33,
            PLAN_SCHEMA_VERSION_V34,
            PLAN_SCHEMA_VERSION_V35,
            PLAN_SCHEMA_VERSION_V36,
            PLAN_SCHEMA_VERSION_V37,
            PLAN_SCHEMA_VERSION_V38,
            PLAN_SCHEMA_VERSION_V39,
            PLAN_SCHEMA_VERSION_V40,
            PLAN_SCHEMA_VERSION_V41,
            PLAN_SCHEMA_VERSION_V42,
        }
        else selection_execution_policy()
    )
    configure_live_environment(
        secret_file=args.secrets_env_file,
        plan=plan,
        execution_policy=policy,
    )
    cells = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase=args.phase,
    )
    if args.model_id is not None:
        cells = [cell for cell in cells if cell.candidate.model_id == args.model_id]
        if not cells:
            raise PoweredRunnerError("requested model is absent from the frozen schedule")
    if args.exclude_model_id:
        excluded = set(args.exclude_model_id)
        available = {candidate.model_id for candidate in candidates}
        if not excluded <= available:
            raise PoweredRunnerError(
                "one or more excluded models are absent from the frozen roster"
            )
        cells = [cell for cell in cells if cell.candidate.model_id not in excluded]
    if args.successor_only:
        successor_ids = set(
            plan.get("execution", {}).get("frontier_refresh_successor", {}).get("new_model_ids", [])
        )
        if not successor_ids:
            raise PoweredRunnerError("plan has no frozen successor-only model set")
        cells = [cell for cell in cells if cell.candidate.model_id in successor_ids]
    if args.task_id:
        requested_tasks = set(args.task_id)
        cells = [cell for cell in cells if str(cell.task["task_id"]) in requested_tasks]
        if {str(cell.task["task_id"]) for cell in cells} != requested_tasks:
            raise PoweredRunnerError("one or more requested tasks are absent from the schedule")
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    frozen_concurrency = plan["execution"]["collection_concurrency"]
    if (
        not args.preflight_only
        and args.model_id is None
        and (
            args.global_concurrency != frozen_concurrency["global"]
            or args.per_model_concurrency != frozen_concurrency["per_model_default"]
        )
    ):
        raise PoweredRunnerError("live collection concurrency differs from the frozen plan")
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
            "pilot_cells": plan["execution"]["pilot"]["cells"],
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
        score_function=score_answer,
        execution_policy=policy,
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
    parser.add_argument("--global-concurrency", type=int, default=24)
    parser.add_argument("--per-model-concurrency", type=int, default=4)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--exclude-model-id", action="append")
    parser.add_argument("--successor-only", action="store_true")
    parser.add_argument("--task-id", action="append")
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
