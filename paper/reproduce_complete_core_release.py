#!/usr/bin/env python3
"""Rebuild the final Fable-inclusive complete-core leaderboard from raw responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flavourbench.epicure_selection_common_core_analysis_v1 import load_complete_common_core
from flavourbench.epicure_selection_complete_core_plan_v84 import (
    build_plan,
    selected_task_ids,
    verify_plan,
    write_plan,
)
from flavourbench.epicure_selection_complete_core_release_v1 import (
    build_release,
    write_release,
)
from flavourbench.epicure_selection_complete_core_sources_v1 import (
    source_graph,
)
from flavourbench.epicure_selection_powered_analysis import _sha256_file
from flavourbench.epicure_selection_route_manifest_v57 import DEEPSEEK_PRO_MODEL_ID


def _load(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"input is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"input is not a JSON object: {path}")
    return value


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "benchmark/powered-v84/plan/epicure-selection-joint-analysis-plan-"
            "2ba71c793c8d4b97eed863ee83fd770b429fdefdffebdeafb241672f634ee507.json"
        ),
    )
    parser.add_argument(
        "--predecessor-v83",
        type=Path,
        default=Path(
            "benchmark/powered-v83/plan/epicure-selection-joint-analysis-plan-"
            "31f45aaf447b9337e07b9b27a75c9706bb6523efec3ad7e2738f76b9fc9d798b.json"
        ),
    )
    parser.add_argument(
        "--parser-source",
        type=Path,
        default=Path("src/flavourbench/selection_response_parser_v3.py"),
    )
    parser.add_argument(
        "--plan-output-directory", type=Path, default=Path("benchmark/powered-v84/plan")
    )
    parser.add_argument(
        "--rebuild-plan",
        action="store_true",
        help="reconstruct the score-blind common-core freeze from every candidate response",
    )
    parser.add_argument(
        "--release-output-directory", type=Path, default=Path("paper/generated/complete-core")
    )
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--permutation-resamples", type=int)
    args = parser.parse_args()

    root = args.root.resolve()
    graph = source_graph(root)
    if args.rebuild_plan:
        from flavourbench.epicure_selection_complete_core_sources_v1 import (
            load_full_primary_panels,
        )

        predecessor_path = root / args.predecessor_v83
        parser_path = root / args.parser_source
        full_1, full_2 = load_full_primary_panels(graph)
        plan = build_plan(
            predecessor=_load(predecessor_path),
            predecessor_path=predecessor_path,
            panel_1_data=full_1,
            panel_1_taskset=graph.panel_1_taskset,
            panel_1_taskset_path=graph.panel_1_taskset_path,
            panel_2_data=full_2,
            panel_2_taskset=graph.panel_2_taskset,
            panel_2_taskset_path=graph.panel_2_taskset_path,
            parser_path=parser_path,
        )
        plan_path = write_plan(plan, root / args.plan_output_directory)
    else:
        plan_path = root / args.plan
        plan = _load(plan_path)
        if not verify_plan(plan):
            raise RuntimeError("frozen complete-core plan failed verification")
    tasks_1, tasks_2 = selected_task_ids(plan)
    allowed = {DEEPSEEK_PRO_MODEL_ID: frozenset({"endpoint_sha256"})}
    selected_1 = load_complete_common_core(
        panel="primary",
        plan=graph.panel_1_plan,
        taskset=graph.panel_1_taskset,
        repeat_panel=graph.panel_1_repeat,
        task_ids=tasks_1,
        model_sources=graph.panel_1_model_sources,
        allowed_source_roster_differences=allowed,
    )
    selected_2 = load_complete_common_core(
        panel="primary",
        plan=graph.panel_2_plan,
        taskset=graph.panel_2_taskset,
        repeat_panel=graph.panel_2_repeat,
        task_ids=tasks_2,
        model_sources=graph.panel_2_model_sources,
        allowed_source_roster_differences=allowed,
    )
    release, leaderboard, pairwise = build_release(
        plan=plan,
        plan_path=plan_path,
        panel_1=selected_1,
        panel_1_taskset=graph.panel_1_taskset,
        panel_1_taskset_path=graph.panel_1_taskset_path,
        panel_2=selected_2,
        panel_2_taskset=graph.panel_2_taskset,
        panel_2_taskset_path=graph.panel_2_taskset_path,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
    )
    release_path = write_release(
        release=release,
        leaderboard=leaderboard,
        pairwise=pairwise,
        directory=root / args.release_output_directory,
    )
    print(
        json.dumps(
            {
                "plan": _display_path(plan_path, root),
                "plan_physical_sha256": _sha256_file(plan_path),
                "release": _display_path(release_path, root),
                "release_physical_sha256": _sha256_file(release_path),
                "models": len(release["analysis"]["models"]),
                "tasks_per_model": release["design"]["primary_tasks_per_model"],
                "resolved_pairs": release["analysis"]["resolved_pair_count"],
                "definitive_top_model_id": release["analysis"]["definitive_top_model_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
