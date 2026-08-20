"""Analyze both frozen powered panels with ingredient-anchor clustered inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .epicure_selection_powered_analysis import (
    PanelData,
    SelectionPoweredAnalysisError,
    _load,
    _sha256,
    _sha256_bytes,
    _sha256_file,
    _write_content_addressed,
    load_panel,
)
from .epicure_selection_powered_analysis_v2 import (
    _leaderboard_csv,
    _pairwise_csv,
    _valid_family_summary,
    analyze_panels,
)
from .epicure_selection_powered_plan_v44 import verify_plan as verify_plan_v44
from .epicure_selection_powered_plan_v45 import verify_plan as verify_plan_v45
from .epicure_selection_powered_plan_v46 import verify_plan as verify_plan_v46
from .epicure_selection_powered_plan_v47 import verify_plan as verify_plan_v47
from .epicure_selection_powered_plan_v48 import verify_plan as verify_joint_plan
from .epicure_selection_powered_plan_v49 import verify_plan as verify_plan_v49
from .epicure_selection_powered_plan_v50 import verify_plan as verify_plan_v50
from .epicure_selection_powered_plan_v51 import verify_plan as verify_joint_plan_v51
from .epicure_selection_powered_plan_v52 import verify_plan as verify_plan_v52
from .epicure_selection_powered_plan_v53 import verify_plan as verify_joint_plan_v53
from .epicure_selection_powered_plan_v54 import verify_plan as verify_plan_v54
from .epicure_selection_powered_plan_v55 import verify_plan as verify_plan_v55
from .epicure_selection_powered_plan_v56 import verify_plan as verify_joint_plan_v56
from .epicure_selection_powered_plan_v58 import verify_plan as verify_plan_v58
from .epicure_selection_powered_plan_v59 import verify_plan as verify_plan_v59
from .epicure_selection_powered_plan_v60 import verify_plan as verify_joint_plan_v60
from .epicure_selection_powered_plan_v62 import verify_plan as verify_plan_v62
from .epicure_selection_powered_plan_v63 import verify_plan as verify_plan_v63
from .epicure_selection_powered_plan_v64 import verify_plan as verify_joint_plan_v64
from .epicure_selection_powered_plan_v65 import verify_plan as verify_plan_v65
from .epicure_selection_powered_plan_v66 import verify_plan as verify_plan_v66
from .epicure_selection_powered_plan_v67 import verify_plan as verify_joint_plan_v67
from .epicure_selection_powered_plan_v74 import verify_plan as verify_plan_v74
from .epicure_selection_powered_plan_v75 import verify_plan as verify_plan_v75
from .epicure_selection_powered_plan_v76 import verify_plan as verify_joint_plan_v76
from .epicure_selection_powered_plan_v77 import verify_plan as verify_joint_plan_v77
from .epicure_selection_powered_plan_v78 import verify_plan as verify_joint_plan_v78
from .epicure_selection_repeat_panel_replication_v1 import (
    verify_repeat_panel as verify_repeat_panel_replication_2,
)
from .epicure_selection_repeat_panel_v2 import verify_repeat_panel
from .epicure_selection_route_manifest_v45 import FABLE_MODEL_ID, QWEN_MODEL_ID
from .epicure_selection_route_manifest_v52 import (
    DEEPSEEK_FLASH_MODEL_ID,
    LUNA_MODEL_ID,
)
from .epicure_selection_route_manifest_v54 import (
    REPLACEMENT_MODEL_IDS as COVERAGE_REPAIR_MODEL_IDS,
)
from .epicure_selection_route_manifest_v57 import DEEPSEEK_PRO_MODEL_ID
from .epicure_selection_taskset_replication_v1 import (
    verify_taskset as verify_taskset_replication_2,
)
from .epicure_selection_taskset_v2 import verify_taskset

ANALYSIS_SCHEMA_VERSION = "flavourbench-selection-powered-joint-analysis-v1"
RELEASE_SCHEMA_VERSION = "flavourbench-selection-powered-joint-release-v1"


def _assert_pin(
    *,
    plan: Mapping[str, Any],
    label: str,
    document: Mapping[str, Any],
    path: Path,
) -> None:
    recorded = plan["inputs"][label]
    if recorded != {
        "semantic_sha256": document["artifact_sha256"],
        "physical_sha256": _sha256_file(path),
    }:
        raise SelectionPoweredAnalysisError(f"joint plan {label} pin differs from input")


def combine_panel_data(left: PanelData, right: PanelData, *, panel: str) -> PanelData:
    """Concatenate two aligned panels without merging or dropping any response cell."""

    if (
        left.model_ids != right.model_ids
        or left.model_names != right.model_names
        or left.slot_ids != right.slot_ids
    ):
        raise SelectionPoweredAnalysisError("joint response panels have different rosters")
    if set(left.task_ids) & set(right.task_ids):
        raise SelectionPoweredAnalysisError("joint response panels reuse a task ID")
    artifacts = left.response_artifact_sha256s + right.response_artifact_sha256s
    if len(set(artifacts)) != len(artifacts):
        raise SelectionPoweredAnalysisError("joint response panels reuse an artifact")
    return PanelData(
        panel=panel,
        model_ids=left.model_ids,
        model_names=left.model_names,
        slot_ids=left.slot_ids,
        task_ids=left.task_ids + right.task_ids,
        families=left.families + right.families,
        scores=np.concatenate((left.scores, right.scores), axis=1),
        completed=np.concatenate((left.completed, right.completed), axis=1),
        parseable=np.concatenate((left.parseable, right.parseable), axis=1),
        selections=tuple(
            left.selections[index] + right.selections[index] for index in range(len(left.model_ids))
        ),
        response_artifact_sha256s=tuple(sorted(artifacts)),
        spend_micros=left.spend_micros + right.spend_micros,
    )


def subset_panel_data(data: PanelData, model_ids: Sequence[str]) -> PanelData:
    """Return an exact ordered model subset without changing any task cells."""

    indices = [data.model_ids.index(model_id) for model_id in model_ids]
    if len(indices) != len(set(indices)) or not indices:
        raise SelectionPoweredAnalysisError("ranked model subset is empty or duplicated")
    return PanelData(
        panel=data.panel,
        model_ids=tuple(data.model_ids[index] for index in indices),
        model_names=tuple(data.model_names[index] for index in indices),
        slot_ids=tuple(data.slot_ids[index] for index in indices),
        task_ids=data.task_ids,
        families=data.families,
        scores=data.scores[indices],
        completed=data.completed[indices],
        parseable=data.parseable[indices],
        selections=tuple(data.selections[index] for index in indices),
        # Preserve the full raw evidence commitment, including diagnostic-only models.
        response_artifact_sha256s=data.response_artifact_sha256s,
        spend_micros=data.spend_micros,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) == 0 or float(np.std(right)) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def replication_stability(
    panel_1: PanelData,
    panel_2: PanelData,
) -> dict[str, Any]:
    """Return a descriptive, response-blindly prespecified cross-panel stability summary."""

    valid_1 = panel_1.completed & panel_1.parseable
    valid_2 = panel_2.completed & panel_2.parseable
    point_1, _, _ = _valid_family_summary(panel_1.scores, valid_1, panel_1.families)
    point_2, _, _ = _valid_family_summary(panel_2.scores, valid_2, panel_2.families)
    ranks_1 = _average_ranks(point_1)
    ranks_2 = _average_ranks(point_2)
    return {
        "status": "descriptive_stability_diagnostic_not_a_selection_rule",
        "panel_1_tasks": len(panel_1.task_ids),
        "panel_2_tasks": len(panel_2.task_ids),
        "model_score_pearson": _correlation(point_1, point_2),
        "model_rank_spearman": _correlation(ranks_1, ranks_2),
        "models": [
            {
                "model_id": model_id,
                "panel_1_score": float(point_1[index]),
                "panel_2_score": float(point_2[index]),
                "panel_2_minus_panel_1": float(point_2[index] - point_1[index]),
                "panel_1_valid_scored": int(valid_1[index].sum()),
                "panel_2_valid_scored": int(valid_2[index].sum()),
            }
            for index, model_id in enumerate(panel_1.model_ids)
        ],
    }


def _source_directory_order(primary: Path, completions: Sequence[Path]) -> Path | tuple[Path, ...]:
    """Preserve the legacy single-path shape unless fallback directories are present."""

    return (primary, *completions) if completions else primary


def _panel_1_model_sources(
    *,
    plan: Mapping[str, Any],
    run_directory: Path,
    source_plan: Mapping[str, Any],
    source_plan_path: Path,
    qwen_plan: Mapping[str, Any],
    qwen_plan_path: Path,
    qwen_run_directory: Path,
    fable_run_directory: Path | None,
    prior_plan_v50: Mapping[str, Any] | None = None,
    prior_plan_v50_path: Path | None = None,
    repair_run_directory: Path | None = None,
    base_completion_run_directories: Sequence[Path] = (),
    coverage_completion_run_directories: Sequence[Path] = (),
    prior_plan_v55: Mapping[str, Any] | None = None,
    prior_plan_v55_path: Path | None = None,
    deepseek_repair_run_directory: Path | None = None,
) -> dict[str, tuple[Path | Sequence[Path], Mapping[str, Any]]]:
    if (
        not (
            verify_plan_v47(plan)
            or verify_plan_v50(plan)
            or verify_plan_v55(plan)
            or verify_plan_v58(plan)
            or verify_plan_v62(plan)
        )
        or not verify_plan_v44(source_plan)
        or not verify_plan_v45(qwen_plan)
    ):
        raise SelectionPoweredAnalysisError("panel 1 plan lineage failed verification")
    if plan["inputs"]["plan_v44_predecessor"] != {
        "semantic_sha256": source_plan["artifact_sha256"],
        "physical_sha256": _sha256_file(source_plan_path),
    }:
        raise SelectionPoweredAnalysisError("panel 1 v44 source pin differs")
    if plan["inputs"]["plan_v45_qwen_source"] != {
        "semantic_sha256": qwen_plan["artifact_sha256"],
        "physical_sha256": _sha256_file(qwen_plan_path),
    }:
        raise SelectionPoweredAnalysisError("panel 1 Qwen source pin differs")
    model_ids = [str(row["model_id"]) for row in plan["roster"]["models"]]
    sources = {
        model_id: (
            _source_directory_order(run_directory, base_completion_run_directories),
            source_plan,
        )
        for model_id in model_ids
    }
    sources[QWEN_MODEL_ID] = (qwen_run_directory, qwen_plan)
    if verify_plan_v58(plan) or verify_plan_v62(plan):
        if (
            fable_run_directory is not None
            or prior_plan_v50 is None
            or prior_plan_v50_path is None
            or repair_run_directory is None
            or prior_plan_v55 is None
            or prior_plan_v55_path is None
            or deepseek_repair_run_directory is None
            or not verify_plan_v50(prior_plan_v50)
            or not verify_plan_v55(prior_plan_v55)
        ):
            raise SelectionPoweredAnalysisError(
                "DeepSeek panel 1 requires v50/v55 predecessors and both repair runs"
            )
        if plan["inputs"]["plan_v55_predecessor"] != {
            "semantic_sha256": prior_plan_v55["artifact_sha256"],
            "physical_sha256": _sha256_file(prior_plan_v55_path),
        }:
            raise SelectionPoweredAnalysisError("panel 1 v55 predecessor pin differs")
        if prior_plan_v55["inputs"]["plan_v50_predecessor"] != {
            "semantic_sha256": prior_plan_v50["artifact_sha256"],
            "physical_sha256": _sha256_file(prior_plan_v50_path),
        }:
            raise SelectionPoweredAnalysisError("panel 1 v50 predecessor pin differs")
        for model_id in COVERAGE_REPAIR_MODEL_IDS:
            sources[model_id] = (
                _source_directory_order(repair_run_directory, coverage_completion_run_directories),
                prior_plan_v55,
            )
        sources[DEEPSEEK_PRO_MODEL_ID] = (deepseek_repair_run_directory, plan)
    elif verify_plan_v55(plan):
        if (
            fable_run_directory is not None
            or prior_plan_v50 is None
            or prior_plan_v50_path is None
            or repair_run_directory is None
            or not verify_plan_v50(prior_plan_v50)
        ):
            raise SelectionPoweredAnalysisError(
                "v55 panel 1 requires its v50 predecessor and coverage-repair run"
            )
        if plan["inputs"]["plan_v50_predecessor"] != {
            "semantic_sha256": prior_plan_v50["artifact_sha256"],
            "physical_sha256": _sha256_file(prior_plan_v50_path),
        }:
            raise SelectionPoweredAnalysisError("panel 1 v50 predecessor pin differs")
        for model_id in COVERAGE_REPAIR_MODEL_IDS:
            sources[model_id] = (
                _source_directory_order(repair_run_directory, coverage_completion_run_directories),
                plan,
            )
    elif verify_plan_v50(plan):
        if fable_run_directory is None:
            raise SelectionPoweredAnalysisError("v50 panel 1 requires the Fable replacement")
        sources[FABLE_MODEL_ID] = (fable_run_directory, plan)
    elif fable_run_directory is not None:
        raise SelectionPoweredAnalysisError("Fable replacement requires a v50 panel plan")
    return sources


def _panel_2_model_sources(
    *,
    plan: Mapping[str, Any],
    run_directory: Path,
    source_plan: Mapping[str, Any],
    source_plan_path: Path,
    luna_run_directory: Path,
    deepseek_flash_run_directory: Path,
    prior_replacement_plan_v52: Mapping[str, Any] | None = None,
    prior_replacement_plan_v52_path: Path | None = None,
    repair_run_directory: Path | None = None,
    base_completion_run_directories: Sequence[Path] = (),
    coverage_completion_run_directories: Sequence[Path] = (),
    prior_plan_v54: Mapping[str, Any] | None = None,
    prior_plan_v54_path: Path | None = None,
    deepseek_repair_run_directory: Path | None = None,
) -> dict[str, tuple[Path | Sequence[Path], Mapping[str, Any]]]:
    if not (
        verify_plan_v52(plan)
        or verify_plan_v54(plan)
        or verify_plan_v59(plan)
        or verify_plan_v63(plan)
    ) or not verify_plan_v49(source_plan):
        raise SelectionPoweredAnalysisError("panel 2 replacement lineage failed verification")
    if plan["inputs"]["plan_v49_predecessor"] != {
        "semantic_sha256": source_plan["artifact_sha256"],
        "physical_sha256": _sha256_file(source_plan_path),
    }:
        raise SelectionPoweredAnalysisError("panel 2 v49 source pin differs")
    model_ids = [str(row["model_id"]) for row in plan["roster"]["models"]]
    sources = {
        model_id: (
            _source_directory_order(run_directory, base_completion_run_directories),
            source_plan,
        )
        for model_id in model_ids
    }
    if verify_plan_v59(plan) or verify_plan_v63(plan):
        if (
            prior_replacement_plan_v52 is None
            or prior_replacement_plan_v52_path is None
            or repair_run_directory is None
            or prior_plan_v54 is None
            or prior_plan_v54_path is None
            or deepseek_repair_run_directory is None
            or not verify_plan_v52(prior_replacement_plan_v52)
            or not verify_plan_v54(prior_plan_v54)
        ):
            raise SelectionPoweredAnalysisError(
                "DeepSeek panel 2 requires v52/v54 predecessors and both repair runs"
            )
        if plan["inputs"]["plan_v54_predecessor"] != {
            "semantic_sha256": prior_plan_v54["artifact_sha256"],
            "physical_sha256": _sha256_file(prior_plan_v54_path),
        }:
            raise SelectionPoweredAnalysisError("panel 2 v54 predecessor pin differs")
        if prior_plan_v54["inputs"]["plan_v52_predecessor"] != {
            "semantic_sha256": prior_replacement_plan_v52["artifact_sha256"],
            "physical_sha256": _sha256_file(prior_replacement_plan_v52_path),
        }:
            raise SelectionPoweredAnalysisError("panel 2 v52 predecessor pin differs")
        sources[LUNA_MODEL_ID] = (luna_run_directory, prior_replacement_plan_v52)
        sources[DEEPSEEK_FLASH_MODEL_ID] = (
            deepseek_flash_run_directory,
            prior_replacement_plan_v52,
        )
        for model_id in COVERAGE_REPAIR_MODEL_IDS:
            sources[model_id] = (
                _source_directory_order(repair_run_directory, coverage_completion_run_directories),
                prior_plan_v54,
            )
        sources[DEEPSEEK_PRO_MODEL_ID] = (deepseek_repair_run_directory, plan)
    elif verify_plan_v54(plan):
        if (
            prior_replacement_plan_v52 is None
            or prior_replacement_plan_v52_path is None
            or repair_run_directory is None
            or not verify_plan_v52(prior_replacement_plan_v52)
        ):
            raise SelectionPoweredAnalysisError(
                "v54 panel 2 requires its v52 predecessor and coverage-repair run"
            )
        if plan["inputs"]["plan_v52_predecessor"] != {
            "semantic_sha256": prior_replacement_plan_v52["artifact_sha256"],
            "physical_sha256": _sha256_file(prior_replacement_plan_v52_path),
        }:
            raise SelectionPoweredAnalysisError("panel 2 v52 predecessor pin differs")
        sources[LUNA_MODEL_ID] = (luna_run_directory, prior_replacement_plan_v52)
        sources[DEEPSEEK_FLASH_MODEL_ID] = (
            deepseek_flash_run_directory,
            prior_replacement_plan_v52,
        )
        for model_id in COVERAGE_REPAIR_MODEL_IDS:
            sources[model_id] = (
                _source_directory_order(repair_run_directory, coverage_completion_run_directories),
                plan,
            )
    else:
        sources[LUNA_MODEL_ID] = (luna_run_directory, plan)
        sources[DEEPSEEK_FLASH_MODEL_ID] = (deepseek_flash_run_directory, plan)
    return sources


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-plan", type=Path, required=True)
    parser.add_argument("--panel-1-plan", type=Path, required=True)
    parser.add_argument("--panel-1-taskset", type=Path, required=True)
    parser.add_argument("--panel-1-repeat-panel", type=Path, required=True)
    parser.add_argument("--panel-1-run-directory", type=Path, required=True)
    parser.add_argument("--panel-1-base-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-1-source-plan-v44", type=Path, required=True)
    parser.add_argument("--panel-1-qwen-plan-v45", type=Path, required=True)
    parser.add_argument("--panel-1-qwen-run-directory", type=Path, required=True)
    parser.add_argument("--panel-1-fable-run-directory", type=Path)
    parser.add_argument("--panel-1-prior-plan-v50", type=Path)
    parser.add_argument("--panel-1-coverage-repair-run-directory", type=Path)
    parser.add_argument("--panel-1-coverage-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-1-prior-plan-v55", type=Path)
    parser.add_argument("--panel-1-deepseek-repair-run-directory", type=Path)
    parser.add_argument("--panel-1-prior-deepseek-run-directory", type=Path)
    parser.add_argument("--panel-1-deepseek-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-1-prior-plan-v62", type=Path)
    parser.add_argument("--panel-1-prior-plan-v65", type=Path)
    parser.add_argument("--panel-1-glm53-run-directory", type=Path)
    parser.add_argument("--panel-1-glm53-repair-run-directory", type=Path, action="append")
    parser.add_argument("--panel-2-plan", type=Path, required=True)
    parser.add_argument("--panel-2-taskset", type=Path, required=True)
    parser.add_argument("--panel-2-repeat-panel", type=Path, required=True)
    parser.add_argument("--panel-2-run-directory", type=Path, required=True)
    parser.add_argument("--panel-2-base-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-2-source-plan-v49", type=Path)
    parser.add_argument("--panel-2-luna-run-directory", type=Path)
    parser.add_argument("--panel-2-deepseek-flash-run-directory", type=Path)
    parser.add_argument("--panel-2-prior-replacement-plan-v52", type=Path)
    parser.add_argument("--panel-2-coverage-repair-run-directory", type=Path)
    parser.add_argument("--panel-2-coverage-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-2-prior-plan-v54", type=Path)
    parser.add_argument("--panel-2-deepseek-repair-run-directory", type=Path)
    parser.add_argument("--panel-2-prior-deepseek-run-directory", type=Path)
    parser.add_argument("--panel-2-deepseek-completion-run-directory", type=Path, action="append")
    parser.add_argument("--panel-2-prior-plan-v63", type=Path)
    parser.add_argument("--panel-2-prior-plan-v66", type=Path)
    parser.add_argument("--panel-2-glm53-run-directory", type=Path)
    parser.add_argument("--panel-2-glm53-repair-run-directory", type=Path, action="append")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)

    joint_plan = _load(args.joint_plan)
    panel_1_plan = _load(args.panel_1_plan)
    panel_1_taskset = _load(args.panel_1_taskset)
    panel_1_repeat = _load(args.panel_1_repeat_panel)
    panel_2_plan = _load(args.panel_2_plan)
    panel_2_taskset = _load(args.panel_2_taskset)
    panel_2_repeat = _load(args.panel_2_repeat_panel)
    source_plan_v44 = _load(args.panel_1_source_plan_v44)
    qwen_plan_v45 = _load(args.panel_1_qwen_plan_v45)

    joint_v51 = verify_joint_plan_v51(joint_plan)
    joint_v53 = verify_joint_plan_v53(joint_plan)
    joint_v56 = verify_joint_plan_v56(joint_plan)
    joint_v60 = verify_joint_plan_v60(joint_plan)
    joint_v64 = verify_joint_plan_v64(joint_plan)
    joint_v67 = verify_joint_plan_v67(joint_plan)
    joint_v76 = verify_joint_plan_v76(joint_plan)
    joint_v77 = verify_joint_plan_v77(joint_plan)
    joint_v78 = verify_joint_plan_v78(joint_plan)
    joint_price_lineage = joint_v77 or joint_v78
    joint_deepseek = joint_v60 or joint_v64 or joint_v67 or joint_v76 or joint_price_lineage
    joint_glm = joint_v67 or joint_v76 or joint_price_lineage
    if not (verify_joint_plan(joint_plan) or joint_v51 or joint_v53 or joint_v56 or joint_deepseek):
        raise SelectionPoweredAnalysisError("joint plan failed semantic verification")
    expected_panels = (
        verify_plan_v74(panel_1_plan) and verify_plan_v75(panel_2_plan)
        if (joint_v76 or joint_price_lineage)
        else verify_plan_v65(panel_1_plan) and verify_plan_v66(panel_2_plan)
        if joint_v67
        else verify_plan_v62(panel_1_plan) and verify_plan_v63(panel_2_plan)
        if joint_v64
        else verify_plan_v58(panel_1_plan) and verify_plan_v59(panel_2_plan)
        if joint_v60
        else verify_plan_v55(panel_1_plan) and verify_plan_v54(panel_2_plan)
        if joint_v56
        else verify_plan_v50(panel_1_plan) and verify_plan_v52(panel_2_plan)
        if joint_v53
        else verify_plan_v50(panel_1_plan) and verify_plan_v49(panel_2_plan)
        if joint_v51
        else verify_plan_v47(panel_1_plan) and verify_plan_v46(panel_2_plan)
    )
    if not expected_panels:
        raise SelectionPoweredAnalysisError("panel analysis plan failed verification")
    if not verify_taskset(panel_1_taskset) or not verify_repeat_panel(
        panel_1_repeat, taskset=panel_1_taskset
    ):
        raise SelectionPoweredAnalysisError("panel 1 task inputs failed verification")
    if not verify_taskset_replication_2(panel_2_taskset) or not verify_repeat_panel_replication_2(
        panel_2_repeat, taskset=panel_2_taskset
    ):
        raise SelectionPoweredAnalysisError("panel 2 task inputs failed verification")
    for label, document, path in (
        ("panel_1_plan", panel_1_plan, args.panel_1_plan),
        ("panel_1_taskset", panel_1_taskset, args.panel_1_taskset),
        ("panel_1_repeat_panel", panel_1_repeat, args.panel_1_repeat_panel),
        ("panel_2_plan", panel_2_plan, args.panel_2_plan),
        ("panel_2_taskset", panel_2_taskset, args.panel_2_taskset),
        ("panel_2_repeat_panel", panel_2_repeat, args.panel_2_repeat_panel),
    ):
        _assert_pin(plan=joint_plan, label=label, document=document, path=path)

    prior_plan_v50 = (
        _load(args.panel_1_prior_plan_v50) if args.panel_1_prior_plan_v50 is not None else None
    )
    prior_plan_v55 = (
        _load(args.panel_1_prior_plan_v55) if args.panel_1_prior_plan_v55 is not None else None
    )
    prior_plan_v65 = (
        _load(args.panel_1_prior_plan_v65) if args.panel_1_prior_plan_v65 is not None else None
    )
    panel_1_lineage_plan = panel_1_plan
    panel_1_glm_plan = panel_1_plan
    if joint_v76 or joint_price_lineage:
        if (
            args.panel_1_prior_plan_v65 is None
            or args.panel_1_prior_plan_v62 is None
            or args.panel_1_glm53_run_directory is None
            or prior_plan_v65 is None
            or not verify_plan_v65(prior_plan_v65)
        ):
            raise SelectionPoweredAnalysisError(
                "v76/v77 requires the exact panel-1 v65/v62 predecessors and GLM-5.3 run"
            )
        if panel_1_plan["inputs"]["plan_v65_predecessor"] != {
            "semantic_sha256": prior_plan_v65["artifact_sha256"],
            "physical_sha256": _sha256_file(args.panel_1_prior_plan_v65),
        }:
            raise SelectionPoweredAnalysisError("panel-1 v65 predecessor pin differs")
        panel_1_lineage_plan = _load(args.panel_1_prior_plan_v62)
        panel_1_glm_plan = prior_plan_v65
        if not verify_plan_v62(panel_1_lineage_plan) or prior_plan_v65["inputs"][
            "plan_v62_predecessor"
        ] != {
            "semantic_sha256": panel_1_lineage_plan["artifact_sha256"],
            "physical_sha256": _sha256_file(args.panel_1_prior_plan_v62),
        }:
            raise SelectionPoweredAnalysisError("panel-1 GLM predecessor pin differs")
    elif joint_v67:
        if args.panel_1_prior_plan_v62 is None or args.panel_1_glm53_run_directory is None:
            raise SelectionPoweredAnalysisError(
                "v67 requires the exact panel-1 v62 predecessor and GLM-5.3 run"
            )
        panel_1_lineage_plan = _load(args.panel_1_prior_plan_v62)
        if not verify_plan_v62(panel_1_lineage_plan) or panel_1_plan["inputs"][
            "plan_v62_predecessor"
        ] != {
            "semantic_sha256": panel_1_lineage_plan["artifact_sha256"],
            "physical_sha256": _sha256_file(args.panel_1_prior_plan_v62),
        }:
            raise SelectionPoweredAnalysisError("panel-1 GLM predecessor pin differs")
    elif (
        args.panel_1_prior_plan_v62 is not None
        or args.panel_1_prior_plan_v65 is not None
        or args.panel_1_glm53_run_directory is not None
        or args.panel_1_glm53_repair_run_directory
    ):
        raise SelectionPoweredAnalysisError("GLM panel-1 sources require joint plan v67 or v76")
    if joint_price_lineage:
        if args.panel_1_prior_deepseek_run_directory is None:
            raise SelectionPoweredAnalysisError(
                "v77 requires the panel-1 prior DeepSeek response directory"
            )
        assert args.panel_1_prior_plan_v62 is not None
        _assert_pin(
            plan=joint_plan,
            label="panel_1_prior_deepseek_plan_v62",
            document=panel_1_lineage_plan,
            path=args.panel_1_prior_plan_v62,
        )
    elif (
        args.panel_1_prior_deepseek_run_directory is not None
        or args.panel_1_deepseek_completion_run_directory
    ):
        raise SelectionPoweredAnalysisError("price-lineage sources require joint plan v77")
    panel_1_coverage_args = (
        args.panel_1_prior_plan_v50,
        args.panel_1_coverage_repair_run_directory,
    )
    panel_1_deepseek_args = (
        args.panel_1_prior_plan_v55,
        args.panel_1_deepseek_repair_run_directory,
    )
    if joint_deepseek:
        if any(value is None for value in panel_1_coverage_args + panel_1_deepseek_args):
            raise SelectionPoweredAnalysisError(
                "joint DeepSeek repair requires panel-1 v50/v55 predecessors and both repair runs"
            )
        if args.panel_1_fable_run_directory is not None:
            raise SelectionPoweredAnalysisError(
                "joint DeepSeek repair supersedes the separate panel-1 Fable run"
            )
    elif joint_v56:
        if any(value is None for value in panel_1_coverage_args):
            raise SelectionPoweredAnalysisError(
                "v56 requires the panel-1 v50 predecessor and complete coverage-repair run"
            )
        if args.panel_1_fable_run_directory is not None:
            raise SelectionPoweredAnalysisError(
                "v56 supersedes the separate panel-1 Fable run with the coverage-repair block"
            )
        if any(value is not None for value in panel_1_deepseek_args):
            raise SelectionPoweredAnalysisError("DeepSeek repair sources require a joint repair")
    elif any(value is not None for value in panel_1_coverage_args + panel_1_deepseek_args):
        raise SelectionPoweredAnalysisError("panel-1 coverage-repair sources require v56")
    model_sources = _panel_1_model_sources(
        plan=panel_1_lineage_plan,
        run_directory=args.panel_1_run_directory,
        source_plan=source_plan_v44,
        source_plan_path=args.panel_1_source_plan_v44,
        qwen_plan=qwen_plan_v45,
        qwen_plan_path=args.panel_1_qwen_plan_v45,
        qwen_run_directory=args.panel_1_qwen_run_directory,
        fable_run_directory=args.panel_1_fable_run_directory,
        prior_plan_v50=prior_plan_v50,
        prior_plan_v50_path=args.panel_1_prior_plan_v50,
        repair_run_directory=args.panel_1_coverage_repair_run_directory,
        base_completion_run_directories=(args.panel_1_base_completion_run_directory or []),
        coverage_completion_run_directories=(args.panel_1_coverage_completion_run_directory or []),
        prior_plan_v55=prior_plan_v55,
        prior_plan_v55_path=args.panel_1_prior_plan_v55,
        deepseek_repair_run_directory=args.panel_1_deepseek_repair_run_directory,
    )
    if joint_glm:
        assert args.panel_1_glm53_run_directory is not None
        model_sources["z-ai/glm-5.3"] = (
            (
                args.panel_1_glm53_run_directory,
                *(args.panel_1_glm53_repair_run_directory or []),
            ),
            panel_1_glm_plan,
        )
    if joint_price_lineage:
        assert args.panel_1_deepseek_repair_run_directory is not None
        assert args.panel_1_prior_deepseek_run_directory is not None
        model_sources[DEEPSEEK_PRO_MODEL_ID] = (
            (args.panel_1_prior_deepseek_run_directory, panel_1_lineage_plan),
            (args.panel_1_deepseek_repair_run_directory, panel_1_plan),
            *(
                (directory, panel_1_plan)
                for directory in (args.panel_1_deepseek_completion_run_directory or [])
            ),
        )
    elif joint_v76:
        assert args.panel_1_deepseek_repair_run_directory is not None
        model_sources[DEEPSEEK_PRO_MODEL_ID] = (
            args.panel_1_deepseek_repair_run_directory,
            panel_1_plan,
        )
    primary_1 = load_panel(
        run_directory=args.panel_1_run_directory,
        panel="primary",
        plan=panel_1_plan,
        taskset=panel_1_taskset,
        repeat_panel=panel_1_repeat,
        model_sources=model_sources,
        allowed_source_roster_differences=(
            {DEEPSEEK_PRO_MODEL_ID: frozenset({"endpoint_sha256"})} if joint_price_lineage else None
        ),
    )
    repeat_1 = load_panel(
        run_directory=args.panel_1_run_directory,
        panel="repeat",
        plan=panel_1_plan,
        taskset=panel_1_taskset,
        repeat_panel=panel_1_repeat,
        model_sources=model_sources,
        allowed_source_roster_differences=(
            {DEEPSEEK_PRO_MODEL_ID: frozenset({"endpoint_sha256"})} if joint_price_lineage else None
        ),
    )
    panel_2_model_sources = None
    panel_2_source_plan = None
    prior_replacement_plan_v52 = None
    prior_plan_v54 = None
    prior_plan_v66 = None
    panel_2_lineage_plan = panel_2_plan
    panel_2_glm_plan = panel_2_plan
    if joint_v76 or joint_price_lineage:
        if (
            args.panel_2_prior_plan_v66 is None
            or args.panel_2_prior_plan_v63 is None
            or args.panel_2_glm53_run_directory is None
        ):
            raise SelectionPoweredAnalysisError(
                "v76/v77 requires the exact panel-2 v66/v63 predecessors and GLM-5.3 run"
            )
        prior_plan_v66 = _load(args.panel_2_prior_plan_v66)
        if not verify_plan_v66(prior_plan_v66) or panel_2_plan["inputs"][
            "plan_v66_predecessor"
        ] != {
            "semantic_sha256": prior_plan_v66["artifact_sha256"],
            "physical_sha256": _sha256_file(args.panel_2_prior_plan_v66),
        }:
            raise SelectionPoweredAnalysisError("panel-2 v66 predecessor pin differs")
        panel_2_lineage_plan = _load(args.panel_2_prior_plan_v63)
        panel_2_glm_plan = prior_plan_v66
        if not verify_plan_v63(panel_2_lineage_plan) or prior_plan_v66["inputs"][
            "plan_v63_predecessor"
        ] != {
            "semantic_sha256": panel_2_lineage_plan["artifact_sha256"],
            "physical_sha256": _sha256_file(args.panel_2_prior_plan_v63),
        }:
            raise SelectionPoweredAnalysisError("panel-2 GLM predecessor pin differs")
    elif joint_v67:
        if args.panel_2_prior_plan_v63 is None or args.panel_2_glm53_run_directory is None:
            raise SelectionPoweredAnalysisError(
                "v67 requires the exact panel-2 v63 predecessor and GLM-5.3 run"
            )
        panel_2_lineage_plan = _load(args.panel_2_prior_plan_v63)
        if not verify_plan_v63(panel_2_lineage_plan) or panel_2_plan["inputs"][
            "plan_v63_predecessor"
        ] != {
            "semantic_sha256": panel_2_lineage_plan["artifact_sha256"],
            "physical_sha256": _sha256_file(args.panel_2_prior_plan_v63),
        }:
            raise SelectionPoweredAnalysisError("panel-2 GLM predecessor pin differs")
    elif (
        args.panel_2_prior_plan_v63 is not None
        or args.panel_2_prior_plan_v66 is not None
        or args.panel_2_glm53_run_directory is not None
        or args.panel_2_glm53_repair_run_directory
    ):
        raise SelectionPoweredAnalysisError("GLM panel-2 sources require joint plan v67 or v76")
    if joint_price_lineage:
        if args.panel_2_prior_deepseek_run_directory is None:
            raise SelectionPoweredAnalysisError(
                "v77 requires the panel-2 prior DeepSeek response directory"
            )
        assert args.panel_2_prior_plan_v63 is not None
        _assert_pin(
            plan=joint_plan,
            label="panel_2_prior_deepseek_plan_v63",
            document=panel_2_lineage_plan,
            path=args.panel_2_prior_plan_v63,
        )
    elif (
        args.panel_2_prior_deepseek_run_directory is not None
        or args.panel_2_deepseek_completion_run_directory
    ):
        raise SelectionPoweredAnalysisError("price-lineage sources require joint plan v77")
    panel_2_replacement_args = (
        args.panel_2_source_plan_v49,
        args.panel_2_luna_run_directory,
        args.panel_2_deepseek_flash_run_directory,
    )
    panel_2_coverage_args = (
        args.panel_2_prior_replacement_plan_v52,
        args.panel_2_coverage_repair_run_directory,
    )
    panel_2_deepseek_args = (
        args.panel_2_prior_plan_v54,
        args.panel_2_deepseek_repair_run_directory,
    )
    if joint_deepseek:
        if any(
            value is None
            for value in panel_2_replacement_args + panel_2_coverage_args + panel_2_deepseek_args
        ):
            raise SelectionPoweredAnalysisError(
                "joint DeepSeek repair requires panel-2 v49/v52/v54 and both repair runs"
            )
        assert args.panel_2_source_plan_v49 is not None
        assert args.panel_2_luna_run_directory is not None
        assert args.panel_2_deepseek_flash_run_directory is not None
        assert args.panel_2_prior_replacement_plan_v52 is not None
        assert args.panel_2_coverage_repair_run_directory is not None
        assert args.panel_2_prior_plan_v54 is not None
        assert args.panel_2_deepseek_repair_run_directory is not None
        panel_2_source_plan = _load(args.panel_2_source_plan_v49)
        prior_replacement_plan_v52 = _load(args.panel_2_prior_replacement_plan_v52)
        prior_plan_v54 = _load(args.panel_2_prior_plan_v54)
        panel_2_model_sources = _panel_2_model_sources(
            plan=panel_2_lineage_plan,
            run_directory=args.panel_2_run_directory,
            source_plan=panel_2_source_plan,
            source_plan_path=args.panel_2_source_plan_v49,
            luna_run_directory=args.panel_2_luna_run_directory,
            deepseek_flash_run_directory=args.panel_2_deepseek_flash_run_directory,
            prior_replacement_plan_v52=prior_replacement_plan_v52,
            prior_replacement_plan_v52_path=args.panel_2_prior_replacement_plan_v52,
            repair_run_directory=args.panel_2_coverage_repair_run_directory,
            base_completion_run_directories=(args.panel_2_base_completion_run_directory or []),
            coverage_completion_run_directories=(
                args.panel_2_coverage_completion_run_directory or []
            ),
            prior_plan_v54=prior_plan_v54,
            prior_plan_v54_path=args.panel_2_prior_plan_v54,
            deepseek_repair_run_directory=args.panel_2_deepseek_repair_run_directory,
        )
    elif joint_v56:
        if any(value is None for value in panel_2_replacement_args + panel_2_coverage_args):
            raise SelectionPoweredAnalysisError(
                "v56 requires base, v52, and complete coverage-repair panel-2 sources"
            )
        assert args.panel_2_source_plan_v49 is not None
        assert args.panel_2_luna_run_directory is not None
        assert args.panel_2_deepseek_flash_run_directory is not None
        assert args.panel_2_prior_replacement_plan_v52 is not None
        assert args.panel_2_coverage_repair_run_directory is not None
        panel_2_source_plan = _load(args.panel_2_source_plan_v49)
        prior_replacement_plan_v52 = _load(args.panel_2_prior_replacement_plan_v52)
        panel_2_model_sources = _panel_2_model_sources(
            plan=panel_2_lineage_plan,
            run_directory=args.panel_2_run_directory,
            source_plan=panel_2_source_plan,
            source_plan_path=args.panel_2_source_plan_v49,
            luna_run_directory=args.panel_2_luna_run_directory,
            deepseek_flash_run_directory=args.panel_2_deepseek_flash_run_directory,
            prior_replacement_plan_v52=prior_replacement_plan_v52,
            prior_replacement_plan_v52_path=args.panel_2_prior_replacement_plan_v52,
            repair_run_directory=args.panel_2_coverage_repair_run_directory,
            base_completion_run_directories=(args.panel_2_base_completion_run_directory or []),
            coverage_completion_run_directories=(
                args.panel_2_coverage_completion_run_directory or []
            ),
        )
    elif joint_v53:
        if any(value is None for value in panel_2_replacement_args):
            raise SelectionPoweredAnalysisError(
                "v53 requires the v49 source and both complete panel-2 replacement runs"
            )
        if any(value is not None for value in panel_2_coverage_args):
            raise SelectionPoweredAnalysisError("coverage-repair sources require a v56 joint plan")
        assert args.panel_2_source_plan_v49 is not None
        assert args.panel_2_luna_run_directory is not None
        assert args.panel_2_deepseek_flash_run_directory is not None
        panel_2_source_plan = _load(args.panel_2_source_plan_v49)
        panel_2_model_sources = _panel_2_model_sources(
            plan=panel_2_lineage_plan,
            run_directory=args.panel_2_run_directory,
            source_plan=panel_2_source_plan,
            source_plan_path=args.panel_2_source_plan_v49,
            luna_run_directory=args.panel_2_luna_run_directory,
            deepseek_flash_run_directory=args.panel_2_deepseek_flash_run_directory,
            base_completion_run_directories=(args.panel_2_base_completion_run_directory or []),
        )
        if any(value is not None for value in panel_2_deepseek_args):
            raise SelectionPoweredAnalysisError("DeepSeek repair sources require a joint repair")
    elif any(
        value is not None
        for value in panel_2_replacement_args + panel_2_coverage_args + panel_2_deepseek_args
    ):
        raise SelectionPoweredAnalysisError(
            "panel-2 replacement sources require a v53 or v56 joint plan"
        )
    if joint_glm:
        assert args.panel_2_glm53_run_directory is not None
        assert panel_2_model_sources is not None
        panel_2_model_sources["z-ai/glm-5.3"] = (
            (
                args.panel_2_glm53_run_directory,
                *(args.panel_2_glm53_repair_run_directory or []),
            ),
            panel_2_glm_plan,
        )
    if joint_price_lineage:
        assert args.panel_2_deepseek_repair_run_directory is not None
        assert args.panel_2_prior_deepseek_run_directory is not None
        assert panel_2_model_sources is not None
        panel_2_model_sources[DEEPSEEK_PRO_MODEL_ID] = (
            (args.panel_2_prior_deepseek_run_directory, panel_2_lineage_plan),
            (args.panel_2_deepseek_repair_run_directory, panel_2_plan),
            *(
                (directory, panel_2_plan)
                for directory in (args.panel_2_deepseek_completion_run_directory or [])
            ),
        )
    elif joint_v76:
        assert args.panel_2_deepseek_repair_run_directory is not None
        assert panel_2_model_sources is not None
        panel_2_model_sources[DEEPSEEK_PRO_MODEL_ID] = (
            args.panel_2_deepseek_repair_run_directory,
            panel_2_plan,
        )

    primary_2 = load_panel(
        run_directory=args.panel_2_run_directory,
        panel="primary",
        plan=panel_2_plan,
        taskset=panel_2_taskset,
        repeat_panel=panel_2_repeat,
        model_sources=panel_2_model_sources,
        allowed_source_roster_differences=(
            {DEEPSEEK_PRO_MODEL_ID: frozenset({"endpoint_sha256"})} if joint_price_lineage else None
        ),
    )
    repeat_2 = load_panel(
        run_directory=args.panel_2_run_directory,
        panel="repeat",
        plan=panel_2_plan,
        taskset=panel_2_taskset,
        repeat_panel=panel_2_repeat,
        model_sources=panel_2_model_sources,
        allowed_source_roster_differences=(
            {DEEPSEEK_PRO_MODEL_ID: frozenset({"endpoint_sha256"})} if joint_price_lineage else None
        ),
    )
    coverage_diagnostics: list[dict[str, Any]] = []
    if joint_v78:
        ranked_model_ids = tuple(
            str(value) for value in joint_plan["eligibility"]["ranked_model_ids"]
        )
        diagnostic_model_ids = tuple(
            str(value) for value in joint_plan["eligibility"]["coverage_diagnostic_model_ids"]
        )
        for model_id in diagnostic_model_ids:
            row: dict[str, Any] = {
                "model_id": model_id,
                "score_status": "coverage_diagnostic_not_ranked",
                "flavourbench_score_reported": False,
                "panels": {},
            }
            for label, primary_panel, repeat_panel in (
                ("panel_1", primary_1, repeat_1),
                ("panel_2", primary_2, repeat_2),
            ):
                index = primary_panel.model_ids.index(model_id)
                repeat_index = repeat_panel.model_ids.index(model_id)
                primary_valid = primary_panel.completed[index] & primary_panel.parseable[index]
                repeat_valid = (
                    repeat_panel.completed[repeat_index] & repeat_panel.parseable[repeat_index]
                )
                row["panels"][label] = {
                    "primary_scheduled": len(primary_panel.task_ids),
                    "primary_completed": int(primary_panel.completed[index].sum()),
                    "primary_parseable": int(primary_panel.parseable[index].sum()),
                    "primary_valid": int(primary_valid.sum()),
                    "repeat_scheduled": len(repeat_panel.task_ids),
                    "repeat_completed": int(repeat_panel.completed[repeat_index].sum()),
                    "repeat_parseable": int(repeat_panel.parseable[repeat_index].sum()),
                    "repeat_valid": int(repeat_valid.sum()),
                }
            coverage_diagnostics.append(row)
        for panel_data in (primary_1, repeat_1, primary_2, repeat_2):
            for model_id in ranked_model_ids:
                index = panel_data.model_ids.index(model_id)
                if not np.all(panel_data.completed[index] & panel_data.parseable[index]):
                    raise SelectionPoweredAnalysisError(
                        "ranked model lacks complete valid coverage: "
                        f"{model_id} in {panel_data.panel}"
                    )
        primary_1 = subset_panel_data(primary_1, ranked_model_ids)
        repeat_1 = subset_panel_data(repeat_1, ranked_model_ids)
        primary_2 = subset_panel_data(primary_2, ranked_model_ids)
        repeat_2 = subset_panel_data(repeat_2, ranked_model_ids)
    primary = combine_panel_data(primary_1, primary_2, panel="joint_primary")
    repeat = combine_panel_data(repeat_1, repeat_2, panel="joint_repeat")
    combined_tasks = list(panel_1_taskset["tasks"]) + list(panel_2_taskset["tasks"])
    combined_repeats = list(panel_1_repeat["tasks"]) + list(panel_2_repeat["tasks"])
    task_by_id = {str(task["task_id"]): task for task in combined_tasks}
    cluster_ids = tuple(str(task["anchor_ingredient"]) for task in combined_tasks)
    repeat_cluster_ids = tuple(
        str(task_by_id[str(task["original_task_id"])]["anchor_ingredient"])
        for task in combined_repeats
    )
    if len(set(cluster_ids)) != int(joint_plan["design"]["unique_anchor_clusters"]):
        raise SelectionPoweredAnalysisError("joint anchor-cluster cardinality drifted")

    analysis = analyze_panels(
        primary=primary,
        repeat=repeat,
        taskset={"tasks": combined_tasks},
        repeat_panel={"tasks": combined_repeats},
        plan=joint_plan,
        cluster_ids=cluster_ids,
        repeat_cluster_ids=repeat_cluster_ids,
    )
    analysis["schema_version"] = ANALYSIS_SCHEMA_VERSION
    analysis["panel_replication"] = replication_stability(primary_1, primary_2)
    analysis["design"] = dict(joint_plan["design"])
    analysis["coverage_diagnostics"] = coverage_diagnostics
    analysis["ranked_model_count"] = len(primary.model_ids)
    analysis["dnf_rows_emitted"] = False

    leaderboard_bytes = _leaderboard_csv(analysis)
    pairwise_bytes = _pairwise_csv(analysis)
    leaderboard_path = _write_content_addressed(
        args.output_directory, "flavourbench-joint-leaderboard-table", leaderboard_bytes
    )
    pairwise_path = _write_content_addressed(
        args.output_directory, "flavourbench-joint-pairwise-table", pairwise_bytes
    )
    release: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": analysis["status"],
        "benchmark": "FlavourBench",
        "track": "Epicure-scored combinatorial culinary decisions",
        "inputs": {
            "joint_plan": {
                "semantic_sha256": joint_plan["artifact_sha256"],
                "physical_sha256": _sha256_file(args.joint_plan),
            },
            "panel_1_plan": {
                "semantic_sha256": panel_1_plan["artifact_sha256"],
                "physical_sha256": _sha256_file(args.panel_1_plan),
            },
            "panel_1_taskset": {
                "semantic_sha256": panel_1_taskset["artifact_sha256"],
                "physical_sha256": _sha256_file(args.panel_1_taskset),
            },
            "panel_1_repeat_panel": {
                "semantic_sha256": panel_1_repeat["artifact_sha256"],
                "physical_sha256": _sha256_file(args.panel_1_repeat_panel),
            },
            "panel_2_plan": {
                "semantic_sha256": panel_2_plan["artifact_sha256"],
                "physical_sha256": _sha256_file(args.panel_2_plan),
            },
            "panel_2_taskset": {
                "semantic_sha256": panel_2_taskset["artifact_sha256"],
                "physical_sha256": _sha256_file(args.panel_2_taskset),
            },
            "panel_2_repeat_panel": {
                "semantic_sha256": panel_2_repeat["artifact_sha256"],
                "physical_sha256": _sha256_file(args.panel_2_repeat_panel),
            },
            "panel_1_primary": {
                "count": len(primary_1.response_artifact_sha256s),
                "artifact_set_sha256": _sha256(list(primary_1.response_artifact_sha256s)),
                "spend_micros": primary_1.spend_micros,
            },
            "panel_1_repeat": {
                "count": len(repeat_1.response_artifact_sha256s),
                "artifact_set_sha256": _sha256(list(repeat_1.response_artifact_sha256s)),
                "spend_micros": repeat_1.spend_micros,
            },
            "panel_2_primary": {
                "count": len(primary_2.response_artifact_sha256s),
                "artifact_set_sha256": _sha256(list(primary_2.response_artifact_sha256s)),
                "spend_micros": primary_2.spend_micros,
            },
            "panel_2_repeat": {
                "count": len(repeat_2.response_artifact_sha256s),
                "artifact_set_sha256": _sha256(list(repeat_2.response_artifact_sha256s)),
                "spend_micros": repeat_2.spend_micros,
            },
            "response_lineage": {
                "panel_1_base_plan_sha256": source_plan_v44["artifact_sha256"],
                "panel_1_qwen_replacement_plan_sha256": qwen_plan_v45["artifact_sha256"],
                "panel_1_superseded_qwen_responses_used": False,
                "panel_1_base_response_selection_rule": (
                    "first_completed_parseable_response_in_frozen_source_directory_order"
                ),
                "panel_1_base_completion_source_directory_count": (
                    1 + len(args.panel_1_base_completion_run_directory or [])
                ),
                "panel_1_fable_replacement_plan_sha256": (
                    panel_1_plan["artifact_sha256"] if (joint_v51 or joint_v53) else None
                ),
                "panel_1_superseded_fable_replacement_plan_sha256": (
                    prior_plan_v50["artifact_sha256"]
                    if (joint_v56 or joint_deepseek) and prior_plan_v50 is not None
                    else None
                ),
                "panel_1_superseded_fable_responses_used": False,
                "panel_1_prior_plan_v50_sha256": (
                    prior_plan_v50["artifact_sha256"] if prior_plan_v50 is not None else None
                ),
                "panel_1_coverage_repair_plan_sha256": (
                    prior_plan_v55["artifact_sha256"]
                    if joint_deepseek and prior_plan_v55 is not None
                    else panel_1_plan["artifact_sha256"]
                    if joint_v56
                    else None
                ),
                "panel_1_coverage_repair_model_ids": (
                    COVERAGE_REPAIR_MODEL_IDS if (joint_v56 or joint_deepseek) else []
                ),
                "panel_1_superseded_coverage_route_responses_used": False,
                "panel_1_coverage_response_selection_rule": (
                    "first_completed_parseable_response_in_frozen_source_directory_order"
                    if args.panel_1_coverage_completion_run_directory
                    else None
                ),
                "panel_1_coverage_completion_source_directory_count": (
                    1 + len(args.panel_1_coverage_completion_run_directory or [])
                    if (joint_v56 or joint_deepseek)
                    else 0
                ),
                "panel_1_coverage_failed_response_artifacts_preserved": bool(
                    args.panel_1_coverage_completion_run_directory
                ),
                "panel_1_coverage_failed_response_artifacts_used_as_score_data": False,
                "panel_1_deepseek_repair_plan_sha256": (
                    panel_1_plan["artifact_sha256"]
                    if (joint_v76 or joint_price_lineage)
                    else panel_1_lineage_plan["artifact_sha256"]
                    if joint_deepseek
                    else None
                ),
                "panel_1_deepseek_repair_model_ids": (
                    [DEEPSEEK_PRO_MODEL_ID] if joint_deepseek else []
                ),
                "panel_1_deepseek_repair_provider_tag": (
                    next(
                        row["provider_tag"]
                        for row in panel_1_plan["roster"]["models"]
                        if row["model_id"] == DEEPSEEK_PRO_MODEL_ID
                    )
                    if joint_deepseek
                    else None
                ),
                "panel_1_superseded_deepseek_route_responses_used": False,
                "panel_1_prior_price_contract_deepseek_responses_used": joint_price_lineage,
                "panel_1_deepseek_response_selection_rule": (
                    "first_completed_parseable_response_in_frozen_source_directory_order"
                    if joint_price_lineage
                    else None
                ),
                "panel_1_deepseek_source_directory_count": (
                    2 + len(args.panel_1_deepseek_completion_run_directory or [])
                    if joint_price_lineage
                    else 1
                    if joint_v76
                    else 0
                ),
                "panel_2_plan_sha256": panel_2_plan["artifact_sha256"],
                "panel_2_base_plan_sha256": (
                    panel_2_source_plan["artifact_sha256"]
                    if panel_2_source_plan is not None
                    else panel_2_plan["artifact_sha256"]
                ),
                "panel_2_reuses_panel_1_responses": False,
                "panel_2_base_response_selection_rule": (
                    "first_completed_parseable_response_in_frozen_source_directory_order"
                ),
                "panel_2_base_completion_source_directory_count": (
                    1 + len(args.panel_2_base_completion_run_directory or [])
                ),
                "panel_2_replacement_plan_sha256": (
                    (
                        prior_replacement_plan_v52["artifact_sha256"]
                        if prior_replacement_plan_v52 is not None
                        else panel_2_plan["artifact_sha256"]
                    )
                    if (joint_v53 or joint_v56 or joint_deepseek)
                    else None
                ),
                "panel_2_replacement_model_ids": (
                    [LUNA_MODEL_ID, DEEPSEEK_FLASH_MODEL_ID]
                    if (joint_v53 or joint_v56 or joint_deepseek)
                    else []
                ),
                "panel_2_coverage_repair_plan_sha256": (
                    prior_plan_v54["artifact_sha256"]
                    if joint_deepseek and prior_plan_v54 is not None
                    else panel_2_plan["artifact_sha256"]
                    if joint_v56
                    else None
                ),
                "panel_2_coverage_repair_model_ids": (
                    COVERAGE_REPAIR_MODEL_IDS if (joint_v56 or joint_deepseek) else []
                ),
                "panel_2_superseded_coverage_route_responses_used": False,
                "panel_2_coverage_response_selection_rule": (
                    "first_completed_parseable_response_in_frozen_source_directory_order"
                    if args.panel_2_coverage_completion_run_directory
                    else None
                ),
                "panel_2_coverage_completion_source_directory_count": (
                    1 + len(args.panel_2_coverage_completion_run_directory or [])
                    if (joint_v56 or joint_deepseek)
                    else 0
                ),
                "panel_2_coverage_failed_response_artifacts_preserved": bool(
                    args.panel_2_coverage_completion_run_directory
                ),
                "panel_2_coverage_failed_response_artifacts_used_as_score_data": False,
                "panel_2_deepseek_repair_plan_sha256": (
                    panel_2_plan["artifact_sha256"]
                    if (joint_v76 or joint_price_lineage)
                    else panel_2_lineage_plan["artifact_sha256"]
                    if joint_deepseek
                    else None
                ),
                "panel_2_deepseek_repair_model_ids": (
                    [DEEPSEEK_PRO_MODEL_ID] if joint_deepseek else []
                ),
                "panel_2_deepseek_repair_provider_tag": (
                    next(
                        row["provider_tag"]
                        for row in panel_2_plan["roster"]["models"]
                        if row["model_id"] == DEEPSEEK_PRO_MODEL_ID
                    )
                    if joint_deepseek
                    else None
                ),
                "panel_2_superseded_deepseek_route_responses_used": False,
                "panel_2_prior_price_contract_deepseek_responses_used": joint_price_lineage,
                "panel_2_deepseek_response_selection_rule": (
                    "first_completed_parseable_response_in_frozen_source_directory_order"
                    if joint_price_lineage
                    else None
                ),
                "panel_2_deepseek_source_directory_count": (
                    2 + len(args.panel_2_deepseek_completion_run_directory or [])
                    if joint_price_lineage
                    else 1
                    if joint_v76
                    else 0
                ),
                "deepseek_cross_contract_allowed_roster_differences": (
                    ["endpoint_sha256"] if joint_price_lineage else []
                ),
                "deepseek_cross_contract_difference_class": (
                    "price_metadata_only" if joint_price_lineage else None
                ),
                "deepseek_coverage_and_parseability_inspected_before_source_freeze": (
                    joint_price_lineage
                ),
                "deepseek_quality_scores_inspected_before_source_freeze": False,
                "panel_2_superseded_route_responses_used": False,
                "glm53_limited_run_model_ids": ["z-ai/glm-5.3"] if joint_glm else [],
                "glm53_panel_1_plan_sha256": (
                    panel_1_glm_plan["artifact_sha256"] if joint_glm else None
                ),
                "glm53_panel_2_plan_sha256": (
                    panel_2_glm_plan["artifact_sha256"] if joint_glm else None
                ),
                "glm53_finite_cli_only": joint_glm,
                "glm53_standing_service": False,
                "glm53_automatic_fallback": False,
                "glm53_response_selection_rule": (
                    "first_completed_parseable_response_in_frozen_source_directory_order"
                    if joint_glm
                    else None
                ),
                "glm53_failed_response_artifacts_preserved": joint_glm,
                "glm53_failed_response_artifacts_used_as_score_data": False,
                "glm53_panel_1_source_directory_count": (
                    1 + len(args.panel_1_glm53_repair_run_directory or []) if joint_glm else 0
                ),
                "glm53_panel_2_source_directory_count": (
                    1 + len(args.panel_2_glm53_repair_run_directory or []) if joint_glm else 0
                ),
            },
        },
        "tables": {
            "leaderboard": {
                "filename": leaderboard_path.name,
                "sha256": _sha256_bytes(leaderboard_bytes),
            },
            "pairwise": {
                "filename": pairwise_path.name,
                "sha256": _sha256_bytes(pairwise_bytes),
            },
        },
        "analysis": analysis,
        "claim_boundary": joint_plan["claim_boundary"],
    }
    release["artifact_sha256"] = _sha256(release)
    release_bytes = (
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    release_path = _write_content_addressed(
        args.output_directory,
        "flavourbench-joint-powered-release",
        release_bytes,
        address=release["artifact_sha256"],
    )
    print(
        json.dumps(
            {
                "release": str(release_path),
                "artifact_sha256": release["artifact_sha256"],
                "leaderboard": str(leaderboard_path),
                "pairwise": str(pairwise_path),
                "status": release["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
