"""Exact response-source graph for the final 27-model common-core release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .epicure_selection_powered_analysis import PanelData, load_panel
from .epicure_selection_powered_joint_analysis_v1 import (
    _panel_1_model_sources,
    _panel_2_model_sources,
)
from .epicure_selection_route_manifest_v57 import DEEPSEEK_PRO_MODEL_ID
from .selection_response_parser_v3 import score_answer_v3


@dataclass(frozen=True)
class CompleteCoreSourceGraph:
    panel_1_plan: dict[str, Any]
    panel_1_plan_path: Path
    panel_1_taskset: dict[str, Any]
    panel_1_taskset_path: Path
    panel_1_repeat: dict[str, Any]
    panel_1_repeat_path: Path
    panel_1_model_sources: dict[str, Any]
    panel_2_plan: dict[str, Any]
    panel_2_plan_path: Path
    panel_2_taskset: dict[str, Any]
    panel_2_taskset_path: Path
    panel_2_repeat: dict[str, Any]
    panel_2_repeat_path: Path
    panel_2_model_sources: dict[str, Any]


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"frozen source is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"frozen source is not a JSON object: {path}")
    return value


def source_graph(root: Path) -> CompleteCoreSourceGraph:
    """Return the explicit, ordered response-source graph rooted at a repository checkout."""

    root = root.resolve()
    taskset_1_path = root / (
        "benchmark/powered-v44/taskset/epicure-selection-taskset-"
        "a33bf28db372090015118371417b0e8ed1254f416d03d2c2c5816a6a752beb41.json"
    )
    repeat_1_path = root / (
        "benchmark/powered-v44/plan/epicure-selection-repeat-panel-"
        "96f766df855b93ad1495ec386c70ad88e42c4f896be24b0538cf1084da3c124a.json"
    )
    plan_1_path = root / (
        "benchmark/powered-v74/plan/epicure-selection-analysis-plan-"
        "55fef84440b6dda3db7ad44ba3947a3b2d81f127c2ad81fce7e6fbd38c8df6c0.json"
    )
    source_plan_1_path = root / (
        "benchmark/powered-v44/plan/epicure-selection-analysis-plan-"
        "dd74a82d4a34500f22ed91178f63497486fd957e67ebc0136bfa3350d3f6d57e.json"
    )
    qwen_plan_path = root / (
        "benchmark/powered-v45/plan/epicure-selection-analysis-plan-"
        "5acce6f61f731a1f823f314694b7da7c6d9fc02b500e5a9eaaff3002e7096acb.json"
    )
    prior_1_v50_path = root / (
        "benchmark/powered-v50/plan/epicure-selection-analysis-plan-"
        "ffd464743c220643b6db285db1c98122b5b7417676bd157d51497fc0e27e4da0.json"
    )
    prior_1_v55_path = root / (
        "benchmark/powered-v55/plan/epicure-selection-analysis-plan-"
        "8577a9a32c5fb266f12b131c309f4543c6fa2cd42538abd16eefbf4c09d578ed.json"
    )
    prior_1_v62_path = root / (
        "benchmark/powered-v62/plan/epicure-selection-analysis-plan-"
        "1c0e229509b35d6cfb988f1abc1985bdb69c143ecbe8b9508cc6dadda91d65a1.json"
    )
    glm_1_path = root / (
        "benchmark/powered-v65/plan/epicure-selection-analysis-plan-"
        "21fad9ca2d5c942578ed5cdd067cf5b1572b19bf6936aaf1a807678a0c0f2ee1.json"
    )
    plan_1 = _load(plan_1_path)
    source_plan_1 = _load(source_plan_1_path)
    qwen_plan = _load(qwen_plan_path)
    prior_1_v50 = _load(prior_1_v50_path)
    prior_1_v55 = _load(prior_1_v55_path)
    prior_1_v62 = _load(prior_1_v62_path)
    glm_1 = _load(glm_1_path)
    base_completion_1 = tuple(
        root / f"benchmark/powered-v77/run-p1-{name}-format-completion-{index}"
        for name, maximum in (
            ("glm52", 3),
            ("kimi", 2),
            ("llama", 2),
            ("mistral", 2),
        )
        for index in range(1, maximum + 1)
    )
    coverage_completion_1 = (
        *(root / f"benchmark/powered-v68/run-p1-glimmer-coverage-completion-{i}" for i in (1, 2)),
        *(root / f"benchmark/powered-v68/run-p1-hy3-coverage-completion-{i}" for i in (1, 2)),
        *(root / f"benchmark/powered-v68/run-p1-minimax-coverage-completion-{i}" for i in (1, 2)),
        *(root / f"benchmark/powered-v68/run-p1-nemotron-coverage-completion-{i}" for i in (1, 2)),
        root / "benchmark/powered-v77/run-p1-fable-coverage-completion-1",
        root / "benchmark/powered-v77/run-p1-nemotron-coverage-completion-3",
    )
    sources_1 = _panel_1_model_sources(
        plan=prior_1_v62,
        run_directory=root / "benchmark/powered-v44/run",
        source_plan=source_plan_1,
        source_plan_path=source_plan_1_path,
        qwen_plan=qwen_plan,
        qwen_plan_path=qwen_plan_path,
        qwen_run_directory=root / "benchmark/powered-v45/run",
        fable_run_directory=None,
        prior_plan_v50=prior_1_v50,
        prior_plan_v50_path=prior_1_v50_path,
        repair_run_directory=root / "benchmark/powered-v55/run-panel1-repair",
        base_completion_run_directories=base_completion_1,
        coverage_completion_run_directories=coverage_completion_1,
        prior_plan_v55=prior_1_v55,
        prior_plan_v55_path=prior_1_v55_path,
        deepseek_repair_run_directory=root / "benchmark/powered-v62/run-panel1-deepseek-repair",
    )
    sources_1["z-ai/glm-5.3"] = (
        (
            root / "benchmark/powered-v65/run-panel1-glm53",
            root / "benchmark/powered-v77/run-p1-glm53-format-completion-1",
            root / "benchmark/powered-v77/run-p1-glm53-format-completion-2",
        ),
        glm_1,
    )
    sources_1[DEEPSEEK_PRO_MODEL_ID] = (
        (root / "benchmark/powered-v62/run-panel1-deepseek-repair", prior_1_v62),
        (root / "benchmark/powered-v74/run-panel1-deepseek-v2", plan_1),
        *(
            (
                root / f"benchmark/powered-v77/run-p1-deepseek-price-lineage-completion-{index}",
                plan_1,
            )
            for index in range(1, 7)
        ),
        (root / "benchmark/powered-v77/run-p1-deepseek-uncertain-completion-1", plan_1),
    )

    taskset_2_path = root / (
        "benchmark/powered-v45/taskset/epicure-selection-taskset-"
        "925ba9d1d4be9c2b7a1e9956ecd6c18d34ffcad22eee28522f16892922c91e3f.json"
    )
    repeat_2_path = root / (
        "benchmark/powered-v45/plan/epicure-selection-repeat-panel-"
        "36d8c12ff883ead78e53406844ad386eb8999168a61d6931fe17135a2c73acfe.json"
    )
    plan_2_path = root / (
        "benchmark/powered-v75/plan/epicure-selection-analysis-plan-"
        "48d9d8f12d6da1910621d15ea7f26750119745ecde3b94fb8dfb3ec5c382cf75.json"
    )
    source_plan_2_path = root / (
        "benchmark/powered-v49/plan/epicure-selection-analysis-plan-"
        "6517dd4d018a4e4406bc34e7d68a8cd815e1c70367c498a8185b8518220b7cf0.json"
    )
    prior_2_v52_path = root / (
        "benchmark/powered-v52/plan/epicure-selection-analysis-plan-"
        "39ae0c4618dc229ce3ba11aed03664e1cbcc4682bd18f86fc6c9b5315b07d2be.json"
    )
    prior_2_v54_path = root / (
        "benchmark/powered-v54/plan/epicure-selection-analysis-plan-"
        "314702bc94a802d530b421ee73a52fb12eea805b43648bd0d9786df785469069.json"
    )
    prior_2_v63_path = root / (
        "benchmark/powered-v63/plan/epicure-selection-analysis-plan-"
        "208c848c6f4b7f372076b5f76f2c65480031a9162d69578250441f054776f91c.json"
    )
    glm_2_path = root / (
        "benchmark/powered-v66/plan/epicure-selection-analysis-plan-"
        "bffde3c99f4af662bc4922fbcb4a1d3d380790f82fa346f52c765ebe4d16511b.json"
    )
    plan_2 = _load(plan_2_path)
    source_plan_2 = _load(source_plan_2_path)
    prior_2_v52 = _load(prior_2_v52_path)
    prior_2_v54 = _load(prior_2_v54_path)
    prior_2_v63 = _load(prior_2_v63_path)
    glm_2 = _load(glm_2_path)
    base_completion_2 = tuple(
        root / f"benchmark/powered-v77/run-p2-{name}-format-completion-{index}"
        for name, maximum in (("glm52", 3), ("llama", 2), ("mistral", 3))
        for index in range(1, maximum + 1)
    )
    coverage_completion_2 = (
        *(root / f"benchmark/powered-v68/run-p2-glimmer-coverage-completion-{i}" for i in (1, 2)),
        *(root / f"benchmark/powered-v68/run-p2-minimax-coverage-completion-{i}" for i in (1, 2)),
        *(root / f"benchmark/powered-v68/run-p2-nemotron-coverage-completion-{i}" for i in (1, 2)),
        root / "benchmark/powered-v77/run-p2-fable-coverage-completion-1",
        root / "benchmark/powered-v77/run-p2-glimmer-coverage-completion-3",
        root / "benchmark/powered-v77/run-p2-glimmer-coverage-completion-4",
        root / "benchmark/powered-v77/run-p2-nemotron-coverage-completion-3",
    )
    sources_2 = _panel_2_model_sources(
        plan=prior_2_v63,
        run_directory=root / "benchmark/powered-v49/run",
        source_plan=source_plan_2,
        source_plan_path=source_plan_2_path,
        luna_run_directory=root / "benchmark/powered-v52/run-luna",
        deepseek_flash_run_directory=root / "benchmark/powered-v52/run-deepseek",
        prior_replacement_plan_v52=prior_2_v52,
        prior_replacement_plan_v52_path=prior_2_v52_path,
        repair_run_directory=root / "benchmark/powered-v54/run-panel2-repair",
        base_completion_run_directories=base_completion_2,
        coverage_completion_run_directories=coverage_completion_2,
        prior_plan_v54=prior_2_v54,
        prior_plan_v54_path=prior_2_v54_path,
        deepseek_repair_run_directory=root / "benchmark/powered-v63/run-panel2-deepseek-repair",
    )
    sources_2["z-ai/glm-5.3"] = (
        (
            root / "benchmark/powered-v66/run-panel2-glm53",
            root / "benchmark/powered-v77/run-p2-glm53-format-completion-1",
            root / "benchmark/powered-v77/run-p2-glm53-format-completion-2",
        ),
        glm_2,
    )
    sources_2[DEEPSEEK_PRO_MODEL_ID] = (
        (root / "benchmark/powered-v63/run-panel2-deepseek-repair", prior_2_v63),
        (root / "benchmark/powered-v75/run-panel2-deepseek", plan_2),
        *(
            (
                root / f"benchmark/powered-v77/run-p2-deepseek-price-lineage-completion-{index}",
                plan_2,
            )
            for index in range(1, 4)
        ),
        (root / "benchmark/powered-v77/run-p2-deepseek-uncertain-completion-1", plan_2),
    )
    return CompleteCoreSourceGraph(
        panel_1_plan=plan_1,
        panel_1_plan_path=plan_1_path,
        panel_1_taskset=_load(taskset_1_path),
        panel_1_taskset_path=taskset_1_path,
        panel_1_repeat=_load(repeat_1_path),
        panel_1_repeat_path=repeat_1_path,
        panel_1_model_sources=sources_1,
        panel_2_plan=plan_2,
        panel_2_plan_path=plan_2_path,
        panel_2_taskset=_load(taskset_2_path),
        panel_2_taskset_path=taskset_2_path,
        panel_2_repeat=_load(repeat_2_path),
        panel_2_repeat_path=repeat_2_path,
        panel_2_model_sources=sources_2,
    )


def load_full_primary_panels(graph: CompleteCoreSourceGraph) -> tuple[PanelData, PanelData]:
    """Verify and load both 640-task candidate panels with parser-v3 validity."""

    common = {
        "panel": "primary",
        "analysis_score_function": score_answer_v3,
        "allowed_source_roster_differences": {
            DEEPSEEK_PRO_MODEL_ID: frozenset({"endpoint_sha256"})
        },
    }
    panel_1 = load_panel(
        run_directory=Path("."),
        plan=graph.panel_1_plan,
        taskset=graph.panel_1_taskset,
        repeat_panel=graph.panel_1_repeat,
        model_sources=graph.panel_1_model_sources,
        **common,
    )
    panel_2 = load_panel(
        run_directory=Path("."),
        plan=graph.panel_2_plan,
        taskset=graph.panel_2_taskset,
        repeat_panel=graph.panel_2_repeat,
        model_sources=graph.panel_2_model_sources,
        **common,
    )
    return panel_1, panel_2
