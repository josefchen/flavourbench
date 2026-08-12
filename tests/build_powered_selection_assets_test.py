from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "paper/build_powered_selection_assets.py"
    spec = importlib.util.spec_from_file_location("build_powered_selection_assets", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release() -> dict[str, object]:
    models = []
    repeatability = []
    for index in range(20):
        model_id = f"vendor/model-{index:02d}"
        repeat = {
            "model_id": model_id,
            "mean_ingredient_set_jaccard": 0.95 - index / 100,
        }
        repeatability.append(repeat)
        models.append(
            {
                "model_id": model_id,
                "model_name": f"Model {index}",
                "eligible": True,
                "availability": {"completed": 640, "parseable": 640},
                "flavourbench_score": 90.0 - index,
                "family_scores": {
                    "substitution": 91.0 - index,
                    "pairing": 90.0 - index,
                    "constraint": 89.0 - index,
                    "cultural_composition": 90.0 - index,
                },
                "score_simultaneous_95_ci": [88.0 - index, 92.0 - index],
                "point_estimate_rank": index + 1,
                "statistical_rank_group": index // 2 + 1,
                "chance_comparison": {
                    "exact_chance_score": 50.0,
                    "holm_significant_above_chance": True,
                },
                "repeatability": repeat,
            }
        )
    pairwise = []
    for left in range(20):
        for right in range(left + 1, 20):
            pairwise.append(
                {
                    "left_model_id": f"vendor/model-{left:02d}",
                    "right_model_id": f"vendor/model-{right:02d}",
                    "holm_significant": right - left > 2,
                    "mean_difference": float(right - left),
                }
            )
    return {
        "inputs": {
            "primary_responses": {"count": 12_800},
            "repeat_responses": {"count": 1_280},
        },
        "analysis": {
            "models": models,
            "pairwise_comparisons": pairwise,
            "repeatability": repeatability,
            "definitive_top_model_id": "vendor/model-00",
            "inference": {"bootstrap_resamples": 50_000, "permutation_resamples": 100_000},
        },
    }


def test_tables_macros_and_figures_render(tmp_path: Path) -> None:
    module = _module()
    release = _release()
    taskset = {
        "counts": {
            "tasks": 640,
            "scored_combinations_per_task": 56,
            "total_prefrozen_selection_scores": 35_840,
        },
        "tasks": [
            {
                "family": family,
                "chance_score_bps": 4_000,
                "optimal_margin_bps": 500,
                "selection_scores_bps": {"ABC": 10_000, "ABD": 5_000},
            }
            for family in module.FAMILIES
        ],
    }
    macros = module._macros(release, taskset)
    leaderboard = module._leaderboard_table(release)
    family = module._family_table(release)
    diagnostics = module._task_diagnostics_table(taskset)
    assert r"\newcommand{\FBModels}{20}" in macros
    assert "vendor/model-00" not in leaderboard
    assert "Regional" in family
    assert "40.0" in diagnostics
    assert r"\begin{tabular}{@{}l r r r r@{}}" in family
    assert r"\begin{tabular}{@{}l r r r@{}}" in diagnostics
    module._configure_plots()
    module._leaderboard_figure(release, tmp_path)
    module._family_heatmap(release, tmp_path)
    module._pairwise_matrix(release, tmp_path)
    module._repeatability_figure(release, tmp_path)
    assert len(list(tmp_path.glob("*.pdf"))) == 4
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_response_map_replaces_only_the_recovery_model(tmp_path: Path) -> None:
    module = _module()
    base = tmp_path / "base"
    recovery = tmp_path / "recovery"

    def write(directory: Path, model_id: str, task_id: str, marker: str) -> None:
        document = {"model_id": model_id, "task_id": task_id, "marker": marker}
        document["artifact_sha256"] = module._sha256(document)
        target = directory / "responses/primary/slot"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"response-{marker}.json").write_text(json.dumps(document))

    write(base, "model/a", "task-1", "base-a")
    write(base, "model/b", "task-1", "superseded-b")
    write(recovery, "model/b", "task-1", "recovery-b")
    rows = module._response_map(
        base,
        2,
        recovery_run_directory=recovery,
        recovery_model_id="model/b",
    )
    assert rows[("model/a", "task-1")]["marker"] == "base-a"
    assert rows[("model/b", "task-1")]["marker"] == "recovery-b"


def test_response_map_combines_multiple_clean_successor_blocks(tmp_path: Path) -> None:
    module = _module()
    base = tmp_path / "base"
    deepseek = tmp_path / "deepseek"
    cohere = tmp_path / "cohere"

    def write(directory: Path, model_id: str, marker: str) -> None:
        document = {"model_id": model_id, "task_id": "task-1", "marker": marker}
        document["artifact_sha256"] = module._sha256(document)
        target = directory / "responses/primary/slot"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"response-{marker}.json").write_text(json.dumps(document))

    write(base, "model/a", "base-a")
    write(base, "model/b", "superseded-b")
    write(base, "model/c", "superseded-c")
    write(base, "model/stale", "preserved-but-excluded")
    write(deepseek, "model/b", "deepseek-b")
    write(cohere, "model/c", "cohere-c")
    rows = module._response_map(
        base,
        3,
        expected_model_ids={"model/a", "model/b", "model/c"},
        response_sources={"model/b": deepseek, "model/c": cohere},
    )
    assert {key: row["marker"] for key, row in rows.items()} == {
        ("model/a", "task-1"): "base-a",
        ("model/b", "task-1"): "deepseek-b",
        ("model/c", "task-1"): "cohere-c",
    }
