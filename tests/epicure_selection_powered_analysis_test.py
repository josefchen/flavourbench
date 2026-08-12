from __future__ import annotations

import json

import numpy as np
import pytest

import flavourbench.epicure_selection_powered_analysis as analysis_module
from flavourbench.epicure_selection_powered_analysis import (
    PanelData,
    SelectionPoweredAnalysisError,
    analyze_panels,
    family_macro_mean,
    family_stratified_bootstrap,
    holm_adjust,
    load_panel,
    paired_sign_flip_pvalues,
)
from flavourbench.epicure_selection_taskset_v1 import FAMILIES


def test_family_macro_mean_weights_families_equally() -> None:
    values = np.asarray([[0.0, 100.0, 50.0, 50.0, 25.0]])
    families = (
        "substitution",
        "substitution",
        "pairing",
        "constraint",
        "cultural_composition",
    )
    assert family_macro_mean(values, families).tolist() == [43.75]


def test_family_bootstrap_is_shared_and_deterministic() -> None:
    families = tuple(family for family in FAMILIES for _ in range(3))
    first = np.arange(12, dtype=float)
    values = np.vstack((first, first + 7.0))
    left = family_stratified_bootstrap(values, families, resamples=101, seed=18)
    right = family_stratified_bootstrap(values, families, resamples=101, seed=18)
    assert np.array_equal(left, right)
    assert np.allclose(left[:, 1] - left[:, 0], 7.0)


def test_sign_flip_and_holm_are_reproducible() -> None:
    differences = np.asarray(
        [
            [4.0] * 12,
            [2.0, -2.0] * 6,
            [-3.0] * 12,
        ]
    )
    observed, pvalues = paired_sign_flip_pvalues(differences, resamples=9_999, seed=20260811)
    repeated, repeated_pvalues = paired_sign_flip_pvalues(
        differences, resamples=9_999, seed=20260811
    )
    assert np.array_equal(observed, repeated)
    assert np.array_equal(pvalues, repeated_pvalues)
    assert observed.tolist() == [4.0, 0.0, -3.0]
    assert pvalues[0] < 0.01
    assert pvalues[1] == 1.0
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def _task(task_id: str, family: str, chance: int = 5_000) -> dict[str, object]:
    return {
        "task_id": task_id,
        "family": family,
        "chance_score_bps": chance,
        "choices": {label: f"ingredient_{label.lower()}" for label in "ABCDEFGH"},
    }


def _panel(
    *,
    panel: str,
    task_ids: tuple[str, ...],
    families: tuple[str, ...],
    scores: np.ndarray,
) -> PanelData:
    model_ids = ("model/a", "model/b", "model/c")
    return PanelData(
        panel=panel,
        model_ids=model_ids,
        model_names=("A", "B", "C"),
        slot_ids=("slot-a", "slot-b", "slot-c"),
        task_ids=task_ids,
        families=families,
        scores=scores,
        completed=np.ones_like(scores, dtype=bool),
        parseable=np.ones_like(scores, dtype=bool),
        selections=tuple(tuple("ABC" for _ in task_ids) for _ in model_ids),
        response_artifact_sha256s=tuple(),
        spend_micros=0,
    )


def test_end_to_end_synthetic_analysis_separates_a_definitive_top() -> None:
    primary_tasks = tuple(
        _task(f"{family}-{index}", family) for family in FAMILIES for index in range(2)
    )
    primary_task_ids = tuple(str(task["task_id"]) for task in primary_tasks)
    primary_families = tuple(str(task["family"]) for task in primary_tasks)
    primary = _panel(
        panel="primary",
        task_ids=primary_task_ids,
        families=primary_families,
        scores=np.asarray([[90.0] * 8, [70.0] * 8, [50.0] * 8]),
    )
    repeat_tasks = tuple(
        {
            **_task(f"repeat-{family}", family),
            "original_task_id": f"{family}-0",
        }
        for family in FAMILIES
    )
    repeat = _panel(
        panel="repeat",
        task_ids=tuple(str(task["task_id"]) for task in repeat_tasks),
        families=tuple(str(task["family"]) for task in repeat_tasks),
        scores=np.asarray([[90.0] * 4, [70.0] * 4, [50.0] * 4]),
    )
    plan = {
        "artifact_sha256": "a" * 64,
        "inference": {"bootstrap_resamples": 1_000, "permutation_resamples": 9_999, "seed": 7},
        "eligibility": {"minimum_completed_tasks": 8},
        "repeatability": {"acceptance_floor": 0.8},
        "outcomes": {"primary_definition": "synthetic equal-family score"},
    }
    result = analyze_panels(
        primary=primary,
        repeat=repeat,
        taskset={"tasks": list(primary_tasks)},
        repeat_panel={"tasks": list(repeat_tasks)},
        plan=plan,
        bootstrap_resamples=1_000,
        permutation_resamples=9_999,
    )
    assert result["status"] == "final_complete"
    assert result["definitive_top_model_id"] == "model/a"
    assert len(result["pairwise_comparisons"]) == 3
    assert all(row["holm_significant"] for row in result["pairwise_comparisons"])
    assert [row["point_estimate_rank"] for row in result["models"]] == [1, 2, 3]
    assert [row["statistical_rank_group"] for row in result["models"]] == [1, 2, 3]
    assert result["models"][0]["repeatability"]["mean_ingredient_set_jaccard"] == 1.0


def test_composite_loader_binds_each_model_to_its_source_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(analysis_module, "MODEL_COUNT", 2)
    monkeypatch.setattr(analysis_module, "TASK_COUNT", 1)
    monkeypatch.setattr(analysis_module, "REPEAT_TASK_COUNT", 1)
    task = {
        "task_id": "task-1",
        "family": "substitution",
        "prompt_sha256": "1" * 64,
        "optimal_selection": "ABC",
    }
    taskset = {"artifact_sha256": "2" * 64, "tasks": [task]}
    repeat_panel = {"artifact_sha256": "3" * 64, "tasks": []}

    def roster_row(model_id: str, slot_id: str, endpoint: str) -> dict[str, object]:
        return {
            "model_id": model_id,
            "model_name": model_id,
            "slot_id": slot_id,
            "canonical_model_slug": f"{model_id}-dated",
            "execution_backend": "openrouter",
            "provider_tag": "provider/fp8",
            "provider_name": "Provider",
            "endpoint_sha256": endpoint,
            "endpoint_execution_sha256": endpoint,
            "backend_contract_sha256": "4" * 64,
            "final_reasoning_effort": "minimal",
        }

    row_a = roster_row("model/a", "slot-a", "a" * 64)
    row_b = roster_row("model/b", "slot-b", "b" * 64)
    predecessor_plan = {
        "artifact_sha256": "5" * 64,
        "inputs": {"route_manifest": {"semantic_sha256": "6" * 64}},
        "roster": {"models": [row_a, roster_row("model/b", "slot-b", "c" * 64)]},
    }
    final_plan = {
        "artifact_sha256": "7" * 64,
        "inputs": {"route_manifest": {"semantic_sha256": "8" * 64}},
        "roster": {"models": [row_a, row_b]},
    }

    def write_failure(
        directory,
        *,
        row: dict[str, object],
        source_plan: dict[str, object],
    ) -> None:
        cell_id = f"cell-{row['slot_id']}"
        document = {
            "schema_version": "flavourbench-powered-response-v1",
            "status": "failed",
            "panel": "primary",
            "plan_sha256": source_plan["artifact_sha256"],
            "manifest_sha256": source_plan["inputs"]["route_manifest"]["semantic_sha256"],
            "taskset_sha256": taskset["artifact_sha256"],
            "repeat_panel_sha256": repeat_panel["artifact_sha256"],
            "cell_id": cell_id,
            "model_id": row["model_id"],
            "task_id": task["task_id"],
            "family": task["family"],
            "slot_id": row["slot_id"],
            "model_name": row["model_name"],
            "canonical_model_slug": row["canonical_model_slug"],
            "execution_backend": row["execution_backend"],
            "endpoint_execution_sha256": row["endpoint_execution_sha256"],
            "backend_contract_sha256": row["backend_contract_sha256"],
            "prompt_sha256": task["prompt_sha256"],
            "optimal_selection": task["optimal_selection"],
            "original_task_id": None,
            "generation": None,
            "scoring": {
                "observed_selection": None,
                "optimal_selection": "ABC",
                "parseable": False,
                "score_bps": 0,
                "score": 0.0,
                "optimal": False,
            },
        }
        document["artifact_sha256"] = analysis_module._sha256(document)
        target = directory / "responses" / "primary" / str(row["slot_id"])
        target.mkdir(parents=True)
        path = target / f"response-{cell_id}-{document['artifact_sha256']}.json"
        path.write_text(json.dumps(document))

    predecessor_run = tmp_path / "predecessor"
    recovery_run = tmp_path / "recovery"
    write_failure(predecessor_run, row=row_a, source_plan=predecessor_plan)
    write_failure(recovery_run, row=row_b, source_plan=final_plan)
    panel = load_panel(
        run_directory=predecessor_run,
        panel="primary",
        plan=final_plan,
        taskset=taskset,
        repeat_panel=repeat_panel,
        model_sources={
            "model/a": (predecessor_run, predecessor_plan),
            "model/b": (recovery_run, final_plan),
        },
    )
    assert panel.model_ids == ("model/a", "model/b")
    assert panel.scores.tolist() == [[0.0], [0.0]]
    with pytest.raises(SelectionPoweredAnalysisError, match="response binding"):
        load_panel(
            run_directory=predecessor_run,
            panel="primary",
            plan=final_plan,
            taskset=taskset,
            repeat_panel=repeat_panel,
            model_sources={"model/b": (recovery_run, final_plan)},
        )
