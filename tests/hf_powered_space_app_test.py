from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def test_powered_space_loads_and_exposes_score_task_and_pairwise_views(
    tmp_path: Path, monkeypatch
) -> None:
    models = []
    for index, model_id in enumerate(("model/a", "model/b")):
        models.append(
            {
                "model_id": model_id,
                "model_name": f"Model {index}",
                "point_estimate_rank": index + 1,
                "statistical_rank_group": index + 1,
                "flavourbench_score": 80.0 - 10 * index,
                "score_simultaneous_95_ci": [75.0 - 10 * index, 85.0 - 10 * index],
                "bootstrap_rank_95_interval": [index + 1, index + 1],
                "availability": {"completed": 640},
                "repeatability": {"mean_ingredient_set_jaccard": 0.9 - index / 10},
                "family_scores": {
                    "substitution": 80.0 - 10 * index,
                    "pairing": 80.0 - 10 * index,
                    "constraint": 80.0 - 10 * index,
                    "cultural_composition": 80.0 - 10 * index,
                },
                "chance_comparison": {"exact_chance_score": 50.0},
                "provider_name": "Provider",
            }
        )
    choices = {label: f"ingredient-{label}" for label in "ABCDEFGH"}
    task = {
        "task_id": "task-1",
        "family": "substitution",
        "prompt": "Choose three ingredients.",
        "prompt_sha256": "1" * 64,
        "choices": choices,
        "optimal_selection": "ABC",
        "selection_scores_bps": {"ABC": 10_000, "ABD": 8_000},
    }
    observations = [
        {
            "model_id": model_id,
            "task_id": "task-1",
            "status": "completed",
            "scoring": {"observed_selection": selection, "score": score},
            "answer_excerpt": f"FINAL_SELECTION: {selection}",
            "answer_truncated": False,
            "actual_model_id": f"{model_id}-dated",
            "actual_provider": "Provider",
            "artifact_sha256": str(index + 2) * 64,
        }
        for index, (model_id, selection, score) in enumerate(
            (("model/a", "ABC", 100.0), ("model/b", "ABD", 80.0))
        )
    ]
    pairwise = [
        {
            "left_model_id": "model/a",
            "right_model_id": "model/b",
            "mean_difference": 10.0,
            "bootstrap_95_ci": [5.0, 15.0],
            "holm_significant": True,
            "holm_p": 0.01,
            "cohen_dz": 0.5,
        }
    ]
    bundle = {
        "schema_version": "flavourbench-powered-space-bundle-v1",
        "release_artifact_sha256": "a" * 64,
        "status": "final_complete",
        "analysis": {
            "inference": {"pairwise_hypotheses": 1},
            "models": models,
            "pairwise_comparisons": pairwise,
        },
        "models": models,
        "tasks": [task],
        "primary_observations": observations,
        "pairwise_comparisons": pairwise,
    }
    bundle["artifact_sha256"] = hashlib.sha256(_canonical(bundle)).hexdigest()
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    monkeypatch.setenv("FLAVOURBENCH_BUNDLE", str(path))
    module_path = Path(__file__).resolve().parents[1] / "hf/space/app.py"
    spec = importlib.util.spec_from_file_location("powered_space_app", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._leaderboard_frame().shape == (2, 8)
    diagnostic = module._completion_diagnostic("model/a")
    assert diagnostic["completed"] == 1
    assert diagnostic["failed"] == 0
    assert diagnostic["conditional_equal_family_score"] == 100.0
    model_summary, family_frame = module._model_detail("Model 0")
    assert "100.00" in model_summary
    assert "1/1" in model_summary
    assert family_frame.loc[0, "Completed-only*"] == 100.0
    detail = module._task_detail("Model 0", "task-1 | substitution")
    assert "100.00" in detail[0]
    assert "FINAL_SELECTION: ABC" in detail[4]
    assert "distinguishable" in module._pair_detail("Model 0", "Model 1")
