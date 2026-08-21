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
                "coverage": {
                    "scheduled": 1,
                    "valid_scored": 1,
                    "valid_scored_per_family": {
                        "substitution": 1,
                        "pairing": 1,
                        "constraint": 1,
                    },
                },
                "panel_replication": {
                    "panel_1": 81.0 - 10 * index,
                    "panel_2": 79.0 - 10 * index,
                    "difference": -2.0,
                },
                "family_scores": {
                    "substitution": 80.0 - 10 * index,
                    "pairing": 80.0 - 10 * index,
                    "constraint": 80.0 - 10 * index,
                },
                "chance_comparison": {"exact_chance_score": 50.0},
                "execution_backend": "provider_direct",
            }
        )
    choices = {label: f"ingredient-{label}" for label in "ABCDEFGH"}
    task = {
        "task_id": "task-1",
        "family": "substitution",
        "anchor_ingredient": "official-anchor",
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
            "scoring": {
                "observed_selection": selection,
                "score": score,
                "parseable": True,
            },
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
        "schema_version": "flavourbench-complete-core-space-bundle-v1",
        "release_artifact_sha256": "a" * 64,
        "status": "final_complete_common_core",
        "design": {
            "panel_count": 2,
            "unique_anchor_clusters": 1,
        },
        "analysis": {
            "inference": {"pairwise_hypotheses": 1},
            "resolved_pair_count": 1,
            "models": models,
            "pairwise_comparisons": pairwise,
        },
        "models": models,
        "tasks": [task],
        "primary_observations": observations,
        "pairwise_comparisons": pairwise,
        "lab_tasks": [
            {
                **task,
                "task_id": "lab-task-1",
                "anchor_ingredient": "training-anchor",
                "lab_split": "train",
            }
        ],
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
    assert "80.00" in model_summary
    assert "1/1" in model_summary
    assert family_frame.loc[0, "Score"] == 80.0
    detail = module._task_detail("Model 0", "task-1 | substitution")
    assert "100.00" in detail[0]
    assert "FINAL_SELECTION: ABC" in detail[4]
    assert "distinguishable" in module._pair_detail("Model 0", "Model 1")
    assert module._score_completion_api("task-1", "FINAL_SELECTION: A,B,C")["reward"] == 1.0
    training_reward = module._training_reward_api("lab-task-1", "FINAL_SELECTION: A,B,C")
    assert training_reward["reward"] == 1.0
    assert training_reward["official_leaderboard_eligible"] is False
    endpoints = module.demo.get_api_info()["named_endpoints"]
    assert "/score_completion" in endpoints
    assert "/score_submission" in endpoints
    assert "/score_uploaded_submission" in endpoints
    assert "/training_reward" in endpoints
