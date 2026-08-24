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
        "stability_analysis": {
            "task_count_stability": [
                {
                    "tasks": 1,
                    "metrics": {
                        "rank_spearman": {
                            "median": 1.0,
                            "p2_5": 1.0,
                            "p97_5": 1.0,
                        },
                        "top_5_overlap": {"median": 1.0},
                        "top_1_preserved": {"mean": 1.0},
                    },
                }
            ],
            "variance_partition": {
                "relative_decision_generalizability_at_534_tasks": 0.936,
                "estimated_balanced_tasks_for_relative_g_0_90": 329,
            },
        },
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
    leaderboard = module._leaderboard_frame()
    assert leaderboard.shape == (2, 7)
    assert list(leaderboard.columns) == [
        "Rank",
        "Model",
        "Score ↑",
        "Simultaneous 95%",
        "Group",
        "Rank 95%",
        "Cells",
    ]
    leaderboard_html = module._leaderboard_html()
    assert "Complete FlavourBench common-core leaderboard" in leaderboard_html
    assert "Model 0" in leaderboard_html
    assert "80.00" in leaderboard_html
    assert "Find a model" in leaderboard_html
    assert "Lab leaders" in leaderboard_html
    assert "data-fb-metric='substitution'" in leaderboard_html
    assert "data-champion-overall='true'" in leaderboard_html
    assert "data-rank-pairing='1'" in leaderboard_html
    assert "data-fb-score-head" in leaderboard_html
    assert "Download JSONL" in leaderboard_html
    insights_html = module._insights_html()
    assert "What the evidence resolves" in insights_html
    assert "Where the leading labs differ" in insights_html
    assert "simultaneous 95 percent interval" in insights_html
    diagnostic = module._completion_diagnostic("model/a")
    assert diagnostic["completed"] == 1
    assert diagnostic["failed"] == 0
    assert diagnostic["conditional_equal_family_score"] == 100.0
    model_summary, family_frame = module._model_detail("Model 0")
    assert "80.00" in model_summary
    assert "1/1" in model_summary
    assert "95% score band" in model_summary
    assert "Backend" not in model_summary
    assert family_frame.loc[0, "Score"] == 80.0
    _, family_html = module._model_detail_ui("Model 0")
    assert "fb-table--family" in family_html
    assert "Random legal choice (exact)" in family_html
    detail = module._task_detail("Model 0", "task-1 | substitution")
    assert "100.00" in detail[0]
    assert "FINAL_SELECTION: ABC" in detail[4]
    detail_ui = module._task_detail_ui("Model 0", "task-1 | substitution")
    assert "fb-choice-label" in detail_ui[2]
    assert "ingredient-A" in detail_ui[2]
    assert "fb-table--score" in detail_ui[3]
    assert "model selection, optimum" in detail_ui[3]
    pair_html = module._pair_detail("Model 0", "Model 1")
    assert "distinguishable" in pair_html
    assert "+0.500" in pair_html
    assert module.LAB_LOGO_URLS["OpenAI"].startswith("/gradio_api/file=")
    assert module.ARCHITECTURE_URL.startswith("/gradio_api/file=")
    assert "data:font/woff2;base64," in module.FONT_CSS
    assert module.HEAD.startswith("<style>")
    assert module._score_completion_api("task-1", "FINAL_SELECTION: A,B,C")["reward"] == 1.0
    training_reward = module._training_reward_api("lab-task-1", "FINAL_SELECTION: A,B,C")
    assert training_reward["reward"] == 1.0
    assert training_reward["official_leaderboard_eligible"] is False
    command = module._command_preview(
        "Hosted OpenAI-compatible endpoint",
        "provider/model",
        "https://example.invalid/v1",
        "LAB_MODEL_API_KEY",
        "12-task smoke test",
    )
    assert "--backend openai-compatible" in command
    assert "--limit 12" in command
    assert "LAB_MODEL_API_KEY" in command
    assert "\n+  --" not in command
    reward_preview = module._reward_preview(
        "lab-task-1 | substitution | train", "FINAL_SELECTION: A,B,C"
    )
    assert "Reward 1.0000" in reward_preview
    assert "cannot alter the public leaderboard" in reward_preview
    endpoints = module.demo.get_api_info()["named_endpoints"]
    assert "/score_completion" in endpoints
    assert "/score_submission" in endpoints
    assert "/score_uploaded_submission" in endpoints
    assert "/training_reward" in endpoints
