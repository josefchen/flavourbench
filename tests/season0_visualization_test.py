from __future__ import annotations

import xml.etree.ElementTree as ET

from flavourbench.season0_visualization import FAMILIES, render_all


def test_publication_svgs_are_valid_xml(tmp_path) -> None:
    model_rows = []
    uplift_rows = []
    for index in range(12):
        model_rows.append(
            {
                "season_model_id": f"model-{index}",
                "display_name": f"Model {index}",
                "provider": "bedrock" if index % 2 == 0 else "openrouter",
                "rating": 1100 - index * 18,
                "rating_lower": 1080 - index * 18,
                "rating_upper": 1120 - index * 18,
                "comparisons": 110,
                "invalid_response_rate": index / 100,
                "mean_arm_cost_usd": 0.001 * (index + 1),
            }
        )
        uplift_rows.append(
            {
                "season_model_id": f"model-{index}",
                "display_name": f"Model {index}",
                "epicure_win_share": 0.62 - index * 0.015,
                "interval_lower": 0.57 - index * 0.015,
                "interval_upper": 0.67 - index * 0.015,
                "epicure_wins": 60,
                "ties": 30,
                "unaided_wins": 30,
                "comparisons": 120,
            }
        )
    analysis = {
        "model_leaderboard": model_rows,
        "uplift_leaderboard": uplift_rows,
        "model_leaderboard_by_family": {
            family: [
                {
                    "season_model_id": row["season_model_id"],
                    "display_name": row["display_name"],
                    "rating": row["rating"] + family_index * 5,
                }
                for row in model_rows
            ]
            for family_index, family in enumerate(FAMILIES)
        },
        "panel_uplift_dimensions": [
            {
                "dimension": dimension,
                "mean_delta": 0.18 - index * 0.03,
                "lower": 0.08 - index * 0.03,
                "upper": 0.28 - index * 0.03,
                "comparisons": 900,
                "task_clusters": 120,
            }
            for index, dimension in enumerate(
                (
                    "task_completion",
                    "constraint_compliance",
                    "coherence",
                    "sensory_promise",
                    "cookability",
                    "clarity",
                    "originality",
                    "evidence_use",
                    "calibration",
                )
            )
        ],
        "operational_metrics": {
            f"model-{index}": {
                "provider": "bedrock" if index % 2 == 0 else "openrouter",
                "epicure_on_tool_use_rate": index / 11,
            }
            for index in range(12)
        },
    }
    paths = render_all(analysis, tmp_path)
    assert len(paths) == 7
    for path in paths.values():
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
