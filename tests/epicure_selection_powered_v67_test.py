from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_selection_powered_plan_v67 import (
    _as_v64,
    build_plan,
    verify_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = next((ROOT / "benchmark/powered-v64/plan").glob("*.json"))
PANEL_1 = next((ROOT / "benchmark/powered-v65/plan").glob("*.json"))
PANEL_2 = next((ROOT / "benchmark/powered-v66/plan").glob("*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_v67_freezes_all_351_pairs_before_glm_results() -> None:
    document = build_plan(
        predecessor=_load(PREDECESSOR),
        predecessor_path=PREDECESSOR,
        panel_1_plan=_load(PANEL_1),
        panel_1_plan_path=PANEL_1,
        panel_2_plan=_load(PANEL_2),
        panel_2_plan_path=PANEL_2,
    )
    assert verify_plan(document)
    assert document["roster"]["model_count"] == 27
    assert document["roster"]["pairwise_hypotheses"] == 351
    assert document["design"]["primary_model_task_cells"] == 34_560
    assert document["design"]["repeat_model_task_cells"] == 3_456
    assert _as_v64(document) == _load(PREDECESSOR)
