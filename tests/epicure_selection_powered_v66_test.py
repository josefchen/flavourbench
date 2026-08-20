from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_selection_powered_plan_v62 import (
    PLAN_SCHEMA_VERSION as V62_SCHEMA,
)
from flavourbench.epicure_selection_powered_plan_v62 import PLAN_VERSION as V62_VERSION
from flavourbench.epicure_selection_powered_plan_v65 import (
    _as_predecessor,
)
from flavourbench.epicure_selection_powered_plan_v65 import (
    build_plan as build_panel_1,
)
from flavourbench.epicure_selection_powered_plan_v65 import (
    verify_plan as verify_panel_1,
)
from flavourbench.epicure_selection_powered_plan_v66 import (
    build_plan as build_panel_2,
)
from flavourbench.epicure_selection_powered_plan_v66 import (
    verify_plan as verify_panel_2,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = next((ROOT / "benchmark/powered-v65/manifest").glob("*.json"))
PANEL_1 = next((ROOT / "benchmark/powered-v62/plan").glob("*.json"))
PANEL_2 = next((ROOT / "benchmark/powered-v63/plan").glob("*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_panel_plans_add_one_complete_glm53_block_without_changing_prior_rows() -> None:
    manifest = _load(MANIFEST)
    panel_1 = build_panel_1(
        predecessor=_load(PANEL_1),
        predecessor_path=PANEL_1,
        manifest=manifest,
        manifest_path=MANIFEST,
    )
    panel_2 = build_panel_2(
        predecessor=_load(PANEL_2),
        predecessor_path=PANEL_2,
        manifest=manifest,
        manifest_path=MANIFEST,
    )
    assert verify_panel_1(panel_1)
    assert verify_panel_2(panel_2)
    assert panel_1["roster"]["model_count"] == panel_2["roster"]["model_count"] == 27
    assert panel_1["roster"]["models"][-1]["model_id"] == "z-ai/glm-5.3"
    assert panel_2["roster"]["models"][-1]["model_id"] == "z-ai/glm-5.3"
    restored = _as_predecessor(
        panel_1,
        predecessor_schema=V62_SCHEMA,
        predecessor_version=V62_VERSION,
        predecessor_key="plan_v62_predecessor",
    )
    assert restored == _load(PANEL_1)
