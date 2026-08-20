from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from flavourbench.epicure_selection_powered_plan_v46 import verify_plan as verify_plan_v46
from flavourbench.epicure_selection_powered_plan_v47 import verify_plan as verify_plan_v47
from flavourbench.epicure_selection_powered_plan_v49 import verify_plan as verify_plan_v49
from flavourbench.epicure_selection_powered_plan_v50 import verify_plan as verify_plan_v50
from flavourbench.epicure_selection_powered_plan_v51 import verify_plan as verify_plan_v51
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.epicure_selection_route_manifest_v45 import FABLE_MODEL_ID, QWEN_MODEL_ID
from flavourbench.epicure_selection_route_manifest_v46 import (
    verify_manifest as verify_manifest_v46,
)
from flavourbench.epicure_selection_route_manifest_v49 import (
    FABLE_BEDROCK_SPEC,
)
from flavourbench.epicure_selection_route_manifest_v49 import (
    verify_manifest as verify_manifest_v49,
)

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR_RELEASE = (
    ROOT / "paper/generated/powered/flavourbench-powered-release-"
    "7aeddf27998b0a8ed0b961cab035e4793305ea120f73ce8f3baa47e4db612cf7.json"
)
MANIFEST_V46 = (
    ROOT / "benchmark/powered-v46/manifest/flavourbench-frontier-refresh-26-"
    "3caaa80fa8c3a646d27b5cad834a2e42fda6edaea82e0c29af9c7e14b1393df7.json"
)
MANIFEST_V49 = (
    ROOT / "benchmark/powered-v49/manifest/flavourbench-frontier-refresh-26-"
    "0251a89259c9cedfc2f2c9660481961b71fadfe80bb8db27536b94ef6476fa33.json"
)
PLAN_V46 = (
    ROOT / "benchmark/powered-v46/plan/epicure-selection-analysis-plan-"
    "61fcdfe6323475910823c35e47367f743fb142d88a06c5ca91883d35bd24c31a.json"
)
PLAN_V47 = (
    ROOT / "benchmark/powered-v47/plan/epicure-selection-analysis-plan-"
    "ed01c0da3a5e6204ddcd436cc1322eff08b06a752f37ca229dd3b15a14f91490.json"
)
PLAN_V49 = (
    ROOT / "benchmark/powered-v49/plan/epicure-selection-analysis-plan-"
    "6517dd4d018a4e4406bc34e7d68a8cd815e1c70367c498a8185b8518220b7cf0.json"
)
PLAN_V50 = (
    ROOT / "benchmark/powered-v50/plan/epicure-selection-analysis-plan-"
    "ffd464743c220643b6db285db1c98122b5b7417676bd157d51497fc0e27e4da0.json"
)
PLAN_V51 = (
    ROOT / "benchmark/powered-v51/plan/epicure-selection-joint-analysis-plan-"
    "ed452423f88069952b30a880e5a001d64040db5a105ef108079fc6c4efd6d26c.json"
)
TASKSET_1 = (
    ROOT / "benchmark/powered-v44/taskset/epicure-selection-taskset-"
    "a33bf28db372090015118371417b0e8ed1254f416d03d2c2c5816a6a752beb41.json"
)
REPEAT_1 = (
    ROOT / "benchmark/powered-v44/plan/epicure-selection-repeat-panel-"
    "96f766df855b93ad1495ec386c70ad88e42c4f896be24b0538cf1084da3c124a.json"
)
TASKSET_2 = (
    ROOT / "benchmark/powered-v45/taskset/epicure-selection-taskset-"
    "925ba9d1d4be9c2b7a1e9956ecd6c18d34ffcad22eee28522f16892922c91e3f.json"
)
REPEAT_2 = (
    ROOT / "benchmark/powered-v45/plan/epicure-selection-repeat-panel-"
    "36d8c12ff883ead78e53406844ad386eb8999168a61d6931fe17135a2c73acfe.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(document: dict[str, object], *, nested: bool) -> dict[str, dict[str, object]]:
    rows = document["models"] if nested else document["roster"]["models"]  # type: ignore[index]
    key = "model" if nested else None
    return {
        str(row[key]["id"] if key else row["model_id"]): row  # type: ignore[index]
        for row in rows  # type: ignore[union-attr]
    }


def test_v49_and_v50_exact_lineage_and_single_route_change() -> None:
    manifest_46 = _load(MANIFEST_V46)
    manifest_49 = _load(MANIFEST_V49)
    plan_46 = _load(PLAN_V46)
    plan_47 = _load(PLAN_V47)
    plan_49 = _load(PLAN_V49)
    plan_50 = _load(PLAN_V50)
    plan_51 = _load(PLAN_V51)
    assert verify_manifest_v46(manifest_46)
    assert verify_manifest_v49(manifest_49)
    assert verify_plan_v46(plan_46)
    assert verify_plan_v47(plan_47)
    assert verify_plan_v49(plan_49)
    assert verify_plan_v50(plan_50)
    assert verify_plan_v51(plan_51)
    assert _sha(MANIFEST_V49) == "54befddcc6a35b3967c387a909da10ec57f20382393621666bcd178e9c47d150"
    assert _sha(PLAN_V49) == "e8b9e1071641b301114ae6ca4ba2b53ffc8d023eaaf850a4e1cee0c2ff96d86d"
    assert _sha(PLAN_V50) == "1972ba9eaa398b6db4ade2de1234d3466d78f3c929cf712e4a5c106f529f755a"
    assert _sha(PLAN_V51) == "bd53050400f76b68b4a2c16fcfa56dc58ac208bdbb8b7eda8f90107759e385a2"
    assert plan_51["inputs"]["panel_1_plan"]["semantic_sha256"] == plan_50["artifact_sha256"]
    assert plan_51["inputs"]["panel_2_plan"]["semantic_sha256"] == plan_49["artifact_sha256"]
    assert plan_51["source_rules"]["panel_1_uses_complete_fable_replacement_block"] is True

    old_manifest_rows = _rows(manifest_46, nested=True)
    new_manifest_rows = _rows(manifest_49, nested=True)
    assert {
        model_id: row for model_id, row in old_manifest_rows.items() if model_id != FABLE_MODEL_ID
    } == {
        model_id: row for model_id, row in new_manifest_rows.items() if model_id != FABLE_MODEL_ID
    }
    fable = new_manifest_rows[FABLE_MODEL_ID]
    assert fable["endpoint"]["tag"] == FABLE_BEDROCK_SPEC["tag"]  # type: ignore[index]
    assert fable["request_policy"]["provider"]["only"] == [  # type: ignore[index]
        FABLE_BEDROCK_SPEC["tag"]
    ]

    old_panel_2 = _rows(plan_46, nested=False)
    new_panel_2 = _rows(plan_49, nested=False)
    assert {
        model_id: row for model_id, row in old_panel_2.items() if model_id != FABLE_MODEL_ID
    } == {model_id: row for model_id, row in new_panel_2.items() if model_id != FABLE_MODEL_ID}
    old_panel_1 = _rows(plan_47, nested=False)
    new_panel_1 = _rows(plan_50, nested=False)
    assert {
        model_id: row for model_id, row in old_panel_1.items() if model_id != FABLE_MODEL_ID
    } == {model_id: row for model_id, row in new_panel_1.items() if model_id != FABLE_MODEL_ID}
    assert new_panel_1[QWEN_MODEL_ID] == old_panel_1[QWEN_MODEL_ID]


def test_v49_and_v50_runner_inputs_schedule_exact_cells() -> None:
    _, tasks_2, repeats_2, plan_49, _, candidates_2 = validate_inputs(
        manifest_path=MANIFEST_V49,
        manifest_sha256="0251a89259c9cedfc2f2c9660481961b71fadfe80bb8db27536b94ef6476fa33",
        taskset_path=TASKSET_2,
        repeat_panel_path=REPEAT_2,
        plan_path=PLAN_V49,
        predecessor_release_path=PREDECESSOR_RELEASE,
    )
    cells_2 = build_cells(
        plan=plan_49,
        taskset=tasks_2,
        repeat_panel=repeats_2,
        candidates=candidates_2,
        phase="all",
    )
    assert len(cells_2) == 26 * (640 + 64)

    _, tasks_1, repeats_1, plan_50, _, candidates_1 = validate_inputs(
        manifest_path=MANIFEST_V49,
        manifest_sha256="0251a89259c9cedfc2f2c9660481961b71fadfe80bb8db27536b94ef6476fa33",
        taskset_path=TASKSET_1,
        repeat_panel_path=REPEAT_1,
        plan_path=PLAN_V50,
        predecessor_release_path=PREDECESSOR_RELEASE,
    )
    fable = [candidate for candidate in candidates_1 if candidate.model_id == FABLE_MODEL_ID]
    cells_1 = build_cells(
        plan=plan_50,
        taskset=tasks_1,
        repeat_panel=repeats_1,
        candidates=fable,
        phase="all",
    )
    assert len(cells_1) == 640 + 64
    assert {cell.candidate.model_id for cell in cells_1} == {FABLE_MODEL_ID}


def test_v49_fail_closed_claims_cannot_be_relaxed() -> None:
    manifest = _load(MANIFEST_V49)
    plan_49 = _load(PLAN_V49)
    plan_50 = _load(PLAN_V50)
    broken_manifest = copy.deepcopy(manifest)
    broken_manifest["fable_route_v49"]["quality_scores_or_selections_used"] = True  # type: ignore[index]
    assert not verify_manifest_v49(broken_manifest)
    broken_49 = copy.deepcopy(plan_49)
    broken_49["execution"]["fable_bedrock_route"]["automatic_fallback"] = True  # type: ignore[index]
    assert not verify_plan_v49(broken_49)
    broken_50 = copy.deepcopy(plan_50)
    broken_50["execution"]["panel_1_composite_v2"]["cross_route_response_pooling"] = True  # type: ignore[index]
    assert not verify_plan_v50(broken_50)
