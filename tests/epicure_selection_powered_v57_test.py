from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flavourbench.epicure_selection_powered_joint_analysis_v1 import (
    _panel_1_model_sources,
    _panel_2_model_sources,
)
from flavourbench.epicure_selection_powered_plan_v58 import verify_plan as verify_plan_v58
from flavourbench.epicure_selection_powered_plan_v59 import verify_plan as verify_plan_v59
from flavourbench.epicure_selection_powered_plan_v60 import verify_plan as verify_plan_v60
from flavourbench.epicure_selection_route_manifest_v57 import (
    DEEPSEEK_PRO_MODEL_ID,
    EXPECTED_CELLS_PER_PANEL,
    PROVIDER_NAME,
    ROUTE_TAG,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    ROOT / "benchmark/powered-v54/manifest/flavourbench-frontier-refresh-26-"
    "ac2c9c1ab64552fb9c367fc5c66e368fac754fdc66ebe566d751e41d382fde06.json"
)
MANIFEST = (
    ROOT / "benchmark/powered-v57/manifest/flavourbench-frontier-refresh-26-"
    "659ec848b9ba34471537c5a8de806d543c5a362a863fc3555930ba55d4af98ed.json"
)
PANEL_1_PREDECESSOR = (
    ROOT / "benchmark/powered-v55/plan/epicure-selection-analysis-plan-"
    "8577a9a32c5fb266f12b131c309f4543c6fa2cd42538abd16eefbf4c09d578ed.json"
)
PANEL_1_PLAN = (
    ROOT / "benchmark/powered-v58/plan/epicure-selection-analysis-plan-"
    "e25b1ba5bdc6c6ef288308dd463bb25e0c7550611ff670a1b5d56d0c53e6e2c4.json"
)
PANEL_2_PREDECESSOR = (
    ROOT / "benchmark/powered-v54/plan/epicure-selection-analysis-plan-"
    "314702bc94a802d530b421ee73a52fb12eea805b43648bd0d9786df785469069.json"
)
PANEL_2_PLAN = (
    ROOT / "benchmark/powered-v59/plan/epicure-selection-analysis-plan-"
    "bcb5b32d3d33a326569fbdc1dc4e15dee822b72a46fe62392df4ddb6f8f66ae4.json"
)
JOINT_PREDECESSOR = (
    ROOT / "benchmark/powered-v56/plan/epicure-selection-joint-analysis-plan-"
    "cc5b423e01445a69e88be7f3be981497cc28a16fa7043c865db546f8499b27f1.json"
)
JOINT_PLAN = (
    ROOT / "benchmark/powered-v60/plan/epicure-selection-joint-analysis-plan-"
    "a6aaa4b8135c1ac8d578da1c84bf2364c6259fd7732f944f04a612f273fda173.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(document: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["model"]["id"]: row for row in document["models"]}


def _plan_rows(document: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["model_id"]: row for row in document["roster"]["models"]}


def test_v57_manifest_changes_only_deepseek_after_two_complete_failed_blocks() -> None:
    source = _load(SOURCE_MANIFEST)
    manifest = _load(MANIFEST)
    assert verify_manifest(manifest)
    assert manifest["content_address"]["digest"] == (
        "659ec848b9ba34471537c5a8de806d543c5a362a863fc3555930ba55d4af98ed"
    )
    assert _sha256(MANIFEST) == ("95d9cab5fe6f7d638c77cc8e8cda22db85eaf6be8fdf8447510bd759d76fa10c")
    before = _rows(source)
    after = _rows(manifest)
    assert list(before) == list(after)
    assert all(
        before[model_id] == after[model_id]
        for model_id in before
        if model_id != DEEPSEEK_PRO_MODEL_ID
    )
    deepseek = after[DEEPSEEK_PRO_MODEL_ID]
    assert deepseek["endpoint"]["tag"] == ROUTE_TAG
    assert deepseek["endpoint"]["provider_name"] == PROVIDER_NAME
    evidence = manifest["deepseek_complete_block_repair_v57"]["failed_block_projections"]
    for panel in ("panel_1", "panel_2"):
        assert evidence[panel]["scheduled_cells"] == EXPECTED_CELLS_PER_PANEL
        assert evidence[panel]["failed_response_artifacts"] == EXPECTED_CELLS_PER_PANEL
        assert evidence[panel]["quality_scores_or_selections_read"] is False


def test_v58_panel_1_plan_replaces_the_entire_deepseek_block_only() -> None:
    predecessor = _load(PANEL_1_PREDECESSOR)
    plan = _load(PANEL_1_PLAN)
    assert verify_plan_v58(plan)
    assert _sha256(PANEL_1_PLAN) == (
        "661a4b9ed6df9f42839865e60218411508334f89c2a5523712911451ebfb12c7"
    )
    before = _plan_rows(predecessor)
    after = _plan_rows(plan)
    assert all(
        before[model_id] == after[model_id]
        for model_id in before
        if model_id != DEEPSEEK_PRO_MODEL_ID
    )
    repair = plan["execution"]["deepseek_complete_block_replacement_v58"]
    assert repair["replacement_model_ids"] == [DEEPSEEK_PRO_MODEL_ID]
    assert repair["replacement_primary_cells_per_model"] == 640
    assert repair["replacement_repeat_cells_per_model"] == 64
    assert repair["selective_failed_cell_retry"] is False
    assert repair["superseded_responses_used"] is False


def test_v59_panel_2_plan_replaces_the_entire_deepseek_block_only() -> None:
    predecessor = _load(PANEL_2_PREDECESSOR)
    plan = _load(PANEL_2_PLAN)
    assert verify_plan_v59(plan)
    assert _sha256(PANEL_2_PLAN) == (
        "3b24aebbf5c93650861c06ac7d0eebfd3ea63e458f649589ab377ff6a54803c3"
    )
    before = _plan_rows(predecessor)
    after = _plan_rows(plan)
    assert all(
        before[model_id] == after[model_id]
        for model_id in before
        if model_id != DEEPSEEK_PRO_MODEL_ID
    )
    repair = plan["execution"]["deepseek_complete_block_replacement_v59"]
    assert repair["replacement_model_ids"] == [DEEPSEEK_PRO_MODEL_ID]
    assert repair["replacement_primary_cells_per_model"] == 640
    assert repair["replacement_repeat_cells_per_model"] == 64
    assert repair["selection_uses_scores_or_selections"] is False
    assert repair["cross_route_response_pooling"] is False


def test_v60_joint_plan_binds_both_deepseek_blocks_without_pooling() -> None:
    predecessor = _load(JOINT_PREDECESSOR)
    plan = _load(JOINT_PLAN)
    assert verify_plan_v60(plan)
    assert _sha256(JOINT_PLAN) == (
        "e9ab9d56b41e629ccd88ddaecb731fc5b22d52845badd1048cf3ee75ad4c01b1"
    )
    assert plan["inputs"]["joint_plan_v56_predecessor"] == {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": "a6614aca7eb9710a706120bdb4a28efb5d4ed9ad058531f0c5f59df8c2326707",
    }
    rules = plan["source_rules"]
    assert rules["panel_1_uses_deepseek_complete_block_repair"] is True
    assert rules["panel_2_uses_deepseek_complete_block_repair"] is True
    assert rules["deepseek_complete_block_repair_model_ids"] == [DEEPSEEK_PRO_MODEL_ID]
    assert rules["deepseek_route_selected_without_quality_scores_or_selections"] is True
    assert rules["superseded_deepseek_route_responses_used"] is False
    assert rules["deepseek_cross_route_response_pooling"] is False


def test_v60_source_maps_use_only_the_final_deepseek_blocks() -> None:
    panel_1 = _load(PANEL_1_PLAN)
    source_44_path = next((ROOT / "benchmark/powered-v44/plan").glob("*dd74a82d*.json"))
    source_44 = _load(source_44_path)
    qwen_45_path = next((ROOT / "benchmark/powered-v45/plan").glob("*5acce6f6*.json"))
    qwen_45 = _load(qwen_45_path)
    prior_50_path = next((ROOT / "benchmark/powered-v50/plan").glob("*ffd46474*.json"))
    prior_50 = _load(prior_50_path)
    prior_55 = _load(PANEL_1_PREDECESSOR)
    sources_1 = _panel_1_model_sources(
        plan=panel_1,
        run_directory=Path("panel-1-base"),
        source_plan=source_44,
        source_plan_path=source_44_path,
        qwen_plan=qwen_45,
        qwen_plan_path=qwen_45_path,
        qwen_run_directory=Path("panel-1-qwen"),
        fable_run_directory=None,
        prior_plan_v50=prior_50,
        prior_plan_v50_path=prior_50_path,
        repair_run_directory=Path("panel-1-coverage"),
        prior_plan_v55=prior_55,
        prior_plan_v55_path=PANEL_1_PREDECESSOR,
        deepseek_repair_run_directory=Path("panel-1-deepseek"),
    )
    assert sources_1[DEEPSEEK_PRO_MODEL_ID] == (Path("panel-1-deepseek"), panel_1)
    assert sources_1["anthropic/claude-opus-5"] == (
        Path("panel-1-coverage"),
        prior_55,
    )

    panel_2 = _load(PANEL_2_PLAN)
    source_49_path = next((ROOT / "benchmark/powered-v49/plan").glob("*6517dd4d*.json"))
    source_49 = _load(source_49_path)
    prior_52_path = next((ROOT / "benchmark/powered-v52/plan").glob("*39ae0c46*.json"))
    prior_52 = _load(prior_52_path)
    prior_54 = _load(PANEL_2_PREDECESSOR)
    sources_2 = _panel_2_model_sources(
        plan=panel_2,
        run_directory=Path("panel-2-base"),
        source_plan=source_49,
        source_plan_path=source_49_path,
        luna_run_directory=Path("panel-2-luna"),
        deepseek_flash_run_directory=Path("panel-2-flash"),
        prior_replacement_plan_v52=prior_52,
        prior_replacement_plan_v52_path=prior_52_path,
        repair_run_directory=Path("panel-2-coverage"),
        prior_plan_v54=prior_54,
        prior_plan_v54_path=PANEL_2_PREDECESSOR,
        deepseek_repair_run_directory=Path("panel-2-deepseek"),
    )
    assert sources_2[DEEPSEEK_PRO_MODEL_ID] == (Path("panel-2-deepseek"), panel_2)
    assert sources_2["anthropic/claude-opus-5"] == (
        Path("panel-2-coverage"),
        prior_54,
    )
