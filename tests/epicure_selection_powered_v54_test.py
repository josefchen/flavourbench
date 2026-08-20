from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flavourbench.epicure_selection_powered_joint_analysis_v1 import (
    _panel_1_model_sources,
    _panel_2_model_sources,
)
from flavourbench.epicure_selection_powered_plan_v54 import verify_plan as verify_plan_v54
from flavourbench.epicure_selection_powered_plan_v55 import verify_plan as verify_plan_v55
from flavourbench.epicure_selection_powered_plan_v56 import verify_plan as verify_plan_v56
from flavourbench.epicure_selection_route_manifest_v54 import (
    REPLACEMENT_MODEL_IDS,
    ROUTE_SPECS,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    ROOT / "benchmark/powered-v52/manifest/"
    "flavourbench-frontier-refresh-26-"
    "e1b8f5df0c79fafc1c9df0b2062078ef063a1710f5f60c5f5b520b892b013929.json"
)
MANIFEST = (
    ROOT / "benchmark/powered-v54/manifest/"
    "flavourbench-frontier-refresh-26-"
    "ac2c9c1ab64552fb9c367fc5c66e368fac754fdc66ebe566d751e41d382fde06.json"
)
PANEL_2_PLAN = (
    ROOT / "benchmark/powered-v54/plan/"
    "epicure-selection-analysis-plan-"
    "314702bc94a802d530b421ee73a52fb12eea805b43648bd0d9786df785469069.json"
)
PANEL_1_PLAN = (
    ROOT / "benchmark/powered-v55/plan/"
    "epicure-selection-analysis-plan-"
    "8577a9a32c5fb266f12b131c309f4543c6fa2cd42538abd16eefbf4c09d578ed.json"
)
JOINT_PLAN = (
    ROOT / "benchmark/powered-v56/plan/"
    "epicure-selection-joint-analysis-plan-"
    "cc5b423e01445a69e88be7f3be981497cc28a16fa7043c865db546f8499b27f1.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v54_manifest_changes_only_the_score_blind_repair_set() -> None:
    source = _load(SOURCE_MANIFEST)
    manifest = _load(MANIFEST)
    assert verify_manifest(manifest)
    assert manifest["content_address"]["digest"] == (
        "ac2c9c1ab64552fb9c367fc5c66e368fac754fdc66ebe566d751e41d382fde06"
    )
    assert _sha256(MANIFEST) == ("4a6ffe621b054be3845e174006d0ffa2312fd0de8034e8d7b9d50771c015ceb5")
    before = {row["model"]["id"]: row for row in source["models"]}
    after = {row["model"]["id"]: row for row in manifest["models"]}
    assert list(before) == list(after)
    for model_id in before:
        if model_id not in REPLACEMENT_MODEL_IDS:
            assert after[model_id] == before[model_id]
    for model_id in REPLACEMENT_MODEL_IDS:
        assert after[model_id]["endpoint"]["tag"] == ROUTE_SPECS[model_id]["tag"]
        assert after[model_id]["request_policy"]["provider"] == {
            "only": [ROUTE_SPECS[model_id]["tag"]],
            "allow_fallbacks": False,
            "data_collection": "deny",
            "require_parameters": True,
        }


def test_v54_panel_2_plan_freezes_full_blocks_and_lower_concurrency() -> None:
    plan = _load(PANEL_2_PLAN)
    assert verify_plan_v54(plan)
    assert _sha256(PANEL_2_PLAN) == (
        "5a3a05eca2f5458f0a196283579888b5a54a3ce3fb6a1ff15952a7a754bb15ee"
    )
    repair = plan["execution"]["complete_coverage_route_replacements_v54"]
    assert repair["replacement_model_ids"] == REPLACEMENT_MODEL_IDS
    assert repair["replacement_primary_cells_per_model"] == 640
    assert repair["replacement_repeat_cells_per_model"] == 64
    assert repair["replacement_blocks_must_be_complete"] is True
    assert repair["superseded_responses_used"] is False
    assert repair["selective_failed_cell_retry"] is False
    assert repair["selection_uses_scores_or_selections"] is False
    assert plan["execution"]["collection_concurrency"]["global"] == 18
    assert plan["execution"]["collection_concurrency"]["per_model_default"] == 2


def test_v55_panel_1_plan_uses_the_same_routes_without_pooling() -> None:
    plan = _load(PANEL_1_PLAN)
    assert verify_plan_v55(plan)
    assert _sha256(PANEL_1_PLAN) == (
        "b995121c8dffac327d084af72da63b301cca9cbf03b18da1bead708ab9f9c120"
    )
    repair = plan["execution"]["complete_coverage_route_replacements_v55"]
    assert repair["replacement_model_ids"] == REPLACEMENT_MODEL_IDS
    assert repair["replacement_primary_cells_per_model"] == 640
    assert repair["replacement_repeat_cells_per_model"] == 64
    assert repair["cross_route_response_pooling"] is False
    assert repair["superseded_responses_used"] is False
    assert repair["selective_failed_cell_retry"] is False
    assert plan["execution"]["collection_concurrency"]["global"] == 18
    assert plan["execution"]["collection_concurrency"]["per_model_default"] == 2


def test_v56_joint_plan_binds_both_complete_coverage_repair_plans() -> None:
    plan = _load(JOINT_PLAN)
    assert verify_plan_v56(plan)
    assert _sha256(JOINT_PLAN) == (
        "a6614aca7eb9710a706120bdb4a28efb5d4ed9ad058531f0c5f59df8c2326707"
    )
    rules = plan["source_rules"]
    assert rules["panel_1_uses_complete_coverage_repair_blocks"] is True
    assert rules["panel_2_uses_complete_coverage_repair_blocks"] is True
    assert rules["complete_coverage_repair_model_ids"] == REPLACEMENT_MODEL_IDS
    assert rules["superseded_coverage_route_responses_used"] is False
    assert rules["cross_route_response_pooling"] is False
    assert rules["selective_failed_cell_retry"] is False


def test_v56_model_source_maps_are_exact_and_do_not_pool_routes() -> None:
    panel_1 = _load(PANEL_1_PLAN)
    panel_2 = _load(PANEL_2_PLAN)
    source_44_path = (
        ROOT / "benchmark/powered-v44/plan/epicure-selection-analysis-plan-"
        "dd74a82d4a34500f22ed91178f63497486fd957e67ebc0136bfa3350d3f6d57e.json"
    )
    source_44 = _load(source_44_path)
    qwen_45_path = (
        ROOT / "benchmark/powered-v45/plan/epicure-selection-analysis-plan-"
        "5acce6f61f731a1f823f314694b7da7c6d9fc02b500e5a9eaaff3002e7096acb.json"
    )
    qwen_45 = _load(qwen_45_path)
    prior_50_path = (
        ROOT / "benchmark/powered-v50/plan/epicure-selection-analysis-plan-"
        "ffd464743c220643b6db285db1c98122b5b7417676bd157d51497fc0e27e4da0.json"
    )
    prior_50 = _load(prior_50_path)
    base_1 = Path("panel-1-base")
    qwen_run = Path("panel-1-qwen")
    repair_1 = Path("panel-1-repair")
    sources_1 = _panel_1_model_sources(
        plan=panel_1,
        run_directory=base_1,
        source_plan=source_44,
        source_plan_path=source_44_path,
        qwen_plan=qwen_45,
        qwen_plan_path=qwen_45_path,
        qwen_run_directory=qwen_run,
        fable_run_directory=None,
        prior_plan_v50=prior_50,
        prior_plan_v50_path=prior_50_path,
        repair_run_directory=repair_1,
    )
    assert sources_1["qwen/qwen3.8-2.4t-a95b"] == (qwen_run, qwen_45)
    assert all(sources_1[model_id] == (repair_1, panel_1) for model_id in REPLACEMENT_MODEL_IDS)

    source_49_path = (
        ROOT / "benchmark/powered-v49/plan/epicure-selection-analysis-plan-"
        "6517dd4d018a4e4406bc34e7d68a8cd815e1c70367c498a8185b8518220b7cf0.json"
    )
    source_49 = _load(source_49_path)
    prior_52_path = (
        ROOT / "benchmark/powered-v52/plan/epicure-selection-analysis-plan-"
        "39ae0c4618dc229ce3ba11aed03664e1cbcc4682bd18f86fc6c9b5315b07d2be.json"
    )
    prior_52 = _load(prior_52_path)
    base_2 = Path("panel-2-base")
    luna_run = Path("panel-2-luna")
    flash_run = Path("panel-2-flash")
    repair_2 = Path("panel-2-repair")
    sources_2 = _panel_2_model_sources(
        plan=panel_2,
        run_directory=base_2,
        source_plan=source_49,
        source_plan_path=source_49_path,
        luna_run_directory=luna_run,
        deepseek_flash_run_directory=flash_run,
        prior_replacement_plan_v52=prior_52,
        prior_replacement_plan_v52_path=prior_52_path,
        repair_run_directory=repair_2,
    )
    assert sources_2["openai/gpt-5.6-luna-pro"] == (luna_run, prior_52)
    assert sources_2["deepseek/deepseek-v4-flash-0731"] == (flash_run, prior_52)
    assert all(sources_2[model_id] == (repair_2, panel_2) for model_id in REPLACEMENT_MODEL_IDS)
