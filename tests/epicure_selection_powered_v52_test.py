from __future__ import annotations

import json
from pathlib import Path

from flavourbench.epicure_selection_powered_joint_analysis_v1 import _panel_2_model_sources
from flavourbench.epicure_selection_powered_plan_v52 import verify_plan
from flavourbench.epicure_selection_powered_plan_v53 import _as_v51
from flavourbench.epicure_selection_powered_plan_v53 import verify_plan as verify_joint_plan
from flavourbench.epicure_selection_route_manifest_v52 import (
    REPLACEMENT_MODEL_IDS,
    ROUTE_SPECS,
    verify_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    ROOT / "benchmark/powered-v49/manifest/"
    "flavourbench-frontier-refresh-26-"
    "0251a89259c9cedfc2f2c9660481961b71fadfe80bb8db27536b94ef6476fa33.json"
)
MANIFEST = (
    ROOT / "benchmark/powered-v52/manifest/"
    "flavourbench-frontier-refresh-26-"
    "e1b8f5df0c79fafc1c9df0b2062078ef063a1710f5f60c5f5b520b892b013929.json"
)
PLAN = (
    ROOT / "benchmark/powered-v52/plan/"
    "epicure-selection-analysis-plan-"
    "39ae0c4618dc229ce3ba11aed03664e1cbcc4682bd18f86fc6c9b5315b07d2be.json"
)
JOINT_PLAN = (
    ROOT / "benchmark/powered-v53/plan/"
    "epicure-selection-joint-analysis-plan-"
    "b8d6f42fd2c3567703961576dca73cabca7e67cf58700cd2e8e59d63132c4805.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v52_manifest_is_exact_and_changes_only_two_routes() -> None:
    source = _load(SOURCE_MANIFEST)
    manifest = _load(MANIFEST)
    assert verify_manifest(manifest)
    assert manifest["content_address"]["digest"] == (
        "e1b8f5df0c79fafc1c9df0b2062078ef063a1710f5f60c5f5b520b892b013929"
    )
    before = {row["model"]["id"]: row for row in source["models"]}
    after = {row["model"]["id"]: row for row in manifest["models"]}
    assert list(before) == list(after)
    for model_id in before:
        if model_id not in REPLACEMENT_MODEL_IDS:
            assert after[model_id] == before[model_id]
    for model_id in REPLACEMENT_MODEL_IDS:
        assert after[model_id]["endpoint"]["tag"] == ROUTE_SPECS[model_id]["tag"]
        assert after[model_id]["request_policy"]["provider"]["only"] == [
            ROUTE_SPECS[model_id]["tag"]
        ]


def test_v52_plan_requires_complete_score_blind_blocks() -> None:
    plan = _load(PLAN)
    assert verify_plan(plan)
    assert plan["artifact_sha256"] == (
        "39ae0c4618dc229ce3ba11aed03664e1cbcc4682bd18f86fc6c9b5315b07d2be"
    )
    replacement = plan["execution"]["panel_2_route_replacements_v52"]
    assert replacement["replacement_model_ids"] == REPLACEMENT_MODEL_IDS
    assert replacement["replacement_primary_cells_per_model"] == 640
    assert replacement["replacement_repeat_cells_per_model"] == 64
    assert replacement["replacement_blocks_must_be_complete"] is True
    assert replacement["superseded_responses_used"] is False
    assert replacement["cross_route_response_pooling"] is False
    assert replacement["selective_failed_cell_retry"] is False
    assert replacement["selection_uses_scores_or_selections"] is False


def test_v53_joint_plan_preserves_inference_and_replaces_only_panel_2_source() -> None:
    plan = _load(JOINT_PLAN)
    assert verify_joint_plan(plan)
    assert plan["artifact_sha256"] == (
        "b8d6f42fd2c3567703961576dca73cabca7e67cf58700cd2e8e59d63132c4805"
    )
    assert _as_v51(plan)["artifact_sha256"] == (
        "ed452423f88069952b30a880e5a001d64040db5a105ef108079fc6c4efd6d26c"
    )
    assert plan["inputs"]["panel_2_plan"]["semantic_sha256"] == (
        "39ae0c4618dc229ce3ba11aed03664e1cbcc4682bd18f86fc6c9b5315b07d2be"
    )
    rules = plan["source_rules"]
    assert rules["panel_2_uses_complete_route_replacement_blocks"] is True
    assert rules["panel_2_replacement_model_ids"] == REPLACEMENT_MODEL_IDS
    assert rules["superseded_panel_2_responses_used"] is False
    assert rules["cross_route_response_pooling"] is False
    assert rules["selective_failed_cell_retry"] is False


def test_panel_2_composite_maps_only_the_two_replacement_directories() -> None:
    plan = _load(PLAN)
    source_plan_path = (
        ROOT / "benchmark/powered-v49/plan/"
        "epicure-selection-analysis-plan-"
        "6517dd4d018a4e4406bc34e7d68a8cd815e1c70367c498a8185b8518220b7cf0.json"
    )
    source_plan = _load(source_plan_path)
    base = Path("base-panel-2")
    luna = Path("luna-replacement")
    deepseek = Path("deepseek-replacement")
    sources = _panel_2_model_sources(
        plan=plan,
        run_directory=base,
        source_plan=source_plan,
        source_plan_path=source_plan_path,
        luna_run_directory=luna,
        deepseek_flash_run_directory=deepseek,
    )
    assert sources["openai/gpt-5.6-luna-pro"] == (luna, plan)
    assert sources["deepseek/deepseek-v4-flash-0731"] == (deepseek, plan)
    assert all(
        directory == base
        for model_id, (directory, _) in sources.items()
        if model_id not in REPLACEMENT_MODEL_IDS
    )
