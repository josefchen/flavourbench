from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flavourbench.epicure_selection_powered_plan_v60 import verify_plan as verify_plan_v60
from flavourbench.epicure_selection_powered_plan_v62 import verify_plan as verify_plan_v62
from flavourbench.epicure_selection_powered_plan_v63 import verify_plan as verify_plan_v63
from flavourbench.epicure_selection_powered_plan_v64 import _as_v60
from flavourbench.epicure_selection_powered_plan_v64 import verify_plan as verify_plan_v64
from flavourbench.epicure_selection_route_manifest_v54 import DEEPSEEK_PRO_MODEL_ID
from flavourbench.epicure_selection_route_manifest_v61 import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
V57 = ROOT / "benchmark/powered-v57/manifest"
V60 = ROOT / "benchmark/powered-v60/plan"
V61 = ROOT / "benchmark/powered-v61/manifest"
V62 = ROOT / "benchmark/powered-v62/plan"
V63 = ROOT / "benchmark/powered-v63/plan"
V64 = ROOT / "benchmark/powered-v64/plan"


def _one(directory: Path) -> tuple[Path, dict[str, object]]:
    paths = list(directory.glob("*.json"))
    assert len(paths) == 1
    return paths[0], json.loads(paths[0].read_text(encoding="utf-8"))


def _physical(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v61_route_successor_is_score_blind_and_changes_only_deepseek() -> None:
    predecessor_path, predecessor = _one(V57)
    path, manifest = _one(V61)
    assert _physical(predecessor_path) == (
        "95d9cab5fe6f7d638c77cc8e8cda22db85eaf6be8fdf8447510bd759d76fa10c"
    )
    assert _physical(path) == "693f391a8d81eb11fe7c93b57e43f631285ba4ec328804bf9219f854b679de31"
    assert manifest["content_address"]["digest"] == (
        "354beb4777c561f4d16a531d2a83bd5589e657e218c58f81233661682c018a1c"
    )
    assert verify_manifest(manifest)
    before = {row["model"]["id"]: row for row in predecessor["models"]}
    after = {row["model"]["id"]: row for row in manifest["models"]}
    assert {
        model_id: row for model_id, row in before.items() if model_id != DEEPSEEK_PRO_MODEL_ID
    } == {model_id: row for model_id, row in after.items() if model_id != DEEPSEEK_PRO_MODEL_ID}
    deepseek = after[DEEPSEEK_PRO_MODEL_ID]
    assert deepseek["endpoint"]["tag"] == "gmicloud/fp8"
    refresh = manifest["deepseek_complete_block_repair_v61"]
    assert refresh["quality_scores_or_selections_used"] is False
    assert refresh["selective_failed_cell_retry"] is False
    assert refresh["complete_replacement_blocks_required"] is True
    assert [
        refresh["baseten_transport_projections"][panel]["completed_response_artifacts"]
        for panel in ("panel_1", "panel_2")
    ] == [70, 72]
    assert [
        refresh["historical_gmicloud_transport_projections"][panel]["completed_response_artifacts"]
        for panel in ("panel_1", "panel_2")
    ] == [699, 704]


def test_v62_v63_and_joint_v64_reproduce_their_frozen_lineage() -> None:
    panel_1_path, panel_1 = _one(V62)
    panel_2_path, panel_2 = _one(V63)
    joint_path, joint = _one(V64)
    predecessor_path, predecessor = _one(V60)
    assert _physical(panel_1_path) == (
        "19a8f1b5da3414e7e877db7486300911b82df7fa37bc3706dc9822ba6e40f642"
    )
    assert _physical(panel_2_path) == (
        "ebf5e7ad7790b4488865d33d5cc173ff8f28c679e6e850b9bd3de4bcbf4e51ff"
    )
    assert _physical(joint_path) == (
        "692e974c77f928ab5b9f0f1ddee9003007c3fc1025ddc012366f9888986224e6"
    )
    assert verify_plan_v62(panel_1)
    assert verify_plan_v63(panel_2)
    assert verify_plan_v64(joint)
    reconstructed = _as_v60(joint)
    assert verify_plan_v60(reconstructed)
    assert reconstructed == predecessor
    assert joint["inputs"]["joint_plan_v60_predecessor"] == {
        "semantic_sha256": predecessor["artifact_sha256"],
        "physical_sha256": _physical(predecessor_path),
    }
    assert joint["inputs"]["panel_1_plan"]["semantic_sha256"] == panel_1["artifact_sha256"]
    assert joint["inputs"]["panel_2_plan"]["semantic_sha256"] == panel_2["artifact_sha256"]
