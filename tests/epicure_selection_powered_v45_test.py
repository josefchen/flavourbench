from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flavourbench.epicure_selection_powered_plan_v45 import verify_plan
from flavourbench.epicure_selection_powered_plan_v46 import verify_plan as verify_plan_v46
from flavourbench.epicure_selection_powered_plan_v47 import verify_plan as verify_plan_v47
from flavourbench.epicure_selection_powered_runner import build_cells, validate_inputs
from flavourbench.epicure_selection_repeat_panel_replication_v1 import verify_repeat_panel
from flavourbench.epicure_selection_route_manifest_v45 import (
    FABLE_MODEL_ID,
    QWEN_MODEL_ID,
    ROUTE_SPECS,
    verify_manifest,
)
from flavourbench.epicure_selection_route_manifest_v46 import (
    verify_manifest as verify_manifest_v46,
)
from flavourbench.epicure_selection_taskset_replication_v1 import verify_taskset
from flavourbench.frontier_contract_runner import load_candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
FIRST_TASKSET = ROOT / (
    "benchmark/powered-v44/taskset/"
    "epicure-selection-taskset-a33bf28db372090015118371417b0e8ed1254f416d03d2c2c5816a6a752beb41.json"
)
FIRST_REPEAT = ROOT / (
    "benchmark/powered-v44/plan/"
    "epicure-selection-repeat-panel-96f766df855b93ad1495ec386c70ad88e42c4f896be24b0538cf1084da3c124a.json"
)
SECOND_TASKSET = ROOT / (
    "benchmark/powered-v45/taskset/"
    "epicure-selection-taskset-925ba9d1d4be9c2b7a1e9956ecd6c18d34ffcad22eee28522f16892922c91e3f.json"
)
SECOND_REPEAT = ROOT / (
    "benchmark/powered-v45/plan/"
    "epicure-selection-repeat-panel-36d8c12ff883ead78e53406844ad386eb8999168a61d6931fe17135a2c73acfe.json"
)
SOURCE_MANIFEST = ROOT / (
    "benchmark/powered-v43/manifest/"
    "flavourbench-frontier-refresh-26-33796dd9a0a4580f15fa79ec9cd50179c2b6ddc7c120c03f1814faf8259f8e9d.json"
)
MANIFEST = ROOT / (
    "benchmark/powered-v45/manifest/"
    "flavourbench-frontier-refresh-26-7c81719d454a35144a3eaacda4392a08d12496787b78e39edb7a66742501f197.json"
)
PLAN = ROOT / (
    "benchmark/powered-v45/plan/"
    "epicure-selection-analysis-plan-5acce6f61f731a1f823f314694b7da7c6d9fc02b500e5a9eaaff3002e7096acb.json"
)
MANIFEST_REPLICATION_2 = ROOT / (
    "benchmark/powered-v46/manifest/"
    "flavourbench-frontier-refresh-26-3caaa80fa8c3a646d27b5cad834a2e42fda6edaea82e0c29af9c7e14b1393df7.json"
)
PLAN_REPLICATION_2 = ROOT / (
    "benchmark/powered-v46/plan/"
    "epicure-selection-analysis-plan-61fcdfe6323475910823c35e47367f743fb142d88a06c5ca91883d35bd24c31a.json"
)
PLAN_PANEL_1_COMPOSITE = ROOT / (
    "benchmark/powered-v47/plan/"
    "epicure-selection-analysis-plan-ed01c0da3a5e6204ddcd436cc1322eff08b06a752f37ca229dd3b15a14f91490.json"
)
PREDECESSOR_RELEASE = ROOT / (
    "paper/generated/powered/"
    "flavourbench-powered-release-7aeddf27998b0a8ed0b961cab035e4793305ea120f73ce8f3baa47e4db612cf7.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_route_recovery_changes_only_fable_and_qwen() -> None:
    source = _load(SOURCE_MANIFEST)
    manifest = _load(MANIFEST)
    assert verify_manifest(manifest)
    assert manifest["status"] == "unranked_candidate"
    source_rows = {row["model"]["id"]: row for row in source["models"]}
    rows = {row["model"]["id"]: row for row in manifest["models"]}
    changed = {FABLE_MODEL_ID, QWEN_MODEL_ID}
    assert set(rows) == set(source_rows)
    assert all(rows[model_id] == source_rows[model_id] for model_id in rows.keys() - changed)
    for model_id, spec in ROUTE_SPECS.items():
        row = rows[model_id]
        assert row["endpoint"]["tag"] == spec["tag"]
        assert row["endpoint"]["provider_name"] == spec["provider"]
        provider_policy = row["request_policy"]["provider"]
        assert provider_policy["allow_fallbacks"] is False
        assert provider_policy["only"] == [spec["tag"]]


def test_v45_plan_requires_complete_replacement_not_failed_cell_retry() -> None:
    plan = _load(PLAN)
    assert verify_plan(plan)
    recovery = plan["execution"]["completion_route_recovery"]
    assert recovery["model_ids"] == [FABLE_MODEL_ID, QWEN_MODEL_ID]
    assert recovery["complete_primary_cells_per_model"] == 640
    assert recovery["complete_repeat_cells_per_model"] == 64
    assert recovery["selective_failed_cell_retry"] is False
    assert recovery["cross_route_response_pooling"] is False
    assert recovery["source_v44_responses_used_in_v45_score"] is False
    assert recovery["quality_score_definition"] == "successful_and_parseable_only"
    assert plan["outcomes"]["failed_content_filtered_or_unparseable"] == (
        "excluded_from_quality_score"
    )
    assert plan["outcomes"]["dnf_classification"] is False
    assert plan["outcomes"]["minimum_coverage_for_score"] is None


def test_v45_manifest_and_plan_preflight_replay() -> None:
    manifest, taskset, repeat, plan, _, candidates = validate_inputs(
        manifest_path=MANIFEST,
        manifest_sha256="7c81719d454a35144a3eaacda4392a08d12496787b78e39edb7a66742501f197",
        taskset_path=FIRST_TASKSET,
        repeat_panel_path=FIRST_REPEAT,
        plan_path=PLAN,
        predecessor_release_path=PREDECESSOR_RELEASE,
    )
    assert (
        manifest["content_address"]["digest"]
        == (plan["inputs"]["route_manifest"]["semantic_sha256"])
    )
    assert len(candidates) == 26
    cells = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase="all",
    )
    assert len(cells) == 26 * (640 + 64)
    assert len({cell.cell_id for cell in cells}) == len(cells)
    load_candidate_manifest(
        MANIFEST,
        expected_digest="7c81719d454a35144a3eaacda4392a08d12496787b78e39edb7a66742501f197",
    )


def test_second_task_panel_is_balanced_and_response_blind() -> None:
    first = _load(FIRST_TASKSET)
    second = _load(SECOND_TASKSET)
    repeat = _load(SECOND_REPEAT)
    assert verify_taskset(second, first_panel=first)
    assert verify_repeat_panel(repeat, taskset=second)
    assert second["replication"]["selection_is_response_blind"] is True
    assert second["replication"]["responses_reused"] is False
    assert second["replication"]["novel_anchor_count"] == 538
    assert second["replication"]["shared_anchor_count"] == 102
    assert second["counts"]["tasks"] == 640
    assert second["counts"]["tasks_per_family"] == 160
    assert second["metric_contract"]["invalid_failed_or_unparseable"] == (
        "excluded_from_quality_score"
    )
    assert second["metric_contract"]["posthoc_item_exclusion"] is False


def test_v45_artifact_physical_hashes_are_stable() -> None:
    expected = {
        MANIFEST: "d13b3e83ffc9b8285749d29ed5cafa0156ecb3da2a86bf3b8741b5675f85bd90",
        PLAN: "7b511b06212f4c393544f451095937e53252144e35133c4308bdadb0a8a200ed",
        SECOND_TASKSET: "4c563b1104d5b59d0cca512c19a5ec5d893c529c04a54afef7f2ba53ee75087d",
        SECOND_REPEAT: "3383d641ec908504c218550be0fe3f7b4e2f907192ba47cfc9c415495874c306",
    }
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in expected.items()
    )


def test_replication_2_route_and_plan_are_frozen_before_calls() -> None:
    source = _load(SOURCE_MANIFEST)
    manifest = _load(MANIFEST_REPLICATION_2)
    plan = _load(PLAN_REPLICATION_2)
    assert verify_manifest_v46(manifest)
    assert verify_plan_v46(plan)
    source_rows = {row["model"]["id"]: row for row in source["models"]}
    rows = {row["model"]["id"]: row for row in manifest["models"]}
    assert all(
        rows[model_id] == source_rows[model_id] for model_id in rows if model_id != QWEN_MODEL_ID
    )
    replication = plan["execution"]["replication_2"]
    assert replication["provider_calls"] == 26 * (640 + 64)
    assert replication["selection_is_response_blind"] is True
    assert replication["first_panel_responses_reused"] is False
    assert replication["shared_anchor_count"] == 102
    assert replication["novel_anchor_count"] == 538


def test_replication_2_runner_inputs_replay_exactly() -> None:
    _, taskset, repeat, plan, _, candidates = validate_inputs(
        manifest_path=MANIFEST_REPLICATION_2,
        manifest_sha256="3caaa80fa8c3a646d27b5cad834a2e42fda6edaea82e0c29af9c7e14b1393df7",
        taskset_path=SECOND_TASKSET,
        repeat_panel_path=SECOND_REPEAT,
        plan_path=PLAN_REPLICATION_2,
        predecessor_release_path=PREDECESSOR_RELEASE,
    )
    cells = build_cells(
        plan=plan,
        taskset=taskset,
        repeat_panel=repeat,
        candidates=candidates,
        phase="all",
    )
    assert len(cells) == 26 * (640 + 64)
    assert len({cell.cell_id for cell in cells}) == len(cells)


def test_replication_2_artifact_physical_hashes_are_stable() -> None:
    assert hashlib.sha256(MANIFEST_REPLICATION_2.read_bytes()).hexdigest() == (
        "d8e3652c4378ae87f71a35d26c665ed8b9e9db77e668e4e93e5c35f59f7c941d"
    )
    assert hashlib.sha256(PLAN_REPLICATION_2.read_bytes()).hexdigest() == (
        "71e0dbb8d823aaa4041d9260606bd7fe45199d682b162d5184c4ce5cd804e6af"
    )


def test_panel_1_composite_replaces_only_the_complete_qwen_block() -> None:
    first = _load(
        ROOT / "benchmark/powered-v44/plan/epicure-selection-analysis-plan-"
        "dd74a82d4a34500f22ed91178f63497486fd957e67ebc0136bfa3350d3f6d57e.json"
    )
    replacement = _load(PLAN)
    composite = _load(PLAN_PANEL_1_COMPOSITE)
    assert verify_plan_v47(composite)
    first_rows = {row["model_id"]: row for row in first["roster"]["models"]}
    replacement_rows = {row["model_id"]: row for row in replacement["roster"]["models"]}
    composite_rows = {row["model_id"]: row for row in composite["roster"]["models"]}
    assert composite_rows[QWEN_MODEL_ID] == replacement_rows[QWEN_MODEL_ID]
    assert all(
        composite_rows[model_id] == row
        for model_id, row in first_rows.items()
        if model_id != QWEN_MODEL_ID
    )
    source = composite["execution"]["panel_1_composite"]
    assert source["replacement_model_ids"] == [QWEN_MODEL_ID]
    assert source["replacement_primary_cells"] == 640
    assert source["replacement_repeat_cells"] == 64
    assert source["replacement_is_complete_model_block"] is True
    assert source["superseded_qwen_responses_used"] is False
    assert source["cross_route_response_pooling"] is False
    assert source["selection_uses_scores_or_selections"] is False


def test_panel_1_composite_artifact_physical_hash_is_stable() -> None:
    assert hashlib.sha256(PLAN_PANEL_1_COMPOSITE.read_bytes()).hexdigest() == (
        "48ac08667e39cfd9bfb6c0f38a7546acbc5cd3c715a8bc2a71d7ab843131ad08"
    )
