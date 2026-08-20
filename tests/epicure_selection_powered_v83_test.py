from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from flavourbench.epicure_selection_powered_plan_v83 import (
    CORE_FAMILIES,
    MODEL_COUNT,
    PAIRWISE_HYPOTHESES,
    PRIMARY_TASKS,
    TASKS_PER_PANEL_FAMILY,
    build_plan,
    verify_plan,
)

ROOT = Path(__file__).resolve().parents[1]
PREDECESSOR = ROOT / (
    "benchmark/powered-v77/plan/epicure-selection-joint-analysis-plan-"
    "9a6a835f956b276f98f4aeba6e20e218a5c2567e599d67602b50d990b9a4d091.json"
)
TASKSET_1 = ROOT / (
    "benchmark/powered-v44/taskset/epicure-selection-taskset-"
    "a33bf28db372090015118371417b0e8ed1254f416d03d2c2c5816a6a752beb41.json"
)
REPEAT_1 = ROOT / (
    "benchmark/powered-v44/plan/epicure-selection-repeat-panel-"
    "96f766df855b93ad1495ec386c70ad88e42c4f896be24b0538cf1084da3c124a.json"
)
SOURCE_PLAN_1 = ROOT / (
    "benchmark/powered-v55/plan/epicure-selection-analysis-plan-"
    "8577a9a32c5fb266f12b131c309f4543c6fa2cd42538abd16eefbf4c09d578ed.json"
)
SOURCE_DIRS_1 = (
    ROOT / "benchmark/powered-v55/run-panel1-repair",
    ROOT / "benchmark/powered-v77/run-p1-fable-coverage-completion-1",
)
TASKSET_2 = ROOT / (
    "benchmark/powered-v45/taskset/epicure-selection-taskset-"
    "925ba9d1d4be9c2b7a1e9956ecd6c18d34ffcad22eee28522f16892922c91e3f.json"
)
REPEAT_2 = ROOT / (
    "benchmark/powered-v45/plan/epicure-selection-repeat-panel-"
    "36d8c12ff883ead78e53406844ad386eb8999168a61d6931fe17135a2c73acfe.json"
)
SOURCE_PLAN_2 = ROOT / (
    "benchmark/powered-v54/plan/epicure-selection-analysis-plan-"
    "314702bc94a802d530b421ee73a52fb12eea805b43648bd0d9786df785469069.json"
)
SOURCE_DIRS_2 = (
    ROOT / "benchmark/powered-v54/run-panel2-repair",
    ROOT / "benchmark/powered-v77/run-p2-fable-coverage-completion-1",
)
PARSER = ROOT / "src/flavourbench/selection_response_parser_v3.py"
PLAN = ROOT / (
    "benchmark/powered-v83/plan/epicure-selection-joint-analysis-plan-"
    "31f45aaf447b9337e07b9b27a75c9706bb6523efec3ad7e2738f76b9fc9d798b.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _physical(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build() -> dict[str, object]:
    return build_plan(
        predecessor=_load(PREDECESSOR),
        predecessor_path=PREDECESSOR,
        panel_1_taskset=_load(TASKSET_1),
        panel_1_taskset_path=TASKSET_1,
        panel_1_repeat=_load(REPEAT_1),
        panel_1_repeat_path=REPEAT_1,
        panel_1_source_plan=_load(SOURCE_PLAN_1),
        panel_1_source_plan_path=SOURCE_PLAN_1,
        panel_1_source_directories=SOURCE_DIRS_1,
        panel_2_taskset=_load(TASKSET_2),
        panel_2_taskset_path=TASKSET_2,
        panel_2_repeat=_load(REPEAT_2),
        panel_2_repeat_path=REPEAT_2,
        panel_2_source_plan=_load(SOURCE_PLAN_2),
        panel_2_source_plan_path=SOURCE_PLAN_2,
        panel_2_source_directories=SOURCE_DIRS_2,
        parser_path=PARSER,
    )


def test_v83_exact_artifact_rebuilds_from_score_blind_fable_validity() -> None:
    document = _load(PLAN)
    assert _physical(PLAN) == "fdbaa1ea87e783ef8b216dd3637671ae8b78666e18de24cfc1ba42506dd9cec3"
    assert document["artifact_sha256"] == (
        "31f45aaf447b9337e07b9b27a75c9706bb6523efec3ad7e2738f76b9fc9d798b"
    )
    assert verify_plan(document)
    if not all(path.is_dir() for path in (*SOURCE_DIRS_1, *SOURCE_DIRS_2)):
        pytest.skip("raw response sources are distributed through the Hugging Face dataset")
    assert _build() == document


def test_v83_has_534_balanced_tasks_and_all_27_models() -> None:
    document = _load(PLAN)
    core = document["common_core"]
    assert core["ranked_models"] == MODEL_COUNT == 27
    assert core["primary_tasks_per_model"] == PRIMARY_TASKS == 534
    assert core["model_task_cells"] == 14_418
    assert core["pairwise_hypotheses"] == PAIRWISE_HYPOTHESES == 351
    assert core["included_families"] == list(CORE_FAMILIES)
    assert core["excluded_family"] == "cultural_composition"
    all_task_ids: list[str] = []
    for panel in ("panel_1", "panel_2"):
        selected = core["panels"][panel]["selected_task_ids_by_family"]
        assert set(selected) == set(CORE_FAMILIES)
        assert all(len(selected[family]) == TASKS_PER_PANEL_FAMILY for family in CORE_FAMILIES)
        panel_ids = [task_id for family in CORE_FAMILIES for task_id in selected[family]]
        assert len(panel_ids) == len(set(panel_ids)) == 267
        all_task_ids.extend(panel_ids)
    assert len(all_task_ids) == len(set(all_task_ids)) == PRIMARY_TASKS


def test_v83_records_exact_fable_availability_without_quality_adaptation() -> None:
    core = _load(PLAN)["common_core"]
    assert core["panels"]["panel_1"]["available_valid_tasks_by_family"] == {
        "constraint": 93,
        "pairing": 89,
        "substitution": 159,
    }
    assert core["panels"]["panel_2"]["available_valid_tasks_by_family"] == {
        "constraint": 94,
        "pairing": 105,
        "substitution": 150,
    }
    assert core["panels"]["panel_1"]["valid_unique_primary_tasks"] == 352
    assert core["panels"]["panel_2"]["valid_unique_primary_tasks"] == 362
    assert core["quality_scores_or_observed_selections_used_for_task_selection"] is False
    assert core["failures_and_unparseable_responses_scored_as_zero"] is False
    assert core["selection_is_conditioned_on_fable_response_validity"] is True
    assert core["full_four_family_26_model_analysis_retained_separately"] is True


def test_v83_verifier_rejects_task_or_claim_drift() -> None:
    document = _load(PLAN)
    drifted = copy.deepcopy(document)
    drifted["common_core"]["failures_and_unparseable_responses_scored_as_zero"] = True
    assert not verify_plan(drifted)
    drifted = copy.deepcopy(document)
    drifted["common_core"]["panels"]["panel_1"]["selected_task_ids_by_family"]["pairing"].pop()
    assert not verify_plan(drifted)
