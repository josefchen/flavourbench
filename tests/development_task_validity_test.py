from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flavourbench.development_task_validity import (
    DevelopmentTaskValidityError,
    build_validity_artifact,
    select_development_tasks,
    verify_scope_review,
    verify_task_bank,
)

ROOT = Path(__file__).resolve().parents[1]
TASK_BANK = ROOT / (
    "data/season0/frozen/"
    "season0-real-task-bank-"
    "1ce969bdee4124fa44bab46a04feda2a0ebeddf4d37c49c0264b48b3833a4313.json"
)
SCOPE_REVIEW = ROOT / "artifacts/expert-calibration/governance/specialist-scope-review-v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_source_bank_and_scope_review_verify_completely() -> None:
    tasks = verify_task_bank(_load(TASK_BANK))
    quarantine = verify_scope_review(
        _load(SCOPE_REVIEW), task_ids={task["task_id"] for task in tasks}
    )

    assert len(tasks) == 120
    assert len(quarantine) == 17
    assert len([task for task in tasks if task["task_id"] not in quarantine]) == 103
    assert sum(task["synthetic"] is not False for task in tasks) == 0


def test_balanced_development_selection_is_quality_first_with_recent_floor() -> None:
    tasks = verify_task_bank(_load(TASK_BANK))
    quarantine = verify_scope_review(
        _load(SCOPE_REVIEW), task_ids={task["task_id"] for task in tasks}
    )
    selected = select_development_tasks(tasks, quarantined=quarantine)

    assert len(selected) == 40
    assert {family: sum(task["family"] == family for task in selected) for family in (
        "substitution",
        "composition",
        "cookability",
        "evidence",
    )} == {
        "substitution": 10,
        "composition": 10,
        "cookability": 10,
        "evidence": 10,
    }
    for family in ("substitution", "composition", "cookability", "evidence"):
        assert sum(
            task["family"] == family
            and task["source"]["created_utc"] >= "2025-01-01T00:00:00Z"
            for task in selected
        ) >= 2
    assert not {task["task_id"] for task in selected}.intersection(quarantine)


def test_validity_artifact_exposes_missing_confirmatory_evidence() -> None:
    artifact = build_validity_artifact(
        task_bank=_load(TASK_BANK), scope_review=_load(SCOPE_REVIEW)
    )

    assert artifact["counts"]["source_integrity_verified"] == 120
    assert artifact["counts"]["general_track_candidates"] == 103
    assert artifact["counts"]["selected_development_tasks"] == 40
    assert artifact["counts"]["synthetic_tasks"] == 0
    assert artifact["counts"]["independently_human_validated_tasks"] == 0
    assert artifact["validity_layers"]["provenance_integrity"]["status"] == "verified"
    assert artifact["validity_layers"]["task_specific_criteria"]["status"] == "missing"
    assert artifact["claim_boundary"]["supports_real_current_model_development_runs"] is True
    assert artifact["claim_boundary"]["supports_official_leaderboard"] is False
    assert all(task["task_specific_success_criteria"] == [] for task in artifact["tasks"])


def test_mutated_or_synthetic_source_fails_closed() -> None:
    bank = _load(TASK_BANK)
    mutated = copy.deepcopy(bank)
    mutated["tasks"][0]["prompt"] += " altered"
    with pytest.raises(DevelopmentTaskValidityError, match="content address"):
        verify_task_bank(mutated)

    synthetic = copy.deepcopy(bank)
    synthetic["synthetic_tasks"] = 1
    synthetic.pop("artifact_sha256")
    from flavourbench.real_task_bank import sha256_json

    synthetic["artifact_sha256"] = sha256_json(synthetic)
    with pytest.raises(DevelopmentTaskValidityError, match="Synthetic|synthetic"):
        verify_task_bank(synthetic)
