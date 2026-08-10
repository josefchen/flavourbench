from __future__ import annotations

import json
from pathlib import Path

from flavourbench.development_task_validity_v2 import (
    FAMILIES,
    build_validity_artifact,
    surface_dependency_reasons,
)

ROOT = Path(__file__).resolve().parents[1]
TASK_BANK = next(
    (ROOT / "data/season0/frozen").glob("season0-real-task-bank-*.json")
)
SCOPE_REVIEW = (
    ROOT / "artifacts/expert-calibration/governance/specialist-scope-review-v1.json"
)


def test_surface_dependency_screen_is_narrow_and_deterministic() -> None:
    assert surface_dependency_reasons("Use https://example.test and the attached photo.") == [
        "external_url_dependency_signal",
        "visual_context_dependency_signal",
    ]
    assert surface_dependency_reasons(
        "Describe how temperature and mixing alter this custard's texture."
    ) == []


def test_real_human_development_pool_excludes_surface_dependencies() -> None:
    artifact = build_validity_artifact(
        task_bank=json.loads(TASK_BANK.read_text(encoding="utf-8")),
        scope_review=json.loads(SCOPE_REVIEW.read_text(encoding="utf-8")),
    )
    assert artifact["schema_version"] == "flavourbench-development-task-validity-v2"
    assert artifact["counts"]["synthetic_tasks"] == 0
    assert artifact["counts"]["selected_development_tasks"] == 40
    assert artifact["counts"]["surface_dependency_quarantined"] > 0
    assert artifact["counts"]["per_family"] == {family: 10 for family in FAMILIES}
    assert all(
        task["surface_dependency_screen"]["status"] == "pass"
        and not surface_dependency_reasons(task["prompt"])
        for task in artifact["tasks"]
    )
    assert artifact["claim_boundary"]["supports_confirmatory_task_validity"] is False
    assert artifact["claim_boundary"]["synthetic_tasks"] == 0
