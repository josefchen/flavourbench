from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(str(ROOT / "scripts/build_human_review_qa_artifact.py"))
build = BUILDER["build"]
load_report = BUILDER["load_report"]
write_projection = BUILDER["write_projection"]
QA_DIR = ROOT / "artifacts/season1/human-review/operational-qa"
QA_V3 = QA_DIR / (
    "restricted-operational-qa-"
    "f1c262e075ccc73a4db0bb3c328e6c90d66d7d01eceaf54dfc9912e8c96e9fea.json"
)
QA_V2 = QA_DIR / (
    "restricted-operational-qa-"
    "9d9886be2fd92319c0ef41e76183c84dfb59c46d324f713da36c14f2643a60fa.json"
)


def test_builder_projects_all_seven_reviewed_quarantined_tasks() -> None:
    report = load_report(QA_V3)
    artifact = build(report)

    assert artifact["lineage"] == {
        "input_schema_version": "flavourbench-human-review-operational-qa-v3",
        "input_artifact_sha256": report["artifact_sha256"],
        "scope_governance_artifact_sha256": report["scope_audit"]["governance_review"][
            "artifact_sha256"
        ],
    }
    assert len(artifact["snapshot"]["datasets"]["scope_quarantine"]) == 7
    task_scope = next(
        row
        for row in artifact["snapshot"]["datasets"]["control_actions"]
        if row["control"] == "Task scope"
    )
    assert task_scope["observed_state"] == "7 reviewed tasks require specialist governance"
    boundary = next(
        block for block in artifact["manifest"]["blocks"] if block["id"] == "boundary"
    )
    assert "7 reviewed tasks require specialist governance" in boundary["body"]
    assert "five reviewed tasks" not in boundary["body"]


def test_builder_rejects_superseded_v2_qa_evidence() -> None:
    with pytest.raises(ValueError, match="unsupported QA evidence schema"):
        load_report(QA_V2)


def test_projection_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    projection = tmp_path / "report_projection.sqlite"
    original = b"historical projection"
    projection.write_bytes(original)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_projection(projection, {"metrics": [{"count": 1}]})

    assert projection.read_bytes() == original
