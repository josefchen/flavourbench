from __future__ import annotations

import json
import runpy
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "paper/flavourbench/readiness-report/build_report_artifact.py"
BUILDER = runpy.run_path(str(SCRIPT))
build = BUILDER["build"]
validate_human_qa = BUILDER["validate_human_qa"]
write_projection = BUILDER["write_projection"]
QA_DIR = ROOT / "flavourbench/artifacts/season1/human-review/operational-qa"
QA_V3 = QA_DIR / (
    "restricted-operational-qa-"
    "f1c262e075ccc73a4db0bb3c328e6c90d66d7d01eceaf54dfc9912e8c96e9fea.json"
)
QA_V2 = QA_DIR / (
    "restricted-operational-qa-"
    "9d9886be2fd92319c0ef41e76183c84dfb59c46d324f713da36c14f2643a60fa.json"
)


def _arguments() -> Namespace:
    return Namespace(
        analysis=ROOT
        / (
            "flavourbench/artifacts/season0/analysis-v6/season0-automated-analysis-"
            "ab45eff77098a97fc05ef7ee5ca689b00724381e4bd8c6f7e4dd60c86fb61d97.json"
        ),
        flow=ROOT / "paper/flavourbench/generated/pilot/pilot-flow.csv",
        models=ROOT / "paper/flavourbench/generated/pilot/pilot-model-uncertainty.csv",
        uplift=ROOT / "paper/flavourbench/generated/pilot/pilot-epicure-robustness.csv",
        bounds=ROOT / "paper/flavourbench/generated/pilot/pilot-preference-bounds.csv",
        reliability=ROOT
        / "paper/flavourbench/generated/pilot/pilot-condition-reliability.csv",
        frontier_bundle=ROOT
        / (
            "flavourbench/artifacts/frontier-refresh/2026-07-28/evidence-bundle-v1/"
            "frontier-contract-evidence-"
            "29515c149ab91a7734f528af5a88fa7d3735644445bdeaf7e3e073a2108d1dd3.json"
        ),
        season_manifest=ROOT
        / (
            "flavourbench/artifacts/season0/manifests/season0-model-manifest-"
            "3919def66686b4bd939c94cdd89659f63ae2afbbf03288413129e2ea8d6b83d2.json"
        ),
        hold_register=ROOT / "governance/hold-register.jsonl",
        human_qa=QA_V3,
    )


def test_report_uses_only_v3_containment_lineage() -> None:
    artifact = build(_arguments())
    containment = artifact["snapshot"]["datasets"]["qa_containment"]

    assert containment == [
        {
            "status": "restricted_operational_qa",
            "schema_version": "flavourbench-human-review-operational-qa-v3",
            "presentations_excluded": 32,
            "artifact_sha256": BUILDER["HUMAN_QA_ARTIFACT_SHA256"],
            "file_sha256": "d1fa8b70e4bc935209a72827aa5c7149fbce644ac410dfa6ce1b85d952697b47",
            "historical_review_session_pool_sha256": BUILDER[
                "HISTORICAL_REVIEW_SESSION_POOL_SHA256"
            ],
            "scope_governance_artifact_sha256": BUILDER[
                "SCOPE_GOVERNANCE_ARTIFACT_SHA256"
            ],
            "governed_quarantine_tasks": 17,
            "reviewed_quarantine_tasks": 7,
            "paper_use": False,
            "ranking_use": False,
            "research_use": False,
        }
    ]
    serialized = json.dumps(containment)
    assert "preference" not in serialized
    assert "repeat" not in serialized


def test_report_rejects_superseded_v2_qa() -> None:
    with pytest.raises(ValueError, match="incomplete v3 lineage|containment contract"):
        validate_human_qa(json.loads(QA_V2.read_text(encoding="utf-8")))


def test_projection_writer_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "report_projection.sqlite"
    path.write_bytes(b"historical")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_projection(path, {"qa_containment": [{"count": 1}]})

    assert path.read_bytes() == b"historical"
