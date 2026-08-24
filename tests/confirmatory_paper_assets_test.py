from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from flavourbench.confirmatory_paper_assets import (
    ConfirmatoryPaperAssetError,
    load_confirmatory_inputs,
    render_macros,
    render_method_table,
)
from flavourbench.season1_method_validation import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "contracts/season1/season1-study-design-v5.json"
BLUEPRINT = ROOT / "contracts/season1/season1-construct-blueprint-v1.json"
METHOD = (
    ROOT / "contracts/season1/method-validation/"
    "season1-statistical-method-validation-"
    "0b4345e523fdaa97d1b406cd1f2165540d0f9ad338bb49f3ac656da73e3c1933.json"
)


def test_confirmatory_assets_reproduce_and_keep_simulation_out_of_scoring() -> None:
    design, method = load_confirmatory_inputs(
        study_design_path=DESIGN,
        method_validation_path=METHOD,
        construct_blueprint_path=BLUEPRINT,
    )

    table = render_method_table(method)
    macros = render_macros(design, method)
    assert "Null with 20\\% ties" in table
    assert "+0.10 half-win effect" in table
    assert "95.2\\%" in table
    assert "88.9\\%" in table
    assert r"\newcommand{\ConfirmatoryResponseArms}{12,800}" in macros
    assert r"\newcommand{\ConfirmatoryTotalPlannedRealArms}{14,560}" in macros
    assert r"\newcommand{\ConfirmatoryPostCollectionRandomAuditTasks}{60}" in macros
    assert r"\newcommand{\ConfirmatoryReliabilityPanelArms}{1,920}" in macros
    assert r"\newcommand{\ConfirmatoryPromptSensitivityArms}{480}" in macros
    assert r"\newcommand{\ConfirmatoryKitchenExecutions}{48}" in macros
    assert r"\newcommand{\ConfirmatoryTaskCount}{240}" in macros
    assert r"\newcommand{\ConfirmatoryEvidenceReviews}{480}" in macros
    assert r"\newcommand{\ConfirmatoryAdmissionDecisions}{1,680}" in macros
    assert r"\newcommand{\ConfirmatoryDistinctPeoplePerTask}{6}" in macros
    assert r"\newcommand{\ConfirmatoryContaminationCalibrationCases}{150}" in macros
    assert r"\newcommand{\ConfirmatoryContaminationParaphraseRecall}{0.85}" in macros
    assert method["claim_boundary"]["scored_benchmark_observations"] == 0
    assert method["claim_boundary"]["leaderboard_use"] is False


def test_confirmatory_assets_reject_a_rehashed_smaller_collection(tmp_path: Path) -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    tampered = deepcopy(design)
    tampered["primary_controlled_collection"]["model_arena"]["total_battles"] = 32
    payload = {key: value for key, value in tampered.items() if key != "artifact_sha256"}
    tampered["artifact_sha256"] = canonical_sha256(payload)
    path = tmp_path / "study-design.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ConfirmatoryPaperAssetError, match="violates the paper contract"):
        load_confirmatory_inputs(
            study_design_path=path,
            method_validation_path=METHOD,
            construct_blueprint_path=BLUEPRINT,
        )
