from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.frontier_model_arena_coverage_assets import (
    CoverageAssetError,
    render_assets,
)

ROOT = Path(__file__).resolve().parents[1]
CORRECTED = ROOT / (
    "artifacts/season1/current-quality-run/frontier-coverage-v4-postrun/"
    "frontier-corrected-development-arena-"
    "234f5b5e3364f0e0f2fddc0f23d47d1d670df509c5707e35cb713183264c5c5e.json"
)
CORRECTED_UPLIFT = ROOT / (
    "artifacts/season1/current-quality-run/release-package-remediation-v1/"
    "frontier-uplift-policy-hold-successor-"
    "e4bbad3fe7ec8e4e6a6f16bb5c6634201e3be040880ab8e8700fc9307c2379b3.json"
)
CORRECTED_COVERAGE = ROOT / (
    "artifacts/season1/current-quality-run/release-package-remediation-v1/"
    "frontier-coverage-policy-hold-successor-"
    "ab17f649e57098d14fc815a087e22c090087301bacf7ccc30f5532dd58c3823d.json"
)


def test_corrected_arena_renders_vector_dependence_assets(tmp_path: Path) -> None:
    first = render_assets(CORRECTED, tmp_path / "first", CORRECTED_UPLIFT, CORRECTED_COVERAGE)
    second = render_assets(CORRECTED, tmp_path / "second", CORRECTED_UPLIFT, CORRECTED_COVERAGE)
    assert first.keys() == second.keys()
    for key in first:
        assert first[key].read_bytes() == second[key].read_bytes()
    provenance = json.loads(first["provenance"].read_text(encoding="utf-8"))
    assert provenance["candidate_comparison_rows"] == 915
    assert provenance["compared_response_arms"] == 188
    assert provenance["missing_model_pair_family_cells"] == 73
    assert provenance["candidate_uplift_pairs"] == 186
    macros = first["macros"].read_text(encoding="utf-8")
    assert r"\newcommand{\FrontierCurrentArenaComparisons}{915}" in macros
    assert r"\newcommand{\FrontierCurrentArenaComparedAnswers}{188}" in macros
    assert r"\newcommand{\FrontierCurrentArenaAnswersAdded}{7}" in macros
    assert r"\newcommand{\FrontierCurrentResponseReuseMaximum}{14}" in macros
    assert r"\newcommand{\FrontierCurrentUpliftPairs}{186}" in macros
    assert provenance["claim_boundary"]["comparison_rows_are_independent"] is False
    assert provenance["claim_boundary"]["official"] is False
    assert first["pdf"].read_bytes().startswith(b"%PDF")
    assert b"<svg" in first["svg"].read_bytes()[:500]


def test_corrected_arena_rejects_forged_content_address(tmp_path: Path) -> None:
    forged = json.loads(CORRECTED.read_text(encoding="utf-8"))
    forged["observed"]["candidate_comparisons"] += 1
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(CoverageAssetError, match="content address"):
        render_assets(path, tmp_path / "output")
