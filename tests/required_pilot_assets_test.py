from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from flavourbench.current_pilot_assets import MODEL_ORDER
from flavourbench.real_task_bank import sha256_json
from flavourbench.required_pilot_assets import (
    EXPECTED_SUMMARY_SHA256,
    RequiredPilotAssetError,
    _verify_summary,
    build_audit_document,
    render_assets,
    verify_required_pilot,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "artifacts/season1/current-quality-run/pilot-v24-required-epicure"
SUMMARY = PILOT_ROOT / (
    "summaries/real-exploratory-summary-"
    "f1a38e30042b9614fa82f5c38b43b98c7c9a18916c4541f21bb12d3bcce8ba70.json"
)


def _pilot():
    return verify_required_pilot(SUMMARY, PILOT_ROOT / "source", PILOT_ROOT / "responses")


def test_required_pilot_reconciles_real_execution_without_quality_claims() -> None:
    pilot = _pilot()

    observed_order = tuple(row["model_id"] for row in pilot.model_rows)
    manifest_order = tuple(row["model_id"] for row in pilot.summary["manifest"]["models"])
    assert observed_order == manifest_order
    assert set(observed_order) == set(MODEL_ORDER)
    assert pilot.counts == {
        "models": 14,
        "tasks": 4,
        "scheduled_pairs": 56,
        "scheduled_arms": 112,
        "normalized_arms": 95,
        "complete_pairs": 43,
        "partial_pairs": 9,
        "failed_pairs": 4,
        "failed_or_partial_pairs": 13,
        "off_responses": 51,
        "on_responses": 44,
        "provider_generations": 334,
        "epicure_calls": 194,
        "epicure_successful_calls": 87,
        "epicure_error_calls": 107,
        "intermediate_ceiling_events": 10,
        "final_length_responses": 0,
        "exact_cost_usd": 4.381703,
        "estimated_cost_usd": 0.337494,
        "combined_measured_cost_usd": 4.719197,
        "kimi_direct_models": 1,
        "openrouter_models": 13,
        "bedrock_models": 0,
        "quality_judgments": 0,
        "synthetic_tasks": 0,
        "synthetic_arms": 0,
    }
    assert all(row["rank_eligible"] is False for row in pilot.model_rows)
    assert all(row["quality_judgments"] == 0 for row in pilot.model_rows)
    assert all(
        any(not event["is_error"] for event in response["response"]["tool_trace"])
        for (work_item_id, condition), response in pilot.responses.items()
        if condition == "epicure_on"
    )


def test_required_pilot_assets_are_static_and_claim_bounded(tmp_path: Path) -> None:
    pilot = _pilot()
    result = render_assets(pilot, tmp_path / "figures", tmp_path / "audit")
    provenance = json.loads(Path(result["provenancePath"]).read_text(encoding="utf-8"))
    audit = build_audit_document(pilot)

    assert provenance["claim_boundary"]["quality_leaderboard_permitted"] is False
    assert provenance["plot_contract"]["static_library"] == "matplotlib"
    assert provenance["plot_contract"]["quality_ordering"] is False
    assert audit["content_address"]["digest"] == result["auditSha256"]
    assert len(provenance["generated_asset_sha256s"]) == 12
    for figure in (
        "current-frontier-pilot-reliability.pdf",
        "current-frontier-pilot-tools.pdf",
        "current-frontier-pilot-efficiency.pdf",
        "current-frontier-pilot-attrition.pdf",
    ):
        assert (tmp_path / "figures" / figure).read_bytes().startswith(b"%PDF")


def test_required_pilot_rejects_a_rehashed_summary_mutation(tmp_path: Path) -> None:
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))
    mutated = deepcopy(document)
    mutated.pop("content_address")
    mutated["official"] = True
    digest = sha256_json(mutated)
    mutated["content_address"] = {
        "algorithm": "sha256",
        "digest": digest,
        "uri": f"sha256:{digest}",
    }
    path = tmp_path / f"real-exploratory-summary-{digest}.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")

    assert digest != EXPECTED_SUMMARY_SHA256
    with pytest.raises(RequiredPilotAssetError, match="frozen required-Epicure"):
        _verify_summary(path)


def test_manuscript_binds_the_current_frontier_generated_provenance() -> None:
    paper_root = ROOT.parent / "paper/flavourbench"
    manuscript = (paper_root / "main.tex").read_text(encoding="utf-8")
    protocol_macros = (
        paper_root
        / "generated/frontier-study/comparison/frontier-protocol-sensitivity-macros.tex"
    ).read_text(encoding="utf-8")
    arena_macros = (
        paper_root
        / "generated/frontier-study/model-arena/frontier-model-arena-review-macros.tex"
    ).read_text(encoding="utf-8")

    assert "frontier-protocol-sensitivity-macros" in manuscript
    assert "frontier-model-arena-review-macros" in manuscript
    assert "50b5e169dbed9cfded2ab4fec097c39043b1bf28ab0ef66247fdc583b56ad0d7" in (
        protocol_macros
    )
    assert "\\newcommand{\\FrontierModelsAtLeastEightPairs}{16}" in protocol_macros
    assert "\\newcommand{\\FrontierMinimumCompletePairs}{8}" in protocol_macros
    assert "407e7fc6413e6d009c942eb51d9603d7cb958f0f282ffe90e1dc8ff28c3b6ac3" in (
        arena_macros
    )
    assert "2f9a355c616d5159f1224023368e6facdcbae540524864b9bcc79deed6935561" in (
        arena_macros
    )
    for superseded in (
        EXPECTED_SUMMARY_SHA256,
        "445c87223fa3550b65b0560af234e8c017c667d095d81cc9508aa009982bda73",
        "66483d372b4619ac505b576f4a9e9ad3c225d0fcabafe04f185e7e13c7589fe5",
        "f4daaef029dfc46d739be479d601938eb75ee73d957b73bd5607762dc6a8e9b2",
        "f0bfa024315c4efdbcc59f8251fdea6fc6fd3592875c9ff0aee65f56ced78e20",
        "c92bedea423cde1d209c39f1b9ce64d7e2734de7c088716f65ab402848118d1b",
    ):
        assert superseded not in manuscript
