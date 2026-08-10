from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from flavourbench.current_pilot_assets import (
    MODEL_ORDER,
    CurrentPilotAssetError,
    _verify_summary,
    render_assets,
    verify_pilot,
)
from flavourbench.real_task_bank import sha256_json

ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "artifacts/season1/current-quality-run/pilot-v19"
SUMMARY = PILOT_ROOT / (
    "summaries/real-exploratory-summary-"
    "f0bfa024315c4efdbcc59f8251fdea6fc6fd3592875c9ff0aee65f56ced78e20.json"
)


def test_real_current_pilot_reconciles_all_frontier_models_without_quality_claims() -> None:
    pilot = verify_pilot(SUMMARY, PILOT_ROOT / "source", PILOT_ROOT / "responses")

    assert tuple(row["model_id"] for row in pilot.model_rows) == MODEL_ORDER
    assert pilot.counts == {
        "models": 14,
        "task_families": 4,
        "scheduled_pairs": 56,
        "scheduled_arms": 112,
        "completed_response_arms": 103,
        "complete_pairs": 47,
        "failed_or_partial_pairs": 9,
        "provider_generations": 166,
        "epicure_calls": 67,
        "epicure_successful_calls": 25,
        "epicure_semantic_error_calls": 42,
        "tool_active_on_arms": 9,
        "completed_on_arms": 47,
        "actual_cost_usd": 3.329832,
        "quality_judgments": 0,
    }
    assert {row["complete_pairs"] for row in pilot.family_rows} == {8, 12, 13, 14}
    assert all(row["quality_judgments"] == 0 for row in pilot.model_rows)
    assert all(row["rank_eligible"] is False for row in pilot.model_rows)


def test_current_pilot_renderer_writes_vector_figures_and_claim_boundary(tmp_path: Path) -> None:
    pilot = verify_pilot(SUMMARY, PILOT_ROOT / "source", PILOT_ROOT / "responses")
    provenance_path = render_assets(pilot, tmp_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert provenance["artifact_sha256"] == sha256_json(
        {key: value for key, value in provenance.items() if key != "artifact_sha256"}
    )
    assert provenance["claim_boundary"]["quality_judgments"] == 0
    assert provenance["claim_boundary"]["rank_eligible"] is False
    assert len(provenance["figures"]) == 3
    assert all(
        (tmp_path / row["figure"]).read_bytes().startswith(b"%PDF")
        for row in provenance["figures"]
    )
    table = (tmp_path / "current-frontier-pilot-table.tex").read_text(encoding="utf-8")
    for name in ("Claude Opus 5", "Claude Sonnet 5", "Kimi K3", "DeepSeek V4 Pro"):
        assert name in table


def test_current_pilot_summary_rejects_rehashed_official_status(tmp_path: Path) -> None:
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

    with pytest.raises(CurrentPilotAssetError, match="claim boundary"):
        _verify_summary(path)
