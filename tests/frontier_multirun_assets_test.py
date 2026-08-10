from pathlib import Path

import pytest

from flavourbench.frontier_multirun_assets import (
    FrontierMultirunAssetError,
    RunInput,
    verify_runs,
    write_assets,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/season1/current-quality-run/pilot-v27-eight-pairs"
RUN = RunInput(
    summary=(
        PILOT / "summaries/real-exploratory-summary-"
        "d0876f6e7b70d9803468b766b4df91f983fcf684c463766bbe9be1b35cda7018.json"
    ),
    sources=PILOT / "source",
    responses=PILOT / "responses",
)


def test_multirun_assets_verify_real_exact_frontier_pilot(tmp_path: Path) -> None:
    pilot = verify_runs([RUN])

    assert pilot.aggregate["synthetic_tasks"] == 0
    assert pilot.aggregate["quality_ranking"] is False
    assert len(pilot.aggregate["tasks"]) == 8
    assert len(pilot.aggregate["task_set_sha256"]) == 64
    assert pilot.aggregate["totals"] == {
        "runs": 1,
        "models": 14,
        "task_families": 4,
        "distinct_tasks": 8,
        "scheduled_pairs": 112,
        "finalized_pairs": 112,
        "complete_pairs": 65,
        "failed_or_partial_pairs": 47,
        "completed_response_arms": 174,
        "provider_generation_ids": 647,
        "epicure_calls": 129,
        "epicure_successful_calls": 103,
        "models_with_at_least_eight_complete_pairs": 2,
        "synthetic_tasks": 0,
        "quality_judgments": 0,
    }
    assert pilot.aggregate["cost"]["known_conservative_exposure_subtotal_usd"] == pytest.approx(
        17.323986733333335
    )
    assert pilot.aggregate["cost"]["provider_charge_complete"] is True
    assert pilot.aggregate["cost"]["unpriced_model_ids"] == []
    assert {row["display_name"] for row in pilot.model_rows} >= {
        "GPT-5.6 Sol (OR pro)",
        "Claude Opus 5",
        "Claude Sonnet 5",
        "DeepSeek V4 Pro",
        "DeepSeek V4 Flash",
        "Kimi K3",
    }
    kimi = next(row for row in pilot.model_rows if row["display_name"] == "Kimi K3")
    assert kimi["canonical_model_slug"] == "k3"
    assert kimi["execution_backend"] == "kimi_direct"
    assert all(row["quality_judgments"] == 0 for row in pilot.model_rows)

    outputs = write_assets(pilot, tmp_path)
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
    assert outputs["reliability_figure"].with_suffix(".svg").is_file()
    assert outputs["cost_latency_figure"].with_suffix(".svg").is_file()
    csv_text = outputs["model_csv"].read_text(encoding="utf-8")
    assert "Claude Opus 5" in csv_text
    assert "Model A" not in csv_text


def test_multirun_assets_reject_duplicate_experimental_units() -> None:
    with pytest.raises(FrontierMultirunAssetError, match="duplicate or invalid experimental unit"):
        verify_runs([RUN, RUN])
