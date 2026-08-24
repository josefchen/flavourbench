from copy import deepcopy
from pathlib import Path

import pytest

from flavourbench.epicure_operational_leaderboard import (
    EpicureOperationalLeaderboardError,
    build_operational_leaderboard,
    render_latex_table,
)
from flavourbench.real_task_bank import sha256_json

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper/flavourbench"
SOURCE = (
    PAPER
    / "generated/frontier-study/high-resource"
    / ("frontier-multirun-c0bd526a2776a25adfbd2c43b98b8f15c143a8cb93b957ba961d0e9efe626688.json")
)
QWEN = (
    ROOT
    / "artifacts/season1/current-quality-run/release-package-remediation-v1"
    / (
        "qwencloud-exploratory-operational-projection-"
        "b2f7790b3eb18d1df083397ce02b5296c549e5ed3ddb3d3f32ea776db3ddca04.json"
    )
)
SCORE_CHART = PAPER / "generated/operational-leaderboard/flavourbench-score-chart.tex"
PROFILE_CHART = PAPER / "generated/operational-leaderboard/flavourbench-operational-profile.tex"
COMPACT_TABLE = (
    PAPER / "generated/operational-leaderboard/flavourbench-leaderboard-compact-table.tex"
)


def test_builds_scoped_public_operational_leaderboard() -> None:
    artifact = build_operational_leaderboard(SOURCE, QWEN)

    assert artifact["artifact_sha256"] == sha256_json(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )
    assert artifact["official_within_scope"] is True
    assert artifact["leaderboard_scope"] == "epicure_grounded_automated_operational"
    assert artifact["totals"]["ranked_models"] == 16
    assert artifact["totals"]["qualified_models"] == 4
    assert artifact["totals"]["provisional_models"] == 12
    assert artifact["totals"]["complete_pairs"] == 110
    assert artifact["totals"]["epicure_successful_calls"] == 207
    assert artifact["claim_boundary"]["quality_judgments"] == 0
    assert artifact["claim_boundary"]["culinary_quality_leaderboard_official"] is False

    rows = artifact["rows"]
    assert rows[0]["model_id"] == "cohere/command-a-plus-05-2026"
    assert rows[0]["operational_rank"] == 1
    assert rows[1]["model_id"] == "anthropic/claude-opus-5"
    tied_rank_three = {row["model_id"] for row in rows if row["operational_rank"] == 3}
    assert tied_rank_three == {
        "anthropic/claude-fable-5",
        "anthropic/claude-sonnet-5",
        "deepseek/deepseek-v4-flash-0731",
        "google/gemini-3.6-flash",
        "moonshotai/kimi-k3",
    }
    assert all(row["automated_operational_rank_eligible"] for row in rows)
    assert not any(row["culinary_quality_rank_eligible"] for row in rows)


def test_qwen_is_visible_but_not_pooled() -> None:
    artifact = build_operational_leaderboard(SOURCE, QWEN)

    extension = artifact["unranked_extensions"]
    assert len(extension) == 1
    assert extension[0]["model_id"] == "qwen3.8-max"
    assert extension[0]["completed_pairs"] == 1
    assert extension[0]["epicure_successful_calls"] == 2
    assert extension[0]["rank"] is None
    assert extension[0]["status"] == "insufficient_comparable_evidence"


def test_source_claim_drift_is_rejected(tmp_path: Path) -> None:
    import json

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source["quality_ranking"] = True
    source["artifact_sha256"] = sha256_json(
        {key: value for key, value in source.items() if key != "artifact_sha256"}
    )
    candidate = tmp_path / "source.json"
    candidate.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")

    with pytest.raises(EpicureOperationalLeaderboardError, match="physical source drift"):
        build_operational_leaderboard(candidate, QWEN)


def test_rows_do_not_mutate_frozen_quality_flags() -> None:
    artifact = build_operational_leaderboard(SOURCE, QWEN)
    mutated = deepcopy(artifact)

    for row in mutated["rows"]:
        row["culinary_quality_rank_eligible"] = True

    assert any(row["culinary_quality_rank_eligible"] for row in mutated["rows"])
    assert not any(row["culinary_quality_rank_eligible"] for row in artifact["rows"])


def test_latex_table_preserves_ranked_panel_and_scope() -> None:
    table = render_latex_table(build_operational_leaderboard(SOURCE, QWEN))

    assert table.count(" \\\\") == 17
    assert "1 & Command A Plus & 75.8 & 12/12 & 100.0 [75.8, 100.0] & 12/15" in table
    assert "3 & Kimi K3 & 52.9 & 7/8 & 87.5 [52.9, 97.8] & 10/13" in table
    assert "5 & Command A Reasoning & 39.1 & 8/12 & 66.7 [39.1, 86.2] & 18/19" in table
    assert "Evidence" not in table
    assert "qualified" not in table
    assert "provisional" not in table
    assert "Qwen" not in table


def test_paper_visuals_match_every_ranked_row() -> None:
    artifact = build_operational_leaderboard(SOURCE, QWEN)
    score_chart = SCORE_CHART.read_text(encoding="utf-8")
    profile_chart = PROFILE_CHART.read_text(encoding="utf-8")
    compact_table = COMPACT_TABLE.read_text(encoding="utf-8")

    for row in artifact["rows"]:
        name = str(row["display_name"])
        score = 100 * float(row["verified_pair_completion_wilson_lower_95"])
        completion = 100 * float(row["verified_pair_completion_rate"])
        tool_rate = 100 * int(row["epicure_successful_calls"]) / int(row["epicure_calls"])
        completed = f"{int(row['verified_complete_pairs'])}/{int(row['scheduled_pairs'])}"

        assert f"{{{name}}}/{score:.1f}/" in score_chart
        assert f"{{{name}}}/{completion:.1f}/{tool_rate:.1f}/" in profile_chart
        assert (
            f"{int(row['operational_rank'])} & {name} & {score:.1f} & {completed}"
            in compact_table
        )
