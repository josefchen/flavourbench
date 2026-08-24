from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from flavourbench.pilot_publication_assets import (
    _effective_comparison_manifest,
    _flow_data,
    _model_data,
    _operational_data,
    _operational_error_data,
    _paired_reliability_data,
    _preference_score_completion_ranges,
    _primary_model_table,
    _standardized_uplift_sensitivities,
    _study_design_figure,
    _uplift_data,
)
from flavourbench.season0_arm_corrections import (
    validate_arm_interpretation_correction,
)
from flavourbench.season0_completion_corrections import (
    validate_completion_interpretation_correction,
)
from flavourbench.season0_judge_protocol import (
    JUDGE_SYSTEM_PROMPT_SHA256,
    JUDGMENT_SCHEMA_SHA256,
)

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "flavourbench"
SERVICE = ROOT / "flavourbench"


def _frozen_analysis() -> dict[str, Any]:
    path = next((SERVICE / "artifacts" / "season0" / "analysis-v6").glob("*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_comparison_manifest() -> dict[str, Any]:
    path = next((SERVICE / "artifacts" / "season0" / "comparisons").glob("*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _completion_interpretation():
    path = next(
        (SERVICE / "artifacts" / "season0" / "corrections").glob(
            "completion-interpretation-correction-*.json"
        )
    )
    correction = json.loads(path.read_text(encoding="utf-8"))
    return validate_completion_interpretation_correction(
        correction=correction,
        arms_dir=SERVICE / "artifacts" / "season0" / "scored-v1" / "arms",
    )


def _effective_comparisons() -> dict[str, Any]:
    return _effective_comparison_manifest(
        _frozen_comparison_manifest(),
        _completion_interpretation(),
    )


def _frozen_task_bank() -> dict[str, Any]:
    path = next((SERVICE / "data" / "season0" / "frozen").glob("season0-real-task-bank-*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_review_queue() -> dict[str, Any]:
    path = next(
        (SERVICE / "data" / "season0" / "frozen").glob(
            "season0-pi-task-review-queue-*.json"
        )
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_curation_audits() -> list[dict[str, Any]]:
    paths = [
        next((SERVICE / "data" / "season0" / "curation").glob("curation-audit-*.json")),
        next(
            (SERVICE / "data" / "season0" / "curation-historical").glob(
                "curation-audit-*.json"
            )
        ),
    ]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_paper_analysis_binds_the_append_only_timeout_correction() -> None:
    correction_path = (
        SERVICE
        / "artifacts"
        / "season0"
        / "corrections"
        / "arm-interpretation-correction-v1.json"
    )
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    validated = validate_arm_interpretation_correction(
        correction=correction,
        arms_dir=SERVICE / "artifacts" / "season0" / "scored-v1" / "arms",
    )
    assert validated is not None
    assert len(validated.arm_ids) == 11
    analysis = _frozen_analysis()
    assert (
        analysis["arm_interpretation_correction_artifact_sha256"]
        == validated.artifact_sha256
    )
    cost_path = next(
        (SERVICE / "artifacts" / "season0" / "scored-v1" / "costs-v3").glob(
            "cost-audit-*.json"
        )
    )
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    assert (
        cost["arm_interpretation_correction_artifact_sha256"]
        == validated.artifact_sha256
    )
    assert cost["arm_interpretation_correction_count"] == 11


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def test_paper_sources_contain_no_generated_calibration_results() -> None:
    """Keep non-provider observations out of every publishable manuscript source."""

    sources = [
        PAPER / "main.tex",
        PAPER / "Makefile",
        PAPER / "SOURCE_NOTES.md",
        ROOT / "paper" / "claim_evidence_map.csv",
    ]
    prohibited = ("simulated_summary", "simulate_calibration")
    for path in sources:
        text = path.read_text(encoding="utf-8").lower()
        matches = [token for token in prohibited if token in text]
        text_without_zero_provenance = re.sub(
            r"\b(?:no|zero)\s+synthetic\b", "", text
        )
        matches.extend(re.findall(r"\bsynthetic\b", text_without_zero_provenance))
        matches.extend(re.findall(r"\bmodel [a-d]\b", text))
        assert not matches, f"non-provider evidence leaked into {path}: {matches}"


def test_paper_data_contains_no_fixture_model_identity() -> None:
    """Published JSON may contain only real provider or methodology records."""

    for path in sorted((PAPER / "data").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        fixture_ids = [
            text for text in _walk_strings(value) if text.startswith("flavourbench/mock-")
        ]
        assert not fixture_ids, f"fixture model identity leaked into {path}: {fixture_ids[:3]}"


def test_paper_discloses_frozen_judging_and_randomization_contracts() -> None:
    """Prevent reproducibility-critical prompt, pairing, and RNG details from drifting out."""

    text = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert JUDGE_SYSTEM_PROMPT_SHA256 in text
    assert JUDGMENT_SCHEMA_SHA256 in text
    assert "c6e9052d19737b39b540dafbd0cea53d1dd0c54b1a04584fd3775ddfe9f35ca7" in text
    assert "parity of SHA-256 over the frozen track, task" in text
    assert "seed 20260721" in text
    assert "seed 20260716" in text
    assert "seed 20260722" in text
    assert "seed 20260723" in text


def test_pilot_weighting_sensitivities_reproduce_frozen_real_cohort() -> None:
    result = _standardized_uplift_sensitivities(_frozen_analysis())
    assert result["comparisons"] == 241
    assert result["task_clusters"] == 102
    assert result["models"] == 12
    assert result["cells_per_task_min"] == 1
    assert result["cells_per_task_max"] == 7
    assert result["cells_per_model_min"] == 5
    assert result["cells_per_model_max"] == 42
    assert math.isclose(result["cell_weighted_estimate"], 0.42738589211618255)
    assert math.isclose(result["equal_task_estimate"], 0.4183356676003735)
    assert math.isclose(result["equal_model_estimate"], 0.4872823698444895)
    assert result["equal_model_complete_replicates"] >= 4900


def test_pilot_cross_family_result_is_selection_not_vote_reversal() -> None:
    analysis = _frozen_analysis()
    rows = _uplift_data(analysis, _standardized_uplift_sensitivities(analysis))
    subset = next(row for row in rows if row["group"] == "admission")
    assert subset["label"] == "Cross-family-admitted subset"
    assert subset["n"] == 133
    assert subset["retained_primary_choice_changes"] == 0
    assert subset["excluded_primary_comparisons"] == 108
    assert (
        subset["excluded_epicure_wins"],
        subset["excluded_ties"],
        subset["excluded_unaided_wins"],
    ) == (15, 42, 51)


def test_pilot_task_flow_is_bound_to_both_curation_audits() -> None:
    rows = _flow_data(
        _frozen_analysis(),
        _frozen_task_bank(),
        _frozen_review_queue(),
        _frozen_comparison_manifest(),
        _frozen_curation_audits(),
    )
    counts = {(row["panel"], row["stage"]): row["count"] for row in rows}
    assert counts[("task", "source_candidates")] == 979
    assert counts[("task", "strict_llm_curator_agreement")] == 462
    assert counts[("task", "selected_tasks")] == 120
    assert counts[("task", "qualified_human_reviewed")] == 0


def test_pilot_condition_attrition_and_separation_are_explicit() -> None:
    analysis = _frozen_analysis()
    operational, matrix = _operational_data(analysis, _effective_comparisons())
    assert sum(int(row["off_success"]) for row in operational) == 1291
    assert sum(int(row["on_success"]) for row in operational) == 1105
    assert {row["outcome"]: row["count"] for row in matrix} == {
        "both_success": 1062,
        "on_failed_off_success": 229,
        "off_failed_on_success": 43,
        "both_failed": 106,
    }
    model_rows = _model_data(analysis)
    assert all("highest_resample_fraction" in row for row in model_rows)
    assert all("rank_one_probability" not in row for row in model_rows)
    table = _primary_model_table(model_rows)
    assert "not estimable" in table
    assert "unbounded" in table
    assert "-2365" not in table


def test_pilot_reliability_and_tool_error_audit_reproduce_real_records() -> None:
    reliability = _paired_reliability_data(_effective_comparisons())[0]
    assert reliability["attempted_cells"] == 1440
    assert reliability["task_clusters"] == 120
    assert reliability["off_success"] == 1291
    assert reliability["on_success"] == 1105
    assert (
        reliability["estimand"]
        == "tool_available_minus_tool_unavailable_realized_success_proportion"
    )
    assert "risk_difference" not in reliability
    assert math.isclose(
        reliability["realized_success_proportion_difference"],
        -0.12916666666666668,
    )
    assert reliability["bootstrap_seed"] == 20260723
    assert reliability["bootstrap_replicates"] == 5000

    arm_paths = sorted((SERVICE / "artifacts" / "season0" / "scored-v1" / "arms").glob("*.json"))
    terminal, errors, summary, stop_rule, arm_set_sha256 = _operational_error_data(
        arm_paths,
        _completion_interpretation(),
    )
    assert [row["total"] for row in terminal] == [267, 156, 10, 2, 20, 17, 11, 1]
    assert [row["error_events"] for row in errors] == [341, 38, 4]
    assert {row["metric"]: row["count"] for row in summary} == {
        "tool_using_arms": 445,
        "trace_events": 1387,
        "successful_trace_events": 1004,
        "error_trace_events": 383,
        "arms_with_error_trace": 227,
        "error_trace_arms_with_successful_final_answer": 59,
    }
    assert {row["metric"]: row["count"] for row in stop_rule} == {
        "arms_stopped_on_second_recorded_mcp_error": 156,
        "both_errors_in_same_tool_round": 95,
        "both_errors_in_round_zero": 86,
    }
    assert arm_set_sha256 == "6dfcbb7fc942458c99ef3331e1d451206584c7ff9134eb33de82a725153b9df2"


def test_pilot_preference_bounds_keep_missingness_visible() -> None:
    rows = _preference_score_completion_ranges(
        _frozen_analysis(),
        _effective_comparisons(),
    )
    planned, admitted, failure_aware = rows
    assert planned["analysis"] == "planned_cell_score_completion"
    assert planned["population"] == 1440
    assert planned["judged_consensuses"] == 241
    assert planned["unresolved_scores"] == 1199
    assert math.isclose(planned["lower_bound"], 103 / 1440)
    assert math.isclose(planned["upper_bound"], (103 + 1199) / 1440)
    assert "tool-available preference" in planned["assumption"]
    assert "Epicure preference" not in planned["assumption"]
    assert math.isclose(
        planned["missing_score_mean_at_neutrality"],
        (0.5 * 1440 - 103) / 1199,
    )

    assert admitted["analysis"] == "admitted_pair_score_completion"
    assert admitted["population"] == 1061
    assert admitted["judged_consensuses"] == 241
    assert admitted["unresolved_scores"] == 820
    assert math.isclose(admitted["lower_bound"], 103 / 1061)
    assert math.isclose(admitted["upper_bound"], (103 + 820) / 1061)
    assert math.isclose(
        admitted["missing_score_mean_at_neutrality"],
        (0.5 * 1061 - 103) / 820,
    )

    assert failure_aware["analysis"] == "failure_aware_score_completion"
    assert failure_aware["population"] == 1334
    assert (
        failure_aware["deterministic_tool_available_wins_from_one_sided_success"],
        failure_aware["deterministic_tool_unavailable_wins_from_one_sided_success"],
        failure_aware["unresolved_scores"],
        failure_aware["excluded_dual_failures"],
    ) == (43, 229, 821, 106)
    assert math.isclose(failure_aware["lower_bound"], 146 / 1334)
    assert math.isclose(failure_aware["upper_bound"], (146 + 821) / 1334)


def test_study_design_figure_separates_contrasts_and_admission() -> None:
    figure = _study_design_figure()
    assert "endpoint contrast" in figure
    assert "paired tool contrast" in figure
    assert "same \\(q,m_1\\)" in figure
    assert "Reliability and admission are outcomes" in figure
    assert "preference\\\\undefined" in figure
    assert "reliability" in figure
    assert "missing" in figure


def test_arxiv_recipe_includes_aggregate_evidence_and_excludes_raw_records() -> None:
    makefile = (PAPER / "Makefile").read_text(encoding="utf-8")
    assert "ARXIV_PILOT_ASSETS := $(PILOT_ASSETS)" in makefile
    assert "pilot-condition-attrition.csv" in makefile
    assert "pilot-condition-reliability.csv" in makefile
    assert "pilot-preference-bounds.csv" in makefile
    assert "pilot-study-design.tex" in makefile
    assert "pilot-uplift-reliability.pdf" in makefile
    assert "pilot-terminal-failures.csv" in makefile
    assert "pilot-mcp-errors.csv" in makefile
    assert "pilot-mcp-summary.csv" in makefile
    assert "pilot-mcp-stop-rule.csv" in makefile
    assert "pilot-endpoint-manifest.tex" in makefile
    assert "MODEL_MANIFEST" in makefile
    assert "pilot-figure-provenance.json" in makefile
    assert "response" not in " ".join(
        line.strip() for line in makefile.splitlines() if "ARXIV_PILOT_ASSETS" in line
    )
