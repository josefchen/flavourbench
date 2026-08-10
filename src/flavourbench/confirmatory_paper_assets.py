"""Render verified Season 1 design and method-validation assets for the paper."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .season1_method_validation import canonical_sha256, verify_artifact


class ConfirmatoryPaperAssetError(RuntimeError):
    """A confirmatory paper input failed its publication boundary checks."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ConfirmatoryPaperAssetError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise ConfirmatoryPaperAssetError(f"{label} must be a JSON object")
    return value


def _verify_embedded_digest(value: Mapping[str, Any]) -> bool:
    embedded = value.get("artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    return isinstance(embedded, str) and embedded == canonical_sha256(payload)


def load_confirmatory_inputs(
    *,
    study_design_path: Path,
    method_validation_path: Path,
    construct_blueprint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    design = _load_json(study_design_path, "Season 1 study design")
    method = _load_json(method_validation_path, "Season 1 method validation")
    blueprint = _load_json(construct_blueprint_path, "Season 1 construct blueprint")

    if not _verify_embedded_digest(design):
        raise ConfirmatoryPaperAssetError("Season 1 study-design hash does not verify")
    if not _verify_embedded_digest(blueprint):
        raise ConfirmatoryPaperAssetError("Season 1 construct-blueprint hash does not verify")
    task_bank = design.get("task_bank")
    primary = design.get("primary_controlled_collection")
    expert = design.get("expert_evaluation")
    claim_boundary = design.get("claim_boundary")
    robustness = design.get("validity_and_robustness")
    release_requirements = design.get("release_requirements")
    top_level_sections = (
        task_bank,
        primary,
        expert,
        claim_boundary,
        robustness,
        release_requirements,
    )
    if not all(isinstance(value, Mapping) for value in top_level_sections):
        raise ConfirmatoryPaperAssetError("Season 1 study design is incomplete")
    assert isinstance(task_bank, Mapping)
    assert isinstance(primary, Mapping)
    assert isinstance(expert, Mapping)
    assert isinstance(claim_boundary, Mapping)
    assert isinstance(robustness, Mapping)
    assert isinstance(release_requirements, Mapping)

    arena = primary.get("model_arena")
    uplift = primary.get("epicure_uplift")
    blueprint_binding = task_bank.get("construct_blueprint")
    splits = task_bank.get("splits")
    admission = task_bank.get("admission")
    contamination = task_bank.get("contamination")
    correction = task_bank.get("correction_policy")
    unique_comparisons = expert.get("minimum_unique_comparisons")
    post_collection_audit = robustness.get("post_collection_item_audit")
    reliability_panel = robustness.get("generation_reliability_panel")
    prompt_sensitivity = robustness.get("prompt_sensitivity_audit")
    cookability_execution = robustness.get("practical_cookability_execution")
    results_release = release_requirements.get("benchmark_results_release")
    if not all(
        isinstance(value, Mapping)
        for value in (
            arena,
            uplift,
            blueprint_binding,
            splits,
            admission,
            contamination,
            correction,
            unique_comparisons,
            post_collection_audit,
            reliability_panel,
            prompt_sensitivity,
            cookability_execution,
            results_release,
        )
    ):
        raise ConfirmatoryPaperAssetError("Season 1 collection contract is incomplete")
    assert isinstance(arena, Mapping)
    assert isinstance(uplift, Mapping)
    assert isinstance(blueprint_binding, Mapping)
    assert isinstance(splits, Mapping)
    assert isinstance(admission, Mapping)
    assert isinstance(contamination, Mapping)
    assert isinstance(correction, Mapping)
    assert isinstance(unique_comparisons, Mapping)
    assert isinstance(post_collection_audit, Mapping)
    assert isinstance(reliability_panel, Mapping)
    assert isinstance(prompt_sensitivity, Mapping)
    assert isinstance(cookability_execution, Mapping)
    assert isinstance(results_release, Mapping)
    detector_calibration = contamination.get("labeled_detection_calibration")
    if not isinstance(detector_calibration, Mapping):
        raise ConfirmatoryPaperAssetError("Season 1 detector calibration is incomplete")

    valid_design = bool(
        design.get("schema_version") == "flavourbench-season1-study-design-v5"
        and design.get("status") == "prospective-design-superseding-v4-before-scored-collection"
        and int(task_bank.get("total") or 0) == 240
        and sum(int(value) for value in splits.values()) == 240
        and int(arena.get("total_battles") or 0) == 3_200
        and int(uplift.get("total_pairs") or 0) == 3_200
        and int(primary.get("total_model_response_arms") or 0) == 12_800
        and int(expert.get("minimum_distinct_independent_raters_per_comparison") or 0) == 2
        and int(unique_comparisons.get("model_arena") or 0) == 800
        and int(unique_comparisons.get("epicure_uplift") or 0) == 800
        and claim_boundary.get("synthetic_observations")
        == "prohibited-from-all-scored-and-supplemental-empirical-evidence"
        and blueprint_binding.get("artifact_sha256") == blueprint.get("artifact_sha256")
        and int(blueprint.get("task_count") or 0) == 240
        and int(admission.get("minimum_human_validity_records") or 0) == 720
        and int(admission.get("minimum_human_evidence_reviews") or 0) == 480
        and int(admission.get("minimum_distinct_verified_task_authors") or 0) == 20
        and int(admission.get("distinct_people_per_task") or 0) == 6
        and admission.get("person_uniqueness_method") == "admin-witnessed-season-hmac-v1"
        and int(detector_calibration.get("minimum_cases") or 0) >= 150
        and float(detector_calibration.get("minimum_overall_precision") or 0) >= 0.95
        and float(detector_calibration.get("minimum_overall_recall") or 0) >= 0.90
        and float(detector_calibration.get("minimum_paraphrase_recall") or 0) >= 0.85
        and detector_calibration.get("required_before_scored_collection") is True
        and correction.get("public_content_addressed_challenges") is True
        and int(correction.get("minimum_independent_adjudicators") or 0) >= 2
        and int(post_collection_audit.get("minimum_random_tasks") or 0) >= 60
        and float(post_collection_audit.get("random_audit_fraction") or 0) >= 0.25
        and post_collection_audit.get("all_anomaly_flagged_tasks") is True
        and int(post_collection_audit.get("minimum_independent_auditors_per_task") or 0) >= 2
        and post_collection_audit.get("auditors_excluded_from_original_task_roles") is True
        and post_collection_audit.get("release_requires_zero_unresolved_material_defects") is True
        and int(reliability_panel.get("task_count") or 0) == 20
        and int(reliability_panel.get("endpoint_count") or 0) == 16
        and int(reliability_panel.get("independent_generations_per_cell") or 0) == 3
        and int(reliability_panel.get("total_panel_arms") or 0) == 1_920
        and int(reliability_panel.get("incremental_arms_beyond_primary") or 0) == 1_280
        and reliability_panel.get("retries_are_not_repetitions") is True
        and int(prompt_sensitivity.get("task_count") or 0) == 20
        and int(prompt_sensitivity.get("endpoint_count") or 0) == 8
        and int(prompt_sensitivity.get("prompt_variants") or 0) == 3
        and int(prompt_sensitivity.get("total_response_arms") or 0) == 480
        and prompt_sensitivity.get("ranking_use") == "development-only-non-ranking-audit"
        and int(cookability_execution.get("task_count") or 0) == 24
        and int(cookability_execution.get("independent_cooks_per_output") or 0) == 2
        and int(cookability_execution.get("total_kitchen_executions") or 0) == 48
        and int(
            robustness.get("total_planned_real_model_response_arms_including_robustness") or 0
        )
        == 14_560
        and results_release.get("all_validity-and-robustness-studies-complete") is True
        and results_release.get("zero-unresolved-material-task-defects") is True
    )
    if not valid_design:
        raise ConfirmatoryPaperAssetError("Season 1 study design violates the paper contract")
    if not verify_artifact(method, reproduce=True):
        raise ConfirmatoryPaperAssetError("Season 1 method validation did not reproduce")
    return design, method


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def _scenario_label(scenario_id: str) -> str:
    labels = {
        "symmetric-null-with-ties": "Null with 20\\% ties",
        "positive-practical-effect-with-ties": "+0.10 half-win effect with 20\\% ties",
    }
    try:
        return labels[scenario_id]
    except KeyError as error:
        raise ConfirmatoryPaperAssetError(
            f"unknown method-validation scenario: {scenario_id}"
        ) from error


def render_method_table(method: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for scenario in method["scenarios"]:
        uplift = scenario["uplift"]
        type_i = uplift["two_sided_type_i_error"]
        power = uplift["one_sided_positive_power"]
        operating_value = _percent(float(type_i if type_i is not None else power))
        operating_label = "type I" if type_i is not None else "power"
        rows.append(
            " & ".join(
                [
                    _scenario_label(str(scenario["scenario_id"])),
                    f"{float(uplift['true_value']):.3f}",
                    f"{float(uplift['mean_estimate']):.3f}",
                    _percent(float(uplift["interval_coverage"])),
                    operating_label,
                    operating_value,
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{tabularx}{\columnwidth}{@{}X r r r l r@{}}",
            r"\toprule",
            r"Scenario & Truth & Mean & Coverage & Check & Rate \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            "",
        ]
    )


def render_macros(design: Mapping[str, Any], method: Mapping[str, Any]) -> str:
    task_bank = design["task_bank"]
    primary = design["primary_controlled_collection"]
    expert = design["expert_evaluation"]
    admission = task_bank["admission"]
    detector_calibration = task_bank["contamination"]["labeled_detection_calibration"]
    robustness = design["validity_and_robustness"]
    post_collection_audit = robustness["post_collection_item_audit"]
    reliability_panel = robustness["generation_reliability_panel"]
    prompt_sensitivity = robustness["prompt_sensitivity_audit"]
    cookability_execution = robustness["practical_cookability_execution"]
    parameters = method["parameters"]
    scenarios = {row["scenario_id"]: row for row in method["scenarios"]}
    null = scenarios["symmetric-null-with-ties"]["uplift"]
    alternative = scenarios["positive-practical-effect-with-ties"]["uplift"]
    values = {
        "ConfirmatoryTaskCount": int(task_bank["total"]),
        "ConfirmatoryScoredTaskCount": int(task_bank["splits"]["scored"]),
        "ConfirmatoryDevelopmentTaskCount": int(task_bank["splits"]["development"]),
        "ConfirmatoryPrivateTaskCount": int(task_bank["splits"]["private_reserve"]),
        "ConfirmatorySurfaceDiagnosticMinimum": int(
            task_bank["minimum_surface_diagnostic_coverage"]
        ),
        "ConfirmatoryBlindSolutionsPerTask": int(
            task_bank["admission"]["blind_prompt_only_solutions_per_task"]
        ),
        "ConfirmatoryReconciliationsPerTask": int(
            task_bank["admission"]["independent_reconciliations_per_task"]
        ),
        "ConfirmatoryAdjudicationsPerTask": int(
            task_bank["admission"]["independent_adjudications_per_task"]
        ),
        "ConfirmatoryValidityRecords": int(admission["minimum_human_validity_records"]),
        "ConfirmatoryEvidenceReviews": int(admission["minimum_human_evidence_reviews"]),
        "ConfirmatoryAdmissionDecisions": int(admission["minimum_human_validity_records"])
        + int(admission["minimum_human_evidence_reviews"])
        + int(task_bank["total"]) * int(admission["blind_prompt_only_solutions_per_task"]),
        "ConfirmatoryDistinctPeoplePerTask": int(admission["distinct_people_per_task"]),
        "ConfirmatoryMinimumTaskAuthors": int(admission["minimum_distinct_verified_task_authors"]),
        "ConfirmatoryContaminationCalibrationCases": int(detector_calibration["minimum_cases"]),
        "ConfirmatoryArenaBattles": int(primary["model_arena"]["total_battles"]),
        "ConfirmatoryUpliftPairs": int(primary["epicure_uplift"]["total_pairs"]),
        "ConfirmatoryResponseArms": int(primary["total_model_response_arms"]),
        "ConfirmatoryTotalPlannedRealArms": int(
            robustness["total_planned_real_model_response_arms_including_robustness"]
        ),
        "ConfirmatoryPostCollectionRandomAuditTasks": int(
            post_collection_audit["minimum_random_tasks"]
        ),
        "ConfirmatoryPostCollectionAuditorsPerTask": int(
            post_collection_audit["minimum_independent_auditors_per_task"]
        ),
        "ConfirmatoryReliabilityTasks": int(reliability_panel["task_count"]),
        "ConfirmatoryReliabilityGenerationsPerCell": int(
            reliability_panel["independent_generations_per_cell"]
        ),
        "ConfirmatoryReliabilityPanelArms": int(reliability_panel["total_panel_arms"]),
        "ConfirmatoryReliabilityIncrementalArms": int(
            reliability_panel["incremental_arms_beyond_primary"]
        ),
        "ConfirmatoryPromptSensitivityArms": int(prompt_sensitivity["total_response_arms"]),
        "ConfirmatoryKitchenExecutionTasks": int(cookability_execution["task_count"]),
        "ConfirmatoryKitchenExecutions": int(
            cookability_execution["total_kitchen_executions"]
        ),
        "ConfirmatoryExpertComparisonsPerTrack": int(
            expert["minimum_unique_comparisons"]["model_arena"]
        ),
        "ConfirmatoryExpertRatersPerComparison": int(
            expert["minimum_distinct_independent_raters_per_comparison"]
        ),
        "ConfirmatoryMonteCarloDatasets": int(parameters["monte_carlo_datasets_per_scenario"]),
        "ConfirmatoryBootstrapReplicates": int(parameters["bootstrap_replicates_per_dataset"]),
    }
    lines = [rf"\newcommand{{\{name}}}{{{value:,}}}" for name, value in values.items()]
    lines.extend(
        [
            rf"\newcommand{{\ConfirmatoryContaminationPrecision}}{{{float(detector_calibration['minimum_overall_precision']):.2f}}}",
            rf"\newcommand{{\ConfirmatoryContaminationRecall}}{{{float(detector_calibration['minimum_overall_recall']):.2f}}}",
            rf"\newcommand{{\ConfirmatoryContaminationParaphraseRecall}}{{{float(detector_calibration['minimum_paraphrase_recall']):.2f}}}",
            rf"\newcommand{{\ConfirmatoryNullCoverage}}{{{_percent(float(null['interval_coverage']))}}}",
            rf"\newcommand{{\ConfirmatoryNullTypeI}}{{{_percent(float(null['two_sided_type_i_error']))}}}",
            rf"\newcommand{{\ConfirmatoryAlternativeCoverage}}{{{_percent(float(alternative['interval_coverage']))}}}",
            rf"\newcommand{{\ConfirmatoryAlternativePower}}{{{_percent(float(alternative['one_sided_positive_power']))}}}",
            rf"\newcommand{{\ConfirmatoryStudyDesignHash}}{{{design['artifact_sha256']}}}",
            rf"\newcommand{{\ConfirmatoryMethodValidationHash}}{{{method['artifact_sha256']}}}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(method: Mapping[str, Any], output: Path) -> None:
    fieldnames = [
        "scenario_id",
        "true_half_win_share",
        "mean_estimate",
        "interval_coverage",
        "two_sided_type_i_error",
        "one_sided_positive_power",
        "monte_carlo_datasets",
        "bootstrap_replicates_per_dataset",
        "scored_benchmark_observations",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario in method["scenarios"]:
            uplift = scenario["uplift"]
            writer.writerow(
                {
                    "scenario_id": scenario["scenario_id"],
                    "true_half_win_share": uplift["true_value"],
                    "mean_estimate": uplift["mean_estimate"],
                    "interval_coverage": uplift["interval_coverage"],
                    "two_sided_type_i_error": uplift["two_sided_type_i_error"],
                    "one_sided_positive_power": uplift["one_sided_positive_power"],
                    "monte_carlo_datasets": scenario["monte_carlo_datasets"],
                    "bootstrap_replicates_per_dataset": scenario[
                        "bootstrap_replicates_per_dataset"
                    ],
                    "scored_benchmark_observations": method["claim_boundary"][
                        "scored_benchmark_observations"
                    ],
                }
            )


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-design", type=Path, required=True)
    parser.add_argument("--method-validation", type=Path, required=True)
    parser.add_argument("--construct-blueprint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    design, method = load_confirmatory_inputs(
        study_design_path=arguments.study_design,
        method_validation_path=arguments.method_validation,
        construct_blueprint_path=arguments.construct_blueprint,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "season1-method-validation-table.tex").write_text(
        render_method_table(method), encoding="utf-8"
    )
    (arguments.output_dir / "season1-confirmatory-macros.tex").write_text(
        render_macros(design, method), encoding="utf-8"
    )
    write_csv(method, arguments.output_dir / "season1-method-validation.csv")


if __name__ == "__main__":
    run()
