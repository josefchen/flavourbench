"""Generate deterministic LaTeX macros and result tables from real Season 0 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .real_task_bank import sha256_json
from .season0_analysis import _implementation_manifest


class PaperAssetError(RuntimeError):
    """A paper input is missing, mixed, or not content-addressed."""


def _implementation_manifest_from_source_root(
    source_root: Path,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    source_hashes = expected_manifest.get("source_sha256")
    dependencies = expected_manifest.get("dependencies")
    if not isinstance(source_hashes, Mapping) or not isinstance(dependencies, Mapping):
        raise PaperAssetError("analysis implementation manifest is incomplete")
    observed: dict[str, str] = {}
    for name in source_hashes:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".py"):
            raise PaperAssetError("analysis implementation source name is invalid")
        path = source_root / name
        try:
            observed[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise PaperAssetError(f"missing frozen analysis source: {name}") from error
    return {
        "source_sha256": observed,
        "dependencies": dict(dependencies),
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise PaperAssetError(f"expected a JSON object: {path}")
    return value


def _verify(document: Mapping[str, Any], label: str) -> str:
    claimed = document.get("artifact_sha256")
    actual = sha256_json(
        {key: value for key, value in document.items() if key != "artifact_sha256"}
    )
    if claimed != actual:
        raise PaperAssetError(f"{label} artifact hash mismatch")
    return actual


def _latex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _integer(value: int | float) -> str:
    return f"{int(value):,}"


def _decimal(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _percent(value: object, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}\\%"


def _rating(row: Mapping[str, Any]) -> str:
    if row.get("rating") is None:
        return "withheld"
    return (
        f"{float(row['rating']):.0f} "
        f"[{float(row['rating_lower']):.0f}, {float(row['rating_upper']):.0f}]"
    )


def _uplift(row: Mapping[str, Any]) -> str:
    return (
        f"{float(row['epicure_win_share']):.3f} "
        f"[{float(row['interval_lower']):.3f}, {float(row['interval_upper']):.3f}]"
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _model_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabularx}{\columnwidth}{@{}rXlrr@{}}",
        r"\toprule",
        r"Order & Endpoint & Diagnostic BT rating [95\% CI] & Valid $n$ & E2E fail \\",
        r"\midrule",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"{rank} & {_latex(row['display_name'])} & {_rating(row)} & "
            f"{int(row.get('comparisons') or 0)} & "
            f"{_percent(row.get('end_to_end_failure_rate') or 0)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return "\n".join(lines)


def _uplift_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabularx}{\columnwidth}{@{}rXlrr@{}}",
        r"\toprule",
        r"Order & Endpoint & Enabled-condition share [95\% CI] & W/T/L & $n$ \\",
        r"\midrule",
    ]
    for rank, row in enumerate(rows, 1):
        wtl = "/".join(
            str(int(row.get(key) or 0)) for key in ("epicure_wins", "ties", "unaided_wins")
        )
        lines.append(
            f"{rank} & {_latex(row['display_name'])} & {_uplift(row)} & {wtl} & "
            f"{int(row.get('comparisons') or 0)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return "\n".join(lines)


def _family_table(analysis: Mapping[str, Any]) -> str:
    arena = analysis["model_leaderboard_by_family"]
    uplift = analysis["uplift_leaderboard_by_family"]
    lines = [
        r"\begin{tabularx}{\columnwidth}{@{}lXX@{}}",
        r"\toprule",
        r"Family & Model Arena leader & Largest estimated Epicure uplift \\",
        r"\midrule",
    ]
    for family in ("substitution", "composition", "cookability", "evidence"):
        arena_rows = [row for row in arena[family] if row.get("rating") is not None]
        uplift_rows = list(uplift[family])
        arena_text = (
            f"{_latex(arena_rows[0]['display_name'])} ({float(arena_rows[0]['rating']):.0f})"
            if arena_rows
            else "withheld (disconnected graph)"
        )
        uplift_text = (
            f"{_latex(uplift_rows[0]['display_name'])} "
            f"({float(uplift_rows[0]['epicure_win_share']):.3f})"
            if uplift_rows
            else "unavailable"
        )
        lines.append(f"{family.title()} & {arena_text} & {uplift_text} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return "\n".join(lines)


def _judge_table(analysis: Mapping[str, Any], judge_manifest: Mapping[str, Any]) -> str:
    stats = analysis["judge_diagnostics"]["judges"]
    lines = [
        r"\begin{tabularx}{\columnwidth}{@{}Xrrr@{}}",
        r"\toprule",
        r"Judge & Swap consistency & Incomplete & Self-excluded \\",
        r"\midrule",
    ]
    for judge in judge_manifest["judges"]:
        row = stats[str(judge["judge_id"])]
        lines.append(
            f"{_latex(judge['display_name'])} & "
            f"{_percent(row.get('orientation_consistency_rate') or 0)} & "
            f"{int(row.get('incomplete_comparisons') or 0)} & "
            f"{int(row.get('self_judgments') or 0)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return "\n".join(lines)


def generate_assets(
    *,
    analysis: Mapping[str, Any],
    comparison_manifest: Mapping[str, Any],
    judge_manifest: Mapping[str, Any],
    target_cost_audit: Mapping[str, Any],
    judgment_summary: Mapping[str, Any],
    output_dir: Path,
    arm_interpretation_correction: Mapping[str, Any] | None = None,
    completion_interpretation_correction: Mapping[str, Any] | None = None,
    implementation_source_root: Path | None = None,
) -> dict[str, str]:
    analysis_sha = _verify(analysis, "analysis")
    comparison_sha = _verify(comparison_manifest, "comparison manifest")
    judge_sha = _verify(judge_manifest, "judge manifest")
    cost_sha = _verify(target_cost_audit, "target cost audit")
    judgment_sha = _verify(judgment_summary, "judgment summary")
    correction_sha = (
        _verify(arm_interpretation_correction, "arm interpretation correction")
        if arm_interpretation_correction is not None
        else None
    )
    completion_correction_sha = (
        _verify(
            completion_interpretation_correction,
            "completion interpretation correction",
        )
        if completion_interpretation_correction is not None
        else None
    )
    if (
        analysis.get("comparison_manifest_artifact_sha256") != comparison_sha
        or analysis.get("judge_manifest_artifact_sha256") != judge_sha
        or analysis.get("target_cost_audit_artifact_sha256") != cost_sha
        or judgment_summary.get("comparison_manifest_artifact_sha256") != comparison_sha
        or judgment_summary.get("judge_manifest_artifact_sha256") != judge_sha
        or analysis.get("arm_interpretation_correction_artifact_sha256") != correction_sha
        or target_cost_audit.get("arm_interpretation_correction_artifact_sha256") != correction_sha
        or analysis.get("completion_interpretation_correction_artifact_sha256")
        != completion_correction_sha
    ):
        raise PaperAssetError("paper inputs are not bound to one comparison/judge protocol")
    expected_implementation = analysis.get("implementation")
    observed_implementation = (
        _implementation_manifest()
        if implementation_source_root is None
        else _implementation_manifest_from_source_root(
            implementation_source_root,
            expected_implementation if isinstance(expected_implementation, Mapping) else {},
        )
    )
    if expected_implementation != observed_implementation:
        raise PaperAssetError("paper build analysis implementation does not match local source")

    if analysis.get("status") != "automated_cohort_analysis_complete":
        raise PaperAssetError("paper build requires a completed automated-cohort analysis")
    if analysis.get("synthetic_arms") != 0 or analysis.get("synthetic_judgments") != 0:
        raise PaperAssetError("paper build refuses synthetic arms or judgments")
    if judgment_summary.get("status") != "collection_complete":
        raise PaperAssetError("paper build requires a completed judgment collection")
    if judgment_summary.get("synthetic_judgments") != 0:
        raise PaperAssetError("paper build refuses synthetic judgments")

    judgment_counts = judgment_summary.get("counts")
    if not isinstance(judgment_counts, Mapping):
        raise PaperAssetError("judgment summary counts are missing")
    planned_judgments = int(judgment_counts.get("planned_judgments") or 0)
    terminal_judgments = int(judgment_counts.get("terminal_judgments") or 0)
    successful_judgments = int(judgment_counts.get("success") or 0)
    failed_judgments = int(judgment_counts.get("failed") or 0)
    provider_attempt_records = int(
        judgment_counts.get("provider_attempt_records") or terminal_judgments
    )
    first_pass_throttles = int(
        judgment_counts.get("first_pass_documented_throttle_rejections") or 0
    )
    recovery_attempts = int(judgment_counts.get("recovery_attempts") or 0)
    planned_recovery_attempts = int(judgment_counts.get("planned_recovery_attempts") or 0)
    recovered_throttles = int(judgment_counts.get("recovered_to_success") or 0)
    recovery_failures = int(judgment_counts.get("recovery_failures") or 0)
    documented_zero_delivery_throttles = int(
        judgment_counts.get("documented_zero_delivery_throttle_rejections") or 0
    )
    expected_judgments = (
        int(comparison_manifest["counts"]["judgable"]) * len(judge_manifest["judges"]) * 2
    )
    judgment_hashes = judgment_summary.get("judgment_artifact_sha256s")
    if not isinstance(judgment_hashes, list):
        raise PaperAssetError("judgment summary artifact registry is missing")
    if not (
        planned_judgments
        == terminal_judgments
        == expected_judgments
        == int(analysis["counts"]["judgment_records"])
    ):
        raise PaperAssetError("judgment collection is incomplete or analysis counts are stale")
    if successful_judgments + failed_judgments != terminal_judgments:
        raise PaperAssetError("judgment terminal status counts do not reconcile")
    if not (
        first_pass_throttles
        == planned_recovery_attempts
        == recovery_attempts
        == recovered_throttles + recovery_failures
        and provider_attempt_records == terminal_judgments + recovery_attempts
        and isinstance(judgment_summary.get("original_collection_summary_artifact_sha256"), str)
        and len(judgment_summary["original_collection_summary_artifact_sha256"]) == 64
        and isinstance(judgment_summary.get("recovery_plan_artifact_sha256"), str)
        and len(judgment_summary["recovery_plan_artifact_sha256"]) == 64
    ):
        raise PaperAssetError("judgment recovery records do not reconcile")
    if len(judgment_hashes) != terminal_judgments or len(set(judgment_hashes)) != len(
        judgment_hashes
    ):
        raise PaperAssetError("judgment artifact registry is incomplete or duplicated")

    model_rows = list(analysis["model_leaderboard"])
    uplift_rows = list(analysis["uplift_leaderboard"])
    operational = list(analysis["operational_metrics"].values())
    successful_arms = sum(int(row["success"]) for row in operational)
    failed_arms = sum(int(row["failed"]) for row in operational)
    tool_calls = sum(int(row["tool_calls"]) for row in operational)
    primary_arena = next((row for row in model_rows if row.get("rating") is not None), None)
    primary_uplift = uplift_rows[0]
    panel_uplift = analysis["panel_uplift"]
    target_cost = float(
        target_cost_audit["cost_usd"].get("combined_conservative_exposure")
        or target_cost_audit["cost_usd"]["combined_attributed"]
    )
    target_cost_counts = target_cost_audit.get("counts") or {}
    target_unattributed_exposure = float(
        target_cost_audit["cost_usd"].get("unattributed_conservative_reservations") or 0
    )
    judge_cost = float(judgment_summary["estimated_cost_usd"])
    judgment_failure_reasons = judgment_summary.get("failure_reasons") or {}
    consensus_available = int(analysis["counts"]["consensus_available"])
    consensus_rows = int(analysis["counts"]["consensus_rows"])
    consensus_records = list(analysis["comparison_consensus"])
    arena_consensus = sum(
        row.get("primary_consensus_available") is True and row.get("track") == "model_arena"
        for row in consensus_records
    )
    uplift_consensus = sum(
        row.get("primary_consensus_available") is True and row.get("track") == "epicure_uplift"
        for row in consensus_records
    )
    source_judgable = int(
        analysis["counts"].get(
            "source_judgable_comparisons", comparison_manifest["counts"]["judgable"]
        )
    )
    effective_judgable = int(
        analysis["counts"].get("effective_judgable_comparisons", consensus_rows)
    )
    completion_exclusions = int(
        analysis["counts"].get("incomplete_final_response_comparison_exclusions", 0)
    )
    completion_arm_count = int(analysis.get("completion_interpretation_correction_count") or 0)
    normal_completion_arms = int(
        analysis.get("arm_validation", {}).get("normal_completion_arms", successful_arms)
    )
    if (
        source_judgable != int(comparison_manifest["counts"]["judgable"])
        or effective_judgable != consensus_rows
        or source_judgable - completion_exclusions != effective_judgable
    ):
        raise PaperAssetError("completion-policy comparison counts do not reconcile")
    overlap = analysis["reference_overlap_audit"]["overall"]
    verbosity = analysis["verbosity_diagnostics"]
    graph = analysis["arena_graph_diagnostics"]["global"]
    family_sensitivity = analysis["judge_family_balanced_sensitivity"]
    family_sensitivity_diagnostics = family_sensitivity["diagnostics"]
    family_sensitivity_graph = family_sensitivity["arena_graph"]
    family_sensitivity_arena = [
        row for row in family_sensitivity["model_leaderboard"] if row.get("rating") is not None
    ]
    family_sensitivity_uplift = family_sensitivity["panel_uplift"]
    if family_sensitivity_graph["connected"] and family_sensitivity_arena:
        sensitivity_leader = family_sensitivity_arena[0]
        family_sensitivity_arena_result = (
            f"remained connected; {_latex(sensitivity_leader['display_name'])} had the highest "
            f"finite diagnostic rating at {_rating(sensitivity_leader)} points."
        )
    else:
        family_sensitivity_arena_result = (
            "was disconnected, so its Bradley--Terry ratings were withheld."
        )
    primary_bootstrap = (
        analysis["arena_task_cluster_bootstrap"]["models"][primary_arena["season_model_id"]]
        if primary_arena
        else None
    )

    macro_values: dict[str, object] = {
        "SeasonZeroAnalysisSHA": analysis_sha,
        "SeasonZeroComparisonSHA": comparison_sha,
        "SeasonZeroJudgeManifestSHA": judge_sha,
        "SeasonZeroTargetCostSHA": cost_sha,
        "SeasonZeroJudgmentSummarySHA": judgment_sha,
        "SeasonZeroFirstPassJudgmentSummarySHA": judgment_summary[
            "original_collection_summary_artifact_sha256"
        ],
        "SeasonZeroRecoveryPlanSHA": judgment_summary["recovery_plan_artifact_sha256"],
        "SeasonZeroTaskCount": "120",
        "SeasonZeroModelCount": "12",
        "SeasonZeroArmCount": _integer(analysis["counts"]["scored_arms"]),
        "SeasonZeroNormalCompletionArmCount": _integer(normal_completion_arms),
        "SeasonZeroCompletionCorrectionArmCount": _integer(completion_arm_count),
        "SeasonZeroSuccessfulArmCount": _integer(successful_arms),
        "SeasonZeroFailedArmCount": _integer(failed_arms),
        "SeasonZeroToolCallCount": _integer(tool_calls),
        "SeasonZeroComparisonCount": _integer(comparison_manifest["counts"]["comparisons"]),
        "SeasonZeroSourceJudgableComparisonCount": _integer(source_judgable),
        "SeasonZeroJudgableComparisonCount": _integer(effective_judgable),
        "SeasonZeroCompletionComparisonExclusionCount": _integer(completion_exclusions),
        "SeasonZeroJudgmentCount": _integer(terminal_judgments),
        "SeasonZeroJudgeProviderAttemptCount": _integer(provider_attempt_records),
        "SeasonZeroSuccessfulJudgmentCount": _integer(successful_judgments),
        "SeasonZeroFailedJudgmentCount": _integer(failed_judgments),
        "SeasonZeroFirstPassThrottledJudgmentCount": _integer(first_pass_throttles),
        "SeasonZeroThrottleRecoveryAttemptCount": _integer(recovery_attempts),
        "SeasonZeroRecoveredThrottleJudgmentCount": _integer(recovered_throttles),
        "SeasonZeroThrottleRecoveryFailureCount": _integer(recovery_failures),
        "SeasonZeroDocumentedZeroDeliveryThrottleCount": _integer(
            documented_zero_delivery_throttles
        ),
        "SeasonZeroOrphanedJudgmentCount": _integer(
            int(judgment_failure_reasons.get("OrphanedRequestEvent", 0))
            + int(judgment_failure_reasons.get("OrphanedRecoveryRequestEvent", 0))
        ),
        "SeasonZeroReadTimeoutJudgmentCount": _integer(
            judgment_failure_reasons.get("ReadTimeoutError", 0)
        ),
        "SeasonZeroThrottledJudgmentCount": _integer(
            judgment_failure_reasons.get("ThrottlingException", 0)
        ),
        "SeasonZeroInvalidJSONJudgmentCount": _integer(
            judgment_failure_reasons.get("JSONDecodeError", 0)
        ),
        "SeasonZeroServiceUnavailableJudgmentCount": _integer(
            judgment_failure_reasons.get("ServiceUnavailableException", 0)
        ),
        "SeasonZeroProtocolErrorJudgmentCount": _integer(
            judgment_failure_reasons.get("JudgmentProtocolError", 0)
        ),
        "SeasonZeroConsensusCount": _integer(consensus_available),
        "SeasonZeroArenaConsensusCount": _integer(arena_consensus),
        "SeasonZeroUpliftConsensusCount": _integer(uplift_consensus),
        "SeasonZeroConsensusCoverage": _percent(consensus_available / consensus_rows),
        "SeasonZeroTargetCostUSD": f"{target_cost:.2f}",
        "SeasonZeroTargetUnattributedCount": _integer(
            target_cost_counts.get("unattributed_arms") or 0
        ),
        "SeasonZeroTargetUnattributedExposureUSD": (f"{target_unattributed_exposure:.2f}"),
        "SeasonZeroOpenRouterZeroChargeRejections": _integer(
            target_cost_counts.get("zero_charge_explicit_rejections") or 0
        ),
        "SeasonZeroRecoveredCostCorrections": _integer(
            target_cost_counts.get("recovered_generation_corrections") or 0
        ),
        "SeasonZeroJudgeCostUSD": f"{judge_cost:.2f}",
        "SeasonZeroCombinedCostUSD": f"{target_cost + judge_cost:.2f}",
        "SeasonZeroArenaGraphConnected": "yes" if graph["connected"] else "no",
        "SeasonZeroReferenceOverlapRate": _percent(
            overlap.get("novel_reference_12gram_match_rate") or 0
        ),
        "SeasonZeroPreferredLongerRate": _percent(
            verbosity.get("preferred_longer_rate_among_unequal") or 0
        ),
        "SeasonZeroFamilyBalancedCoverage": _percent(
            family_sensitivity_diagnostics.get("coverage") or 0
        ),
        "SeasonZeroFamilyBalancedArenaResult": family_sensitivity_arena_result,
        "SeasonZeroFamilyBalancedPanelUpliftEstimate": (
            f"{float(family_sensitivity_uplift['task_cluster_win_share']):.3f} "
            f"[{float(family_sensitivity_uplift['task_cluster_interval_lower']):.3f}, "
            f"{float(family_sensitivity_uplift['task_cluster_interval_upper']):.3f}]"
        ),
        "SeasonZeroFamilyBalancedPanelUpliftN": _integer(
            family_sensitivity_uplift["valid_comparisons"]
        ),
        "SeasonZeroTopModel": _latex(primary_arena["display_name"])
        if primary_arena
        else "withheld",
        "SeasonZeroTopModelRating": _rating(primary_arena) if primary_arena else "withheld",
        "SeasonZeroTopModelRankOneProbability": (
            _percent(primary_bootstrap["rank_one_probability"])
            if primary_bootstrap and primary_bootstrap.get("rank_one_probability") is not None
            else "unavailable"
        ),
        "SeasonZeroTopModelClusterInterval": (
            f"[{float(primary_bootstrap['rating_interval_lower']):.0f}, "
            f"{float(primary_bootstrap['rating_interval_upper']):.0f}]"
            if primary_bootstrap and primary_bootstrap.get("rating_interval_lower") is not None
            else "unavailable"
        ),
        "SeasonZeroTopUpliftModel": _latex(primary_uplift["display_name"]),
        "SeasonZeroTopUpliftEstimate": _uplift(primary_uplift),
        "SeasonZeroPanelUpliftEstimate": (
            f"{float(panel_uplift['task_cluster_win_share']):.3f} "
            f"[{float(panel_uplift['task_cluster_interval_lower']):.3f}, "
            f"{float(panel_uplift['task_cluster_interval_upper']):.3f}]"
        ),
        "SeasonZeroPanelUpliftWTL": "/".join(
            str(int(panel_uplift[key])) for key in ("epicure_wins", "ties", "unaided_wins")
        ),
        "SeasonZeroPanelUpliftN": _integer(panel_uplift["valid_comparisons"]),
        "SeasonZeroPanelUpliftTaskCount": _integer(panel_uplift["task_clusters"]),
        "SeasonZeroPanelUpliftScore": _decimal(
            int(panel_uplift["epicure_wins"]) + int(panel_uplift["ties"]) / 2,
            1,
        ),
        "SeasonZeroFamilyBalancedPanelUpliftWTL": "/".join(
            str(int(family_sensitivity_uplift[key]))
            for key in ("epicure_wins", "ties", "unaided_wins")
        ),
        "SeasonZeroFamilyBalancedPanelUpliftTaskCount": _integer(
            family_sensitivity_uplift["task_clusters"]
        ),
        "SeasonZeroFamilyExcludedPanelUpliftN": _integer(
            int(panel_uplift["valid_comparisons"])
            - int(family_sensitivity_uplift["valid_comparisons"])
        ),
        "SeasonZeroFamilyExcludedPanelUpliftWTL": "/".join(
            str(int(panel_uplift[key]) - int(family_sensitivity_uplift[key]))
            for key in ("epicure_wins", "ties", "unaided_wins")
        ),
        "SeasonZeroArenaMinN": _integer(
            min(int(row.get("comparisons") or 0) for row in model_rows)
        ),
        "SeasonZeroArenaMaxN": _integer(
            max(int(row.get("comparisons") or 0) for row in model_rows)
        ),
        "SeasonZeroArenaTaskClusters": _integer(
            analysis["arena_task_cluster_bootstrap"]["task_clusters"]
        ),
        "SeasonZeroArenaBootstrapSuccessful": _integer(
            analysis["arena_task_cluster_bootstrap"]["successful_replicates"]
        ),
        "SeasonZeroArenaBootstrapDisconnected": _integer(
            analysis["arena_task_cluster_bootstrap"]["disconnected_replicates"]
        ),
    }
    macros = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macro_values.items()
    )
    pilot_macro_names = (
        "SeasonZeroAnalysisSHA",
        "SeasonZeroArmCount",
        "SeasonZeroNormalCompletionArmCount",
        "SeasonZeroCompletionCorrectionArmCount",
        "SeasonZeroSuccessfulArmCount",
        "SeasonZeroFailedArmCount",
        "SeasonZeroToolCallCount",
        "SeasonZeroComparisonCount",
        "SeasonZeroSourceJudgableComparisonCount",
        "SeasonZeroJudgableComparisonCount",
        "SeasonZeroCompletionComparisonExclusionCount",
        "SeasonZeroJudgmentCount",
        "SeasonZeroJudgeProviderAttemptCount",
        "SeasonZeroConsensusCount",
        "SeasonZeroArenaConsensusCount",
        "SeasonZeroUpliftConsensusCount",
        "SeasonZeroConsensusCoverage",
        "SeasonZeroPreferredLongerRate",
        "SeasonZeroPanelUpliftEstimate",
        "SeasonZeroPanelUpliftWTL",
        "SeasonZeroPanelUpliftN",
        "SeasonZeroPanelUpliftTaskCount",
        "SeasonZeroPanelUpliftScore",
        "SeasonZeroFamilyBalancedPanelUpliftN",
        "SeasonZeroFamilyBalancedPanelUpliftEstimate",
        "SeasonZeroFamilyBalancedPanelUpliftWTL",
        "SeasonZeroFamilyBalancedPanelUpliftTaskCount",
        "SeasonZeroFamilyExcludedPanelUpliftN",
        "SeasonZeroFamilyExcludedPanelUpliftWTL",
        "SeasonZeroArenaMinN",
        "SeasonZeroArenaMaxN",
        "SeasonZeroArenaTaskClusters",
        "SeasonZeroArenaBootstrapSuccessful",
        "SeasonZeroArenaBootstrapDisconnected",
        "SeasonZeroCombinedCostUSD",
    )
    pilot_macros = "\n".join(
        f"\\newcommand{{\\{name}}}{{{macro_values[name]}}}" for name in pilot_macro_names
    )
    paths = {
        "macros": output_dir / "season0_results_macros.tex",
        "pilot_macros": output_dir / "pilot_results_macros.tex",
        "model_table": output_dir / "season0_model_table.tex",
        "uplift_table": output_dir / "season0_uplift_table.tex",
        "family_table": output_dir / "season0_family_table.tex",
        "judge_table": output_dir / "season0_judge_table.tex",
    }
    _write(paths["macros"], macros)
    _write(paths["pilot_macros"], pilot_macros)
    _write(paths["model_table"], _model_table(model_rows))
    _write(paths["uplift_table"], _uplift_table(uplift_rows))
    _write(paths["family_table"], _family_table(analysis))
    _write(paths["judge_table"], _judge_table(analysis, judge_manifest))
    return {key: str(path) for key, path in paths.items()}


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--judge-manifest", type=Path, required=True)
    parser.add_argument("--target-cost-audit", type=Path, required=True)
    parser.add_argument("--judgment-summary", type=Path, required=True)
    parser.add_argument("--arm-interpretation-correction", type=Path, required=True)
    parser.add_argument("--completion-interpretation-correction", type=Path, required=True)
    parser.add_argument("--implementation-source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate_assets(
        analysis=_load(args.analysis),
        comparison_manifest=_load(args.comparison_manifest),
        judge_manifest=_load(args.judge_manifest),
        target_cost_audit=_load(args.target_cost_audit),
        judgment_summary=_load(args.judgment_summary),
        output_dir=args.output_dir,
        arm_interpretation_correction=_load(args.arm_interpretation_correction),
        completion_interpretation_correction=_load(args.completion_interpretation_correction),
        implementation_source_root=args.implementation_source_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
