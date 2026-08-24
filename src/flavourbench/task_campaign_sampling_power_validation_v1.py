"""Build a blocked, exact-frame statistical design validation for sampling v2.

This CLI is offline and coordinate-only.  It verifies content-addressed local
sources, runs deterministic simulated operating-characteristic checks, and emits
a strict NO-GO candidate.  It cannot activate a study or create evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scipy import stats

from .real_task_bank import sha256_json
from .sampling_power_engine_v1 import (
    DEFAULT_SCENARIOS,
    FrameSpec,
    build_frame_spec,
    run_validation,
)
from .task_campaign_human_sampling_successor_v2 import (
    materialize_sampling_frame_v2,
    verify_sampling_artifact_v2,
)
from .task_campaign_study_design_successor import verify_successor_design

SCHEMA_VERSION = "flavourbench-season1-sampling-power-validation-v1-candidate"
STATUS = "blocked_no_go_exact_frame_power_not_validated"
DEFAULT_DATASETS = 500
DEFAULT_SEED = 20260809
SUPERSEDED_ARTIFACT_SEMANTIC_SHA256 = (
    "8cd9217f6e59b47de0f66f1fc6c45fe5766b8729d3f808ee83cfb2e5147c7a9d"
)
SUPERSEDED_ARTIFACT_PHYSICAL_SHA256 = (
    "5a91e9e9b18d7004313a80f0c1754c33ec8012797ee54709b6e6534c0937d421"
)

FLAVOURBENCH_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PAPER_ROOT = FLAVOURBENCH_ROOT.parent

SAMPLING_V2_SEMANTIC_SHA256 = "34e52469f335d65bc3369726b06ff226ca6b1df0f43121b42695bc79bb1a1dc2"
SAMPLING_V2_PHYSICAL_SHA256 = "d753c003d86072de81b7663fae6327018649c062ed045b629baad1632973040f"
REQUESTED_TRUNCATED_PHYSICAL_LITERAL = (
    "d753c003d86072de81b7663fae6327018649c062ed045b629baad1632973040"
)
SAMPLING_V1_SEMANTIC_SHA256 = "5a0b1bbeb20564c9e8fde78b958bbed723ee0cc3395c809267c3775adeed95f8"
SAMPLING_V1_PHYSICAL_SHA256 = "6c7371daa9506cdf5dcee38c19ee48dd2938f36d259c1ee04b7c849c49977039"
DESIGN_SEMANTIC_SHA256 = "e9d31fffbd0e6a7791c04e0cc0b0c4308bfac91745099e0e685c38224479f59e"
DESIGN_PHYSICAL_SHA256 = "6affdc8f80e59476254834d8edc588c471a5bd7e86145e66448e4fb7b90118af"
LEGACY_V5_DESIGN_SEMANTIC_SHA256 = (
    "7a63cfd6117338a3af16a422d5ee3458298fdc0ff2fd0abfe45fe851a7e54506"
)
LEGACY_V5_DESIGN_PHYSICAL_SHA256 = (
    "57080b61171ad81d2d0d40307939ad9681db9a37452262b1baec4613d2b477cd"
)
K16_ALTERNATIVE_SEMANTIC_SHA256 = "675cdb81bcbd54cf3532025ae70069723d7e9843b0eeeb92f1ea38bee7c58278"
K16_ALTERNATIVE_PHYSICAL_SHA256 = "f26bac6aa12a6dfca4ab0ee8ff7f2a0814a2214e8f21f9292d09221ec5104740"
ARENA_POLICY_SEMANTIC_SHA256 = "bdc0fa93c6365cdcd45694d1d5500d82ccbd622f3be897be9217e252855ffff5"
ARENA_POLICY_PHYSICAL_SHA256 = "02adfc4a32e2690c1f8f5ddce6edba3f1974159956027b88003a484f5a0655bc"
OLD_ARENA_MC_SEMANTIC_SHA256 = "2ba9f4d5b9b2f5a231edf085ea55b6ab67097780c12be86868082dbbdac95351"
OLD_ARENA_MC_PHYSICAL_SHA256 = "ef259b46e2e549ab163fe7f695a9619adb07cc66afecb4a89cb6d102424b937b"
PRODUCTION_STATISTICS_PHYSICAL_SHA256 = (
    "4739a404be983956ff6d178dbc1521c8c5fb5fea3699014687f1e7c40ce2e452"
)
OLD_ARENA_MC_ENGINE_PHYSICAL_SHA256 = (
    "e157419b99de8b5547ba2543931a3d2ab9906598594ae860adb5c3a06715e3f2"
)
VALIDATION_ENGINE_PHYSICAL_SHA256 = (
    "dac31aa2327311a6a5ee4d10d35b603b62ba1a3e682036594f0b1bd2e3233777"
)

SAMPLING_V2_REFERENCE = (
    "flavourbench/artifacts/season1/human-judgment-sampling-v2-candidate/"
    f"human-judgment-sampling-v2-candidate-{SAMPLING_V2_SEMANTIC_SHA256}.json"
)
SAMPLING_V1_REFERENCE = (
    "flavourbench/artifacts/season1/human-judgment-sampling-v1-candidate/"
    f"human-judgment-sampling-v1-candidate-{SAMPLING_V1_SEMANTIC_SHA256}.json"
)
DESIGN_REFERENCE = (
    "flavourbench/artifacts/season1/study-design-v6-candidate/"
    f"study-design-v6-candidate-{DESIGN_SEMANTIC_SHA256}.json"
)
LEGACY_V5_DESIGN_REFERENCE = "flavourbench/contracts/season1/season1-study-design-v5.json"
K16_ALTERNATIVE_REFERENCE = (
    "flavourbench/artifacts/season1/study-design-16-model-alternative-v1-candidate/"
    "study-design-16-model-alternative-v1-candidate-"
    f"{K16_ALTERNATIVE_SEMANTIC_SHA256}.json"
)
ARENA_POLICY_REFERENCE = "flavourbench/contracts/season1/season1-arena-inference-acceptance-v1.json"
OLD_ARENA_MC_REFERENCE = (
    "flavourbench/contracts/season1/method-validation/season1-arena-production-monte-carlo-v1.json"
)
PRODUCTION_STATISTICS_REFERENCE = "flavourbench/src/flavourbench/season1_statistics.py"
OLD_ARENA_MC_ENGINE_REFERENCE = "flavourbench/src/flavourbench/season1_arena_monte_carlo.py"
VALIDATION_ENGINE_REFERENCE = "flavourbench/src/flavourbench/sampling_power_engine_v1.py"
SUPERSEDED_ARTIFACT_REFERENCE = (
    "flavourbench/artifacts/season1/sampling-power-validation-v1-candidate/"
    "sampling-power-validation-v1-candidate-"
    f"{SUPERSEDED_ARTIFACT_SEMANTIC_SHA256}.json"
)

DEFAULT_SAMPLING_V2 = EVALUATION_PAPER_ROOT / SAMPLING_V2_REFERENCE
DEFAULT_DESIGN = EVALUATION_PAPER_ROOT / DESIGN_REFERENCE
DEFAULT_LEGACY_V5_DESIGN = EVALUATION_PAPER_ROOT / LEGACY_V5_DESIGN_REFERENCE
DEFAULT_K16_ALTERNATIVE = EVALUATION_PAPER_ROOT / K16_ALTERNATIVE_REFERENCE
DEFAULT_ARENA_POLICY = EVALUATION_PAPER_ROOT / ARENA_POLICY_REFERENCE
DEFAULT_OLD_ARENA_MC = EVALUATION_PAPER_ROOT / OLD_ARENA_MC_REFERENCE
DEFAULT_PRODUCTION_STATISTICS = EVALUATION_PAPER_ROOT / PRODUCTION_STATISTICS_REFERENCE
DEFAULT_OLD_ARENA_MC_ENGINE = EVALUATION_PAPER_ROOT / OLD_ARENA_MC_ENGINE_REFERENCE
DEFAULT_VALIDATION_ENGINE = EVALUATION_PAPER_ROOT / VALIDATION_ENGINE_REFERENCE
DEFAULT_SUPERSEDED_ARTIFACT = EVALUATION_PAPER_ROOT / SUPERSEDED_ARTIFACT_REFERENCE
DEFAULT_OUTPUT_DIR = FLAVOURBENCH_ROOT / "artifacts/season1/sampling-power-validation-v1-candidate"


class SamplingPowerValidationError(RuntimeError):
    """A source binding, validation contract, or no-replace write failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SamplingPowerValidationError(message)


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SamplingPowerValidationError(f"cannot open regular source: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"source is not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_bound_bytes(path: Path, expected_sha256: str) -> bytes:
    data = _read_regular_bytes(path)
    _require(
        hashlib.sha256(data).hexdigest() == expected_sha256,
        f"physical digest mismatch: {path}",
    )
    return data


def _load_bound_json(path: Path, *, physical_sha256: str, semantic_sha256: str) -> dict[str, Any]:
    try:
        document = json.loads(_read_bound_bytes(path, physical_sha256).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SamplingPowerValidationError(f"invalid bound JSON: {path}") from error
    _require(isinstance(document, dict), f"bound JSON is not an object: {path}")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(
        document.get("artifact_sha256") == semantic_sha256 and sha256_json(body) == semantic_sha256,
        f"semantic digest mismatch: {path}",
    )
    return document


def _load_sources(
    *,
    sampling_v2_path: Path,
    design_path: Path,
    legacy_v5_design_path: Path,
    k16_alternative_path: Path,
    arena_policy_path: Path,
    old_arena_mc_path: Path,
    production_statistics_path: Path,
    old_arena_mc_engine_path: Path,
    validation_engine_path: Path,
    superseded_artifact_path: Path,
) -> tuple[
    FrameSpec,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    sampling = _load_bound_json(
        sampling_v2_path,
        physical_sha256=SAMPLING_V2_PHYSICAL_SHA256,
        semantic_sha256=SAMPLING_V2_SEMANTIC_SHA256,
    )
    verify_sampling_artifact_v2(sampling)
    design = _load_bound_json(
        design_path,
        physical_sha256=DESIGN_PHYSICAL_SHA256,
        semantic_sha256=DESIGN_SEMANTIC_SHA256,
    )
    verify_successor_design(design)
    legacy_v5 = _load_bound_json(
        legacy_v5_design_path,
        physical_sha256=LEGACY_V5_DESIGN_PHYSICAL_SHA256,
        semantic_sha256=LEGACY_V5_DESIGN_SEMANTIC_SHA256,
    )
    k16_alternative = _load_bound_json(
        k16_alternative_path,
        physical_sha256=K16_ALTERNATIVE_PHYSICAL_SHA256,
        semantic_sha256=K16_ALTERNATIVE_SEMANTIC_SHA256,
    )
    policy = _load_bound_json(
        arena_policy_path,
        physical_sha256=ARENA_POLICY_PHYSICAL_SHA256,
        semantic_sha256=ARENA_POLICY_SEMANTIC_SHA256,
    )
    old_mc = _load_bound_json(
        old_arena_mc_path,
        physical_sha256=OLD_ARENA_MC_PHYSICAL_SHA256,
        semantic_sha256=OLD_ARENA_MC_SEMANTIC_SHA256,
    )
    _read_bound_bytes(production_statistics_path, PRODUCTION_STATISTICS_PHYSICAL_SHA256)
    _read_bound_bytes(old_arena_mc_engine_path, OLD_ARENA_MC_ENGINE_PHYSICAL_SHA256)
    _read_bound_bytes(validation_engine_path, VALIDATION_ENGINE_PHYSICAL_SHA256)
    _load_bound_json(
        superseded_artifact_path,
        physical_sha256=SUPERSEDED_ARTIFACT_PHYSICAL_SHA256,
        semantic_sha256=SUPERSEDED_ARTIFACT_SEMANTIC_SHA256,
    )
    frame = build_frame_spec(materialize_sampling_frame_v2(sampling))
    _require(frame.roster_size == 14, "bound source is not the exact 14-model frame")
    _require(frame.task_count == 80, "bound source is not the exact 80-task frame")
    _require(
        design["candidate_model_panel"]["model_count"] == frame.roster_size,
        "sampling roster and source design disagree",
    )
    _require(
        legacy_v5.get("model_panel", {}).get("candidate_count") == 16
        and legacy_v5.get("task_bank", {}).get("splits", {}).get("scored") == 160
        and legacy_v5.get("primary_controlled_collection", {})
        .get("model_arena", {})
        .get("total_battles")
        == 3200
        and legacy_v5.get("primary_controlled_collection", {})
        .get("epicure_uplift", {})
        .get("total_pairs")
        == 3200,
        "legacy v5 full production layout changed",
    )
    _require(
        k16_alternative.get("candidate_model_panel", {}).get("model_count") == 16
        and k16_alternative.get("status") == "blocked_offline_16_model_alternative_not_authorized",
        "K16 two-lane candidate boundary changed",
    )
    return frame, policy, old_mc, legacy_v5, k16_alternative


def _source_commitments() -> list[dict[str, Any]]:
    return [
        {
            "role": "exact_human_judgment_sampling_v2_starting_frame",
            "reference_path": SAMPLING_V2_REFERENCE,
            "semantic_sha256": SAMPLING_V2_SEMANTIC_SHA256,
            "physical_sha256": SAMPLING_V2_PHYSICAL_SHA256,
        },
        {
            "role": "sampling_v2_inherited_v1_coordinate_frame",
            "reference_path": SAMPLING_V1_REFERENCE,
            "semantic_sha256": SAMPLING_V1_SEMANTIC_SHA256,
            "physical_sha256": SAMPLING_V1_PHYSICAL_SHA256,
        },
        {
            "role": "source_14_model_80_task_study_design_v6",
            "reference_path": DESIGN_REFERENCE,
            "semantic_sha256": DESIGN_SEMANTIC_SHA256,
            "physical_sha256": DESIGN_PHYSICAL_SHA256,
        },
        {
            "role": "legacy_full_k16_240_task_160_scored_design_comparator",
            "reference_path": LEGACY_V5_DESIGN_REFERENCE,
            "semantic_sha256": LEGACY_V5_DESIGN_SEMANTIC_SHA256,
            "physical_sha256": LEGACY_V5_DESIGN_PHYSICAL_SHA256,
        },
        {
            "role": "blocked_current_two_lane_k16_roster_and_80_task_design_comparator",
            "reference_path": K16_ALTERNATIVE_REFERENCE,
            "semantic_sha256": K16_ALTERNATIVE_SEMANTIC_SHA256,
            "physical_sha256": K16_ALTERNATIVE_PHYSICAL_SHA256,
        },
        {
            "role": "frozen_production_arena_acceptance_policy_v1_nontransferability_reference",
            "reference_path": ARENA_POLICY_REFERENCE,
            "semantic_sha256": ARENA_POLICY_SEMANTIC_SHA256,
            "physical_sha256": ARENA_POLICY_PHYSICAL_SHA256,
        },
        {
            "role": "old_k16_production_monte_carlo_contract_nontransferability_reference",
            "reference_path": OLD_ARENA_MC_REFERENCE,
            "semantic_sha256": OLD_ARENA_MC_SEMANTIC_SHA256,
            "physical_sha256": OLD_ARENA_MC_PHYSICAL_SHA256,
        },
        {
            "role": "current_production_bt_and_crossed_cluster_statistics_engine",
            "reference_path": PRODUCTION_STATISTICS_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": PRODUCTION_STATISTICS_PHYSICAL_SHA256,
        },
        {
            "role": "old_k16_monte_carlo_engine_nontransferability_reference",
            "reference_path": OLD_ARENA_MC_ENGINE_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": OLD_ARENA_MC_ENGINE_PHYSICAL_SHA256,
        },
        {
            "role": "exact_frame_coordinate_only_validation_engine",
            "reference_path": VALIDATION_ENGINE_REFERENCE,
            "semantic_sha256": None,
            "physical_sha256": VALIDATION_ENGINE_PHYSICAL_SHA256,
        },
        {
            "role": (
                "superseded_sampling_power_candidate_with_proved_planning_and_weighting_defects"
            ),
            "reference_path": SUPERSEDED_ARTIFACT_REFERENCE,
            "semantic_sha256": SUPERSEDED_ARTIFACT_SEMANTIC_SHA256,
            "physical_sha256": SUPERSEDED_ARTIFACT_PHYSICAL_SHA256,
        },
    ]


def _exact_binomial_rate_bounds(rate: float | None, trials: int) -> tuple[float, float]:
    """Return a two-sided 95% Clopper-Pearson interval for a simulated rate."""

    if rate is None:
        return 0.0, 1.0
    _require(trials > 0, "Monte Carlo trial count must be positive")
    successes = int(round(rate * trials))
    _require(
        0 <= successes <= trials and abs(rate - successes / trials) <= 0.5 / trials + 1e-12,
        "reported Monte Carlo rate is incompatible with its trial count",
    )
    interval = stats.binomtest(successes, trials).proportion_ci(
        confidence_level=0.95,
        method="exact",
    )
    return float(interval.low), float(interval.high)


def _candidate_checks(validation: Mapping[str, Any], datasets: int) -> list[dict[str, Any]]:
    rows = {row["scenario_id"]: row for row in validation["scenario_results"]}
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, observed: Any, threshold: str) -> None:
        checks.append(
            {
                "check_id": check_id,
                "core": True,
                "passed": bool(passed),
                "observed": observed,
                "prespecified_threshold": threshold,
            }
        )

    add(
        "monte_carlo_datasets_per_scenario",
        datasets >= DEFAULT_DATASETS,
        datasets,
        f">= {DEFAULT_DATASETS}",
    )
    add(
        "worst_case_binomial_mcse",
        0.5 / datasets**0.5 <= 0.025,
        round(0.5 / datasets**0.5, 6),
        "<= 0.025",
    )
    for scenario_id in (
        "null_complete_moderate_dependence",
        "null_high_task_rater_dependence",
    ):
        row = rows[scenario_id]["overall_uplift"]
        coverage_lower, _ = _exact_binomial_rate_bounds(row["coverage"], datasets)
        _, type_i_upper = _exact_binomial_rate_bounds(row["two_sided_type_i_error"], datasets)
        add(
            f"{scenario_id}_overall_coverage",
            coverage_lower >= 0.90,
            {
                "rate": row["coverage"],
                "exact_binomial_95_lower": round(coverage_lower, 6),
            },
            "exact binomial Monte Carlo 95% lower bound >= 0.90",
        )
        add(
            f"{scenario_id}_overall_type_i",
            type_i_upper <= 0.08,
            {
                "rate": row["two_sided_type_i_error"],
                "exact_binomial_95_upper": round(type_i_upper, 6),
            },
            "exact binomial Monte Carlo 95% upper bound <= 0.08",
        )

    for scenario_id in (
        "calibrated_0_08_complete",
        "calibrated_0_08_high_dependence",
        "mcar_5pct_responses_and_ratings",
        "plausible_rater_dropout",
    ):
        row = rows[scenario_id]["overall_uplift"]
        power_lower, _ = _exact_binomial_rate_bounds(row["one_sided_power"], datasets)
        coverage_lower, _ = _exact_binomial_rate_bounds(row["coverage"], datasets)
        add(
            f"{scenario_id}_overall_power_at_0_08",
            power_lower >= 0.80,
            {
                "rate": row["one_sided_power"],
                "exact_binomial_95_lower": round(power_lower, 6),
            },
            "exact binomial Monte Carlo 95% lower bound >= 0.80",
        )
        add(
            f"{scenario_id}_overall_coverage",
            coverage_lower >= 0.90,
            {
                "rate": row["coverage"],
                "exact_binomial_95_lower": round(coverage_lower, 6),
            },
            "exact binomial Monte Carlo 95% lower bound >= 0.90",
        )
        add(
            f"{scenario_id}_absolute_bias",
            abs(row["bias"]) <= 0.02,
            row["bias"],
            "absolute bias <= 0.02 half-win share",
        )

    mnar = rows["outcome_dependent_missingness"]["overall_uplift"]
    mnar_coverage_lower, _ = _exact_binomial_rate_bounds(mnar["coverage"], datasets)
    add(
        "outcome_dependent_missingness_overall_coverage",
        mnar_coverage_lower >= 0.90,
        {
            "rate": mnar["coverage"],
            "exact_binomial_95_lower": round(mnar_coverage_lower, 6),
        },
        "exact binomial Monte Carlo 95% lower bound >= 0.90",
    )
    add(
        "outcome_dependent_missingness_absolute_bias",
        abs(mnar["bias"]) <= 0.02,
        mnar["bias"],
        "absolute bias <= 0.02 half-win share",
    )

    crossover = rows["family_crossover_zero_overall"]["overall_uplift"]
    crossover_lower, _ = _exact_binomial_rate_bounds(crossover["coverage"], datasets)
    add(
        "family_crossover_zero_overall_coverage",
        crossover_lower >= 0.90,
        {
            "rate": crossover["coverage"],
            "exact_binomial_95_lower": round(crossover_lower, 6),
        },
        "exact binomial Monte Carlo 95% lower bound >= 0.90",
    )

    for scenario_id in ("calibrated_0_08_complete", "calibrated_0_08_high_dependence"):
        row = rows[scenario_id]
        family_power = row["family_uplift"]["minimum_one_sided_detection_power"]
        family_power_lower, _ = _exact_binomial_rate_bounds(family_power, datasets)
        add(
            f"{scenario_id}_family_power_at_0_08",
            family_power_lower >= 0.80,
            {
                "rate": family_power,
                "exact_binomial_95_lower": round(family_power_lower, 6),
            },
            "minimum family exact-binomial Monte Carlo 95% power lower bound >= 0.80",
        )
        add(
            f"{scenario_id}_family_precision",
            row["family_uplift"]["mean_simultaneous_ci_halfwidth"] <= 0.10,
            row["family_uplift"]["mean_simultaneous_ci_halfwidth"],
            "mean 4-family simultaneous CI half-width <= 0.10",
        )
        model_power = row["per_model_uplift"]["minimum_one_sided_detection_power"]
        model_power_lower, _ = _exact_binomial_rate_bounds(model_power, datasets)
        add(
            f"{scenario_id}_per_model_power",
            model_power_lower >= 0.80,
            {
                "rate": model_power,
                "exact_binomial_95_lower": round(model_power_lower, 6),
            },
            "minimum model exact-binomial Monte Carlo 95% power lower bound >= 0.80",
        )
        add(
            f"{scenario_id}_per_model_precision",
            row["per_model_uplift"]["mean_bonferroni_ci_halfwidth"] <= 0.10,
            row["per_model_uplift"]["mean_bonferroni_ci_halfwidth"],
            "mean roster-family simultaneous CI half-width <= 0.10",
        )
        arena = row["arena_ranking"]
        power_lower, _ = _exact_binomial_rate_bounds(
            arena["focal_shifted_model_identified_top_power"], datasets
        )
        add(
            f"{scenario_id}_arena_proxy_top_identification_at_50_elo",
            power_lower >= 0.80,
            {
                "rate": arena["focal_shifted_model_identified_top_power"],
                "exact_binomial_95_lower": round(power_lower, 6),
            },
            "diagnostic BT/sandwich proxy exact-binomial Monte Carlo 95% lower bound >= 0.80",
        )

    for scenario_id in rows:
        missing = rows[scenario_id]["missingness_and_connectivity"]
        add(
            f"{scenario_id}_full_graph_connectivity",
            missing["full_graph_connected_rate"] >= 0.99,
            missing["full_graph_connected_rate"],
            ">= 0.99",
        )
        add(
            f"{scenario_id}_family_graph_connectivity",
            missing["all_four_family_graphs_connected_rate"] >= 0.99,
            missing["all_four_family_graphs_connected_rate"],
            ">= 0.99",
        )

    reliability = rows["calibrated_0_08_complete"]["reliability"]
    reliability_lower, _ = _exact_binomial_rate_bounds(
        reliability["overall_exact_agreement_coverage"], datasets
    )
    add(
        "reliability_overall_coverage",
        reliability_lower >= 0.90,
        {
            "rate": reliability["overall_exact_agreement_coverage"],
            "exact_binomial_95_lower": round(reliability_lower, 6),
        },
        "exact binomial Monte Carlo 95% lower bound >= 0.90",
    )
    add(
        "reliability_overall_precision",
        reliability["mean_ci_halfwidth"] <= 0.05,
        reliability["mean_ci_halfwidth"],
        "mean task-cluster CI half-width <= 0.05",
    )
    add(
        "reliability_track_precision",
        reliability["mean_track_simultaneous_ci_halfwidth"] <= 0.06,
        reliability["mean_track_simultaneous_ci_halfwidth"],
        "mean two-track simultaneous CI half-width <= 0.06",
    )
    return checks


def _old_contract_reconciliation(
    frame: FrameSpec, validation: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    diagnostics = validation["frame_diagnostics"]
    support = diagnostics["arena_support"]
    policy_global = policy["global_fit"]
    policy_family = policy["family_specific_fit"]
    policy_pair = policy["pairwise_reporting"]
    policy_simulation = policy["simulation_gate"]
    checks = [
        {
            "requirement": "global admitted scored tasks",
            "required": policy_global["required_admitted_scored_tasks"],
            "observed": frame.task_count,
            "passed": frame.task_count >= policy_global["required_admitted_scored_tasks"],
        },
        {
            "requirement": "global admitted scored tasks per family",
            "required": policy_global["required_admitted_scored_tasks_per_family"],
            "observed": min(diagnostics["tasks_per_family"].values()),
            "passed": min(diagnostics["tasks_per_family"].values())
            >= policy_global["required_admitted_scored_tasks_per_family"],
        },
        {
            "requirement": "global comparisons per model",
            "required": policy_global["minimum_unique_comparisons_per_model"],
            "observed": support["minimum_global_comparisons_per_model"],
            "passed": support["minimum_global_comparisons_per_model"]
            >= policy_global["minimum_unique_comparisons_per_model"],
        },
        {
            "requirement": "family comparisons per model",
            "required": policy_family["minimum_unique_comparisons_per_model"],
            "observed": support["minimum_family_comparisons_per_model"],
            "passed": support["minimum_family_comparisons_per_model"]
            >= policy_family["minimum_unique_comparisons_per_model"],
        },
        {
            "requirement": "shared task clusters per reported pair",
            "required": policy_pair["minimum_shared_task_clusters_for_interval"],
            "observed": support["minimum_shared_task_clusters_per_pair"],
            "passed": support["minimum_shared_task_clusters_per_pair"]
            >= policy_pair["minimum_shared_task_clusters_for_interval"],
        },
        {
            "requirement": "production Monte Carlo roster size",
            "required": policy_simulation["models"],
            "observed": frame.roster_size,
            "passed": frame.roster_size == policy_simulation["models"],
        },
        {
            "requirement": "old exact production layout",
            "required": "K16; 160 tasks; 20 arena comparisons/task; 3,200 arena comparisons",
            "observed": (
                f"K{frame.roster_size}; {frame.task_count} tasks; "
                f"10 arena comparisons/task; {len(frame.arena_task)} arena comparisons"
            ),
            "passed": False,
        },
        {
            "requirement": "validated surrogate permitted",
            "required": policy_simulation["validated_surrogate_permitted"],
            "observed": (
                "production-equivalent hierarchical BT point-estimator plus finite-cluster "
                "sandwich diagnostic proxy"
            ),
            "passed": False,
        },
    ]
    return {
        "status": "non_transferable_and_current_frame_fails_frozen_v1",
        "old_pass_or_contract_may_not_be_reused": True,
        "checks": checks,
        "failed_requirements": [row["requirement"] for row in checks if not row["passed"]],
    }


def _estimands() -> dict[str, Any]:
    return {
        "tie_score": {"loss": 0.0, "tie": 0.5, "win": 1.0},
        "primary": {
            "overall_epicure_uplift": {
                "estimand": (
                    "equal-family mean of scheduled-comparison Epicure-on half-win share "
                    "minus 0.5 over the exact 800-comparison frame"
                ),
                "inference": (
                    "comparison means first; task means second; four family means equally "
                    "weighted; Welch-Satterthwaite finite-cluster t interval"
                ),
                "missingness_target": "full intent-to-evaluate schedule, not observed-only rows",
            },
            "arena_model_ranking": {
                "production_estimand": (
                    "family-standardized weighted Bradley-Terry model ratings and ranks under "
                    "season1_statistics.py"
                ),
                "production_interval": (
                    "family-stratified task-cluster by crossed rater-cluster bootstrap"
                ),
                "current_status": "withheld; frozen v1 structural and simulation gates fail",
                "diagnostic_only": (
                    "exact-frame comparison-aggregated BT point fit with production-equivalent "
                    "hierarchical weights, CR1 task-cluster sandwich, a conservative 19-df cap, "
                    "and Bonferroni 91-pair screen"
                ),
            },
        },
        "confirmatory_families": {
            "per_model_epicure_uplift": {
                "count": "roster_size parameter (14 in bound frame)",
                "multiplicity": "Bonferroni simultaneous intervals across roster",
                "production_contract_status": (
                    "withheld: existing controlled uplift requires 200 pairs/model and "
                    "50/model/family; exact frame has 57-58 and 14-15"
                ),
            },
            "family_epicure_uplift": {
                "count": 4,
                "multiplicity": "Bonferroni simultaneous intervals across four families",
                "small_cluster_caution": "20 task clusters per family; t(19), never row-level n",
            },
        },
        "reliability": {
            "estimand": "exact same-rater categorical agreement on 400 concealed repeats",
            "inference": "task-cluster interval overall and two-track simultaneous intervals",
            "not_identifiable": (
                "per-human-rater reliability and rater-component variance; frame has abstract "
                "slots but zero assigned identities"
            ),
        },
        "exploratory_only": [
            "individual model-pair contrasts below support floor",
            "complete 14-model total order",
            "model-by-family uplift interactions",
            "family-specific arena ranks",
            "post-hoc missingness and rater-subgroup contrasts",
        ],
        "independence_boundary": {
            "independent_comparisons_claimed": False,
            "independent_ratings_claimed": False,
            "1600_comparisons_used_as_independent_n": False,
            "3200_primary_ratings_used_as_independent_n": False,
            "primary_cluster_count": 80,
            "smallest_family_cluster_count": 20,
        },
    }


def _full_benchmark_comparison(
    legacy_v5: Mapping[str, Any],
    k16_alternative: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare reduced frames with the full frozen-layout scientific target."""

    v5_primary = legacy_v5["primary_controlled_collection"]
    arena_comparisons = int(v5_primary["model_arena"]["total_battles"])
    uplift_comparisons = int(v5_primary["epicure_uplift"]["total_pairs"])
    comparisons = arena_comparisons + uplift_comparisons
    raters_per_comparison = int(legacy_v5["expert_evaluation"]["ratings_per_comparison"])
    primary_ratings = comparisons * raters_per_comparison
    repeat_rate = float(legacy_v5["expert_evaluation"]["reliability_repeat_rate"])
    repeats = round(primary_ratings * repeat_rate)
    total_presentations = primary_ratings + repeats
    _require(
        (arena_comparisons, uplift_comparisons, primary_ratings, repeats, total_presentations)
        == (3200, 3200, 12800, 1600, 14400),
        "full-layout human workload changed",
    )
    return {
        "minimum_structurally_aligned_candidate": (
            "frozen-v5 K16 240-task/160-scored full primary layout, after freezing an "
            "eligible exact 16-model roster and regenerating all addresses"
        ),
        "recommendation_scope": (
            "structurally aligned with the frozen numeric gates but not a power recommendation; "
            "the exact production simulation must pass before activation, and further arena "
            "expansion may be required"
        ),
        "design_options": {
            "bound_reduced_k14_v6_human_frame": {
                "models": 14,
                "scored_task_clusters": 80,
                "arena_comparisons": 800,
                "uplift_comparisons": 800,
                "primary_ratings": 3200,
                "repeat_presentations": 400,
                "total_rating_presentations": 3600,
                "ranking_and_per_model_uplift": "NO_GO",
            },
            "blocked_reduced_k16_two_lane_human_frame": {
                "source_semantic_sha256": K16_ALTERNATIVE_SEMANTIC_SHA256,
                "models": 16,
                "scored_task_clusters": 80,
                "arena_generation_comparisons": int(k16_alternative["arithmetic"]["arena_battles"]),
                "uplift_generation_comparisons": int(k16_alternative["arithmetic"]["uplift_pairs"]),
                "human_arena_comparisons": 800,
                "human_uplift_comparisons": 800,
                "primary_ratings": 3200,
                "repeat_presentations": 400,
                "total_rating_presentations": 3600,
                "pair_task_support": "6-7 versus required 10",
                "ranking_and_per_model_uplift": "NO_GO",
                "roster_boundary": (
                    "unofficial: Qwen 3.8 Max is a mutable alias; Kimi K3 immutability is "
                    "unproven; all routes remain unranked and unauthorized"
                ),
            },
            "full_v5_k16_primary_layout": {
                "source_semantic_sha256": LEGACY_V5_DESIGN_SEMANTIC_SHA256,
                "task_bank_total": int(legacy_v5["task_bank"]["total"]),
                "scored_task_clusters": int(legacy_v5["task_bank"]["splits"]["scored"]),
                "scored_task_clusters_per_family": 40,
                "arena_comparisons": arena_comparisons,
                "uplift_comparisons": uplift_comparisons,
                "arena_comparisons_per_model": 400,
                "arena_comparisons_per_model_family": 100,
                "arena_pair_task_support_under_exact_production_layout": "26-27",
                "uplift_comparisons_per_model": 200,
                "uplift_comparisons_per_model_family": 50,
                "production_structure_matches_frozen_numeric_gates": True,
                "production_power_validated": False,
            },
        },
        "full_layout_human_rating_workload": {
            "unique_arena_comparisons": arena_comparisons,
            "unique_uplift_comparisons": uplift_comparisons,
            "unique_comparisons": comparisons,
            "distinct_raters_per_comparison": raters_per_comparison,
            "primary_rating_presentations": primary_ratings,
            "concealed_repeat_rate_of_primary_ratings": repeat_rate,
            "concealed_repeat_presentations": repeats,
            "total_rating_presentations": total_presentations,
            "rating_hours": {
                "at_3_minutes_each": 720,
                "at_4_minutes_each": 960,
                "at_5_minutes_each": 1200,
                "at_8_minutes_each": 1920,
            },
            "authorization_or_cost_claim": False,
        },
        "production_bt_bootstrap_validation": {
            "method_is_implemented_and_distributable": True,
            "frozen_required_datasets_per_scenario": 2000,
            "frozen_scenarios": 8,
            "frozen_bootstrap_replicates_per_dataset": 5000,
            "nominal_total_bootstrap_refits": 80_000_000,
            "completed_exact_final_roster_receipts": 0,
            "can_be_validated": True,
            "validated_now": False,
            "required_action": (
                "freeze the final eligible K16 roster and full 160-task coordinate frame, then "
                "run and seal the exact production weighted BT plus crossed task/rater bootstrap "
                "over all frozen scenarios; no surrogate or old-roster result may transfer"
            ),
        },
    }


def _conclusions(
    frame: FrameSpec,
    validation: Mapping[str, Any],
    candidate_checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    diagnostics = validation["frame_diagnostics"]
    failed = [str(row["check_id"]) for row in candidate_checks if not row["passed"]]
    uplift_appearances = Counter(frame.uplift_model.tolist())
    family_uplift_appearances = Counter(
        (int(model), int(family))
        for model, family in zip(frame.uplift_model, frame.uplift_family, strict=True)
    )
    return {
        "decision": "NO_GO_BOUNDED_EXACT_FRAME_POWER_NOT_VALIDATED",
        "power_validated": False,
        "precision_validated": False,
        "type_i_error_validated_for_official_method": False,
        "failed_candidate_core_thresholds": failed,
        "minimum_structurally_aligned_full_benchmark_candidate": {
            "design": (
                "frozen-v5 K16 full layout: 240 tasks, 160 scored, 3,200 arena "
                "comparisons, and 3,200 uplift comparisons"
            ),
            "power_sufficiency_unknown": True,
            "may_require_more_arena_sampling": True,
            "recommended_for_activation": False,
            "executable_now": False,
            "conditions": (
                "freeze an exact eligible K16 two-lane roster, regenerate content addresses, "
                "and pass the exact production 2,000-dataset by 5,000-bootstrap gate before "
                "authorizing or funding 14,400 human rating presentations; expand the arena "
                "layout and revalidate if the 50-Elo power gate fails"
            ),
        },
        "arena_14_model_ranking": {
            "decision": "NO_GO_WITHHOLD_RANKS_AND_PAIRWISE_INTERVALS",
            "reason": (
                "pair support is "
                f"{diagnostics['arena_support']['minimum_shared_task_clusters_per_pair']}-"
                f"{diagnostics['arena_support']['maximum_shared_task_clusters_per_pair']} "
                "task "
                "clusters versus frozen floor 10; old K16 simulation is non-transferable"
            ),
            "exact_remediation": [
                (
                    "minimum structurally aligned route: restore the v5 K16 160-scored-task, "
                    "3,200-arena layout after the final two-lane roster is identity-eligible; "
                    "run the exact production gate and expand further if 50-Elo power fails"
                ),
                (
                    "for global K14 pair intervals, select at least 910 arena comparisons with "
                    "every one of 91 pairs in at least 10 distinct tasks, then validate the exact "
                    "K14 production BT plus crossed task/rater bootstrap"
                ),
                (
                    "to retain frozen-v1-like family ranks, expand to at least 40 tasks/family "
                    "and 100 comparisons/model/family (at least 700/family, 2,800 total), with "
                    "all pair-support and bootstrap-connectivity floors verified"
                ),
                (
                    "otherwise freeze a narrower non-ranking global merit estimand in a new "
                    "acceptance-policy version and suppress all pair-specific intervals"
                ),
            ],
        },
        "per_model_epicure_uplift": {
            "decision": "NO_GO_CONFIRMATORY_PER_MODEL_UPLIFT",
            "observed_comparisons_per_model": {
                "minimum": min(uplift_appearances.values()),
                "maximum": max(uplift_appearances.values()),
                "minimum_per_family": min(family_uplift_appearances.values()),
                "maximum_per_family": max(family_uplift_appearances.values()),
            },
            "exact_remediation": (
                "minimum structurally aligned route: restore the full K16 3,200-uplift layout, "
                "which supplies 200 "
                "comparisons/model and 50/model/family; K14 would require 2,800 balanced "
                "comparisons, while narrowing to overall uplift is an explicitly secondary option"
            ),
        },
        "overall_epicure_uplift": {
            "decision": "NO_GO_UNTIL_ALL_CORE_MISSINGNESS_AND_RATER_SCENARIOS_PASS",
            "scope": (
                "the 80-task aggregate is the narrowest plausible confirmatory estimand; any "
                "passing complete-case screen does not validate outcome-dependent missingness"
            ),
            "exact_remediation": (
                "freeze response/rating missingness bounds or a justified weighting model, bind "
                "actual rater assignment, and rerun all core calibration thresholds; add tasks "
                "if 0.08 half-win power or precision remains below threshold"
            ),
        },
        "family_effects": {
            "decision": "NO_GO_CONFIRMATORY_FAMILY_EFFECTS_UNLESS_FAMILY_THRESHOLDS_PASS",
            "reason": "only 20 task clusters per family; ratings cannot substitute for tasks",
            "exact_remediation": (
                "add independent tasks per family according to the saved power curve or narrow "
                "family effects to exploratory heterogeneity checks"
            ),
        },
        "reliability": {
            "decision": "OVERALL_SCREEN_ONLY; NO_PER_RATER_CLAIM",
            "exact_remediation": (
                "bind a rater-assignment roster before collection and add repeats per actual "
                "rater if per-rater reliability is required; add repeats/tasks if overall or "
                "track half-width thresholds fail"
            ),
        },
        "exploratory_contrasts": {
            "decision": "EXPLORATORY_ONLY_NOT_ACTIVATION_OR_PAPER_CLAIMS",
            "multiplicity": "label and report all contrasts; no cherry-picked rank declaration",
        },
    }


def _claim_boundary() -> dict[str, Any]:
    return {
        "activation_authorized": False,
        "provider_calls_authorized": False,
        "network_calls_authorized": False,
        "model_calls_authorized": False,
        "epicure_calls_authorized": False,
        "database_calls_authorized": False,
        "deployment_authorized": False,
        "human_contact_authorized": False,
        "human_judgment_collection_authorized": False,
        "compensation_or_spend_authorized": False,
        "official": False,
        "rank_eligible": False,
        "rank_authorized": False,
        "paper_or_public_claim_authorized": False,
        "observed_judgments": 0,
        "fabricated_judgments": 0,
        "human_judgments": 0,
        "reviewer_identities_assigned": 0,
        "quality_observations": 0,
        "simulated_draws_persisted_as_judgments": False,
        "simulation_is_model_quality_evidence": False,
        "research_result": False,
    }


def build_validation_artifact(
    *,
    datasets: int = DEFAULT_DATASETS,
    seed: int = DEFAULT_SEED,
    sampling_v2_path: Path = DEFAULT_SAMPLING_V2,
    design_path: Path = DEFAULT_DESIGN,
    legacy_v5_design_path: Path = DEFAULT_LEGACY_V5_DESIGN,
    k16_alternative_path: Path = DEFAULT_K16_ALTERNATIVE,
    arena_policy_path: Path = DEFAULT_ARENA_POLICY,
    old_arena_mc_path: Path = DEFAULT_OLD_ARENA_MC,
    production_statistics_path: Path = DEFAULT_PRODUCTION_STATISTICS,
    old_arena_mc_engine_path: Path = DEFAULT_OLD_ARENA_MC_ENGINE,
    validation_engine_path: Path = DEFAULT_VALIDATION_ENGINE,
    superseded_artifact_path: Path = DEFAULT_SUPERSEDED_ARTIFACT,
) -> dict[str, Any]:
    """Build the deterministic exact-frame validation from bound local bytes."""

    _require(datasets >= 20, "at least 20 deterministic datasets are required")
    frame, policy, _old_mc, legacy_v5, k16_alternative = _load_sources(
        sampling_v2_path=sampling_v2_path,
        design_path=design_path,
        legacy_v5_design_path=legacy_v5_design_path,
        k16_alternative_path=k16_alternative_path,
        arena_policy_path=arena_policy_path,
        old_arena_mc_path=old_arena_mc_path,
        production_statistics_path=production_statistics_path,
        old_arena_mc_engine_path=old_arena_mc_engine_path,
        validation_engine_path=validation_engine_path,
        superseded_artifact_path=superseded_artifact_path,
    )
    validation = run_validation(
        frame,
        scenarios=DEFAULT_SCENARIOS,
        datasets=datasets,
        seed=seed,
    )
    checks = _candidate_checks(validation, datasets)
    old_contract = _old_contract_reconciliation(frame, validation, policy)
    conclusions = _conclusions(frame, validation, checks)
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "artifact_role": "offline_exact_frame_statistical_design_validation_successor",
        "supersession": {
            "supersedes_semantic_sha256": SUPERSEDED_ARTIFACT_SEMANTIC_SHA256,
            "supersedes_physical_sha256": SUPERSEDED_ARTIFACT_PHYSICAL_SHA256,
            "supersedes_reference_path": SUPERSEDED_ARTIFACT_REFERENCE,
            "reason": (
                "correct per-model task-grain planning variance, use production-equivalent "
                "hierarchical arena point weights under within-task missingness, and replace "
                "boundary-degenerate Wald Monte Carlo acceptance bounds with exact binomial bounds"
            ),
            "prior_no_go_decision_preserved": True,
        },
        "source_commitments": _source_commitments(),
        "physical_digest_reconciliation": {
            "requested_literal": REQUESTED_TRUNCATED_PHYSICAL_LITERAL,
            "requested_literal_hex_length": len(REQUESTED_TRUNCATED_PHYSICAL_LITERAL),
            "is_valid_sha256": False,
            "verified_complete_physical_sha256": SAMPLING_V2_PHYSICAL_SHA256,
            "normalization_or_silent_acceptance": False,
            "disposition": "missing final nibble f restored only from verified source bytes",
        },
        "prespecified_estimands_and_inference": _estimands(),
        "reduced_vs_full_benchmark_comparison": _full_benchmark_comparison(
            legacy_v5, k16_alternative
        ),
        "simulation_contract": {
            "datasets_per_scenario": datasets,
            "seed": seed,
            "scenario_ids": [scenario.scenario_id for scenario in DEFAULT_SCENARIOS],
            "scenario_set_fixed_in_bound_engine_before_artifact_build": True,
            "external_preregistration_claimed": False,
            "deterministic_seed_streams": True,
            "monte_carlo_rate_mcse": "sqrt(p*(1-p)/datasets)",
            "monte_carlo_acceptance_bounds": (
                "two-sided 95% exact Clopper-Pearson interval; no boundary-degenerate "
                "Wald acceptance bound"
            ),
            "trinary_ties": True,
            "two_ratings_per_comparison": True,
            "task_cluster_effects": True,
            "comparison_shared_effects": True,
            "cross_task_rater_effects": True,
            "abstract_rater_pool_sensitivity": [8, 12, 32],
            "family_heterogeneity": True,
            "mcar_responses_and_ratings": True,
            "outcome_dependent_missing_responses_and_ratings": True,
            "outcome_dependent_rater_dropout": True,
            "graph_connectivity_checked": True,
            "repeats_checked_at_exact_rate": 0.125,
            "nested_production_bootstrap_run": False,
            "diagnostic_arena_proxy_only": True,
            "diagnostic_arena_method": (
                "comparison-aggregated fractional-outcome Bradley-Terry point equation with "
                "production-equivalent equal-family/equal-task/equal-battle hierarchical "
                "weights, including within-task missingness; CR1 task sandwich/Bonferroni "
                "intervals remain a non-production proxy instead of the official crossed "
                "task/rater bootstrap"
            ),
            "production_method_validation_claimed": False,
        },
        "old_frozen_arena_v1_reconciliation": old_contract,
        "validation_results": validation,
        "prespecified_candidate_acceptance": {
            "status": "fail" if any(not row["passed"] for row in checks) else "pass",
            "all_core_thresholds_must_pass": True,
            "checks": checks,
            "failed_check_ids": [row["check_id"] for row in checks if not row["passed"]],
            "passing_proxy_cannot_override_frozen_policy_failure": True,
        },
        "conclusions": conclusions,
        "claim_boundary": _claim_boundary(),
    }
    return {**body, "artifact_sha256": sha256_json(body)}


def verify_validation_artifact(
    document: Mapping[str, Any],
    *,
    datasets: int | None = None,
    seed: int | None = None,
) -> None:
    """Rebuild and require exact equality, including all false boundaries."""

    _require(isinstance(document, Mapping), "validation artifact must be an object")
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(
        document.get("artifact_sha256") == sha256_json(body),
        "validation semantic digest mismatch",
    )
    parameters = document.get("simulation_contract")
    _require(isinstance(parameters, Mapping), "simulation contract is absent")
    expected = build_validation_artifact(
        datasets=int(datasets if datasets is not None else parameters["datasets_per_scenario"]),
        seed=int(seed if seed is not None else parameters["seed"]),
    )
    _require(document == expected, "validation artifact differs from exact deterministic build")
    boundary = document.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "claim boundary is absent")
    boolean_values = [value for value in boundary.values() if isinstance(value, bool)]
    _require(boolean_values and not any(boolean_values), "a claim-boundary flag became true")
    _require(document["conclusions"]["power_validated"] is False, "power was declared valid")
    _require(
        document["prespecified_candidate_acceptance"]["status"] == "fail",
        "NO-GO artifact unexpectedly passed",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_validation_artifact(document: Mapping[str, Any], output_dir: Path) -> Path:
    """Publish with explicit O_EXCL temporary creation and no-replace hard link."""

    verify_validation_artifact(document)
    _require(not output_dir.is_symlink(), "output directory may not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(output_dir.is_dir() and not output_dir.is_symlink(), "invalid output directory")
    rendered = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = str(document["artifact_sha256"])
    destination = output_dir / f"sampling-power-validation-v1-candidate-{digest}.json"
    if destination.exists() or destination.is_symlink():
        _require(_read_regular_bytes(destination) == rendered, "existing final artifact conflicts")
        return destination

    temporary: Path | None = None
    descriptor: int | None = None
    for nonce in range(1000):
        candidate = output_dir / f".sampling-power-v1-{os.getpid()}-{nonce}"
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o644,
            )
            temporary = candidate
            break
        except FileExistsError:
            continue
    _require(descriptor is not None and temporary is not None, "cannot reserve O_EXCL temp file")
    try:
        offset = 0
        while offset < len(rendered):
            offset += os.write(descriptor, rendered[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise SamplingPowerValidationError(
                "final artifact appeared during no-replace hard-link publication"
            ) from error
        _fsync_directory(output_dir)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-candidate", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_validation_artifact(datasets=args.datasets, seed=args.seed)
    if args.write_candidate:
        print(write_validation_artifact(document, args.output_dir))
    else:
        print(document["artifact_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
