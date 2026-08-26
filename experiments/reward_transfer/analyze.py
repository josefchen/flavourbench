#!/usr/bin/env python3
"""Run the prespecified inference for the sealed Epicure transfer evaluations."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from flavourbench.lab import PRIMARY_FAMILIES
from flavourbench.reward_transfer import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_RESULTS,
    TRAINED_CONDITIONS,
    RewardTransferError,
    crossed_seed_anchor_bootstrap,
    evaluation_rng_seed,
    load_plan,
    load_verified_evaluation,
    matched_anchor_sign_flip,
    semantic_sha256,
    stratified_point_estimate,
    summarize_run,
    write_json,
)


def _aligned_values(
    rows: list[dict[str, Any]], tasks: list[dict[str, Any]], field: str
) -> np.ndarray:
    by_id = {str(row["task_id"]): row for row in rows}
    return np.asarray([float(by_id[str(task["task_id"])][field]) for task in tasks])


def _matrix(
    rows_by_run: dict[tuple[str, int | None], list[dict[str, Any]]],
    tasks: list[dict[str, Any]],
    condition: str,
    seeds: list[int],
    field: str = "score",
) -> np.ndarray:
    return np.stack(
        [_aligned_values(rows_by_run[(condition, seed)], tasks, field) for seed in seeds]
    )


def _condition_summary(
    *,
    condition: str,
    seeds: list[int | None],
    rows_by_run: dict[tuple[str, int | None], list[dict[str, Any]]],
) -> dict[str, Any]:
    runs = []
    for seed in seeds:
        summary = summarize_run(rows_by_run[(condition, seed)])
        runs.append({"training_seed": seed, **summary})
    score_values = [float(run["score_unconditional"]) for run in runs]
    parse_values = [float(run["parse_rate"]) for run in runs]
    optimum_values = [float(run["exact_optimum_rate"]) for run in runs]
    conditional_values = [
        float(run["score_conditional_on_parse"])
        for run in runs
        if run["score_conditional_on_parse"] is not None
    ]
    per_family = []
    for family in PRIMARY_FAMILIES:
        family_runs = [
            next(row for row in run["per_family"] if row["family"] == family) for run in runs
        ]
        family_conditional = [
            float(row["score_conditional_on_parse"])
            for row in family_runs
            if row["score_conditional_on_parse"] is not None
        ]
        per_family.append(
            {
                "family": family,
                "score_unconditional_mean": mean(
                    float(row["score_unconditional"]) for row in family_runs
                ),
                "parse_rate_mean": mean(float(row["parse_rate"]) for row in family_runs),
                "score_conditional_on_parse_mean": (
                    mean(family_conditional) if family_conditional else None
                ),
                "exact_optimum_rate_mean": mean(
                    float(row["exact_optimum_rate"]) for row in family_runs
                ),
            }
        )
    return {
        "condition": condition,
        "replicate_runs": len(runs),
        "score_unconditional_mean": mean(score_values),
        "score_unconditional_seed_range": [min(score_values), max(score_values)],
        "score_unconditional_seed_sd": stdev(score_values) if len(score_values) > 1 else None,
        "parse_rate_mean": mean(parse_values),
        "parse_rate_seed_range": [min(parse_values), max(parse_values)],
        "score_conditional_on_parse_mean": (
            mean(conditional_values) if conditional_values else None
        ),
        "exact_optimum_rate_mean": mean(optimum_values),
        "per_family": per_family,
        "runs": runs,
    }


def _contrast(
    *,
    label: str,
    differences: np.ndarray,
    families: list[str],
    panels: list[str],
    protocol_hash: str,
    bootstrap_resamples: int,
    sign_flip_resamples: int,
    practical_threshold: float | None,
) -> dict[str, Any]:
    bootstrap_seed = evaluation_rng_seed(protocol_hash, f"{label}:crossed-bootstrap")
    sign_flip_seed = evaluation_rng_seed(protocol_hash, f"{label}:anchor-sign-flip")
    point, interval = crossed_seed_anchor_bootstrap(
        differences,
        families,
        panels,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    tested_point, p_value = matched_anchor_sign_flip(
        differences,
        families,
        panels,
        resamples=sign_flip_resamples,
        seed=sign_flip_seed,
    )
    if abs(point - tested_point) > 1e-12:
        raise RewardTransferError("bootstrap and sign-flip point estimates differ")
    anchor_differences = differences.mean(axis=0)
    result: dict[str, Any] = {
        "label": label,
        "estimate_points": point,
        "confidence_interval_95": interval,
        "bootstrap": {
            "method": "crossed matched-seed and within-family-by-panel anchor bootstrap",
            "resamples": bootstrap_resamples,
            "rng_seed": bootstrap_seed,
        },
        "two_sided_sign_flip_p": p_value,
        "sign_flip": {
            "unit": "anchor difference averaged across matched training seeds",
            "resamples": sign_flip_resamples,
            "rng_seed": sign_flip_seed,
            "monte_carlo_plus_one_correction": True,
        },
        "anchor_diagnostics": {
            "anchors": int(differences.shape[1]),
            "mean_difference_points": float(anchor_differences.mean()),
            "median_difference_points": float(np.median(anchor_differences)),
            "sd_difference_points": float(anchor_differences.std(ddof=1)),
            "positive_fraction": float(np.mean(anchor_differences > 0)),
            "zero_fraction": float(np.mean(anchor_differences == 0)),
        },
        "training_seed_estimates_points": [
            stratified_point_estimate(row, families, panels) for row in differences
        ],
    }
    if practical_threshold is not None:
        detected = interval[0] > 0 and p_value < 0.05
        threshold_met = point >= practical_threshold
        if detected and threshold_met:
            interpretation = "positive_and_meets_preregistered_practical_threshold"
        elif detected:
            interpretation = "positive_but_below_preregistered_practical_threshold"
        elif interval[1] < 0 and p_value < 0.05:
            interpretation = "negative_difference_detected"
        else:
            interpretation = "inconclusive_at_preregistered_alpha"
        result["alpha"] = 0.05
        result["practical_gain_threshold_points"] = practical_threshold
        result["statistically_detected_positive"] = detected
        result["point_estimate_meets_practical_threshold"] = threshold_met
        result["interpretation"] = interpretation
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("primary", "public"), required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--sign-flip-resamples", type=int)
    args = parser.parse_args()
    plan = load_plan()
    frozen_bootstrap = 50_000
    frozen_sign_flip = 100_000
    if args.bootstrap_resamples is not None and args.bootstrap_resamples != frozen_bootstrap:
        raise RewardTransferError("confirmatory bootstrap count is frozen at 50,000")
    if args.sign_flip_resamples is not None and args.sign_flip_resamples != frozen_sign_flip:
        raise RewardTransferError("confirmatory sign-flip count is frozen at 100,000")

    directory = args.results / args.split
    master, tasks, rows_by_run = load_verified_evaluation(
        directory,
        split=args.split,
        gate_path=args.results / "evaluation-gate.json",
        checkpoints=args.checkpoints,
    )
    output = directory / "analysis.json"
    if output.exists():
        raise RewardTransferError(f"analysis is already sealed: {output}")
    seeds = [int(seed) for seed in plan["seeds"]]
    families = [str(task["family"]) for task in tasks]
    panels = [str(task.get("source_panel") or task.get("release_panel")) for task in tasks]
    control = _matrix(rows_by_run, tasks, "sft_format_control", seeds)
    treatment = _matrix(rows_by_run, tasks, "sft_epicure_optimum", seeds)
    base = _aligned_values(rows_by_run[("pretrained_base", None)], tasks, "score")
    primary_label = (
        "primary:sft_epicure_optimum_minus_sft_format_control"
        if args.split == "primary"
        else "public_replication:sft_epicure_optimum_minus_sft_format_control"
    )
    primary_contrast = _contrast(
        label=primary_label,
        differences=treatment - control,
        families=families,
        panels=panels,
        protocol_hash=plan["artifact_sha256"],
        bootstrap_resamples=frozen_bootstrap,
        sign_flip_resamples=frozen_sign_flip,
        practical_threshold=float(plan["inference"]["practical_gain_threshold_points"]),
    )
    treatment_minus_base = _contrast(
        label=f"{args.split}:secondary:sft_epicure_optimum_minus_pretrained_base",
        differences=treatment - base[None, :],
        families=families,
        panels=panels,
        protocol_hash=plan["artifact_sha256"],
        bootstrap_resamples=frozen_bootstrap,
        sign_flip_resamples=frozen_sign_flip,
        practical_threshold=None,
    )
    control_minus_base = _contrast(
        label=f"{args.split}:secondary:sft_format_control_minus_pretrained_base",
        differences=control - base[None, :],
        families=families,
        panels=panels,
        protocol_hash=plan["artifact_sha256"],
        bootstrap_resamples=frozen_bootstrap,
        sign_flip_resamples=frozen_sign_flip,
        practical_threshold=None,
    )
    per_task = []
    for index, task in enumerate(tasks):
        per_task.append(
            {
                "task_id": task["task_id"],
                "anchor_ingredient": task["anchor_ingredient"],
                "family": task["family"],
                "panel": panels[index],
                "pretrained_base_score": float(base[index]),
                "format_control_seed_mean_score": float(control[:, index].mean()),
                "epicure_optimum_seed_mean_score": float(treatment[:, index].mean()),
                "treatment_minus_control_points": float(
                    (treatment[:, index] - control[:, index]).mean()
                ),
            }
        )

    analysis: dict[str, Any] = {
        "schema_version": "flavourbench-reward-transfer-analysis-v1",
        "status": (
            "primary_analysis_complete_before_public_replication"
            if args.split == "primary"
            else "public_replication_analysis_complete"
        ),
        "split": args.split,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_artifact_sha256": plan["artifact_sha256"],
        "evaluation_gate_artifact_sha256": master["evaluation_gate_artifact_sha256"],
        "evaluation_manifest_artifact_sha256": master["artifact_sha256"],
        "primary_analysis_artifact_sha256": master.get("primary_analysis_artifact_sha256"),
        "tasks": len(tasks),
        "training_seeds": seeds,
        "estimand": (
            "equal-family, equal-source-panel mean score difference with unparseable completions "
            "retained at zero"
        ),
        "condition_summaries": [
            _condition_summary(
                condition="pretrained_base",
                seeds=[None],
                rows_by_run=rows_by_run,
            ),
            *[
                _condition_summary(
                    condition=condition,
                    seeds=seeds,
                    rows_by_run=rows_by_run,
                )
                for condition in TRAINED_CONDITIONS
            ],
        ],
        "confirmatory_contrast" if args.split == "primary" else "replication_contrast": (
            primary_contrast
        ),
        "secondary_contrasts": [treatment_minus_base, control_minus_base],
        "multiplicity": (
            "One preregistered confirmatory contrast; no multiplicity adjustment. Base comparisons "
            "and the public replication are secondary."
        ),
        "rng_policy": (
            "All analysis seeds are SHA-256 derivations of the prospective protocol hash and a "
            "fixed analysis label; they are independent of observed outcomes."
        ),
        "chance_baseline_score": stratified_point_estimate(
            [float(task["chance_score_bps"]) / 100.0 for task in tasks],
            families,
            panels,
        ),
        "per_task": per_task,
        "claim_boundary": plan["claim_boundary"],
    }
    analysis["artifact_sha256"] = semantic_sha256(analysis)
    write_json(output, analysis)
    print(f"{output} {analysis['artifact_sha256']}")


if __name__ == "__main__":
    main()
