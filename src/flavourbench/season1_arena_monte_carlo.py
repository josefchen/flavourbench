from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .season1_arena_acceptance import (
    ARENA_INFERENCE_POLICY_SHA256,
    FAMILIES,
    canonical_sha256,
    evaluate_arena_inference_acceptance,
    load_arena_inference_policy,
)
from .season1_statistics import (
    ArenaObservation,
    _arena_rank_point,
    _cluster_multipliers,
    _hierarchical_weights,
    _numpy_bt_refit,
    _stratified_task_multipliers,
    full_roster_components,
)

SCHEMA_VERSION = "flavourbench-season1-arena-production-monte-carlo-v1"
RESULT_SCHEMA_VERSION = "flavourbench-season1-arena-production-monte-carlo-result-v1"
SOURCE_BUNDLE_SCHEMA_VERSION = "flavourbench-season1-arena-engine-source-bundle-v1"
DISTRIBUTED_RECEIPT_SCHEMA_VERSION = (
    "flavourbench-season1-arena-distributed-receipts-v1"
)
MODEL_COUNT = 16
TASKS_PER_FAMILY = 40
BATTLES_PER_TASK = 20
RATERS_PER_COMPARISON = 2
RATER_COUNT = 64
SCENARIOS = (
    "null_tie_rate_0.10",
    "null_tie_rate_0.40",
    "single_model_50_elo_shift",
    "family_crossover_zero_global_effect",
    "task_icc_0.05_and_0.25_with_observed_response_reuse",
    "crossed_rater_severity_and_side_bias",
    "observed_endpoint_family_missingness_and_near_disconnection",
    "both_bad_0_and_0.20_with_10_percent_anomalous_or_repeat_raters",
)


class MonteCarloContractError(ValueError):
    """A shard or aggregate violates the frozen simulation contract."""


def engine_source_bundle() -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    paths = {
        "src/flavourbench/season1_arena_acceptance.py": package
        / "season1_arena_acceptance.py",
        "src/flavourbench/season1_statistics.py": package / "season1_statistics.py",
        "src/flavourbench/season1_arena_monte_carlo.py": package
        / "season1_arena_monte_carlo.py",
    }
    body = {
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "policy_sha256": ARENA_INFERENCE_POLICY_SHA256,
        "source_file_sha256s": {
            relative: hashlib.sha256(path.read_bytes()).hexdigest()
            for relative, path in paths.items()
        },
    }
    return {**body, "artifact_sha256": canonical_sha256(body)}


def _round_robin_pairs(models: Sequence[str]) -> list[tuple[str, str]]:
    rotation = list(models)
    pairs: list[tuple[str, str]] = []
    for _ in range(len(models) - 1):
        pairs.extend(
            (rotation[index], rotation[-index - 1])
            for index in range(len(models) // 2)
        )
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return pairs


def build_production_layout() -> dict[str, Any]:
    models = [f"model-{index:02d}" for index in range(MODEL_COUNT)]
    round_robin = _round_robin_pairs(models)
    battles: list[dict[str, Any]] = []
    tasks: dict[str, str] = {}
    per_family_pair_cursor = 0
    for family in FAMILIES:
        for task_index in range(TASKS_PER_FAMILY):
            task_id = f"{family}-task-{task_index:02d}"
            tasks[task_id] = family
            for local_index in range(BATTLES_PER_TASK):
                first, second = round_robin[
                    (per_family_pair_cursor + local_index) % len(round_robin)
                ]
                battle_ordinal = len(battles)
                battles.append(
                    {
                        "battle_id": f"battle-{battle_ordinal:04d}",
                        "task_id": task_id,
                        "family": family,
                        "model_a": first,
                        "model_b": second,
                        "rater_ids": [
                            f"rater-{(battle_ordinal * 2) % RATER_COUNT:02d}",
                            f"rater-{(battle_ordinal * 2 + 1) % RATER_COUNT:02d}",
                        ],
                    }
                )
            per_family_pair_cursor += BATTLES_PER_TASK

    appearances = Counter(
        model_id
        for battle in battles
        for model_id in (battle["model_a"], battle["model_b"])
    )
    family_appearances = Counter(
        (battle["family"], model_id)
        for battle in battles
        for model_id in (battle["model_a"], battle["model_b"])
    )
    model_family_tasks: dict[tuple[str, str], set[str]] = {
        (model_id, family): set() for model_id in models for family in FAMILIES
    }
    for battle in battles:
        for model_id in (battle["model_a"], battle["model_b"]):
            model_family_tasks[(model_id, battle["family"])].add(battle["task_id"])
    if (
        len(battles) != 3_200
        or set(appearances.values()) != {400}
        or set(family_appearances.values()) != {100}
        or min(map(len, model_family_tasks.values())) < 20
    ):
        raise MonteCarloContractError("deterministic production layout is unbalanced")
    payload = {
        "schema_version": "flavourbench-season1-arena-production-layout-v1",
        "model_ids": models,
        "tasks": tasks,
        "battles": battles,
        "counts": {
            "models": len(models),
            "families": len(FAMILIES),
            "admitted_scored_tasks": len(tasks),
            "tasks_per_family": TASKS_PER_FAMILY,
            "battles": len(battles),
            "endpoint_appearances": sum(appearances.values()),
            "endpoint_appearances_per_model": min(appearances.values()),
            "endpoint_appearances_per_model_family": min(family_appearances.values()),
            "raters_per_comparison": RATERS_PER_COMPARISON,
        },
    }
    return {**payload, "artifact_sha256": canonical_sha256(payload)}


def _dataset_seed(scenario: str, dataset_index: int) -> int:
    digest = hashlib.sha256(
        f"{ARENA_INFERENCE_POLICY_SHA256}:{scenario}:{dataset_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _scenario_settings(scenario: str, dataset_index: int) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise MonteCarloContractError(f"unknown frozen scenario: {scenario}")
    settings: dict[str, Any] = {
        "tie_rate": 0.10,
        "shift_elo": 0.0,
        "task_icc": 0.05,
        "rater_side_bias_sd_elo": 0.0,
        "both_bad_rate": 0.0,
        "anomalous_rater_fraction": 0.0,
        "family_crossover": False,
        "sparse_missingness": False,
    }
    if scenario == "null_tie_rate_0.40":
        settings["tie_rate"] = 0.40
    elif scenario == "single_model_50_elo_shift":
        settings["shift_elo"] = 50.0
    elif scenario == "family_crossover_zero_global_effect":
        settings["family_crossover"] = True
    elif scenario == "task_icc_0.05_and_0.25_with_observed_response_reuse":
        settings["task_icc"] = 0.05 if dataset_index % 2 == 0 else 0.25
    elif scenario == "crossed_rater_severity_and_side_bias":
        settings["rater_side_bias_sd_elo"] = 35.0
    elif scenario == "observed_endpoint_family_missingness_and_near_disconnection":
        settings["sparse_missingness"] = True
    elif scenario == "both_bad_0_and_0.20_with_10_percent_anomalous_or_repeat_raters":
        settings["both_bad_rate"] = 0.0 if dataset_index % 2 == 0 else 0.20
        settings["anomalous_rater_fraction"] = 0.10
    return settings


def _simulate_observations(
    layout: Mapping[str, Any],
    *,
    scenario: str,
    dataset_index: int,
) -> tuple[list[ArenaObservation], dict[str, float]]:
    rng = np.random.default_rng(_dataset_seed(scenario, dataset_index))
    settings = _scenario_settings(scenario, dataset_index)
    models = list(layout["model_ids"])
    model_elo = {model_id: 0.0 for model_id in models}
    model_elo[models[0]] = float(settings["shift_elo"])
    family_delta = {
        "composition": 50.0,
        "cookability": -50.0,
        "evidence": 50.0,
        "substitution": -50.0,
    }
    rater_bias = {
        f"rater-{index:02d}": float(
            rng.normal(0.0, float(settings["rater_side_bias_sd_elo"]))
        )
        for index in range(RATER_COUNT)
    }
    anomalous_count = round(RATER_COUNT * float(settings["anomalous_rater_fraction"]))
    anomalous_raters = {
        f"rater-{index:02d}" for index in range(int(anomalous_count))
    }
    task_icc = float(settings["task_icc"])
    logistic_variance = math.pi**2 / 3.0
    task_sd_logit = math.sqrt(task_icc / (1.0 - task_icc) * logistic_variance)
    task_effect = {
        task_id: float(rng.normal(0.0, task_sd_logit))
        for task_id in layout["tasks"]
    }
    observations: list[ArenaObservation] = []
    for battle in layout["battles"]:
        if settings["sparse_missingness"]:
            sparse_model = models[-1]
            if battle["family"] in {"cookability", "evidence", "substitution"} and sparse_model in {
                battle["model_a"],
                battle["model_b"],
            }:
                continue
            if (
                battle["family"] == "composition"
                and sparse_model in {battle["model_a"], battle["model_b"]}
                and battle["task_id"] != "composition-task-00"
            ):
                continue
        first = str(battle["model_a"])
        second = str(battle["model_b"])
        family = str(battle["family"])
        base_delta = model_elo[first] - model_elo[second]
        if settings["family_crossover"]:
            base_delta += family_delta[family] * (
                int(first == models[0]) - int(second == models[0])
            )
        for rater_id in battle["rater_ids"]:
            if rng.random() < float(settings["both_bad_rate"]):
                continue
            logit = (
                base_delta * math.log(10.0) / 400.0
                + task_effect[str(battle["task_id"])]
                + rater_bias[str(rater_id)] * math.log(10.0) / 400.0
            )
            first_win = rng.random() < 1.0 / (1.0 + math.exp(-logit))
            if str(rater_id) in anomalous_raters:
                first_win = not first_win
            outcome = (
                0.5
                if rng.random() < float(settings["tie_rate"])
                else 1.0
                if first_win
                else 0.0
            )
            observations.append(
                ArenaObservation(
                    observation_id=(
                        f"mc-{scenario}-{dataset_index}-{battle['battle_id']}-{rater_id}"
                    ),
                    task_id=str(battle["task_id"]),
                    family=family,
                    battle_id=str(battle["battle_id"]),
                    rater_id=str(rater_id),
                    model_a=first,
                    model_b=second,
                    response_a_id=f"{battle['task_id']}-{first}",
                    response_b_id=f"{battle['task_id']}-{second}",
                    outcome=outcome,
                )
            )
    # The crossover is balanced across the four equally weighted families, so
    # its prespecified global truth remains zero.
    return observations, model_elo


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), (0.025, 0.975), method="linear")
    return float(low), float(high)


def _fit_dataset(
    observations: Sequence[ArenaObservation],
    roster: Sequence[str],
    *,
    bootstrap_replicates: int,
    seed: int,
    duplication_check: bool,
) -> dict[str, Any]:
    task_family = {row.task_id: row.family for row in observations}
    base_weights = _hierarchical_weights(observations)
    point = _arena_rank_point(observations, roster, base_weights)
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    rating_samples: dict[str, list[float]] = {model_id: [] for model_id in roster}
    duplicate_samples: list[float] = []
    global_connected = 0
    family_connected: Counter[str] = Counter()
    duplicated = [
        *observations,
        *[
            ArenaObservation(
                **{
                    **asdict(row),
                    "observation_id": f"exact-duplicate-{row.observation_id}",
                }
            )
            for row in observations
        ],
    ]
    duplicate_base_weights = _hierarchical_weights(duplicated) if duplication_check else None
    for _ in range(bootstrap_replicates):
        task_multipliers = _stratified_task_multipliers(rng, task_family)
        rater_multipliers = _cluster_multipliers(
            rng, [row.rater_id for row in observations]
        )
        weights = np.asarray(
            [
                base_weights[index]
                * task_multipliers[row.task_id]
                * rater_multipliers[row.rater_id]
                for index, row in enumerate(observations)
            ]
        )
        active_edges = [
            (row.model_a, row.model_b)
            for index, row in enumerate(observations)
            if weights[index] > 0
        ]
        for family in FAMILIES:
            family_edges = [
                (row.model_a, row.model_b)
                for index, row in enumerate(observations)
                if row.family == family and weights[index] > 0
            ]
            if len(full_roster_components(roster, family_edges)) == 1:
                family_connected[family] += 1
        if len(full_roster_components(roster, active_edges)) != 1:
            continue
        global_connected += 1
        fitted = _numpy_bt_refit(observations, roster, weights)
        for model_id in roster:
            rating_samples[model_id].append(fitted[model_id])
        if duplication_check and duplicate_base_weights is not None:
            duplicate_weights = np.asarray(
                [
                    duplicate_base_weights[index]
                    * task_multipliers[row.task_id]
                    * rater_multipliers[row.rater_id]
                    for index, row in enumerate(duplicated)
                ]
            )
            duplicate_fit = _numpy_bt_refit(duplicated, roster, duplicate_weights)
            duplicate_samples.append(
                duplicate_fit[roster[0]] - duplicate_fit[roster[1]]
            )
    if not rating_samples[roster[0]]:
        raise MonteCarloContractError("all bootstrap replicates were disconnected")
    contrast_samples = np.asarray(rating_samples[roster[0]]) - np.asarray(
        rating_samples[roster[1]]
    )
    low, high = _percentile_interval(contrast_samples)
    duplicate_width_delta = None
    if duplication_check:
        duplicate_low, duplicate_high = _percentile_interval(duplicate_samples)
        duplicate_width_delta = (duplicate_high - duplicate_low) - (high - low)
    point_delta = point[roster[0]] - point[roster[1]]
    return {
        "point_elo_difference": round(point_delta, 9),
        "point_ratings": {
            model_id: round(value, 9) for model_id, value in point.items()
        },
        "interval_lower": round(low, 9),
        "interval_upper": round(high, 9),
        "global_bootstrap_connected_rate": round(
            global_connected / bootstrap_replicates, 9
        ),
        "family_bootstrap_connected_rates": {
            family: round(family_connected[family] / bootstrap_replicates, 9)
            for family in FAMILIES
        },
        "successful_bootstrap_replicates": len(contrast_samples),
        "rating_samples": {
            model_id: [round(value, 9) for value in values]
            for model_id, values in rating_samples.items()
        },
        "duplicate_interval_width_delta": (
            round(duplicate_width_delta, 12)
            if duplicate_width_delta is not None
            else None
        ),
    }


def _sparse_bootstrap_connectivity(
    observations: Sequence[ArenaObservation],
    roster: Sequence[str],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    task_family = {row.task_id: row.family for row in observations}
    base_weights = _hierarchical_weights(observations)
    rng = np.random.default_rng(seed ^ 0x5A5A5A5A)
    global_connected = 0
    family_connected: Counter[str] = Counter()
    for _ in range(bootstrap_replicates):
        task_multipliers = _stratified_task_multipliers(rng, task_family)
        rater_multipliers = _cluster_multipliers(
            rng, [row.rater_id for row in observations]
        )
        active = [
            row
            for index, row in enumerate(observations)
            if base_weights[index]
            * task_multipliers[row.task_id]
            * rater_multipliers[row.rater_id]
            > 0
        ]
        if len(
            full_roster_components(
                roster, [(row.model_a, row.model_b) for row in active]
            )
        ) == 1:
            global_connected += 1
        for family in FAMILIES:
            family_edges = [
                (row.model_a, row.model_b) for row in active if row.family == family
            ]
            if len(full_roster_components(roster, family_edges)) == 1:
                family_connected[family] += 1
    return {
        "bootstrap_replicates_executed": bootstrap_replicates,
        "global_bootstrap_connected_rate": round(
            global_connected / bootstrap_replicates, 9
        ),
        "family_bootstrap_connected_rates": {
            family: round(family_connected[family] / bootstrap_replicates, 9)
            for family in FAMILIES
        },
    }


def run_dataset(
    *,
    scenario: str,
    dataset_index: int,
    bootstrap_replicates: int,
    production_mode: bool,
) -> dict[str, Any]:
    policy = load_arena_inference_policy()
    source_bundle = engine_source_bundle()
    required_bootstraps = int(policy["simulation_gate"]["bootstrap_replicates"])
    minimum_datasets = int(policy["simulation_gate"]["minimum_datasets_per_scenario"])
    if dataset_index < 0 or dataset_index >= minimum_datasets:
        raise MonteCarloContractError("dataset index is outside the frozen production range")
    if production_mode and bootstrap_replicates != required_bootstraps:
        raise MonteCarloContractError("production shards require exactly 5,000 bootstraps")
    if bootstrap_replicates < 2:
        raise MonteCarloContractError("development shards require at least two bootstraps")
    layout = build_production_layout()
    observations, truth_by_model = _simulate_observations(
        layout,
        scenario=scenario,
        dataset_index=dataset_index,
    )
    if scenario == "observed_endpoint_family_missingness_and_near_disconnection":
        connectivity = _sparse_bootstrap_connectivity(
            observations,
            layout["model_ids"],
            bootstrap_replicates=bootstrap_replicates,
            seed=_dataset_seed(scenario, dataset_index),
        )
        gate = evaluate_arena_inference_acceptance(
            observations,
            layout["model_ids"],
            view="all",
            admitted_tasks=layout["tasks"],
            comparison_raters=None,
            postcollection_item_audit=None,
            policy=policy,
        )
        analysis = {
            **connectivity,
            "ranking_status": gate["withholding_status"],
            "acceptance_deficit_codes": sorted(
                {str(item["code"]) for item in gate["deficits"]}
            ),
            "ratings": None,
            "confidence_intervals": None,
        }
    else:
        fitted = _fit_dataset(
            observations,
            layout["model_ids"],
            bootstrap_replicates=bootstrap_replicates,
            seed=_dataset_seed(scenario, dataset_index),
            duplication_check=dataset_index == 0,
        )
        rating_samples = fitted.pop("rating_samples")
        point_by_model = fitted.pop("point_ratings")
        pairwise_interval_count = 0
        pairwise_coverage_count = 0
        null_pairwise_count = 0
        null_rejection_count = 0
        shift_50_pairwise_count = 0
        shift_50_detection_count = 0
        probability_absolute_bias_sum = 0.0
        for first_index, first in enumerate(layout["model_ids"]):
            for second in layout["model_ids"][first_index + 1 :]:
                samples = np.asarray(rating_samples[first]) - np.asarray(
                    rating_samples[second]
                )
                low, high = _percentile_interval(samples)
                truth_delta = truth_by_model[first] - truth_by_model[second]
                estimate_delta = point_by_model[first] - point_by_model[second]
                pairwise_interval_count += 1
                pairwise_coverage_count += int(low <= truth_delta <= high)
                if truth_delta == 0.0:
                    null_pairwise_count += 1
                    null_rejection_count += int(not (low <= 0.0 <= high))
                if abs(truth_delta) == 50.0:
                    shift_50_pairwise_count += 1
                    shift_50_detection_count += int(
                        low > 0.0 if truth_delta > 0 else high < 0.0
                    )
                truth_probability = 1.0 / (
                    1.0 + 10.0 ** (-truth_delta / 400.0)
                )
                estimate_probability = 1.0 / (
                    1.0 + 10.0 ** (-estimate_delta / 400.0)
                )
                probability_absolute_bias_sum += abs(
                    estimate_probability - truth_probability
                )
        target_truth_delta = truth_by_model[layout["model_ids"][0]] - truth_by_model[
            layout["model_ids"][1]
        ]
        truth_probability = 1.0 / (
            1.0 + 10.0 ** (-target_truth_delta / 400.0)
        )
        estimate_probability = 1.0 / (
            1.0 + 10.0 ** (-float(fitted["point_elo_difference"]) / 400.0)
        )
        analysis = {
            **fitted,
            "bootstrap_replicates_executed": bootstrap_replicates,
            "truth_elo_difference": target_truth_delta,
            "truth_win_probability": round(truth_probability, 9),
            "estimated_win_probability": round(estimate_probability, 9),
            "probability_scale_absolute_bias": round(
                abs(estimate_probability - truth_probability), 9
            ),
            "interval_covers_truth": (
                fitted["interval_lower"]
                <= target_truth_delta
                <= fitted["interval_upper"]
            ),
            "null_rejected": not (
                fitted["interval_lower"] <= 0.0 <= fitted["interval_upper"]
            ),
            "positive_effect_detected": fitted["interval_lower"] > 0.0,
            "pairwise_interval_count": pairwise_interval_count,
            "pairwise_coverage_count": pairwise_coverage_count,
            "null_pairwise_count": null_pairwise_count,
            "null_rejection_count": null_rejection_count,
            "shift_50_pairwise_count": shift_50_pairwise_count,
            "shift_50_detection_count": shift_50_detection_count,
            "probability_absolute_bias_sum": round(
                probability_absolute_bias_sum, 9
            ),
        }
    body = {
        "schema_version": SCHEMA_VERSION,
        "policy_sha256": policy["artifact_sha256"],
        "layout_sha256": layout["artifact_sha256"],
        "engine_source_bundle_sha256": source_bundle["artifact_sha256"],
        "scenario": scenario,
        "dataset_index": dataset_index,
        "dataset_seed": _dataset_seed(scenario, dataset_index),
        "bootstrap_replicates": bootstrap_replicates,
        "engine": "production_equation_exact_task_by_rater_cluster_bootstrap",
        "production_mode": production_mode,
        "status": "completed" if production_mode else "development_only_completed",
        "analysis": analysis,
        "claim_boundary": {
            "counts_toward_production_gate": production_mode,
            "aggregate_acceptance_claimed": False,
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
        },
    }
    return {**body, "record_sha256": canonical_sha256(body)}


def _verified_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MonteCarloContractError("checkpoint contains a non-object record")
            digest = value.get("record_sha256")
            payload = {key: item for key, item in value.items() if key != "record_sha256"}
            if digest != canonical_sha256(payload):
                raise MonteCarloContractError("checkpoint record content address failed")
            records.append(value)
    return records


def aggregate_production_results(paths: Sequence[Path]) -> dict[str, Any]:
    policy = load_arena_inference_policy()
    source_bundle = engine_source_bundle()
    gate = policy["simulation_gate"]
    required_datasets = int(gate["minimum_datasets_per_scenario"])
    required_bootstraps = int(gate["bootstrap_replicates"])
    records = _verified_records(paths)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("scenario")), int(record.get("dataset_index", -1)))
        if key in by_key:
            raise MonteCarloContractError("duplicate scenario/dataset checkpoint record")
        if (
            key[0] not in SCENARIOS
            or record.get("policy_sha256") != policy["artifact_sha256"]
            or record.get("engine_source_bundle_sha256")
            != source_bundle["artifact_sha256"]
            or record.get("bootstrap_replicates") != required_bootstraps
            or record.get("production_mode") is not True
            or record.get("status") != "completed"
            or record.get("claim_boundary", {}).get("counts_toward_production_gate") is not True
            or record.get("analysis", {}).get("bootstrap_replicates_executed")
            != required_bootstraps
        ):
            raise MonteCarloContractError("checkpoint record is not production-gate eligible")
        by_key[key] = record
    complete = all(
        {(scenario, index) for index in range(required_datasets)}.issubset(by_key)
        for scenario in SCENARIOS
    ) and len(by_key) == len(SCENARIOS) * required_datasets
    if not complete:
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "required_not_yet_executed",
            "policy_sha256": policy["artifact_sha256"],
            "engine_source_bundle_sha256": source_bundle["artifact_sha256"],
            "completed_records": len(by_key),
            "required_records": len(SCENARIOS) * required_datasets,
            "acceptance": None,
            "claim_boundary": {
                "production_gate_complete": False,
                "pass_claimed": False,
            },
        }
        return {**payload, "artifact_sha256": canonical_sha256(payload)}

    layout_sha256s = {str(record.get("layout_sha256")) for record in by_key.values()}
    if len(layout_sha256s) != 1 or next(iter(layout_sha256s)) != build_production_layout()[
        "artifact_sha256"
    ]:
        raise MonteCarloContractError("complete result contains multiple or unknown layouts")
    record_set_sha256 = canonical_sha256(
        {
            "record_sha256s": sorted(
                str(record["record_sha256"]) for record in by_key.values()
            )
        }
    )

    inferential = [
        record
        for record in by_key.values()
        if record["scenario"]
        != "observed_endpoint_family_missingness_and_near_disconnection"
    ]
    interval_count = sum(
        int(record["analysis"]["pairwise_interval_count"]) for record in inferential
    )
    coverage_count = sum(
        int(record["analysis"]["pairwise_coverage_count"]) for record in inferential
    )
    null_count = sum(
        int(record["analysis"]["null_pairwise_count"]) for record in inferential
    )
    null_rejections = sum(
        int(record["analysis"]["null_rejection_count"]) for record in inferential
    )
    shift_count = sum(
        int(record["analysis"]["shift_50_pairwise_count"]) for record in inferential
    )
    shift_detections = sum(
        int(record["analysis"]["shift_50_detection_count"]) for record in inferential
    )
    probability_bias_sum = sum(
        float(record["analysis"]["probability_absolute_bias_sum"])
        for record in inferential
    )
    if not all((interval_count, null_count, shift_count)):
        raise MonteCarloContractError("complete result lacks required pairwise estimands")
    coverage = coverage_count / interval_count
    type_i_error = null_rejections / null_count
    power = shift_detections / shift_count
    probability_bias = probability_bias_sum / interval_count
    duplicate_deltas = [
        abs(float(record["analysis"]["duplicate_interval_width_delta"]))
        for record in inferential
        if record["analysis"].get("duplicate_interval_width_delta") is not None
    ]
    sparse_records = [
        record
        for record in by_key.values()
        if record["scenario"]
        == "observed_endpoint_family_missingness_and_near_disconnection"
    ]
    sparse_invalid_records = sum(
        not (
            record["analysis"]["ranking_status"] == policy["withholding_status"]
            and record["analysis"]["ratings"] is None
            and record["analysis"]["confidence_intervals"] is None
        )
        for record in sparse_records
    )
    acceptance_policy = gate["acceptance"]
    checks = {
        "pairwise_difference_coverage": (
            float(acceptance_policy["pairwise_difference_coverage_lower"])
            <= coverage
            <= float(acceptance_policy["pairwise_difference_coverage_upper"])
        ),
        "type_i_error": (
            float(acceptance_policy["type_i_error_lower"])
            <= type_i_error
            <= float(acceptance_policy["type_i_error_upper"])
        ),
        "power_at_50_elo": power >= float(acceptance_policy["minimum_power_at_50_elo"]),
        "probability_scale_absolute_bias": probability_bias
        <= float(acceptance_policy["maximum_probability_scale_absolute_bias"]),
        "no_interval_narrowing_under_exact_row_duplication": bool(duplicate_deltas)
        and max(duplicate_deltas) <= 1e-8,
        "deterministic_sparse_anchor_and_disconnected_withholding": all(
            record["analysis"]["ranking_status"]
            == policy["withholding_status"]
            and record["analysis"]["ratings"] is None
            and record["analysis"]["confidence_intervals"] is None
            for record in sparse_records
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "policy_sha256": policy["artifact_sha256"],
        "engine_source_bundle_sha256": source_bundle["artifact_sha256"],
        "layout_sha256": next(iter(layout_sha256s)),
        "checkpoint_record_set_sha256": record_set_sha256,
        "scenario_dataset_counts": {
            scenario: sum(key[0] == scenario for key in by_key) for scenario in SCENARIOS
        },
        "completed_records": len(by_key),
        "required_records": len(SCENARIOS) * required_datasets,
        "bootstrap_replicates_per_record": required_bootstraps,
        "metrics": {
            "pairwise_difference_coverage": round(coverage, 9),
            "pairwise_intervals_evaluated": interval_count,
            "type_i_error": round(type_i_error, 9),
            "null_pairwise_intervals_evaluated": null_count,
            "power_at_50_elo": round(power, 9),
            "shift_50_pairwise_intervals_evaluated": shift_count,
            "probability_scale_absolute_bias": round(probability_bias, 9),
            "maximum_duplicate_interval_width_delta": round(max(duplicate_deltas), 12),
            "duplication_checks_completed": len(duplicate_deltas),
            "sparse_withheld_records": len(sparse_records),
            "sparse_invalid_records": sparse_invalid_records,
        },
        "acceptance": checks,
        "claim_boundary": {
            "production_gate_complete": True,
            "pass_claimed": status == "pass",
            "model_quality_evidence": False,
        },
    }
    return {**payload, "artifact_sha256": canonical_sha256(payload)}


def bind_distributed_receipts(
    document: Mapping[str, Any],
    *,
    execution_contract_sha256: str,
    execution_manifest_sha256: str,
    shard_result_set_sha256: str,
    shard_count: int,
) -> dict[str, Any]:
    """Bind a complete engine result to independently verified shard receipts."""

    unsealed = dict(document)
    original_digest = unsealed.pop("artifact_sha256", None)
    if (
        original_digest != canonical_sha256(unsealed)
        or unsealed.get("status") not in {"pass", "fail"}
        or "distributed_receipts" in unsealed
        or shard_count != len(SCENARIOS) * int(
            load_arena_inference_policy()["simulation_gate"][
                "minimum_datasets_per_scenario"
            ]
        )
    ):
        raise MonteCarloContractError("only a complete unsealed result may bind receipts")
    for digest in (
        execution_contract_sha256,
        execution_manifest_sha256,
        shard_result_set_sha256,
        str(unsealed.get("checkpoint_record_set_sha256", "")),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise MonteCarloContractError("distributed receipt contains an invalid digest")
    receipt_body = {
        "schema_version": DISTRIBUTED_RECEIPT_SCHEMA_VERSION,
        "execution_contract_sha256": execution_contract_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
        "shard_result_set_sha256": shard_result_set_sha256,
        "shard_count": shard_count,
        "dataset_record_count": int(unsealed["completed_records"]),
        "checkpoint_record_set_sha256": unsealed["checkpoint_record_set_sha256"],
        "unsealed_engine_result_sha256": original_digest,
        "policy_sha256": unsealed["policy_sha256"],
        "layout_sha256": unsealed["layout_sha256"],
        "engine_source_bundle_sha256": unsealed["engine_source_bundle_sha256"],
        "claim_boundary": {
            "synthetic_method_validation_only": True,
            "model_quality_evidence": False,
        },
    }
    receipt = {
        **receipt_body,
        "artifact_sha256": canonical_sha256(receipt_body),
    }
    sealed = {**unsealed, "distributed_receipts": receipt}
    return {**sealed, "artifact_sha256": canonical_sha256(sealed)}


def verify_production_result(document: Mapping[str, Any]) -> bool:
    policy = load_arena_inference_policy()
    source_bundle = engine_source_bundle()
    digest = document.get("artifact_sha256")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    acceptance = document.get("acceptance")
    counts = document.get("scenario_dataset_counts")
    metrics = document.get("metrics")
    receipt = document.get("distributed_receipts")
    thresholds = policy["simulation_gate"]["acceptance"]
    required_datasets = int(policy["simulation_gate"]["minimum_datasets_per_scenario"])
    pair_count = MODEL_COUNT * (MODEL_COUNT - 1) // 2
    if not isinstance(metrics, Mapping) or not isinstance(receipt, Mapping):
        return False
    receipt_payload = {
        key: value for key, value in receipt.items() if key != "artifact_sha256"
    }
    unsealed_payload = {
        key: value
        for key, value in payload.items()
        if key != "distributed_receipts"
    }
    try:
        from .season1_arena_distributed import load_execution_contract

        execution_contract_sha256 = load_execution_contract()["artifact_sha256"]
    except (ImportError, KeyError, OSError, RuntimeError, ValueError):
        return False
    try:
        expected_acceptance = {
            "pairwise_difference_coverage": (
                float(thresholds["pairwise_difference_coverage_lower"])
                <= float(metrics["pairwise_difference_coverage"])
                <= float(thresholds["pairwise_difference_coverage_upper"])
            ),
            "type_i_error": (
                float(thresholds["type_i_error_lower"])
                <= float(metrics["type_i_error"])
                <= float(thresholds["type_i_error_upper"])
            ),
            "power_at_50_elo": float(metrics["power_at_50_elo"])
            >= float(thresholds["minimum_power_at_50_elo"]),
            "probability_scale_absolute_bias": float(
                metrics["probability_scale_absolute_bias"]
            )
            <= float(thresholds["maximum_probability_scale_absolute_bias"]),
            "no_interval_narrowing_under_exact_row_duplication": int(
                metrics["duplication_checks_completed"]
            )
            == 7
            and abs(float(metrics["maximum_duplicate_interval_width_delta"])) <= 1e-8,
            "deterministic_sparse_anchor_and_disconnected_withholding": int(
                metrics["sparse_withheld_records"]
            )
            == int(policy["simulation_gate"]["minimum_datasets_per_scenario"])
            and int(metrics["sparse_invalid_records"]) == 0,
        }
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        digest == canonical_sha256(payload)
        and document.get("schema_version") == RESULT_SCHEMA_VERSION
        and document.get("status") == "pass"
        and document.get("policy_sha256") == policy["artifact_sha256"]
        and document.get("engine_source_bundle_sha256")
        == source_bundle["artifact_sha256"]
        and receipt.get("artifact_sha256") == canonical_sha256(receipt_payload)
        and receipt.get("schema_version") == DISTRIBUTED_RECEIPT_SCHEMA_VERSION
        and receipt.get("execution_contract_sha256")
        == execution_contract_sha256
        and isinstance(receipt.get("execution_manifest_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", str(receipt.get("execution_manifest_sha256"))
        )
        is not None
        and isinstance(receipt.get("shard_result_set_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", str(receipt.get("shard_result_set_sha256"))
        )
        is not None
        and receipt.get("shard_count") == len(SCENARIOS) * required_datasets
        and receipt.get("dataset_record_count")
        == document.get("completed_records")
        and receipt.get("checkpoint_record_set_sha256")
        == document.get("checkpoint_record_set_sha256")
        and receipt.get("unsealed_engine_result_sha256")
        == canonical_sha256(unsealed_payload)
        and receipt.get("policy_sha256") == document.get("policy_sha256")
        and receipt.get("layout_sha256") == document.get("layout_sha256")
        and receipt.get("engine_source_bundle_sha256")
        == document.get("engine_source_bundle_sha256")
        and isinstance(receipt.get("claim_boundary"), Mapping)
        and receipt["claim_boundary"].get("synthetic_method_validation_only") is True
        and receipt["claim_boundary"].get("model_quality_evidence") is False
        and document.get("layout_sha256") == build_production_layout()["artifact_sha256"]
        and isinstance(document.get("checkpoint_record_set_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", str(document.get("checkpoint_record_set_sha256"))
        )
        is not None
        and document.get("completed_records")
        == len(SCENARIOS) * required_datasets
        and document.get("required_records") == document.get("completed_records")
        and document.get("bootstrap_replicates_per_record")
        == int(policy["simulation_gate"]["bootstrap_replicates"])
        and isinstance(counts, Mapping)
        and set(counts) == set(SCENARIOS)
        and all(
            counts.get(scenario)
            == required_datasets
            for scenario in SCENARIOS
        )
        and isinstance(metrics.get("pairwise_intervals_evaluated"), int)
        and isinstance(metrics.get("shift_50_pairwise_intervals_evaluated"), int)
        and isinstance(metrics.get("null_pairwise_intervals_evaluated"), int)
        and metrics.get("pairwise_intervals_evaluated")
        == (len(SCENARIOS) - 1) * required_datasets * pair_count
        and metrics.get("shift_50_pairwise_intervals_evaluated")
        == required_datasets * (MODEL_COUNT - 1)
        and metrics.get("null_pairwise_intervals_evaluated")
        == metrics.get("pairwise_intervals_evaluated")
        - metrics.get("shift_50_pairwise_intervals_evaluated")
        and isinstance(acceptance, Mapping)
        and dict(acceptance) == expected_acceptance
        and all(expected_acceptance.values())
        and isinstance(document.get("claim_boundary"), Mapping)
        and document["claim_boundary"].get("production_gate_complete") is True
        and document["claim_boundary"].get("pass_claimed") is True
        and document["claim_boundary"].get("model_quality_evidence") is False
    )


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    layout_command = commands.add_parser("layout")
    layout_command.add_argument("--output", type=Path, required=True)
    shard = commands.add_parser("run-shard")
    shard.add_argument("--scenario", choices=SCENARIOS, required=True)
    shard.add_argument("--start", type=int, required=True)
    shard.add_argument("--count", type=int, required=True)
    shard.add_argument("--bootstrap-replicates", type=int, default=5_000)
    shard.add_argument("--development-mode", action="store_true")
    shard.add_argument("--checkpoint", type=Path, required=True)
    shard.add_argument("--progress-every", type=int, default=1)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.command == "layout":
        _write_json(arguments.output, build_production_layout())
        return
    if arguments.command == "aggregate":
        _write_json(
            arguments.output,
            aggregate_production_results(arguments.checkpoints),
        )
        return

    completed = {
        int(record["dataset_index"])
        for record in _verified_records([arguments.checkpoint])
    } if arguments.checkpoint.exists() else set()
    for offset in range(arguments.count):
        dataset_index = arguments.start + offset
        if dataset_index in completed:
            continue
        result = run_dataset(
            scenario=arguments.scenario,
            dataset_index=dataset_index,
            bootstrap_replicates=arguments.bootstrap_replicates,
            production_mode=not arguments.development_mode,
        )
        _append_record(arguments.checkpoint, result)
        if (offset + 1) % arguments.progress_every == 0:
            print(
                json.dumps(
                    {
                        "scenario": arguments.scenario,
                        "completed_dataset_index": dataset_index,
                        "checkpoint": str(arguments.checkpoint),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    run()
