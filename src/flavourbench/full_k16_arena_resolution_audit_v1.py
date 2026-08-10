"""Development-only full-K16 arena resolution and remediation audit.

The audit reuses the frozen production Bradley--Terry point estimator and the
crossed task/rater bootstrap.  It creates synthetic outcomes only, writes
append-only content-addressed artifacts, and never counts toward the frozen
production method-validation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .season1_arena_acceptance import ARENA_INFERENCE_POLICY_SHA256, FAMILIES
from .season1_arena_monte_carlo import (
    RATER_COUNT,
    TASKS_PER_FAMILY,
    _dataset_seed,
    _fit_dataset,
    _percentile_interval,
    _round_robin_pairs,
    _simulate_observations,
    build_production_layout,
)
from .season1_statistics import ArenaObservation

SCHEMA_VERSION = "flavourbench-full-k16-arena-resolution-audit-v1-fresh-successor-candidate"
STATUS = "development_only_no_go_not_production_method_validation"
MODEL_COUNT = 16
FOCAL_MODEL = "model-00"
PEER_COUNT = MODEL_COUNT - 1
TARGET_ELO = 50.0
FROZEN_POWER_TARGET = 0.80
CONFIDENCE_LEVEL = 0.95
REPEAT_RATE = 0.125
DEFAULT_WORKERS = 8
DEFAULT_BASELINE_DATASETS = 100
DEFAULT_BASELINE_BOOTSTRAPS = 200
DEFAULT_SCREEN_DATASETS = 40
DEFAULT_SCREEN_BOOTSTRAPS = 100
DEFAULT_CONFIRM_DATASETS = 100
DEFAULT_CONFIRM_BOOTSTRAPS = 200
FRESH_CONFIRMATION_DATASET_START = 140
FRESH_CONFIRMATION_DATASET_END = 239
FRESH_CONFIRMATION_LINEAGE = "generated_fresh_successor_indices_140_239"

FLAVOURBENCH_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PAPER_ROOT = FLAVOURBENCH_ROOT.parent
DEFAULT_OUTPUT_DIR = (
    FLAVOURBENCH_ROOT / "artifacts/season1/full-k16-arena-resolution-audit-v1-candidate"
)

POLICY_REFERENCE = "flavourbench/contracts/season1/season1-arena-inference-acceptance-v1.json"
V5_DESIGN_REFERENCE = "flavourbench/contracts/season1/season1-study-design-v5.json"
PRODUCTION_ENGINE_REFERENCE = "flavourbench/src/flavourbench/season1_arena_monte_carlo.py"
PRODUCTION_STATISTICS_REFERENCE = "flavourbench/src/flavourbench/season1_statistics.py"
AUDIT_ENGINE_REFERENCE = "flavourbench/src/flavourbench/full_k16_arena_resolution_audit_v1.py"
PROVISIONAL_PREDECESSOR_REFERENCE = (
    "flavourbench/artifacts/season1/full-k16-arena-resolution-audit-v1-candidate/"
    "full-k16-arena-resolution-audit-v1-candidate-"
    "b59dfc07280f972b83e00c5699a0f28e3da61135f2da22cfaf4638a4d1391910.json"
)
INVALIDATED_SUCCESSOR_REFERENCE = (
    "flavourbench/artifacts/season1/full-k16-arena-resolution-audit-v1-candidate/"
    "full-k16-arena-resolution-audit-v1-candidate-"
    "64f9f2f8afad51ded9f3be8b84d0c91c0259eae78c5b1086099dbb9f4f26eb1a.json"
)

POLICY_PHYSICAL_SHA256 = "02adfc4a32e2690c1f8f5ddce6edba3f1974159956027b88003a484f5a0655bc"
POLICY_SEMANTIC_SHA256 = "bdc0fa93c6365cdcd45694d1d5500d82ccbd622f3be897be9217e252855ffff5"
V5_DESIGN_PHYSICAL_SHA256 = "57080b61171ad81d2d0d40307939ad9681db9a37452262b1baec4613d2b477cd"
V5_DESIGN_SEMANTIC_SHA256 = "7a63cfd6117338a3af16a422d5ee3458298fdc0ff2fd0abfe45fe851a7e54506"
PRODUCTION_ENGINE_PHYSICAL_SHA256 = (
    "e157419b99de8b5547ba2543931a3d2ab9906598594ae860adb5c3a06715e3f2"
)
PRODUCTION_STATISTICS_PHYSICAL_SHA256 = (
    "4739a404be983956ff6d178dbc1521c8c5fb5fea3699014687f1e7c40ce2e452"
)
PROVISIONAL_PREDECESSOR_PHYSICAL_SHA256 = (
    "82caec6dba223c089b972775236ea47e5ef380b1beffd437461b6160b58ff893"
)
PROVISIONAL_PREDECESSOR_SEMANTIC_SHA256 = (
    "b59dfc07280f972b83e00c5699a0f28e3da61135f2da22cfaf4638a4d1391910"
)
INVALIDATED_SUCCESSOR_PHYSICAL_SHA256 = (
    "5685a70d168ba6dde901eb5481e80e9ad3574eb3e1de4131f84c4e53dff23920"
)
INVALIDATED_SUCCESSOR_SEMANTIC_SHA256 = (
    "64f9f2f8afad51ded9f3be8b84d0c91c0259eae78c5b1086099dbb9f4f26eb1a"
)


class ArenaResolutionAuditError(RuntimeError):
    """A bound input, deterministic simulation, or no-replace write failed."""


@dataclass(frozen=True)
class LayoutSpec:
    config_id: str
    distinct_pairs_per_task: int
    raters_per_comparison: int


BASELINE = LayoutSpec("p20_r2", 20, 2)
REQUESTED_EXPANSIONS = (
    LayoutSpec("p40_r2", 40, 2),
    LayoutSpec("p80_r2", 80, 2),
    LayoutSpec("p20_r4", 20, 4),
)
MAXIMAL_PAIR_EXPANSION = LayoutSpec("p120_r2", 120, 2)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArenaResolutionAuditError(message)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _physical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArenaResolutionAuditError(f"cannot open bound source: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"bound source is not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verify_bound_sources() -> list[dict[str, Any]]:
    commitments = [
        (
            "frozen_arena_inference_policy",
            POLICY_REFERENCE,
            POLICY_PHYSICAL_SHA256,
            POLICY_SEMANTIC_SHA256,
        ),
        (
            "frozen_full_k16_v5_design",
            V5_DESIGN_REFERENCE,
            V5_DESIGN_PHYSICAL_SHA256,
            V5_DESIGN_SEMANTIC_SHA256,
        ),
        (
            "exact_production_layout_and_bootstrap_engine",
            PRODUCTION_ENGINE_REFERENCE,
            PRODUCTION_ENGINE_PHYSICAL_SHA256,
            None,
        ),
        (
            "exact_production_bt_statistics_engine",
            PRODUCTION_STATISTICS_REFERENCE,
            PRODUCTION_STATISTICS_PHYSICAL_SHA256,
            None,
        ),
        (
            "provisional_overlapping_confirmation_predecessor",
            PROVISIONAL_PREDECESSOR_REFERENCE,
            PROVISIONAL_PREDECESSOR_PHYSICAL_SHA256,
            PROVISIONAL_PREDECESSOR_SEMANTIC_SHA256,
        ),
        (
            "invalidated_not_fresh_confirmation_predecessor",
            INVALIDATED_SUCCESSOR_REFERENCE,
            INVALIDATED_SUCCESSOR_PHYSICAL_SHA256,
            INVALIDATED_SUCCESSOR_SEMANTIC_SHA256,
        ),
    ]
    result: list[dict[str, Any]] = []
    for role, reference, physical, semantic in commitments:
        path = EVALUATION_PAPER_ROOT / reference
        data = _read_regular_bytes(path)
        _require(hashlib.sha256(data).hexdigest() == physical, f"digest mismatch: {path}")
        if semantic is not None:
            try:
                document = json.loads(data)
            except json.JSONDecodeError as error:
                raise ArenaResolutionAuditError(f"invalid bound JSON: {path}") from error
            body = {key: value for key, value in document.items() if key != "artifact_sha256"}
            _require(
                document.get("artifact_sha256") == semantic and _semantic_sha256(body) == semantic,
                f"semantic digest mismatch: {path}",
            )
        result.append(
            {
                "role": role,
                "reference_path": reference,
                "physical_sha256": physical,
                "semantic_sha256": semantic,
            }
        )
    audit_path = Path(__file__).resolve()
    result.append(
        {
            "role": "development_only_resolution_audit_engine",
            "reference_path": AUDIT_ENGINE_REFERENCE,
            "physical_sha256": _physical_sha256(audit_path),
            "semantic_sha256": None,
        }
    )
    return result


def _load_bound_predecessor(
    *, reference: str, physical_sha256: str, semantic_sha256: str, label: str
) -> dict[str, Any]:
    path = EVALUATION_PAPER_ROOT / reference
    data = _read_regular_bytes(path)
    _require(
        hashlib.sha256(data).hexdigest() == physical_sha256,
        f"{label} physical digest mismatch",
    )
    try:
        document = json.loads(data)
    except json.JSONDecodeError as error:
        raise ArenaResolutionAuditError(f"invalid {label} JSON") from error
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(
        document.get("artifact_sha256") == semantic_sha256
        and _semantic_sha256(body) == semantic_sha256,
        f"{label} semantic digest mismatch",
    )
    for record in document.get("dataset_records", []):
        record_body = {key: value for key, value in record.items() if key != "record_sha256"}
        _require(
            record.get("record_sha256") == _semantic_sha256(record_body),
            f"{label} record digest mismatch",
        )
    return document


def _load_provisional_predecessor() -> dict[str, Any]:
    return _load_bound_predecessor(
        reference=PROVISIONAL_PREDECESSOR_REFERENCE,
        physical_sha256=PROVISIONAL_PREDECESSOR_PHYSICAL_SHA256,
        semantic_sha256=PROVISIONAL_PREDECESSOR_SEMANTIC_SHA256,
        label="provisional predecessor",
    )


def _load_invalidated_successor() -> dict[str, Any]:
    return _load_bound_predecessor(
        reference=INVALIDATED_SUCCESSOR_REFERENCE,
        physical_sha256=INVALIDATED_SUCCESSOR_PHYSICAL_SHA256,
        semantic_sha256=INVALIDATED_SUCCESSOR_SEMANTIC_SHA256,
        label="invalidated successor",
    )


def _artifact_confirmation_records(
    document: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    final_conditions = [
        row for row in document["condition_results"] if row["stage"] == "fresh_final_confirmation"
    ]
    _require(len(final_conditions) == 1, "predecessor final condition is not unique")
    condition = final_conditions[0]
    records = [
        row
        for row in document["dataset_records"]
        if row["config_id"] == condition["config_id"]
        and row["shift_elo"] == condition["shift_elo"]
        and row["bootstrap_replicates"] == condition["bootstraps_per_dataset"]
    ]
    _require(
        len(records) == condition["datasets"] == 100,
        "predecessor confirmation does not contain exactly 100 records",
    )
    return records


def _simulation_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the DGP identity, intentionally excluding hashes and lineage labels."""

    identity = {
        "layout_sha256": str(record["layout_sha256"]),
        "config_id": str(record["config_id"]),
        "shift_elo": float(record["shift_elo"]),
        "dataset_index": int(record["dataset_index"]),
        "dataset_seed": int(record["dataset_seed"]),
    }
    _require(bool(identity["layout_sha256"]), "simulation identity has an empty layout hash")
    _require(bool(identity["config_id"]), "simulation identity has an empty config id")
    _require(identity["dataset_index"] >= 0, "simulation identity has a negative index")
    return identity


def _simulation_identity_key(record: Mapping[str, Any]) -> tuple[str, str, float, int, int]:
    identity = _simulation_identity(record)
    return (
        identity["layout_sha256"],
        identity["config_id"],
        identity["shift_elo"],
        identity["dataset_index"],
        identity["dataset_seed"],
    )


def _simulation_identity_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return _simulation_identity_keys_sha256(
        {_simulation_identity_key(record) for record in records}
    )


def _simulation_identity_keys_sha256(
    keys: set[tuple[str, str, float, int, int]],
) -> str:
    serialized = sorted(
        _canonical_bytes(
            {
                "layout_sha256": key[0],
                "config_id": key[1],
                "shift_elo": key[2],
                "dataset_index": key[3],
                "dataset_seed": key[4],
            }
        ).decode()
        for key in keys
    )
    return _semantic_sha256({"simulation_identities": serialized})


def _layout_rater_id(battle_ordinal: int, spec: LayoutSpec, offset: int) -> str:
    ordinal = (battle_ordinal * spec.raters_per_comparison + offset) % RATER_COUNT
    return f"rater-{ordinal:02d}"


def build_layout(spec: LayoutSpec) -> dict[str, Any]:
    _require(
        1 <= spec.distinct_pairs_per_task <= MODEL_COUNT * (MODEL_COUNT - 1) // 2,
        "distinct pairs per task must be between 1 and 120",
    )
    _require(spec.raters_per_comparison >= 2, "at least two raters are required")
    models = [f"model-{index:02d}" for index in range(MODEL_COUNT)]
    round_robin = _round_robin_pairs(models)
    tasks: dict[str, str] = {}
    battles: list[dict[str, Any]] = []
    per_family_pair_cursor = 0
    for family in FAMILIES:
        for task_index in range(TASKS_PER_FAMILY):
            task_id = f"{family}-task-{task_index:02d}"
            tasks[task_id] = family
            task_pairs: set[tuple[str, str]] = set()
            for local_index in range(spec.distinct_pairs_per_task):
                first, second = round_robin[
                    (per_family_pair_cursor + local_index) % len(round_robin)
                ]
                pair = tuple(sorted((first, second)))
                _require(pair not in task_pairs, f"duplicate pair within {task_id}")
                task_pairs.add(pair)
                battle_ordinal = len(battles)
                battles.append(
                    {
                        "battle_id": f"battle-{battle_ordinal:04d}",
                        "task_id": task_id,
                        "family": family,
                        "model_a": first,
                        "model_b": second,
                        "rater_ids": [
                            _layout_rater_id(battle_ordinal, spec, offset)
                            for offset in range(spec.raters_per_comparison)
                        ],
                    }
                )
            per_family_pair_cursor += spec.distinct_pairs_per_task

    appearances = Counter(
        model_id for battle in battles for model_id in (battle["model_a"], battle["model_b"])
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
    counts = {
        "models": MODEL_COUNT,
        "families": len(FAMILIES),
        "scored_tasks": len(tasks),
        "distinct_pairs_per_task": spec.distinct_pairs_per_task,
        "unique_comparisons": len(battles),
        "raters_per_comparison": spec.raters_per_comparison,
        "primary_human_presentations": len(battles) * spec.raters_per_comparison,
        "concealed_repeat_presentations_at_12_5_percent": round(
            len(battles) * spec.raters_per_comparison * REPEAT_RATE
        ),
        "total_human_presentations_with_repeats": round(
            len(battles) * spec.raters_per_comparison * (1.0 + REPEAT_RATE)
        ),
        "unique_response_identities_reused": len(tasks) * MODEL_COUNT,
        "endpoint_appearances_per_model_minimum": min(appearances.values()),
        "endpoint_appearances_per_model_maximum": max(appearances.values()),
        "endpoint_appearances_per_model_family_minimum": min(family_appearances.values()),
        "endpoint_appearances_per_model_family_maximum": max(family_appearances.values()),
        "unique_task_clusters_per_model_family_minimum": min(map(len, model_family_tasks.values())),
    }
    payload = {
        "schema_version": "flavourbench-full-k16-arena-resolution-layout-v1",
        "config_id": spec.config_id,
        "model_ids": models,
        "tasks": tasks,
        "battles": battles,
        "counts": counts,
    }
    result = {**payload, "artifact_sha256": _semantic_sha256(payload)}
    if spec == BASELINE:
        frozen = build_production_layout()
        comparable_keys = (
            "battle_id",
            "task_id",
            "family",
            "model_a",
            "model_b",
            "rater_ids",
        )
        _require(
            list(tasks.items()) == list(frozen["tasks"].items())
            and len(battles) == len(frozen["battles"])
            and all(
                all(left[key] == right[key] for key in comparable_keys)
                for left, right in zip(battles, frozen["battles"], strict=True)
            ),
            "baseline layout does not reproduce the frozen production layout",
        )
    return result


def _simulate_shift_observations(
    layout: Mapping[str, Any], *, shift_elo: float, dataset_index: int
) -> tuple[list[ArenaObservation], dict[str, float], int]:
    seed = _dataset_seed("single_model_50_elo_shift", dataset_index)
    if layout["config_id"] == BASELINE.config_id and shift_elo == TARGET_ELO:
        observations, truth = _simulate_observations(
            layout,
            scenario="single_model_50_elo_shift",
            dataset_index=dataset_index,
        )
        return observations, truth, seed

    rng = np.random.default_rng(seed)
    models = list(layout["model_ids"])
    model_elo = {model_id: 0.0 for model_id in models}
    model_elo[models[0]] = float(shift_elo)
    # Preserve the frozen DGP draw order: 64 zero-SD rater draws, then task
    # effects, then both-bad, win, and tie uniforms for every observation.
    rater_bias = {f"rater-{index:02d}": float(rng.normal(0.0, 0.0)) for index in range(RATER_COUNT)}
    task_icc = 0.05
    task_sd_logit = math.sqrt(task_icc / (1.0 - task_icc) * (math.pi**2 / 3.0))
    task_effect = {task_id: float(rng.normal(0.0, task_sd_logit)) for task_id in layout["tasks"]}
    observations: list[ArenaObservation] = []
    for battle in layout["battles"]:
        first = str(battle["model_a"])
        second = str(battle["model_b"])
        base_delta = model_elo[first] - model_elo[second]
        for rater_id in battle["rater_ids"]:
            if rng.random() < 0.0:
                continue
            logit = (
                base_delta * math.log(10.0) / 400.0
                + task_effect[str(battle["task_id"])]
                + rater_bias[str(rater_id)] * math.log(10.0) / 400.0
            )
            first_win = rng.random() < 1.0 / (1.0 + math.exp(-logit))
            outcome = 0.5 if rng.random() < 0.10 else 1.0 if first_win else 0.0
            observations.append(
                ArenaObservation(
                    observation_id=(
                        f"resolution-{layout['config_id']}-{shift_elo:g}-{dataset_index}-"
                        f"{battle['battle_id']}-{rater_id}"
                    ),
                    task_id=str(battle["task_id"]),
                    family=str(battle["family"]),
                    battle_id=str(battle["battle_id"]),
                    rater_id=str(rater_id),
                    model_a=first,
                    model_b=second,
                    response_a_id=f"{battle['task_id']}-{first}",
                    response_b_id=f"{battle['task_id']}-{second}",
                    outcome=outcome,
                )
            )
    return observations, model_elo, seed


def _uniform_peer_index(dataset_index: int) -> int:
    digest = hashlib.sha256(
        f"{ARENA_INFERENCE_POLICY_SHA256}:uniform-peer:{dataset_index}".encode()
    ).digest()
    return 1 + int.from_bytes(digest[:8], "big") % PEER_COUNT


def run_dataset(
    *,
    spec: LayoutSpec,
    shift_elo: float,
    dataset_index: int,
    bootstrap_replicates: int,
    duplication_check: bool | None = None,
    record_lineage: str = "generated_current_run",
) -> dict[str, Any]:
    _require(dataset_index >= 0, "dataset index must be nonnegative")
    _require(bootstrap_replicates >= 2, "at least two bootstrap replicates required")
    _require(bool(record_lineage), "record lineage must be nonempty")
    layout = build_layout(spec)
    observations, truth, seed = _simulate_shift_observations(
        layout, shift_elo=shift_elo, dataset_index=dataset_index
    )
    fitted = _fit_dataset(
        observations,
        layout["model_ids"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        duplication_check=(dataset_index == 0 if duplication_check is None else duplication_check),
    )
    samples = fitted.pop("rating_samples")
    points = fitted.pop("point_ratings")
    focal_samples = np.asarray(samples[FOCAL_MODEL], dtype=float)
    peer_rows: list[dict[str, Any]] = []
    truth_probability = 1.0 / (1.0 + 10.0 ** (-shift_elo / 400.0))
    for peer_index in range(1, MODEL_COUNT):
        peer = layout["model_ids"][peer_index]
        contrasts = focal_samples - np.asarray(samples[peer], dtype=float)
        lower, upper = _percentile_interval(contrasts)
        point = float(points[FOCAL_MODEL] - points[peer])
        estimated_probability = 1.0 / (1.0 + 10.0 ** (-point / 400.0))
        peer_rows.append(
            {
                "peer_model_id": peer,
                "point_elo": round(point, 9),
                "interval_lower": round(lower, 9),
                "interval_upper": round(upper, 9),
                "positive_effect_detected": bool(lower > 0.0),
                "interval_covers_truth": bool(lower <= shift_elo <= upper),
                "probability_scale_absolute_bias": round(
                    abs(estimated_probability - truth_probability), 9
                ),
            }
        )
    selected_index = _uniform_peer_index(dataset_index)
    selected = peer_rows[selected_index - 1]
    fixed = peer_rows[0]
    body = {
        "config_id": spec.config_id,
        "layout_sha256": layout["artifact_sha256"],
        "shift_elo": shift_elo,
        "dataset_index": dataset_index,
        "dataset_seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "record_lineage": record_lineage,
        "engine": "exact_production_bt_plus_crossed_task_rater_bootstrap",
        "fixed_peer_model_id": fixed["peer_model_id"],
        "fixed_peer_detected": fixed["positive_effect_detected"],
        "uniform_peer_model_id": selected["peer_model_id"],
        "uniform_peer_detected": selected["positive_effect_detected"],
        "focal_point_ranked_first": bool(
            points[FOCAL_MODEL]
            > max(value for model, value in points.items() if model != FOCAL_MODEL)
        ),
        "all_15_pointwise_intervals_positive": all(
            row["positive_effect_detected"] for row in peer_rows
        ),
        "detected_peer_count": sum(row["positive_effect_detected"] for row in peer_rows),
        "covered_peer_count": sum(row["interval_covers_truth"] for row in peer_rows),
        "mean_probability_scale_absolute_bias": round(
            float(np.mean([row["probability_scale_absolute_bias"] for row in peer_rows])),
            9,
        ),
        "global_bootstrap_connected_rate": fitted["global_bootstrap_connected_rate"],
        "family_bootstrap_connected_rates": fitted["family_bootstrap_connected_rates"],
        "successful_bootstrap_replicates": fitted["successful_bootstrap_replicates"],
        "duplicate_interval_width_delta": fitted["duplicate_interval_width_delta"],
        "peer_intervals": peer_rows,
        "claim_boundary": {
            "counts_toward_production_gate": False,
            "production_method_validation_complete": False,
            "model_quality_evidence": False,
            "human_judgments_created": False,
        },
    }
    return {**body, "record_sha256": _semantic_sha256(body)}


def _exact_rate(successes: int, trials: int) -> dict[str, Any]:
    _require(0 <= successes <= trials and trials > 0, "invalid binomial count")
    interval = stats.binomtest(successes, trials).proportion_ci(
        confidence_level=CONFIDENCE_LEVEL, method="exact"
    )
    point = successes / trials
    return {
        "successes": successes,
        "trials": trials,
        "point_estimate": round(point, 9),
        "mc_standard_error": round(math.sqrt(point * (1.0 - point) / trials), 9),
        "clopper_pearson_95_lower": round(float(interval.low), 9),
        "clopper_pearson_95_upper": round(float(interval.high), 9),
    }


def aggregate_condition(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(records), "cannot aggregate an empty condition")
    config_ids = {str(row["config_id"]) for row in records}
    shifts = {float(row["shift_elo"]) for row in records}
    bootstraps = {int(row["bootstrap_replicates"]) for row in records}
    _require(
        len(config_ids) == len(shifts) == len(bootstraps) == 1,
        "condition records disagree",
    )
    fixed = _exact_rate(sum(bool(row["fixed_peer_detected"]) for row in records), len(records))
    marginal = _exact_rate(sum(bool(row["uniform_peer_detected"]) for row in records), len(records))
    simultaneous = _exact_rate(
        sum(bool(row["all_15_pointwise_intervals_positive"]) for row in records),
        len(records),
    )
    point_top = _exact_rate(
        sum(bool(row["focal_point_ranked_first"]) for row in records), len(records)
    )
    per_peer: list[dict[str, Any]] = []
    for peer_index in range(PEER_COUNT):
        peer_id = str(records[0]["peer_intervals"][peer_index]["peer_model_id"])
        rate = _exact_rate(
            sum(
                bool(row["peer_intervals"][peer_index]["positive_effect_detected"])
                for row in records
            ),
            len(records),
        )
        per_peer.append({"peer_model_id": peer_id, **rate})
    minimum_connectivity = min(
        [float(row["global_bootstrap_connected_rate"]) for row in records]
        + [
            float(value)
            for row in records
            for value in row["family_bootstrap_connected_rates"].values()
        ]
    )
    pooled_descriptive = sum(int(row["detected_peer_count"]) for row in records) / (
        len(records) * PEER_COUNT
    )
    return {
        "config_id": next(iter(config_ids)),
        "shift_elo": next(iter(shifts)),
        "datasets": len(records),
        "bootstraps_per_dataset": next(iter(bootstraps)),
        "fixed_focal_vs_model_01_detection": fixed,
        "fixed_focal_vs_model_01_coverage": _exact_rate(
            sum(bool(row["peer_intervals"][0]["interval_covers_truth"]) for row in records),
            len(records),
        ),
        "fixed_focal_vs_model_01_mean_point_elo": round(
            float(np.mean([row["peer_intervals"][0]["point_elo"] for row in records])),
            9,
        ),
        "fixed_focal_vs_model_01_mean_probability_absolute_bias": round(
            float(
                np.mean(
                    [row["peer_intervals"][0]["probability_scale_absolute_bias"] for row in records]
                )
            ),
            9,
        ),
        "frozen_average_marginal_pair_power_uniform_peer_estimator": marginal,
        "all_15_pointwise_intervals_positive_same_dataset": simultaneous,
        "focal_point_ranked_first": point_top,
        "descriptive_all_peer_detection_fraction_correlated_not_binomial": round(
            pooled_descriptive, 9
        ),
        "per_peer_detection": per_peer,
        "coverage_fraction": round(
            sum(int(row["covered_peer_count"]) for row in records) / (len(records) * PEER_COUNT),
            9,
        ),
        "mean_probability_scale_absolute_bias": round(
            float(np.mean([row["mean_probability_scale_absolute_bias"] for row in records])),
            9,
        ),
        "minimum_global_or_family_bootstrap_connectivity": round(minimum_connectivity, 9),
        "maximum_absolute_duplicate_interval_width_delta": (
            round(
                max(
                    abs(float(row["duplicate_interval_width_delta"]))
                    for row in records
                    if row["duplicate_interval_width_delta"] is not None
                ),
                12,
            )
            if any(row["duplicate_interval_width_delta"] is not None for row in records)
            else None
        ),
        "frozen_50_elo_power_target": FROZEN_POWER_TARGET,
        "frozen_target_lower_bound_passes": bool(
            next(iter(shifts)) == TARGET_ELO
            and marginal["clopper_pearson_95_lower"] >= FROZEN_POWER_TARGET
        ),
        "simultaneous_top_identification_lower_bound_passes": bool(
            next(iter(shifts)) == TARGET_ELO
            and simultaneous["clopper_pearson_95_lower"] >= FROZEN_POWER_TARGET
        ),
    }


def _worker(payload: tuple[LayoutSpec, float, int, int, bool, str]) -> dict[str, Any]:
    spec, shift, index, bootstraps, duplication_check, record_lineage = payload
    return run_dataset(
        spec=spec,
        shift_elo=shift,
        dataset_index=index,
        bootstrap_replicates=bootstraps,
        duplication_check=duplication_check,
        record_lineage=record_lineage,
    )


def run_condition(
    *,
    spec: LayoutSpec,
    shift_elo: float,
    datasets: int,
    bootstrap_replicates: int,
    workers: int,
    dataset_start: int = 0,
    record_lineage: str = "generated_current_run",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(
        datasets > 0 and workers > 0 and dataset_start >= 0,
        "datasets and workers must be positive and dataset start nonnegative",
    )
    payloads = [
        (
            spec,
            shift_elo,
            index,
            bootstrap_replicates,
            index == dataset_start,
            record_lineage,
        )
        for index in range(dataset_start, dataset_start + datasets)
    ]
    if workers == 1:
        records = [_worker(payload) for payload in payloads]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_worker, payloads))
    records.sort(key=lambda row: int(row["dataset_index"]))
    return records, aggregate_condition(records)


def _human_workload(layout: Mapping[str, Any]) -> dict[str, Any]:
    counts = layout["counts"]
    return {
        "scored_tasks": counts["scored_tasks"],
        "distinct_pairs_per_task": counts["distinct_pairs_per_task"],
        "unique_comparisons": counts["unique_comparisons"],
        "raters_per_comparison": counts["raters_per_comparison"],
        "primary_human_presentations": counts["primary_human_presentations"],
        "concealed_repeat_rate": REPEAT_RATE,
        "concealed_repeat_presentations": counts["concealed_repeat_presentations_at_12_5_percent"],
        "total_human_presentations": counts["total_human_presentations_with_repeats"],
        "unique_response_identities_reused": counts["unique_response_identities_reused"],
        "provider_price_or_money_claim": False,
    }


def _screen_selection(aggregates: Sequence[Mapping[str, Any]]) -> str:
    eligible = [
        row
        for row in aggregates
        if row["frozen_average_marginal_pair_power_uniform_peer_estimator"][
            "clopper_pearson_95_upper"
        ]
        >= FROZEN_POWER_TARGET
    ]
    _require(bool(eligible), "no screened layout can plausibly reach the frozen target")
    eligible.sort(
        key=lambda row: (
            -float(
                row["frozen_average_marginal_pair_power_uniform_peer_estimator"]["point_estimate"]
            ),
            str(row["config_id"]),
        )
    )
    return str(eligible[0]["config_id"])


def build_audit_artifact(
    *,
    workers: int = DEFAULT_WORKERS,
    baseline_datasets: int = DEFAULT_BASELINE_DATASETS,
    baseline_bootstraps: int = DEFAULT_BASELINE_BOOTSTRAPS,
    screen_datasets: int = DEFAULT_SCREEN_DATASETS,
    screen_bootstraps: int = DEFAULT_SCREEN_BOOTSTRAPS,
    confirm_datasets: int = DEFAULT_CONFIRM_DATASETS,
    confirm_bootstraps: int = DEFAULT_CONFIRM_BOOTSTRAPS,
    enforce_minimums: bool = True,
    reuse_verified_predecessor_stages: bool = True,
) -> dict[str, Any]:
    if enforce_minimums:
        _require(
            baseline_datasets >= 100
            and baseline_bootstraps >= 200
            and screen_datasets >= 40
            and screen_bootstraps >= 100
            and confirm_datasets >= 100
            and confirm_bootstraps >= 200,
            "binding audit cannot weaken the documented two-stage Monte Carlo plan",
        )
    sources = _verify_bound_sources()
    specs = {spec.config_id: spec for spec in (BASELINE, *REQUESTED_EXPANSIONS)}
    layouts = {config_id: build_layout(spec) for config_id, spec in specs.items()}
    all_records: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    screen_aggregates: list[dict[str, Any]] = []
    if reuse_verified_predecessor_stages:
        _require(
            baseline_datasets == DEFAULT_BASELINE_DATASETS
            and baseline_bootstraps == DEFAULT_BASELINE_BOOTSTRAPS
            and screen_datasets == DEFAULT_SCREEN_DATASETS
            and screen_bootstraps == DEFAULT_SCREEN_BOOTSTRAPS
            and confirm_datasets == DEFAULT_CONFIRM_DATASETS
            and confirm_bootstraps == DEFAULT_CONFIRM_BOOTSTRAPS
            and FRESH_CONFIRMATION_DATASET_END - FRESH_CONFIRMATION_DATASET_START + 1
            == DEFAULT_CONFIRM_DATASETS,
            "predecessor reuse requires the exact binding baseline, screen, "
            "and fresh-confirmation plan",
        )
        predecessor = _load_provisional_predecessor()
        invalidated_successor = _load_invalidated_successor()
        reusable_conditions = [
            dict(row)
            for row in predecessor["condition_results"]
            if row["stage"] in {"baseline_confirmation", "bounded_screen"}
        ]
        invalidated_reusable_conditions = [
            dict(row)
            for row in invalidated_successor["condition_results"]
            if row["stage"] in {"baseline_confirmation", "bounded_screen"}
        ]
        _require(
            invalidated_reusable_conditions == reusable_conditions,
            "predecessor baseline/screen aggregates differ",
        )
        reusable_keys = {
            (
                str(row["config_id"]),
                float(row["shift_elo"]),
                int(row["bootstraps_per_dataset"]),
            ): int(row["datasets"])
            for row in reusable_conditions
        }
        reused_records = [
            dict(row)
            for row in predecessor["dataset_records"]
            if (
                str(row["config_id"]),
                float(row["shift_elo"]),
                int(row["bootstrap_replicates"]),
            )
            in reusable_keys
            and int(row["dataset_index"])
            < reusable_keys[
                (
                    str(row["config_id"]),
                    float(row["shift_elo"]),
                    int(row["bootstrap_replicates"]),
                )
            ]
        ]
        excluded_records = [
            row
            for row in predecessor["dataset_records"]
            if row["record_sha256"] not in {reused["record_sha256"] for reused in reused_records}
        ]
        _require(
            len(reusable_conditions) == 6
            and len(reused_records) == 420
            and len(excluded_records) == 100,
            "predecessor stage partition is not 420 reusable plus 100 excluded",
        )
        for condition in reusable_conditions:
            matching = [
                row
                for row in reused_records
                if row["config_id"] == condition["config_id"]
                and row["shift_elo"] == condition["shift_elo"]
                and row["bootstrap_replicates"] == condition["bootstraps_per_dataset"]
            ]
            recomputed = aggregate_condition(matching)
            expected = {key: value for key, value in condition.items() if key != "stage"}
            _require(recomputed == expected, "reused predecessor aggregate mismatch")
        all_records.extend(reused_records)
        conditions.extend(reusable_conditions)
        screen_aggregates.extend(
            row for row in reusable_conditions if row["stage"] == "bounded_screen"
        )
        predecessor_documents = (
            (
                PROVISIONAL_PREDECESSOR_REFERENCE,
                PROVISIONAL_PREDECESSOR_PHYSICAL_SHA256,
                PROVISIONAL_PREDECESSOR_SEMANTIC_SHA256,
                predecessor,
            ),
            (
                INVALIDATED_SUCCESSOR_REFERENCE,
                INVALIDATED_SUCCESSOR_PHYSICAL_SHA256,
                INVALIDATED_SUCCESSOR_SEMANTIC_SHA256,
                invalidated_successor,
            ),
        )
        predecessor_identity_sets = {
            semantic: {_simulation_identity_key(row) for row in document["dataset_records"]}
            for _, _, semantic, document in predecessor_documents
        }
        predecessor_confirmation_records = {
            semantic: _artifact_confirmation_records(document)
            for _, _, semantic, document in predecessor_documents
        }
        predecessor_retained_hash_sets = {
            semantic: {
                row["record_sha256"]
                for row in document["dataset_records"]
                if row not in predecessor_confirmation_records[semantic]
            }
            for _, _, semantic, document in predecessor_documents
        }
        reused_hashes = {row["record_sha256"] for row in reused_records}
        _require(
            all(
                predecessor_retained_hash_sets[semantic] == reused_hashes
                for _, _, semantic, _ in predecessor_documents
            ),
            "predecessors do not share the same 420 baseline/screen records",
        )
        prior_observed_identity_keys = set().union(*predecessor_identity_sets.values())
        stage_lineage = {
            "baseline_and_screen_source": "verified_provisional_predecessor_records",
            "reused_record_count": len(reused_records),
            "reused_record_set_sha256": _semantic_sha256(
                {"record_sha256s": sorted(row["record_sha256"] for row in reused_records)}
            ),
            "excluded_overlapping_confirmation_record_count": len(excluded_records),
            "excluded_overlapping_confirmation_record_set_sha256": _semantic_sha256(
                {"record_sha256s": sorted(row["record_sha256"] for row in excluded_records)}
            ),
            "excluded_predecessor_confirmation_record_hash_union_count": len(
                {
                    row["record_sha256"]
                    for rows in predecessor_confirmation_records.values()
                    for row in rows
                }
            ),
            "excluded_predecessor_confirmation_record_hash_union_sha256": (
                _semantic_sha256(
                    {
                        "record_sha256s": sorted(
                            {
                                row["record_sha256"]
                                for rows in predecessor_confirmation_records.values()
                                for row in rows
                            }
                        )
                    }
                )
            ),
            "excluded_predecessor_confirmation_simulation_identity_union_count": len(
                {
                    _simulation_identity_key(row)
                    for rows in predecessor_confirmation_records.values()
                    for row in rows
                }
            ),
            "excluded_predecessor_confirmation_simulation_identity_union_sha256": (
                _simulation_identity_keys_sha256(
                    {
                        _simulation_identity_key(row)
                        for rows in predecessor_confirmation_records.values()
                        for row in rows
                    }
                )
            ),
            "excluded_records_used_in_successor": False,
            "prior_observed_predecessors": [
                {
                    "reference_path": reference,
                    "physical_sha256": physical,
                    "semantic_sha256": semantic,
                    "dataset_record_count": len(document["dataset_records"]),
                    "unique_simulation_identity_count": len(predecessor_identity_sets[semantic]),
                    "simulation_identity_set_sha256": _simulation_identity_set_sha256(
                        document["dataset_records"]
                    ),
                    "confirmation_record_count": len(predecessor_confirmation_records[semantic]),
                    "confirmation_record_set_sha256": _semantic_sha256(
                        {
                            "record_sha256s": sorted(
                                row["record_sha256"]
                                for row in predecessor_confirmation_records[semantic]
                            )
                        }
                    ),
                    "confirmation_unique_simulation_identity_count": len(
                        {
                            _simulation_identity_key(row)
                            for row in predecessor_confirmation_records[semantic]
                        }
                    ),
                    "confirmation_simulation_identity_set_sha256": (
                        _simulation_identity_set_sha256(predecessor_confirmation_records[semantic])
                    ),
                    "fresh_confirmation_overlap_count": 0,
                    "status": "superseded_not_for_use",
                }
                for reference, physical, semantic, document in predecessor_documents
            ],
            "prior_observed_union_unique_simulation_identity_count": len(
                prior_observed_identity_keys
            ),
            "prior_observed_union_simulation_identity_set_sha256": (
                _simulation_identity_keys_sha256(prior_observed_identity_keys)
            ),
            "failed_no_write_execution_note": (
                "a disjoint confirmation completed but failed closed before artifact "
                "construction because the optional duplication-delta aggregate was empty; "
                "no artifact or partial result was retained"
            ),
        }
    else:
        for shift in (50.0, 75.0, 100.0):
            records, aggregate = run_condition(
                spec=BASELINE,
                shift_elo=shift,
                datasets=baseline_datasets,
                bootstrap_replicates=baseline_bootstraps,
                workers=workers,
            )
            all_records.extend(records)
            conditions.append({"stage": "baseline_confirmation", **aggregate})
        for spec in REQUESTED_EXPANSIONS:
            records, aggregate = run_condition(
                spec=spec,
                shift_elo=TARGET_ELO,
                datasets=screen_datasets,
                bootstrap_replicates=screen_bootstraps,
                workers=workers,
            )
            all_records.extend(records)
            tagged = {"stage": "bounded_screen", **aggregate}
            conditions.append(tagged)
            screen_aggregates.append(tagged)
        stage_lineage = {
            "baseline_and_screen_source": "generated_in_current_nonbinding_test_run",
            "reused_record_count": 0,
            "excluded_records_used_in_successor": False,
        }

    selected_config = _screen_selection(screen_aggregates)
    selected_spec = specs[selected_config]
    confirmation_dataset_start = (
        FRESH_CONFIRMATION_DATASET_START if reuse_verified_predecessor_stages else screen_datasets
    )
    confirmation_lineage = (
        FRESH_CONFIRMATION_LINEAGE
        if reuse_verified_predecessor_stages
        else "generated_current_nonbinding_test_run"
    )
    planned_fresh_identity_keys: set[tuple[str, str, float, int, int]] | None = None
    if reuse_verified_predecessor_stages:
        planned_fresh_identity_keys = {
            (
                str(layouts[selected_config]["artifact_sha256"]),
                selected_config,
                TARGET_ELO,
                dataset_index,
                _dataset_seed("single_model_50_elo_shift", dataset_index),
            )
            for dataset_index in range(
                FRESH_CONFIRMATION_DATASET_START,
                FRESH_CONFIRMATION_DATASET_END + 1,
            )
        }
        _require(
            len(planned_fresh_identity_keys) == DEFAULT_CONFIRM_DATASETS,
            "fresh confirmation plan does not contain exactly 100 identities",
        )
        _require(
            not planned_fresh_identity_keys.intersection(prior_observed_identity_keys),
            "planned confirmation overlaps predecessor simulation identity",
        )
        _require(
            all(
                not planned_fresh_identity_keys.intersection(identity_set)
                for identity_set in predecessor_identity_sets.values()
            ),
            "planned confirmation overlaps an individual predecessor identity",
        )
        for predecessor_audit in stage_lineage["prior_observed_predecessors"]:
            predecessor_audit["planned_fresh_confirmation_overlap_count"] = 0
        stage_lineage.update(
            {
                "fresh_confirmation_preflight_completed_before_execution": True,
                "planned_fresh_confirmation_dataset_indices": [
                    FRESH_CONFIRMATION_DATASET_START,
                    FRESH_CONFIRMATION_DATASET_END,
                ],
                "planned_fresh_confirmation_record_lineage": (FRESH_CONFIRMATION_LINEAGE),
                "planned_fresh_confirmation_unique_simulation_identity_count": len(
                    planned_fresh_identity_keys
                ),
                "planned_fresh_confirmation_simulation_identity_set_sha256": (
                    _simulation_identity_keys_sha256(planned_fresh_identity_keys)
                ),
                "planned_fresh_confirmation_overlap_with_prior_observed_union": 0,
            }
        )
    records, confirmation = run_condition(
        spec=selected_spec,
        shift_elo=TARGET_ELO,
        datasets=confirm_datasets,
        bootstrap_replicates=confirm_bootstraps,
        workers=workers,
        dataset_start=confirmation_dataset_start,
        record_lineage=confirmation_lineage,
    )
    if reuse_verified_predecessor_stages:
        expected_indices = set(
            range(FRESH_CONFIRMATION_DATASET_START, FRESH_CONFIRMATION_DATASET_END + 1)
        )
        observed_indices = {int(row["dataset_index"]) for row in records}
        fresh_identity_keys = {_simulation_identity_key(row) for row in records}
        _require(observed_indices == expected_indices, "fresh confirmation indices changed")
        _require(len(records) == len(fresh_identity_keys) == 100, "fresh identities are not unique")
        _require(
            all(row.get("record_lineage") == FRESH_CONFIRMATION_LINEAGE for row in records),
            "fresh confirmation lineage changed",
        )
        _require(
            all(
                int(row["dataset_seed"])
                == _dataset_seed("single_model_50_elo_shift", int(row["dataset_index"]))
                for row in records
            ),
            "fresh confirmation dataset seed changed",
        )
        _require(
            not fresh_identity_keys.intersection(prior_observed_identity_keys),
            "fresh confirmation overlaps predecessor simulation identity",
        )
        _require(
            fresh_identity_keys == planned_fresh_identity_keys,
            "executed confirmation identities differ from the preflighted plan",
        )
        for predecessor_audit in stage_lineage["prior_observed_predecessors"]:
            predecessor_audit["fresh_confirmation_overlap_count"] = len(
                fresh_identity_keys.intersection(
                    predecessor_identity_sets[predecessor_audit["semantic_sha256"]]
                )
            )
        stage_lineage.update(
            {
                "fresh_confirmation_record_count": len(records),
                "fresh_confirmation_dataset_indices": [
                    FRESH_CONFIRMATION_DATASET_START,
                    FRESH_CONFIRMATION_DATASET_END,
                ],
                "fresh_confirmation_record_lineage": FRESH_CONFIRMATION_LINEAGE,
                "fresh_confirmation_unique_simulation_identity_count": len(fresh_identity_keys),
                "fresh_confirmation_simulation_identity_set_sha256": (
                    _simulation_identity_set_sha256(records)
                ),
                "fresh_confirmation_overlap_with_prior_observed_union": 0,
            }
        )
    all_records.extend(records)
    confirmation = {"stage": "fresh_final_confirmation", **confirmation}
    conditions.append(confirmation)

    frozen_pass = bool(confirmation["frozen_target_lower_bound_passes"])
    simultaneous_pass = bool(confirmation["simultaneous_top_identification_lower_bound_passes"])
    baseline_50 = next(
        row
        for row in conditions
        if row["stage"] == "baseline_confirmation" and row["shift_elo"] == 50.0
    )
    record_set_sha256 = _semantic_sha256(
        {"record_sha256s": sorted(row["record_sha256"] for row in all_records)}
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "audit_date": "2026-08-09",
        "supersession": {
            "all_predecessors_retained_append_only": True,
            "superseded_artifacts": [
                {
                    "reference_path": PROVISIONAL_PREDECESSOR_REFERENCE,
                    "semantic_sha256": PROVISIONAL_PREDECESSOR_SEMANTIC_SHA256,
                    "physical_sha256": PROVISIONAL_PREDECESSOR_PHYSICAL_SHA256,
                    "status": "superseded_not_for_use",
                    "defect": (
                        "confirmation reused bounded-screen simulation identities at "
                        "dataset indices 0 through 39"
                    ),
                },
                {
                    "reference_path": INVALIDATED_SUCCESSOR_REFERENCE,
                    "semantic_sha256": INVALIDATED_SUCCESSOR_SEMANTIC_SHA256,
                    "physical_sha256": INVALIDATED_SUCCESSOR_PHYSICAL_SHA256,
                    "status": "superseded_not_for_use",
                    "defect": (
                        "claimed-fresh confirmation indices 40 through 139 overlapped "
                        "previously exposed confirmation identities 40 through 99"
                    ),
                },
            ],
            "reason": (
                "this successor uses entirely unseen simulation identities at dataset "
                "indices 140 through 239 and excludes identities exposed by both predecessors"
            ),
        },
        "stage_lineage": stage_lineage,
        "source_commitments": sources,
        "layout_commitments": [
            {
                "config_id": config_id,
                "layout_sha256": layout["artifact_sha256"],
                "counts": layout["counts"],
            }
            for config_id, layout in layouts.items()
        ],
        "simulation_contract": {
            "data_generating_process": {
                "models": MODEL_COUNT,
                "scored_tasks": TASKS_PER_FAMILY * len(FAMILIES),
                "families": list(FAMILIES),
                "focal_model": FOCAL_MODEL,
                "peer_truth_elo": 0.0,
                "focal_truth_elo_shifts": [50.0, 75.0, 100.0],
                "tie_probability": 0.10,
                "task_icc": 0.05,
                "rater_count": RATER_COUNT,
                "rater_bias_sd_elo": 0.0,
                "same_16_response_identities_reused_within_each_task": True,
                "unique_response_identities": TASKS_PER_FAMILY * len(FAMILIES) * MODEL_COUNT,
                "synthetic_outcomes_only": True,
            },
            "estimator": "frozen arena-rank 0.1.1 Bradley-Terry point fit",
            "interval": (
                "frozen 95% percentile interval from crossed family-stratified task "
                "and global rater-cluster bootstrap"
            ),
            "common_random_number_seed": (
                "frozen single_model_50_elo_shift dataset seed for each dataset index"
            ),
            "uniform_peer_selection": (
                "independent SHA-256 domain ARENA_POLICY_SHA256:uniform-peer:dataset_index; "
                "one of 15 peers selected before reading simulated outcomes"
            ),
            "two_stage_plan": {
                "baseline": {
                    "conditions": ["p20_r2@50", "p20_r2@75", "p20_r2@100"],
                    "datasets_each": baseline_datasets,
                    "bootstraps_each": baseline_bootstraps,
                },
                "screen": {
                    "conditions": ["p40_r2@50", "p80_r2@50", "p20_r4@50"],
                    "datasets_each": screen_datasets,
                    "bootstraps_each": screen_bootstraps,
                },
                "advance_rule": (
                    "among layouts whose exact 95% CP upper bound reaches 0.80, "
                    "advance the highest uniform-peer marginal point estimate; "
                    "break ties by config_id"
                ),
                "selected_final_config": selected_config,
                "screen_dataset_indices": [0, screen_datasets - 1],
                "confirmation_dataset_indices": [
                    confirmation_dataset_start,
                    confirmation_dataset_start + confirm_datasets - 1,
                ],
                "screen_confirmation_dataset_overlap": len(
                    set(range(screen_datasets)).intersection(
                        range(
                            confirmation_dataset_start,
                            confirmation_dataset_start + confirm_datasets,
                        )
                    )
                ),
                "fresh_confirmation_datasets": confirm_datasets,
                "fresh_confirmation_bootstraps_each": confirm_bootstraps,
            },
            "production_gate_required_but_not_run": {
                "datasets_per_scenario": 2000,
                "bootstrap_replicates": 5000,
                "scenarios": 8,
                "nominal_bootstrap_refits": 80_000_000,
            },
        },
        "estimand_separation": {
            "frozen_primary_50_elo_estimand": (
                "average marginal power for focal model versus a uniformly selected one "
                "of 15 exchangeable peers"
            ),
            "frozen_primary_estimator": (
                "one preselected peer binary detection per independent dataset"
            ),
            "frozen_primary_acceptance": (
                "two-sided exact 95% Clopper-Pearson lower bound must be >= 0.80"
            ),
            "fixed_peer_diagnostic": "focal model-00 versus fixed model-01",
            "simultaneous_top_diagnostic": (
                "all 15 separate pointwise 95% bootstrap interval lower endpoints exceed "
                "zero in the same dataset; not a simultaneous-confidence procedure"
            ),
            "correlated_all_peer_fraction": (
                "descriptive only; 15 within-dataset detections are not treated as "
                "independent binomial trials"
            ),
        },
        "condition_results": conditions,
        "dataset_records": sorted(
            all_records,
            key=lambda row: (
                str(row["config_id"]),
                float(row["shift_elo"]),
                int(row["bootstrap_replicates"]),
                int(row["dataset_index"]),
            ),
        ),
        "record_set_sha256": record_set_sha256,
        "human_presentation_workloads": {
            config_id: _human_workload(layout) for config_id, layout in layouts.items()
        },
        "decision": {
            "baseline_frozen_marginal_power_lower_bound": baseline_50[
                "frozen_average_marginal_pair_power_uniform_peer_estimator"
            ]["clopper_pearson_95_lower"],
            "selected_final_config": selected_config,
            "selected_final_frozen_marginal_power_lower_bound": confirmation[
                "frozen_average_marginal_pair_power_uniform_peer_estimator"
            ]["clopper_pearson_95_lower"],
            "selected_final_simultaneous_top_lower_bound": confirmation[
                "all_15_pointwise_intervals_positive_same_dataset"
            ]["clopper_pearson_95_lower"],
            "frozen_50_elo_marginal_target_passed": frozen_pass,
            "simultaneous_top_resolution_passed": simultaneous_pass,
            "overall_verdict": "NO-GO",
            "reason": (
                "development-only evidence cannot complete the frozen production gate; "
                "any failed exact lower-bound target additionally blocks the tested layout"
            ),
            "next_required_expansion_if_selected_candidate_fails": {
                "config_id": MAXIMAL_PAIR_EXPANSION.config_id,
                "distinct_pairs_per_task": 120,
                "raters_per_comparison": 2,
                "primary_human_presentations": 38_400,
                "concealed_repeat_presentations": 4_800,
                "total_human_presentations": 43_200,
                "status": "quantified_not_simulated_or_authorized",
            },
        },
        "claim_boundary": {
            "development_only": True,
            "production_method_validation_complete": False,
            "counts_toward_frozen_gate": False,
            "model_quality_evidence": False,
            "human_judgments_created": False,
            "human_study_authorized": False,
            "provider_calls_made": False,
            "deployment_or_activation_authorized": False,
            "paper_or_public_claim_authorized": False,
            "official_benchmark_or_rank_authority": False,
        },
    }
    return {**body, "artifact_sha256": _semantic_sha256(body)}


def verify_audit_artifact(document: Mapping[str, Any]) -> None:
    body = {key: value for key, value in document.items() if key != "artifact_sha256"}
    _require(document.get("artifact_sha256") == _semantic_sha256(body), "semantic digest mismatch")
    _require(document.get("schema_version") == SCHEMA_VERSION, "schema mismatch")
    _require(document.get("status") == STATUS, "status boundary changed")
    supersession = document.get("supersession", {})
    expected_superseded = {
        (
            PROVISIONAL_PREDECESSOR_REFERENCE,
            PROVISIONAL_PREDECESSOR_PHYSICAL_SHA256,
            PROVISIONAL_PREDECESSOR_SEMANTIC_SHA256,
        ),
        (
            INVALIDATED_SUCCESSOR_REFERENCE,
            INVALIDATED_SUCCESSOR_PHYSICAL_SHA256,
            INVALIDATED_SUCCESSOR_SEMANTIC_SHA256,
        ),
    }
    superseded_artifacts = supersession.get("superseded_artifacts", [])
    observed_superseded = {
        (
            row.get("reference_path"),
            row.get("physical_sha256"),
            row.get("semantic_sha256"),
        )
        for row in superseded_artifacts
    }
    _require(
        supersession.get("all_predecessors_retained_append_only") is True
        and observed_superseded == expected_superseded
        and len(superseded_artifacts) == 2
        and all(row.get("status") == "superseded_not_for_use" for row in superseded_artifacts),
        "two-predecessor supersession changed",
    )
    stage_lineage = document.get("stage_lineage", {})
    binding_successor = stage_lineage.get("baseline_and_screen_source") == (
        "verified_provisional_predecessor_records"
    )
    if binding_successor:
        _require(
            stage_lineage.get("reused_record_count") == 420
            and stage_lineage.get("excluded_overlapping_confirmation_record_count") == 100
            and stage_lineage.get("excluded_records_used_in_successor") is False,
            "predecessor record-lineage boundary changed",
        )
    boundary = document.get("claim_boundary", {})
    _require(boundary.get("development_only") is True, "development boundary changed")
    _require(
        not any(value for key, value in boundary.items() if key != "development_only"),
        "an authority flag is true",
    )
    _require(document["decision"]["overall_verdict"] == "NO-GO", "NO-GO changed")
    _require(
        document["simulation_contract"]["production_gate_required_but_not_run"][
            "nominal_bootstrap_refits"
        ]
        == 80_000_000,
        "production workload changed",
    )
    two_stage = document["simulation_contract"]["two_stage_plan"]
    screen_indices = set(
        range(
            two_stage["screen_dataset_indices"][0],
            two_stage["screen_dataset_indices"][1] + 1,
        )
    )
    confirmation_indices = set(
        range(
            two_stage["confirmation_dataset_indices"][0],
            two_stage["confirmation_dataset_indices"][1] + 1,
        )
    )
    _require(
        not screen_indices.intersection(confirmation_indices)
        and two_stage["screen_confirmation_dataset_overlap"] == 0,
        "screen and confirmation datasets overlap",
    )
    confirmation_records = [
        row
        for row in document["dataset_records"]
        if row["config_id"] == two_stage["selected_final_config"]
        and row["shift_elo"] == TARGET_ELO
        and row["bootstrap_replicates"] == two_stage["fresh_confirmation_bootstraps_each"]
    ]
    observed_confirmation_indices = {int(row["dataset_index"]) for row in confirmation_records}
    _require(
        len(confirmation_records) == two_stage["fresh_confirmation_datasets"]
        and observed_confirmation_indices == confirmation_indices,
        "confirmation record indices do not match the disjoint contract",
    )
    observed_confirmation_lineages = {row.get("record_lineage") for row in confirmation_records}
    _require(
        observed_confirmation_lineages
        == {
            FRESH_CONFIRMATION_LINEAGE
            if binding_successor
            else "generated_current_nonbinding_test_run"
        },
        "confirmation execution lineage is not bound",
    )
    if binding_successor:
        _require(
            confirmation_indices
            == set(
                range(
                    FRESH_CONFIRMATION_DATASET_START,
                    FRESH_CONFIRMATION_DATASET_END + 1,
                )
            ),
            "binding confirmation indices changed",
        )
        predecessor = _load_provisional_predecessor()
        invalidated_successor = _load_invalidated_successor()
        predecessor_excluded = [
            row
            for row in predecessor["dataset_records"]
            if row["config_id"] == two_stage["selected_final_config"]
            and row["shift_elo"] == TARGET_ELO
            and row["bootstrap_replicates"] == two_stage["fresh_confirmation_bootstraps_each"]
        ]
        excluded_hashes = {row["record_sha256"] for row in predecessor_excluded}
        reused_hashes = {
            row["record_sha256"]
            for row in predecessor["dataset_records"]
            if row["record_sha256"] not in excluded_hashes
        }
        current_hashes = {row["record_sha256"] for row in document["dataset_records"]}
        _require(
            len(reused_hashes) == 420
            and len(excluded_hashes) == 100
            and reused_hashes.issubset(current_hashes)
            and not excluded_hashes.intersection(current_hashes)
            and stage_lineage["reused_record_set_sha256"]
            == _semantic_sha256({"record_sha256s": sorted(reused_hashes)})
            and stage_lineage["excluded_overlapping_confirmation_record_set_sha256"]
            == _semantic_sha256({"record_sha256s": sorted(excluded_hashes)}),
            "predecessor record reuse or exclusion mismatch",
        )
    for record in document["dataset_records"]:
        record_body = {key: value for key, value in record.items() if key != "record_sha256"}
        _require(
            record.get("record_sha256") == _semantic_sha256(record_body),
            "dataset record digest mismatch",
        )
    record_hashes = [str(record["record_sha256"]) for record in document["dataset_records"]]
    _require(
        len(record_hashes) == len(set(record_hashes)),
        "dataset record digests are not unique",
    )
    _require(
        document.get("record_set_sha256")
        == _semantic_sha256({"record_sha256s": sorted(record_hashes)}),
        "record-set digest mismatch",
    )
    layout_by_config = {
        row["config_id"]: row["layout_sha256"] for row in document["layout_commitments"]
    }
    _require(
        all(
            layout_by_config.get(record["config_id"]) == record["layout_sha256"]
            for record in document["dataset_records"]
        ),
        "record layout/config identity does not match a bound layout",
    )
    for condition in document["condition_results"]:
        matching = [
            record
            for record in document["dataset_records"]
            if record["config_id"] == condition["config_id"]
            and record["shift_elo"] == condition["shift_elo"]
            and record["bootstrap_replicates"] == condition["bootstraps_per_dataset"]
        ]
        expected = {key: value for key, value in condition.items() if key != "stage"}
        _require(
            aggregate_condition(matching) == expected,
            "condition aggregate mismatch",
        )

    if binding_successor:
        _require(
            len(document["dataset_records"]) == 520,
            "binding successor does not contain exactly 520 records",
        )
        _require(
            all(
                int(record["dataset_seed"])
                == _dataset_seed(
                    "single_model_50_elo_shift",
                    int(record["dataset_index"]),
                )
                for record in confirmation_records
            ),
            "fresh confirmation seed contract changed",
        )
        predecessor_documents = (
            (
                PROVISIONAL_PREDECESSOR_REFERENCE,
                PROVISIONAL_PREDECESSOR_PHYSICAL_SHA256,
                PROVISIONAL_PREDECESSOR_SEMANTIC_SHA256,
                predecessor,
            ),
            (
                INVALIDATED_SUCCESSOR_REFERENCE,
                INVALIDATED_SUCCESSOR_PHYSICAL_SHA256,
                INVALIDATED_SUCCESSOR_SEMANTIC_SHA256,
                invalidated_successor,
            ),
        )
        predecessor_confirmation_hash_sets: dict[str, set[str]] = {}
        predecessor_retained_hash_sets: dict[str, set[str]] = {}
        predecessor_identity_sets: dict[str, set[tuple[str, str, float, int, int]]] = {}
        for _, _, semantic, predecessor_document in predecessor_documents:
            confirmation_hashes = {
                str(record["record_sha256"])
                for record in predecessor_document["dataset_records"]
                if record["config_id"] == two_stage["selected_final_config"]
                and record["shift_elo"] == TARGET_ELO
                and record["bootstrap_replicates"]
                == two_stage["fresh_confirmation_bootstraps_each"]
            }
            predecessor_confirmation_hash_sets[semantic] = confirmation_hashes
            predecessor_retained_hash_sets[semantic] = {
                str(record["record_sha256"])
                for record in predecessor_document["dataset_records"]
                if record["record_sha256"] not in confirmation_hashes
            }
            predecessor_identity_sets[semantic] = {
                _simulation_identity_key(record)
                for record in predecessor_document["dataset_records"]
            }

        current_hashes = set(record_hashes)
        _require(
            all(
                len(predecessor_confirmation_hash_sets[semantic]) == 100
                and not predecessor_confirmation_hash_sets[semantic].intersection(current_hashes)
                for semantic in predecessor_confirmation_hash_sets
            ),
            "a predecessor confirmation record was reused",
        )
        retained_hashes = predecessor_retained_hash_sets[PROVISIONAL_PREDECESSOR_SEMANTIC_SHA256]
        _require(
            len(retained_hashes) == 420
            and retained_hashes
            == predecessor_retained_hash_sets[INVALIDATED_SUCCESSOR_SEMANTIC_SHA256]
            and retained_hashes.issubset(current_hashes),
            "the common 420-record predecessor lineage changed",
        )
        predecessor_confirmation_hash_union = set().union(
            *predecessor_confirmation_hash_sets.values()
        )
        predecessor_confirmation_identity_union = {
            _simulation_identity_key(record)
            for _, _, _, predecessor_document in predecessor_documents
            for record in _artifact_confirmation_records(predecessor_document)
        }
        _require(
            len(predecessor_confirmation_hash_union) == 200
            and len(predecessor_confirmation_identity_union) == 140
            and stage_lineage.get("excluded_predecessor_confirmation_record_hash_union_count")
            == 200
            and stage_lineage.get("excluded_predecessor_confirmation_record_hash_union_sha256")
            == _semantic_sha256({"record_sha256s": sorted(predecessor_confirmation_hash_union)})
            and stage_lineage.get(
                "excluded_predecessor_confirmation_simulation_identity_union_count"
            )
            == 140
            and stage_lineage.get(
                "excluded_predecessor_confirmation_simulation_identity_union_sha256"
            )
            == _simulation_identity_keys_sha256(predecessor_confirmation_identity_union),
            "predecessor confirmation exclusion union changed",
        )

        prior_union = set().union(*predecessor_identity_sets.values())
        fresh_identities = {_simulation_identity_key(record) for record in confirmation_records}
        _require(
            len(fresh_identities) == 100 and not fresh_identities.intersection(prior_union),
            "fresh confirmation overlaps a predecessor simulation identity",
        )
        _require(
            stage_lineage.get("fresh_confirmation_preflight_completed_before_execution") is True
            and stage_lineage.get("planned_fresh_confirmation_dataset_indices")
            == [FRESH_CONFIRMATION_DATASET_START, FRESH_CONFIRMATION_DATASET_END]
            and stage_lineage.get("planned_fresh_confirmation_record_lineage")
            == FRESH_CONFIRMATION_LINEAGE
            and stage_lineage.get("planned_fresh_confirmation_unique_simulation_identity_count")
            == 100
            and stage_lineage.get("planned_fresh_confirmation_simulation_identity_set_sha256")
            == _simulation_identity_keys_sha256(fresh_identities)
            and stage_lineage.get("planned_fresh_confirmation_overlap_with_prior_observed_union")
            == 0
            and stage_lineage.get("fresh_confirmation_record_count") == 100
            and stage_lineage.get("fresh_confirmation_dataset_indices")
            == [FRESH_CONFIRMATION_DATASET_START, FRESH_CONFIRMATION_DATASET_END]
            and stage_lineage.get("fresh_confirmation_record_lineage") == FRESH_CONFIRMATION_LINEAGE
            and stage_lineage.get("fresh_confirmation_unique_simulation_identity_count") == 100
            and stage_lineage.get("fresh_confirmation_simulation_identity_set_sha256")
            == _simulation_identity_keys_sha256(fresh_identities)
            and stage_lineage.get("fresh_confirmation_overlap_with_prior_observed_union") == 0
            and stage_lineage.get("prior_observed_union_unique_simulation_identity_count")
            == len(prior_union)
            and stage_lineage.get("prior_observed_union_simulation_identity_set_sha256")
            == _simulation_identity_keys_sha256(prior_union),
            "fresh confirmation identity-lineage metadata changed",
        )

        predecessor_audits = stage_lineage.get("prior_observed_predecessors", [])
        audits_by_semantic = {audit.get("semantic_sha256"): audit for audit in predecessor_audits}
        _require(
            len(predecessor_audits) == len(audits_by_semantic) == 2,
            "predecessor identity audit cardinality changed",
        )
        for reference, physical, semantic, predecessor_document in predecessor_documents:
            identity_set = predecessor_identity_sets[semantic]
            identity_audit = audits_by_semantic.get(semantic, {})
            predecessor_confirmation = _artifact_confirmation_records(predecessor_document)
            confirmation_identity_set = {
                _simulation_identity_key(record) for record in predecessor_confirmation
            }
            _require(
                identity_audit.get("reference_path") == reference
                and identity_audit.get("physical_sha256") == physical
                and identity_audit.get("dataset_record_count")
                == len(predecessor_document["dataset_records"])
                and identity_audit.get("unique_simulation_identity_count") == len(identity_set)
                and identity_audit.get("simulation_identity_set_sha256")
                == _simulation_identity_keys_sha256(identity_set)
                and identity_audit.get("confirmation_record_count") == 100
                and identity_audit.get("confirmation_record_set_sha256")
                == _semantic_sha256(
                    {
                        "record_sha256s": sorted(
                            record["record_sha256"] for record in predecessor_confirmation
                        )
                    }
                )
                and identity_audit.get("confirmation_unique_simulation_identity_count") == 100
                and identity_audit.get("confirmation_simulation_identity_set_sha256")
                == _simulation_identity_keys_sha256(confirmation_identity_set)
                and identity_audit.get("planned_fresh_confirmation_overlap_count") == 0
                and identity_audit.get("fresh_confirmation_overlap_count") == 0
                and identity_audit.get("status") == "superseded_not_for_use",
                "predecessor simulation-identity audit mismatch",
            )


def write_audit_artifact(document: Mapping[str, Any], output_dir: Path) -> Path:
    verify_audit_artifact(document)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = str(document["artifact_sha256"])
    destination = output_dir / f"full-k16-arena-resolution-audit-v1-candidate-{digest}.json"
    data = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    if destination.exists():
        _require(destination.read_bytes() == data, "existing content address has different bytes")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".full-k16-resolution-v1-", suffix=".json", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _require(destination.read_bytes() == data, "content-address collision")
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_audit_artifact(workers=args.workers)
    path = write_audit_artifact(artifact, args.output_dir)
    print(path)
    print(artifact["artifact_sha256"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
