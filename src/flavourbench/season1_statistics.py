from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Literal

import numpy as np

from .season1_arena_acceptance import (
    ArenaInferenceAcceptanceError,
    evaluate_arena_inference_acceptance,
    load_arena_inference_policy,
)

ARENA_RANK_VERSION = "0.1.1"
CONTROLLED_BOOTSTRAP_REPLICATES = 5_000
CONTROLLED_BOOTSTRAP_SEED = 20260801
FAMILIES = ("composition", "cookability", "evidence", "substitution")


class StatisticalContractError(ValueError):
    """The analysis input violates the prospective statistical contract."""


@dataclass(frozen=True)
class ArenaObservation:
    observation_id: str
    task_id: str
    family: str
    battle_id: str
    rater_id: str
    model_a: str
    model_b: str
    response_a_id: str
    response_b_id: str
    outcome: float

    def __post_init__(self) -> None:
        if not all(
            (
                self.observation_id,
                self.task_id,
                self.family,
                self.battle_id,
                self.rater_id,
                self.model_a,
                self.model_b,
                self.response_a_id,
                self.response_b_id,
            )
        ):
            raise StatisticalContractError("arena observations require complete cluster identities")
        if (
            self.model_a == self.model_b
            or self.response_a_id == self.response_b_id
            or self.outcome not in {0.0, 0.5, 1.0}
        ):
            raise StatisticalContractError("arena observation has an invalid matchup or outcome")


@dataclass(frozen=True)
class UpliftJudgment:
    judgment_id: str
    task_id: str
    family: str
    battle_id: str
    rater_id: str
    model_id: str
    choice: Literal["epicure_win", "tie", "unaided_win", "both_bad"]

    @property
    def outcome(self) -> float | None:
        return {
            "epicure_win": 1.0,
            "tie": 0.5,
            "unaided_win": 0.0,
            "both_bad": None,
        }[self.choice]


@dataclass(frozen=True)
class UpliftScheduledPair:
    pair_id: str
    task_id: str
    family: str
    model_id: str
    repetition_index: int
    epicure_valid: bool
    unaided_valid: bool

    def __post_init__(self) -> None:
        if not all((self.pair_id, self.task_id, self.family, self.model_id)):
            raise StatisticalContractError("uplift schedule rows require complete identities")
        if self.repetition_index < 1:
            raise StatisticalContractError(
                "provider retries are not scheduled generation repetitions"
            )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def full_roster_components(
    roster: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
) -> list[list[str]]:
    competitors = sorted(set(roster))
    if len(competitors) != len(roster):
        raise StatisticalContractError("frozen model roster contains duplicates")
    roster_set = set(competitors)
    adjacency = {model_id: set() for model_id in competitors}
    for first, second in comparisons:
        if first not in roster_set or second not in roster_set:
            raise StatisticalContractError("comparison contains a model outside the frozen roster")
        adjacency[first].add(second)
        adjacency[second].add(first)
    components: list[list[str]] = []
    unseen = set(competitors)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        unseen -= component
        components.append(sorted(component))
    return sorted(components, key=lambda values: (values[0], len(values)))


def _hierarchical_weights(
    rows: Sequence[ArenaObservation | UpliftJudgment],
    *,
    include_both_bad: bool = False,
) -> np.ndarray:
    active = [
        row
        for row in rows
        if include_both_bad or not isinstance(row, UpliftJudgment) or row.choice != "both_bad"
    ]
    if not active:
        return np.zeros(len(rows), dtype=float)
    families = sorted({row.family for row in active})
    tasks_by_family: dict[str, set[str]] = defaultdict(set)
    battles_by_task: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_by_battle: Counter[tuple[str, str, str]] = Counter()
    for row in active:
        tasks_by_family[row.family].add(row.task_id)
        battles_by_task[(row.family, row.task_id)].add(row.battle_id)
        rows_by_battle[(row.family, row.task_id, row.battle_id)] += 1
    weights = np.zeros(len(rows), dtype=float)
    for index, row in enumerate(rows):
        if row not in active:
            continue
        weights[index] = (
            1.0
            / len(families)
            / len(tasks_by_family[row.family])
            / len(battles_by_task[(row.family, row.task_id)])
            / rows_by_battle[(row.family, row.task_id, row.battle_id)]
        )
    return weights


def _schedule_weights(rows: Sequence[UpliftScheduledPair]) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=float)
    families = sorted({row.family for row in rows})
    tasks_by_family: dict[str, set[str]] = defaultdict(set)
    pairs_by_task: Counter[tuple[str, str]] = Counter()
    for row in rows:
        tasks_by_family[row.family].add(row.task_id)
        pairs_by_task[(row.family, row.task_id)] += 1
    return np.asarray(
        [
            1.0
            / len(families)
            / len(tasks_by_family[row.family])
            / pairs_by_task[(row.family, row.task_id)]
            for row in rows
        ],
        dtype=float,
    )


def _arena_rank_point(
    observations: Sequence[ArenaObservation],
    roster: Sequence[str],
    weights: np.ndarray,
) -> dict[str, float]:
    try:
        import jax.numpy as jnp
        from arena_rank.models.bradley_terry import BradleyTerry
        from arena_rank.utils.data_utils import PairDataset
    except Exception as exc:  # pragma: no cover - production dependency fence
        raise StatisticalContractError("arena-rank is unavailable") from exc
    if version("arena-rank") != ARENA_RANK_VERSION:
        raise StatisticalContractError("official analysis requires arena-rank 0.1.1")
    index = {model_id: position for position, model_id in enumerate(roster)}
    pairs = jnp.asarray(
        [[index[row.model_a], index[row.model_b]] for row in observations],
        dtype=jnp.int32,
    )
    outcomes = jnp.asarray([row.outcome for row in observations], dtype=jnp.float64)
    normalized_weights = weights / weights.sum() * len(weights)
    dataset = PairDataset(
        competitors=list(roster),
        pairs=pairs,
        outcomes=outcomes,
        counts=jnp.ones(len(observations), dtype=jnp.float64),
        weights=jnp.asarray(normalized_weights, dtype=jnp.float64),
        opt_weights=jnp.asarray(normalized_weights, dtype=jnp.float64),
    )
    model = BradleyTerry(len(roster)).fit(dataset)
    logits = np.array(model.params["ratings"], dtype=float, copy=True)
    logits -= logits.mean()
    ratings = 1000.0 + logits * (400.0 / math.log(10.0))
    return {model_id: float(ratings[index[model_id]]) for model_id in roster}


def _numpy_bt_refit(
    observations: Sequence[ArenaObservation],
    roster: Sequence[str],
    weights: np.ndarray,
) -> dict[str, float]:
    index = {model_id: position for position, model_id in enumerate(roster)}
    design = np.zeros((len(observations), len(roster) - 1), dtype=float)
    target = np.asarray([row.outcome for row in observations], dtype=float)
    for row_index, row in enumerate(observations):
        first = index[row.model_a]
        second = index[row.model_b]
        if first < len(roster) - 1:
            design[row_index, first] += 1.0
        if second < len(roster) - 1:
            design[row_index, second] -= 1.0
    beta = np.zeros(len(roster) - 1, dtype=float)
    normalized = weights / weights.sum() * len(weights)
    ridge = 1e-8
    for _ in range(250):
        logits = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        variance = np.maximum(probability * (1.0 - probability), 1e-10)
        hessian = design.T @ ((normalized * variance)[:, None] * design)
        hessian += ridge * np.eye(len(beta))
        gradient = design.T @ (normalized * (target - probability)) - ridge * beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    logits = np.append(beta, 0.0)
    logits -= logits.mean()
    ratings = 1000.0 + logits * (400.0 / math.log(10.0))
    return {model_id: float(ratings[index[model_id]]) for model_id in roster}


def _stratified_task_multipliers(
    rng: np.random.Generator,
    task_family: dict[str, str],
) -> Counter[str]:
    result: Counter[str] = Counter()
    for family in sorted(set(task_family.values())):
        tasks = sorted(task_id for task_id, value in task_family.items() if value == family)
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        result.update(str(task_id) for task_id in sampled)
    return result


def _cluster_multipliers(
    rng: np.random.Generator,
    identities: Sequence[str],
) -> Counter[str]:
    """Return ordinary cluster-bootstrap multiplicities for crossed clusters."""

    clusters = sorted(set(identities))
    if not clusters:
        return Counter()
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    return Counter(str(value) for value in sampled)


def _percentile(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, (0.025, 0.975), method="linear")
    return float(low), float(high)


def analyze_controlled_arena(
    observations: Sequence[ArenaObservation],
    roster: Sequence[str],
    *,
    bootstrap_replicates: int = CONTROLLED_BOOTSTRAP_REPLICATES,
    seed: int = CONTROLLED_BOOTSTRAP_SEED,
    view: str = "all",
    admitted_tasks: Mapping[str, str] | None = None,
    comparison_raters: Mapping[str, Sequence[str]] | None = None,
    postcollection_item_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    ordered_roster = tuple(sorted(roster))
    ordered = tuple(sorted(observations, key=lambda row: row.observation_id))
    if len(ordered_roster) < 2:
        raise StatisticalContractError("arena analysis requires at least two frozen models")
    if bootstrap_replicates < 1:
        raise StatisticalContractError("bootstrap replicate count must be positive")
    task_family: dict[str, str] = {}
    response_coordinates: dict[str, tuple[str, str]] = {}
    for row in ordered:
        if row.task_id in task_family and task_family[row.task_id] != row.family:
            raise StatisticalContractError("task identity maps to multiple families")
        task_family[row.task_id] = row.family
        for response_id, model_id in (
            (row.response_a_id, row.model_a),
            (row.response_b_id, row.model_b),
        ):
            coordinate = (row.task_id, model_id)
            prior = response_coordinates.setdefault(response_id, coordinate)
            if prior != coordinate:
                raise StatisticalContractError(
                    "a response arm is reused across task or model clusters"
                )
    response_row_counts = Counter(
        response_id
        for row in ordered
        for response_id in (row.response_a_id, row.response_b_id)
    )
    response_battle_sets: dict[str, set[str]] = defaultdict(set)
    for row in ordered:
        response_battle_sets[row.response_a_id].add(row.battle_id)
        response_battle_sets[row.response_b_id].add(row.battle_id)
    evidence_units = {
        "raw_preference_rows": len(ordered),
        "unique_task_clusters": len(task_family),
        "unique_response_arms": len(response_coordinates),
        "unique_battles": len({row.battle_id for row in ordered}),
        "unique_raters": len({row.rater_id for row in ordered}),
        "maximum_rows_per_reused_response_arm": max(response_row_counts.values(), default=0),
        "maximum_battles_per_reused_response_arm": max(
            (len(values) for values in response_battle_sets.values()),
            default=0,
        ),
        "primary_resampling_unit": "task_cluster",
        "response_reuse_policy": (
            "response arms are identity-bound to one task and model and remain locked "
            "inside that task in every bootstrap replicate"
        ),
        "comparison_rows_treated_as_independent": False,
        "scalar_effective_sample_size_claimed": False,
    }
    try:
        acceptance_policy = load_arena_inference_policy()
        structural_acceptance = evaluate_arena_inference_acceptance(
            ordered,
            ordered_roster,
            view=view,
            admitted_tasks=admitted_tasks,
            comparison_raters=comparison_raters,
            postcollection_item_audit=postcollection_item_audit,
            policy=acceptance_policy,
        )
    except ArenaInferenceAcceptanceError as exc:
        raise StatisticalContractError(f"arena acceptance policy failed: {exc}") from exc
    components = full_roster_components(
        ordered_roster,
        [(row.model_a, row.model_b) for row in ordered],
    )
    degrees = Counter(
        model_id
        for row in ordered
        for model_id in (row.model_a, row.model_b)
    )
    input_sha256 = _canonical_sha256(
        {
            "roster": ordered_roster,
            "observations": [asdict(row) for row in ordered],
            "bootstrap_replicates": bootstrap_replicates,
            "seed": seed,
            "view": view,
            "admitted_tasks": sorted((admitted_tasks or task_family).items()),
            "comparison_raters": {
                key: sorted(set(values))
                for key, values in sorted((comparison_raters or {}).items())
            },
            "postcollection_item_audit_sha256": (
                postcollection_item_audit.get("artifact_sha256")
                if postcollection_item_audit is not None
                else None
            ),
            "acceptance_policy_sha256": acceptance_policy["artifact_sha256"],
        }
    )
    if structural_acceptance["status"] != "pass":
        return {
            "schema_version": "flavourbench-controlled-arena-analysis-v1",
            "input_sha256": input_sha256,
            "ranking_status": structural_acceptance["withholding_status"],
            "comparison_components": components,
            "evidence_units": evidence_units,
            "bootstrap_seed": seed,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_connected_replicates": None,
            "bootstrap_successful_replicates": None,
            "bootstrap_connected_rate": None,
            "family_bootstrap_connected_rates": {
                family: None
                for family in (FAMILIES if view == "all" else (view,))
            },
            "pairwise_win_probability": {
                first: {second: None for second in ordered_roster if second != first}
                for first in ordered_roster
            },
            "pairwise_win_probability_interval": {
                first: {second: None for second in ordered_roster if second != first}
                for first in ordered_roster
            },
            "pairwise_reporting_support": structural_acceptance[
                "pairwise_reporting_support"
            ],
            "rows": [
                {
                    "competitor_id": model_id,
                    "rating": None,
                    "rating_lower": None,
                    "rating_upper": None,
                    "comparison_degree": degrees[model_id],
                    "unique_tasks": len(
                        {
                            row.task_id
                            for row in ordered
                            if model_id in {row.model_a, row.model_b}
                        }
                    ),
                    "unique_battles": len(
                        {
                            row.battle_id
                            for row in ordered
                            if model_id in {row.model_a, row.model_b}
                        }
                    ),
                }
                for model_id in ordered_roster
            ],
            "statistical_acceptance": structural_acceptance,
        }

    base_weights = _hierarchical_weights(ordered)
    point = _arena_rank_point(ordered, ordered_roster, base_weights)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {model_id: [] for model_id in ordered_roster}
    ranks: dict[str, list[int]] = {model_id: [] for model_id in ordered_roster}
    pairwise: dict[tuple[str, str], list[float]] = {
        (first, second): []
        for first in ordered_roster
        for second in ordered_roster
        if first != second
    }
    connected = 0
    successful = 0
    bootstrap_families = FAMILIES if view == "all" else (view,)
    family_connected: Counter[str] = Counter()
    for _ in range(bootstrap_replicates):
        task_multipliers = _stratified_task_multipliers(rng, task_family)
        rater_multipliers = _cluster_multipliers(
            rng,
            [row.rater_id for row in ordered],
        )
        weights = np.asarray(
            [
                base_weights[index]
                * task_multipliers[row.task_id]
                * rater_multipliers[row.rater_id]
                for index, row in enumerate(ordered)
            ],
            dtype=float,
        )
        active_edges = [
            (row.model_a, row.model_b)
            for index, row in enumerate(ordered)
            if weights[index] > 0
        ]
        for family in bootstrap_families:
            family_edges = [
                (row.model_a, row.model_b)
                for index, row in enumerate(ordered)
                if row.family == family and weights[index] > 0
            ]
            if len(full_roster_components(ordered_roster, family_edges)) == 1:
                family_connected[family] += 1
        if len(full_roster_components(ordered_roster, active_edges)) != 1:
            continue
        connected += 1
        try:
            fitted = _numpy_bt_refit(ordered, ordered_roster, weights)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            continue
        successful += 1
        ordered_rank = sorted(ordered_roster, key=lambda value: (-fitted[value], value))
        rank_index = {model_id: index + 1 for index, model_id in enumerate(ordered_rank)}
        for model_id in ordered_roster:
            samples[model_id].append(fitted[model_id])
            ranks[model_id].append(rank_index[model_id])
        for first, second in pairwise:
            probability = 1.0 / (
                1.0 + 10.0 ** ((fitted[second] - fitted[first]) / 400.0)
            )
            pairwise[(first, second)].append(probability)
    connected_rate = connected / bootstrap_replicates
    family_connected_rates = {
        family: family_connected[family] / bootstrap_replicates
        for family in bootstrap_families
    }
    try:
        final_acceptance = evaluate_arena_inference_acceptance(
            ordered,
            ordered_roster,
            view=view,
            admitted_tasks=admitted_tasks,
            comparison_raters=comparison_raters,
            postcollection_item_audit=postcollection_item_audit,
            bootstrap_connected_rate=connected_rate,
            family_bootstrap_connected_rates=family_connected_rates,
            include_bootstrap=True,
            policy=acceptance_policy,
        )
    except ArenaInferenceAcceptanceError as exc:
        raise StatisticalContractError(f"arena acceptance policy failed: {exc}") from exc
    if successful != connected or successful == 0:
        final_acceptance["status"] = "fail"
        final_acceptance["deficits"].append(
            {
                "code": "bootstrap_refit_failures",
                "connected_replicates": connected,
                "successful_replicates": successful,
                "required_successful_replicates": connected,
            }
        )
    acceptance = final_acceptance["status"] == "pass"
    rows: list[dict[str, object]] = []
    for model_id in sorted(ordered_roster, key=lambda value: (-point[value], value)):
        values = np.asarray(samples[model_id], dtype=float)
        low, high = _percentile(values) if acceptance else (None, None)
        rank_values = np.asarray(ranks[model_id], dtype=int)
        rows.append(
            {
                "competitor_id": model_id,
                "rating": round(point[model_id], 6) if acceptance else None,
                "rating_lower": round(low, 6) if low is not None else None,
                "rating_upper": round(high, 6) if high is not None else None,
                "comparison_degree": degrees[model_id],
                "unique_tasks": len({
                    row.task_id
                    for row in ordered
                    if model_id in {row.model_a, row.model_b}
                }),
                "unique_battles": len({
                    row.battle_id
                    for row in ordered
                    if model_id in {row.model_a, row.model_b}
                }),
                "unique_response_arms": len(
                    {
                        response_id
                        for row in ordered
                        if model_id in {row.model_a, row.model_b}
                        for response_id, response_model in (
                            (row.response_a_id, row.model_a),
                            (row.response_b_id, row.model_b),
                        )
                        if response_model == model_id
                    }
                ),
                "unique_raters": len({
                    row.rater_id
                    for row in ordered
                    if model_id in {row.model_a, row.model_b}
                }),
                "rank_one_probability": (
                    round(float(np.mean(rank_values == 1)), 6)
                    if acceptance
                    else None
                ),
                "top_three_probability": (
                    round(float(np.mean(rank_values <= min(3, len(ordered_roster)))), 6)
                    if acceptance
                    else None
                ),
            }
        )
    pairwise_payload = {
        first: {
            second: (
                round(float(np.mean(pairwise[(first, second)])), 6)
                if acceptance
                else None
            )
            for second in ordered_roster
            if second != first
        }
        for first in ordered_roster
    }
    pairwise_interval_payload: dict[str, dict[str, dict[str, float] | None]] = {
        first: {} for first in ordered_roster
    }
    pairwise_support = final_acceptance["pairwise_reporting_support"]
    for first in ordered_roster:
        for second in ordered_roster:
            if first == second:
                continue
            support = pairwise_support[first][second]
            values = np.asarray(pairwise[(first, second)], dtype=float)
            if acceptance and support["interval_reportable"] and len(values):
                low, high = _percentile(values)
                pairwise_interval_payload[first][second] = {
                    "lower": round(low, 6),
                    "upper": round(high, 6),
                }
            else:
                pairwise_interval_payload[first][second] = None
    return {
        "schema_version": "flavourbench-controlled-arena-analysis-v1",
        "input_sha256": input_sha256,
        "ranking_status": (
            "estimated" if acceptance else final_acceptance["withholding_status"]
        ),
        "point_estimator": "arena-rank==0.1.1 weighted Bradley-Terry",
        "interval_method": (
            "family-stratified task-cluster by crossed rater-cluster bootstrap"
        ),
        "dependence_handling": (
            "family-standardized hierarchical weights; crossed task and rater cluster "
            "resampling; shared response arms locked within task clusters"
        ),
        "bootstrap_seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_connected_replicates": connected,
        "bootstrap_successful_replicates": successful,
        "bootstrap_connected_rate": round(connected_rate, 6),
        "family_bootstrap_connected_rates": {
            family: round(rate, 6) for family, rate in family_connected_rates.items()
        },
        "comparison_components": components,
        "evidence_units": evidence_units,
        "pairwise_win_probability": pairwise_payload,
        "pairwise_win_probability_interval": pairwise_interval_payload,
        "pairwise_reporting_support": pairwise_support,
        "rows": rows,
        "statistical_acceptance": final_acceptance,
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total > 0 else None


def _uplift_point(
    judgments: Sequence[UpliftJudgment],
    schedule: Sequence[UpliftScheduledPair],
    model_id: str,
    *,
    task_multipliers: Counter[str] | None = None,
    rater_multipliers: Counter[str] | None = None,
) -> dict[str, object]:
    model_judgments = [row for row in judgments if row.model_id == model_id]
    preference_rows = [row for row in model_judgments if row.outcome is not None]
    preference_weights = _hierarchical_weights(preference_rows)
    if task_multipliers is not None:
        preference_weights *= np.asarray(
            [task_multipliers[row.task_id] for row in preference_rows], dtype=float
        )
    if rater_multipliers is not None:
        preference_weights *= np.asarray(
            [rater_multipliers[row.rater_id] for row in preference_rows], dtype=float
        )
    preference_values = np.asarray([float(row.outcome) for row in preference_rows], dtype=float)
    preference = _weighted_mean(preference_values, preference_weights)
    family_preference = {
        family: _weighted_mean(
            np.asarray(
                [float(row.outcome) for row in preference_rows if row.family == family],
                dtype=float,
            ),
            preference_weights[
                np.asarray([row.family == family for row in preference_rows], dtype=bool)
            ],
        )
        for family in FAMILIES
    }

    model_schedule = [row for row in schedule if row.model_id == model_id]
    schedule_weights = _schedule_weights(model_schedule)
    if task_multipliers is not None:
        schedule_weights *= np.asarray(
            [task_multipliers[row.task_id] for row in model_schedule], dtype=float
        )
    completion_values = np.asarray(
        [int(row.epicure_valid) - int(row.unaided_valid) for row in model_schedule],
        dtype=float,
    )
    completion_delta = _weighted_mean(completion_values, schedule_weights)

    judgments_by_pair: dict[str, list[UpliftJudgment]] = defaultdict(list)
    for row in model_judgments:
        judgments_by_pair[row.battle_id].append(row)
    observed: list[tuple[UpliftScheduledPair, float]] = []
    missing: list[UpliftScheduledPair] = []
    observed_both_bad: list[UpliftScheduledPair] = []
    for pair in model_schedule:
        pair_judgments = judgments_by_pair.get(pair.pair_id, [])
        nonbad = [float(row.outcome) for row in pair_judgments if row.outcome is not None]
        if pair.epicure_valid and pair.unaided_valid and nonbad:
            observed.append((pair, float(np.mean(nonbad))))
        elif pair.epicure_valid and pair.unaided_valid and pair_judgments and all(
            row.choice == "both_bad" for row in pair_judgments
        ):
            observed_both_bad.append(pair)
        else:
            missing.append(pair)

    bound_rows = [pair for pair, _value in observed] + missing
    bound_weights = _schedule_weights(bound_rows)
    if task_multipliers is not None:
        bound_weights *= np.asarray(
            [task_multipliers[row.task_id] for row in bound_rows], dtype=float
        )
    observed_values = np.asarray(
        [value for _pair, value in observed] + [0.0 for _pair in missing],
        dtype=float,
    )
    missing_mask = np.asarray(
        [False for _pair, _value in observed] + [True for _pair in missing],
        dtype=bool,
    )
    lower = _weighted_mean(observed_values, bound_weights)
    upper_values = observed_values.copy()
    upper_values[missing_mask] = 1.0
    upper = _weighted_mean(upper_values, bound_weights)
    return {
        "preference": preference,
        "family_preference": family_preference,
        "completion_delta": completion_delta,
        "missingness_lower": lower,
        "missingness_upper": upper,
        "scheduled_pairs": len(model_schedule),
        "both_valid_pairs": sum(
            row.epicure_valid and row.unaided_valid for row in model_schedule
        ),
        "epicure_only_valid_pairs": sum(
            row.epicure_valid and not row.unaided_valid for row in model_schedule
        ),
        "unaided_only_valid_pairs": sum(
            row.unaided_valid and not row.epicure_valid for row in model_schedule
        ),
        "neither_valid_pairs": sum(
            not row.epicure_valid and not row.unaided_valid for row in model_schedule
        ),
        "judged_preference_pairs": len(observed),
        "judged_both_bad_pairs": len(observed_both_bad),
        "missing_preference_pairs": len(missing),
    }


def analyze_controlled_uplift(
    judgments: Sequence[UpliftJudgment],
    schedule: Sequence[UpliftScheduledPair],
    roster: Sequence[str],
    *,
    bootstrap_replicates: int = CONTROLLED_BOOTSTRAP_REPLICATES,
    seed: int = CONTROLLED_BOOTSTRAP_SEED,
) -> dict[str, object]:
    ordered_roster = tuple(sorted(roster))
    ordered_judgments = tuple(sorted(judgments, key=lambda row: row.judgment_id))
    ordered_schedule = tuple(sorted(schedule, key=lambda row: row.pair_id))
    if bootstrap_replicates < 1 or not ordered_schedule:
        raise StatisticalContractError("uplift analysis requires a schedule and bootstrap")
    if any(row.model_id not in ordered_roster for row in (*ordered_judgments, *ordered_schedule)):
        raise StatisticalContractError("uplift input contains a model outside the frozen roster")
    schedule_pairs = {row.pair_id for row in ordered_schedule}
    if any(row.battle_id not in schedule_pairs for row in ordered_judgments):
        raise StatisticalContractError(
            "uplift judgment is not bound to the intent-to-evaluate schedule"
        )
    task_family: dict[str, str] = {}
    for row in ordered_schedule:
        if row.task_id in task_family and task_family[row.task_id] != row.family:
            raise StatisticalContractError("scheduled task identity maps to multiple families")
        task_family[row.task_id] = row.family

    points = {
        model_id: _uplift_point(ordered_judgments, ordered_schedule, model_id)
        for model_id in ordered_roster
    }
    rng = np.random.default_rng(seed)
    preference_samples: dict[str, list[float]] = defaultdict(list)
    completion_samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap_replicates):
        multipliers = _stratified_task_multipliers(rng, task_family)
        rater_multipliers = _cluster_multipliers(
            rng,
            [row.rater_id for row in ordered_judgments],
        )
        for model_id in ordered_roster:
            estimate = _uplift_point(
                ordered_judgments,
                ordered_schedule,
                model_id,
                task_multipliers=multipliers,
                rater_multipliers=rater_multipliers,
            )
            if estimate["preference"] is not None:
                preference_samples[model_id].append(float(estimate["preference"]))
            if estimate["completion_delta"] is not None:
                completion_samples[model_id].append(float(estimate["completion_delta"]))

    rows: list[dict[str, object]] = []
    accepted = True
    for model_id in ordered_roster:
        point = points[model_id]
        preference_values = np.asarray(preference_samples[model_id], dtype=float)
        completion_values = np.asarray(completion_samples[model_id], dtype=float)
        family_counts = Counter(
            row.family for row in ordered_schedule if row.model_id == model_id
        )
        model_accepted = bool(
            point["preference"] is not None
            and len(preference_values) == bootstrap_replicates
            and len(completion_values) == bootstrap_replicates
            and int(point["scheduled_pairs"]) >= 200
            and all(family_counts[family] >= 50 for family in FAMILIES)
            and all(point["family_preference"][family] is not None for family in FAMILIES)
        )
        accepted &= model_accepted
        preference_interval = _percentile(preference_values) if model_accepted else (None, None)
        completion_interval = _percentile(completion_values) if model_accepted else (None, None)
        rows.append(
            {
                "competitor_id": model_id,
                "epicure_win_share": (
                    round(float(point["preference"]), 6)
                    if point["preference"] is not None
                    else None
                ),
                "interval_lower": (
                    round(preference_interval[0], 6)
                    if preference_interval[0] is not None
                    else None
                ),
                "interval_upper": (
                    round(preference_interval[1], 6)
                    if preference_interval[1] is not None
                    else None
                ),
                "family_epicure_win_share": {
                    family: (
                        round(float(point["family_preference"][family]), 6)
                        if point["family_preference"][family] is not None
                        else None
                    )
                    for family in FAMILIES
                },
                "paired_completion_delta": (
                    round(float(point["completion_delta"]), 6)
                    if point["completion_delta"] is not None
                    else None
                ),
                "completion_interval_lower": (
                    round(completion_interval[0], 6)
                    if completion_interval[0] is not None
                    else None
                ),
                "completion_interval_upper": (
                    round(completion_interval[1], 6)
                    if completion_interval[1] is not None
                    else None
                ),
                "missingness_lower": (
                    round(float(point["missingness_lower"]), 6)
                    if point["missingness_lower"] is not None
                    else None
                ),
                "missingness_upper": (
                    round(float(point["missingness_upper"]), 6)
                    if point["missingness_upper"] is not None
                    else None
                ),
                **{
                    key: point[key]
                    for key in (
                        "scheduled_pairs",
                        "both_valid_pairs",
                        "epicure_only_valid_pairs",
                        "unaided_only_valid_pairs",
                        "neither_valid_pairs",
                        "judged_preference_pairs",
                        "judged_both_bad_pairs",
                        "missing_preference_pairs",
                    )
                },
                "provisional": not model_accepted,
            }
        )
    input_sha256 = _canonical_sha256(
        {
            "roster": ordered_roster,
            "judgments": [asdict(row) for row in ordered_judgments],
            "schedule": [asdict(row) for row in ordered_schedule],
            "bootstrap_replicates": bootstrap_replicates,
            "seed": seed,
        }
    )
    return {
        "schema_version": "flavourbench-controlled-uplift-analysis-v1",
        "input_sha256": input_sha256,
        "ranking_status": "estimated" if accepted else "withheld_underpowered_or_incomplete",
        "estimand": "family-standardized conditional preference plus paired-completion ITT",
        "interval_method": (
            "family-stratified task-cluster by crossed rater-cluster bootstrap"
        ),
        "dependence_handling": (
            "matched response pair is the battle unit; repeated ballots are weighted "
            "within battle; task and rater identities are resampled as crossed clusters"
        ),
        "bootstrap_seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "evidence_units": {
            "raw_judgment_rows": len(ordered_judgments),
            "unique_task_clusters": len(task_family),
            "unique_scheduled_pairs": len(ordered_schedule),
            "unique_raters": len({row.rater_id for row in ordered_judgments}),
            "comparison_rows_treated_as_independent": False,
            "primary_resampling_units": ["task_cluster", "rater_cluster"],
        },
        "rows": sorted(
            rows,
            key=lambda row: (
                -(row["epicure_win_share"] if row["epicure_win_share"] is not None else -1),
                str(row["competitor_id"]),
            ),
        ),
        "statistical_acceptance": {
            "status": "pass" if accepted else "fail",
            "minimum_pairs_per_model": 200,
            "minimum_pairs_per_model_family": 50,
        },
    }
