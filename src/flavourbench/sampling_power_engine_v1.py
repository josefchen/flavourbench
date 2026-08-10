"""Deterministic design validation for an abstract human-judgment sampling frame.

The engine consumes coordinates only.  It never reads judgments, provider state,
reviewer identities, a database, or the network.  Ratings created here are
explicitly simulated and cannot be imported as FlavourBench evidence.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

FAMILIES = ("substitution", "composition", "cookability", "evidence")
TIE_VALUES = np.asarray((0.0, 0.5, 1.0), dtype=float)
# For the bound K14 frame, this panel-centred focal merit produces a 50 Elo
# focal-versus-each-other-model probability contrast under the BT scale.
K14_FOCAL_MERIT_FOR_50_ELO = 0.06635860902206844


class SamplingPowerEngineError(ValueError):
    """The coordinate frame, scenario, or finite-cluster analysis is invalid."""


@dataclass(frozen=True)
class FrameSpec:
    """Numeric view of a materialized sampling frame, parameterized by roster size."""

    model_ids: tuple[str, ...]
    task_ordinals: tuple[int, ...]
    task_family: np.ndarray
    arena_task: np.ndarray
    arena_family: np.ndarray
    arena_first: np.ndarray
    arena_second: np.ndarray
    arena_comparison_ids: tuple[str, ...]
    uplift_task: np.ndarray
    uplift_family: np.ndarray
    uplift_model: np.ndarray
    uplift_comparison_ids: tuple[str, ...]
    repeat_track: np.ndarray
    repeat_task: np.ndarray
    repeat_source_track_index: np.ndarray
    repeat_source_rater_slot: np.ndarray
    primary_slots: int

    @property
    def roster_size(self) -> int:
        return len(self.model_ids)

    @property
    def task_count(self) -> int:
        return len(self.task_ordinals)


@dataclass(frozen=True)
class Scenario:
    """Prespecified data-generating scenario in half-win-share units."""

    scenario_id: str
    core: bool
    null: bool
    uplift_family_effects: tuple[float, float, float, float]
    arena_family_shifts: tuple[float, float, float, float]
    tie_probability: float = 0.20
    task_sd: float = 0.05
    comparison_sd: float = 0.025
    rater_sd: float = 0.015
    rater_count: int = 32
    response_missing_rate: float = 0.0
    rating_missing_rate: float = 0.0
    outcome_dependent_missing_slope: float = 0.0
    dropout_fraction: float = 0.0
    repeat_copy_probability: float = 0.72

    @property
    def overall_uplift(self) -> float:
        return float(sum(self.uplift_family_effects) / len(self.uplift_family_effects))

    def validate(self) -> None:
        if not self.scenario_id:
            raise SamplingPowerEngineError("scenario_id is required")
        if len(self.uplift_family_effects) != len(FAMILIES):
            raise SamplingPowerEngineError("one uplift effect is required per family")
        if len(self.arena_family_shifts) != len(FAMILIES):
            raise SamplingPowerEngineError("one arena shift is required per family")
        if not 0.0 <= self.tie_probability < 1.0:
            raise SamplingPowerEngineError("tie_probability must lie in [0, 1)")
        if self.rater_count < 2:
            raise SamplingPowerEngineError("at least two simulated raters are required")
        for value in (
            self.response_missing_rate,
            self.rating_missing_rate,
            self.dropout_fraction,
            self.repeat_copy_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise SamplingPowerEngineError("probabilities must lie in [0, 1]")
        if self.null and (
            not math.isclose(self.overall_uplift, 0.0, abs_tol=1e-12)
            or any(
                not math.isclose(value, 0.0, abs_tol=1e-12) for value in self.arena_family_shifts
            )
        ):
            raise SamplingPowerEngineError("null scenario has a non-null primary effect")


DEFAULT_SCENARIOS = (
    Scenario(
        "null_complete_moderate_dependence",
        core=True,
        null=True,
        uplift_family_effects=(0.0, 0.0, 0.0, 0.0),
        arena_family_shifts=(0.0, 0.0, 0.0, 0.0),
    ),
    Scenario(
        "null_high_task_rater_dependence",
        core=True,
        null=True,
        uplift_family_effects=(0.0, 0.0, 0.0, 0.0),
        arena_family_shifts=(0.0, 0.0, 0.0, 0.0),
        task_sd=0.10,
        comparison_sd=0.05,
        rater_sd=0.04,
        rater_count=8,
    ),
    Scenario(
        "calibrated_0_08_complete",
        core=True,
        null=False,
        uplift_family_effects=(0.08, 0.08, 0.08, 0.08),
        arena_family_shifts=(K14_FOCAL_MERIT_FOR_50_ELO,) * 4,
    ),
    Scenario(
        "calibrated_0_08_high_dependence",
        core=True,
        null=False,
        uplift_family_effects=(0.08, 0.08, 0.08, 0.08),
        arena_family_shifts=(K14_FOCAL_MERIT_FOR_50_ELO,) * 4,
        task_sd=0.10,
        comparison_sd=0.05,
        rater_sd=0.04,
        rater_count=8,
    ),
    Scenario(
        "family_crossover_zero_overall",
        core=True,
        null=False,
        uplift_family_effects=(0.08, 0.08, -0.08, -0.08),
        arena_family_shifts=(
            K14_FOCAL_MERIT_FOR_50_ELO,
            K14_FOCAL_MERIT_FOR_50_ELO,
            -K14_FOCAL_MERIT_FOR_50_ELO,
            -K14_FOCAL_MERIT_FOR_50_ELO,
        ),
    ),
    Scenario(
        "mcar_5pct_responses_and_ratings",
        core=True,
        null=False,
        uplift_family_effects=(0.08, 0.08, 0.08, 0.08),
        arena_family_shifts=(K14_FOCAL_MERIT_FOR_50_ELO,) * 4,
        response_missing_rate=0.05,
        rating_missing_rate=0.05,
    ),
    Scenario(
        "outcome_dependent_missingness",
        core=True,
        null=False,
        uplift_family_effects=(0.08, 0.08, 0.08, 0.08),
        arena_family_shifts=(K14_FOCAL_MERIT_FOR_50_ELO,) * 4,
        response_missing_rate=0.03,
        rating_missing_rate=0.03,
        outcome_dependent_missing_slope=0.16,
    ),
    Scenario(
        "plausible_rater_dropout",
        core=True,
        null=False,
        uplift_family_effects=(0.08, 0.08, 0.08, 0.08),
        arena_family_shifts=(K14_FOCAL_MERIT_FOR_50_ELO,) * 4,
        rating_missing_rate=0.02,
        rater_sd=0.035,
        rater_count=12,
        dropout_fraction=0.25,
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SamplingPowerEngineError(message)


def _index_rows(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, row in enumerate(rows):
        value = str(row[key])
        _require(value not in result, f"duplicate {key}: {value}")
        result[value] = index
    return result


def build_frame_spec(materialized: Mapping[str, Sequence[Mapping[str, Any]]]) -> FrameSpec:
    """Validate and convert any compatible materialized roster/frame."""

    arena = list(materialized.get("arena_comparisons", ()))
    uplift = list(materialized.get("uplift_comparisons", ()))
    slots = list(materialized.get("primary_judgment_slots", ()))
    repeats = list(materialized.get("concealed_repeat_presentations", ()))
    _require(arena and uplift and slots and repeats, "materialized frame is incomplete")
    families = {str(row["family"]) for row in (*arena, *uplift)}
    _require(families == set(FAMILIES), "frame families changed")
    task_ordinals = tuple(sorted({int(row["design_slot_ordinal"]) for row in (*arena, *uplift)}))
    task_index = {ordinal: index for index, ordinal in enumerate(task_ordinals)}
    task_family_by_ordinal: dict[int, str] = {}
    for row in (*arena, *uplift):
        ordinal = int(row["design_slot_ordinal"])
        family = str(row["family"])
        prior = task_family_by_ordinal.setdefault(ordinal, family)
        _require(prior == family, "one task ordinal maps to multiple families")
    family_index = {family: index for index, family in enumerate(FAMILIES)}
    task_family = np.asarray(
        [family_index[task_family_by_ordinal[value]] for value in task_ordinals], dtype=int
    )
    models = sorted(
        {
            *(str(model) for row in arena for model in row["model_ids"]),
            *(str(row["model_id"]) for row in uplift),
        }
    )
    model_index = {model: index for index, model in enumerate(models)}
    arena_index = _index_rows(arena, "comparison_id")
    uplift_index = _index_rows(uplift, "comparison_id")
    slot_index = _index_rows(slots, "judgment_slot_id")
    _require(len(slots) == 2 * (len(arena) + len(uplift)), "two primary slots are required")
    for row in (*arena, *uplift):
        ids = tuple(str(value) for value in row["judgment_slot_ids"])
        _require(len(ids) == 2 and len(set(ids)) == 2, "two distinct slot IDs are required")
        _require(all(value in slot_index for value in ids), "comparison slot is missing")

    repeat_track: list[int] = []
    repeat_source: list[int] = []
    repeat_rater: list[int] = []
    repeat_task: list[int] = []
    for row in repeats:
        track = str(row["track"])
        comparison_id = str(row["source_comparison_id"])
        source_slot = slots[slot_index[str(row["source_judgment_slot_id"])]]
        rater_slot = int(source_slot["rater_slot"]) - 1
        _require(rater_slot in {0, 1}, "repeat source rater slot changed")
        _require(bool(row.get("same_rater_as_source_required")), "repeat rater changed")
        if track == "model_arena":
            _require(comparison_id in arena_index, "repeat arena source is missing")
            repeat_track.append(0)
            repeat_source.append(arena_index[comparison_id])
        elif track == "epicure_uplift":
            _require(comparison_id in uplift_index, "repeat uplift source is missing")
            repeat_track.append(1)
            repeat_source.append(uplift_index[comparison_id])
        else:
            raise SamplingPowerEngineError("unknown repeat track")
        repeat_rater.append(rater_slot)
        repeat_task.append(task_index[int(row["design_slot_ordinal"])])

    return FrameSpec(
        model_ids=tuple(models),
        task_ordinals=task_ordinals,
        task_family=task_family,
        arena_task=np.asarray(
            [task_index[int(row["design_slot_ordinal"])] for row in arena], dtype=int
        ),
        arena_family=np.asarray([family_index[str(row["family"])] for row in arena], dtype=int),
        arena_first=np.asarray([model_index[str(row["model_ids"][0])] for row in arena]),
        arena_second=np.asarray([model_index[str(row["model_ids"][1])] for row in arena]),
        arena_comparison_ids=tuple(str(row["comparison_id"]) for row in arena),
        uplift_task=np.asarray(
            [task_index[int(row["design_slot_ordinal"])] for row in uplift], dtype=int
        ),
        uplift_family=np.asarray([family_index[str(row["family"])] for row in uplift], dtype=int),
        uplift_model=np.asarray([model_index[str(row["model_id"])] for row in uplift]),
        uplift_comparison_ids=tuple(str(row["comparison_id"]) for row in uplift),
        repeat_track=np.asarray(repeat_track, dtype=int),
        repeat_task=np.asarray(repeat_task, dtype=int),
        repeat_source_track_index=np.asarray(repeat_source, dtype=int),
        repeat_source_rater_slot=np.asarray(repeat_rater, dtype=int),
        primary_slots=len(slots),
    )


def frame_diagnostics(frame: FrameSpec) -> dict[str, Any]:
    """Return exact counts and connectivity without outcomes."""

    arena_by_task = Counter(frame.arena_task.tolist())
    uplift_by_task = Counter(frame.uplift_task.tolist())
    family_tasks = Counter(frame.task_family.tolist())
    repeat_categories = Counter(
        (int(track), int(slot))
        for track, slot in zip(frame.repeat_track, frame.repeat_source_rater_slot, strict=True)
    )
    pair_tasks: dict[tuple[int, int], set[int]] = {
        (first, second): set()
        for first in range(frame.roster_size)
        for second in range(first + 1, frame.roster_size)
    }
    model_tasks: dict[tuple[int, int], set[int]] = {
        (model, family): set()
        for model in range(frame.roster_size)
        for family in range(len(FAMILIES))
    }
    global_model_comparisons: Counter[int] = Counter()
    family_model_comparisons: Counter[tuple[int, int]] = Counter()
    for task, family, first, second in zip(
        frame.arena_task,
        frame.arena_family,
        frame.arena_first,
        frame.arena_second,
        strict=True,
    ):
        pair_tasks[tuple(sorted((int(first), int(second))))].add(int(task))
        for model in (int(first), int(second)):
            model_tasks[model, int(family)].add(int(task))
            global_model_comparisons[model] += 1
            family_model_comparisons[model, int(family)] += 1
    return {
        "roster_size": frame.roster_size,
        "tasks": frame.task_count,
        "tasks_per_family": {
            FAMILIES[index]: family_tasks[index] for index in range(len(FAMILIES))
        },
        "arena_comparisons": len(frame.arena_task),
        "uplift_comparisons": len(frame.uplift_task),
        "comparisons_per_task": {
            "arena_min": min(arena_by_task.values()),
            "arena_max": max(arena_by_task.values()),
            "uplift_min": min(uplift_by_task.values()),
            "uplift_max": max(uplift_by_task.values()),
        },
        "ratings_per_comparison": 2,
        "primary_rating_slots": frame.primary_slots,
        "repeat_presentations": len(frame.repeat_track),
        "repeat_rate_of_primary_slots": round(len(frame.repeat_track) / frame.primary_slots, 8),
        "repeat_categories": {
            f"{('arena', 'uplift')[track]}_slot_{slot + 1}": repeat_categories[track, slot]
            for track in range(2)
            for slot in range(2)
        },
        "complete_arena_graph_connected": _connected(
            frame.roster_size,
            frame.arena_first,
            frame.arena_second,
            np.ones(len(frame.arena_task), bool),
        ),
        "complete_arena_family_graphs_connected": {
            family: _connected(
                frame.roster_size,
                frame.arena_first,
                frame.arena_second,
                frame.arena_family == family_index,
            )
            for family_index, family in enumerate(FAMILIES)
        },
        "arena_support": {
            "minimum_shared_task_clusters_per_pair": min(map(len, pair_tasks.values())),
            "maximum_shared_task_clusters_per_pair": max(map(len, pair_tasks.values())),
            "minimum_global_comparisons_per_model": min(global_model_comparisons.values()),
            "maximum_global_comparisons_per_model": max(global_model_comparisons.values()),
            "minimum_family_comparisons_per_model": min(family_model_comparisons.values()),
            "maximum_family_comparisons_per_model": max(family_model_comparisons.values()),
            "minimum_unique_task_clusters_per_model_family": min(map(len, model_tasks.values())),
            "maximum_unique_task_clusters_per_model_family": max(map(len, model_tasks.values())),
        },
    }


def _assignment(ids: Sequence[str], rater_count: int) -> np.ndarray:
    assigned = np.empty((len(ids), 2), dtype=int)
    for index, identifier in enumerate(ids):
        digest = hashlib.sha256(identifier.encode()).digest()
        first = int.from_bytes(digest[:8], "big") % rater_count
        second = (first + 1 + int.from_bytes(digest[8:16], "big") % (rater_count - 1)) % rater_count
        assigned[index] = (first, second)
    return assigned


def _draw_trinary(
    mean: np.ndarray, tie_probability: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(mean, 0.025, 0.975)
    tie = np.minimum(tie_probability, 1.9 * np.minimum(clipped, 1.0 - clipped))
    loss = 1.0 - clipped - 0.5 * tie
    uniform = rng.random(clipped.shape)
    outcome = np.where(uniform < loss, 0.0, np.where(uniform < loss + tie, 0.5, 1.0))
    probabilities = np.stack((loss, tie, clipped - 0.5 * tie), axis=-1)
    return outcome, probabilities


def _comparison_means(values: np.ndarray) -> np.ndarray:
    observed = np.isfinite(values)
    count = observed.sum(axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def _stratified_t_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    group_count: int = 4,
) -> tuple[float, float, float, float, float] | None:
    means: list[float] = []
    variance_terms: list[float] = []
    dfs: list[int] = []
    for group in range(group_count):
        subset = values[(groups == group) & np.isfinite(values)]
        if len(subset) < 2:
            return None
        means.append(float(subset.mean()))
        variance_terms.append(float(subset.var(ddof=1) / len(subset) / group_count**2))
        dfs.append(len(subset) - 1)
    estimate = float(np.mean(means))
    variance = float(sum(variance_terms))
    se = math.sqrt(max(variance, 0.0))
    denominator = sum(value**2 / df for value, df in zip(variance_terms, dfs, strict=True))
    df = variance**2 / denominator if denominator > 0 else float(min(dfs))
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, max(df, 1.0)))
    return estimate, se, float(df), estimate - critical * se, estimate + critical * se


def _single_group_interval(
    values: np.ndarray, *, alpha: float
) -> tuple[float, float, float, float, float] | None:
    subset = values[np.isfinite(values)]
    if len(subset) < 2:
        return None
    estimate = float(subset.mean())
    se = float(subset.std(ddof=1) / math.sqrt(len(subset)))
    df = float(len(subset) - 1)
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return estimate, se, df, estimate - critical * se, estimate + critical * se


def _task_values(
    comparison_values: np.ndarray, comparison_tasks: np.ndarray, task_count: int
) -> np.ndarray:
    result = np.full(task_count, np.nan)
    for task in range(task_count):
        subset = comparison_values[(comparison_tasks == task) & np.isfinite(comparison_values)]
        if len(subset):
            result[task] = float(subset.mean())
    return result


def _arena_design(frame: FrameSpec) -> tuple[np.ndarray, np.ndarray]:
    model_count = frame.roster_size
    x = np.zeros((len(frame.arena_task), model_count - 1), dtype=float)
    for row, (first, second) in enumerate(zip(frame.arena_first, frame.arena_second, strict=True)):
        if first < model_count - 1:
            x[row, first] += 1.0
        else:
            x[row] -= 1.0
        if second < model_count - 1:
            x[row, second] -= 1.0
        else:
            x[row] += 1.0
    transform = np.vstack((np.eye(model_count - 1), -np.ones((1, model_count - 1))))
    return x, transform


def _fit_arena(
    frame: FrameSpec,
    comparison_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    observed = np.isfinite(comparison_values)
    if observed.sum() <= frame.roster_size:
        return None
    x_all, transform = _arena_design(frame)
    x = x_all[observed]
    y = comparison_values[observed] - 0.5
    families = frame.arena_family[observed]
    tasks = frame.arena_task[observed]
    family_counts = np.bincount(families, minlength=len(FAMILIES))
    if np.any(family_counts == 0):
        return None
    # Match the production hierarchical point-estimator weights after the two
    # rater rows for each battle have been reduced to their comparison mean:
    # equal families, equal observed tasks within family, and equal observed
    # battles within task. Equal-comparison family weights are equivalent only
    # on the complete balanced frame and diverge when individual battles are
    # missing inside a task.
    tasks_per_family = {
        family: len(np.unique(tasks[families == family])) for family in range(len(FAMILIES))
    }
    battles_per_task = Counter(
        (int(family), int(task)) for family, task in zip(families, tasks, strict=True)
    )
    weights = np.asarray(
        [
            1.0
            / len(FAMILIES)
            / tasks_per_family[int(family)]
            / battles_per_task[int(family), int(task)]
            for family, task in zip(families, tasks, strict=True)
        ]
    )
    beta = np.zeros(x.shape[1], dtype=float)
    ridge = 1e-8
    normalized = weights / weights.sum() * len(weights)
    hessian = np.empty((x.shape[1], x.shape[1]), dtype=float)
    for _ in range(250):
        probability = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30.0, 30.0)))
        variance = np.maximum(probability * (1.0 - probability), 1e-10)
        hessian = x.T @ ((normalized * variance)[:, None] * x)
        hessian += ridge * np.eye(len(beta))
        gradient = x.T @ (normalized * ((y + 0.5) - probability)) - ridge * beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    probability = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30.0, 30.0)))
    residual = (y + 0.5) - probability
    bread = np.linalg.pinv(hessian, rcond=1e-12)
    unique_tasks = np.unique(tasks)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for task in unique_tasks:
        mask = tasks == task
        score = x[mask].T @ (normalized[mask] * residual[mask])
        meat += np.outer(score, score)
    clusters = len(unique_tasks)
    parameters = x.shape[1]
    observations = len(y)
    if clusters <= 1 or observations <= parameters:
        return None
    scale = clusters / (clusters - 1) * (observations - 1) / (observations - parameters)
    covariance = transform @ (bread @ meat @ bread * scale) @ transform.T
    merits = transform @ beta
    return merits, covariance, clusters


def _arena_truth(frame: FrameSpec, shifts: Sequence[float]) -> np.ndarray:
    model_count = frame.roster_size
    expected = np.empty(len(frame.arena_task), dtype=float)
    for family in range(len(FAMILIES)):
        strength = np.full(model_count, -float(shifts[family]) / (model_count - 1))
        strength[0] = float(shifts[family])
        mask = frame.arena_family == family
        expected[mask] = (
            0.5 + strength[frame.arena_first[mask]] - strength[frame.arena_second[mask]]
        )
    fit = _fit_arena(frame, expected)
    _require(fit is not None, "complete arena truth is unidentified")
    return fit[0]


def _pairwise_arena_metrics(
    merits: np.ndarray,
    covariance: np.ndarray,
    truth: np.ndarray,
    *,
    clusters: int,
) -> dict[str, bool | float]:
    pairs = [(left, right) for left in range(len(merits)) for right in range(left + 1, len(merits))]
    critical = float(stats.t.ppf(1.0 - 0.05 / (2.0 * len(pairs)), max(min(clusters - 1, 19), 1)))
    covered = True
    any_null_rejection = False
    top_identified = True
    point_rank_one = True
    focal_lower_bounds: list[float] = []
    halfwidths: list[float] = []
    for left, right in pairs:
        estimate = float(merits[left] - merits[right])
        variance = float(
            covariance[left, left] + covariance[right, right] - 2.0 * covariance[left, right]
        )
        halfwidth = critical * math.sqrt(max(variance, 0.0))
        low, high = estimate - halfwidth, estimate + halfwidth
        target = float(truth[left] - truth[right])
        covered &= low <= target <= high
        any_null_rejection |= high < 0.0 or low > 0.0
        halfwidths.append(halfwidth)
        if left == 0:
            point_rank_one &= estimate > 0.0
            top_identified &= low > 0.0
            focal_lower_bounds.append(low)
    return {
        "simultaneous_coverage": covered,
        "null_fwer_rejection": any_null_rejection,
        "shifted_model_point_rank_one": point_rank_one,
        "shifted_model_identified_top": top_identified,
        "minimum_shifted_model_pair_lower_bound": min(focal_lower_bounds),
        "median_pairwise_halfwidth": float(np.median(halfwidths)),
    }


def _connected(
    model_count: int, first: np.ndarray, second: np.ndarray, observed: np.ndarray
) -> bool:
    adjacency: list[set[int]] = [set() for _ in range(model_count)]
    for left, right in zip(first[observed], second[observed], strict=True):
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    seen: set[int] = set()
    pending = [0]
    while pending:
        value = pending.pop()
        if value in seen:
            continue
        seen.add(value)
        pending.extend(adjacency[value] - seen)
    return len(seen) == model_count


def _dropout_mask(
    assignment: np.ndarray,
    tasks: np.ndarray,
    scenario: Scenario,
    rater_effect: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros(assignment.shape, dtype=bool)
    if scenario.dropout_fraction <= 0.0:
        return mask
    # More severe simulated raters are more likely to drop out.  This makes the
    # stress test outcome-dependent while preserving an explicit DGP target.
    order = np.argsort(rater_effect)[::-1]
    dropout_count = max(1, round(scenario.rater_count * scenario.dropout_fraction))
    dropout_raters = set(int(value) for value in order[:dropout_count])
    lower = max(1, len(np.unique(tasks)) // 3)
    upper = max(2, 2 * len(np.unique(tasks)) // 3)
    cutoffs = {rater: int(rng.integers(lower, upper)) for rater in dropout_raters}
    for row in range(len(tasks)):
        for slot in range(2):
            rater = int(assignment[row, slot])
            if rater in cutoffs and int(tasks[row]) >= cutoffs[rater]:
                mask[row, slot] = True
    return mask


def _simulate_dataset(
    frame: FrameSpec,
    scenario: Scenario,
    rng: np.random.Generator,
    arena_assignment: np.ndarray,
    uplift_assignment: np.ndarray,
) -> dict[str, Any]:
    model_count = frame.roster_size
    rater_effect = rng.normal(0.0, scenario.rater_sd, size=scenario.rater_count)
    rater_effect -= rater_effect.mean()
    uplift_task_effect = rng.normal(0.0, scenario.task_sd, size=frame.task_count)
    arena_task_model = rng.normal(
        0.0, scenario.task_sd / math.sqrt(2.0), size=(frame.task_count, model_count)
    )
    arena_pair_effect = rng.normal(0.0, scenario.comparison_sd, size=len(frame.arena_task))
    uplift_pair_effect = rng.normal(0.0, scenario.comparison_sd, size=len(frame.uplift_task))

    arena_mean = np.empty((len(frame.arena_task), 2), dtype=float)
    for family in range(len(FAMILIES)):
        strength = np.full(model_count, -scenario.arena_family_shifts[family] / (model_count - 1))
        strength[0] = scenario.arena_family_shifts[family]
        mask = frame.arena_family == family
        base = (
            0.5
            + strength[frame.arena_first[mask]]
            - strength[frame.arena_second[mask]]
            + arena_task_model[frame.arena_task[mask], frame.arena_first[mask]]
            - arena_task_model[frame.arena_task[mask], frame.arena_second[mask]]
            + arena_pair_effect[mask]
        )
        arena_mean[mask] = base[:, None] + rater_effect[arena_assignment[mask]]

    uplift_base = np.asarray(scenario.uplift_family_effects)[frame.uplift_family]
    # Fixed, mean-zero model heterogeneity is present in every scenario so the
    # per-model family is not accidentally validated by a homogeneous toy DGP.
    model_heterogeneity = np.linspace(-0.025, 0.025, model_count)
    model_heterogeneity -= model_heterogeneity.mean()
    uplift_mean = (
        0.5
        + uplift_base
        + model_heterogeneity[frame.uplift_model]
        + uplift_task_effect[frame.uplift_task]
        + uplift_pair_effect
    )[:, None] + rater_effect[uplift_assignment]

    arena_y, arena_probabilities = _draw_trinary(arena_mean, scenario.tie_probability, rng)
    uplift_y, uplift_probabilities = _draw_trinary(uplift_mean, scenario.tie_probability, rng)

    arena_response_probability = np.full(len(frame.arena_task), scenario.response_missing_rate)
    uplift_response_probability = np.full(len(frame.uplift_task), scenario.response_missing_rate)
    if scenario.outcome_dependent_missing_slope:
        arena_response_probability += (
            scenario.outcome_dependent_missing_slope
            * np.clip(0.5 - arena_mean.mean(axis=1), 0.0, 0.5)
            * 2.0
        )
        uplift_response_probability += (
            scenario.outcome_dependent_missing_slope
            * np.clip(0.5 - uplift_mean.mean(axis=1), 0.0, 0.5)
            * 2.0
        )
    arena_response_missing = rng.random(len(frame.arena_task)) < np.clip(
        arena_response_probability, 0.0, 0.95
    )
    uplift_response_missing = rng.random(len(frame.uplift_task)) < np.clip(
        uplift_response_probability, 0.0, 0.95
    )

    arena_rating_probability = np.full(arena_y.shape, scenario.rating_missing_rate)
    uplift_rating_probability = np.full(uplift_y.shape, scenario.rating_missing_rate)
    if scenario.outcome_dependent_missing_slope:
        arena_rating_probability += scenario.outcome_dependent_missing_slope * (1.0 - arena_y)
        uplift_rating_probability += scenario.outcome_dependent_missing_slope * (1.0 - uplift_y)
    arena_missing = rng.random(arena_y.shape) < np.clip(arena_rating_probability, 0.0, 0.95)
    uplift_missing = rng.random(uplift_y.shape) < np.clip(uplift_rating_probability, 0.0, 0.95)
    arena_missing |= arena_response_missing[:, None]
    uplift_missing |= uplift_response_missing[:, None]
    arena_missing |= _dropout_mask(arena_assignment, frame.arena_task, scenario, rater_effect, rng)
    uplift_missing |= _dropout_mask(
        uplift_assignment, frame.uplift_task, scenario, rater_effect, rng
    )
    arena_y[arena_missing] = np.nan
    uplift_y[uplift_missing] = np.nan

    arena_comparison = _comparison_means(arena_y)
    uplift_comparison = _comparison_means(uplift_y)
    uplift_task = _task_values(uplift_comparison, frame.uplift_task, frame.task_count)
    overall = _stratified_t_interval(uplift_task - 0.5, frame.task_family, alpha=0.05)

    family_intervals: list[tuple[float, float, float, float, float] | None] = []
    for family in range(len(FAMILIES)):
        family_intervals.append(
            _single_group_interval(
                uplift_task[frame.task_family == family] - 0.5,
                alpha=0.05 / len(FAMILIES),
            )
        )

    model_intervals: list[tuple[float, float, float, float, float] | None] = []
    model_truth: list[float] = []
    for model in range(model_count):
        mask = frame.uplift_model == model
        model_task = _task_values(
            np.where(mask, uplift_comparison, np.nan),
            frame.uplift_task,
            frame.task_count,
        )
        model_intervals.append(
            _stratified_t_interval(
                model_task - 0.5,
                frame.task_family,
                alpha=0.05 / model_count,
            )
        )
        family_weights = Counter(frame.uplift_family[mask].tolist())
        true = np.mean(
            [
                scenario.uplift_family_effects[family] + model_heterogeneity[model]
                for family in range(len(FAMILIES))
                if family_weights[family]
            ]
        )
        model_truth.append(float(true))

    arena_fit = _fit_arena(frame, arena_comparison)
    arena_truth = _arena_truth(frame, scenario.arena_family_shifts)
    arena_metrics = None
    if arena_fit is not None:
        arena_metrics = _pairwise_arena_metrics(
            arena_fit[0], arena_fit[1], arena_truth, clusters=arena_fit[2]
        )

    observed_arena = np.isfinite(arena_comparison)
    graph_full = _connected(model_count, frame.arena_first, frame.arena_second, observed_arena)
    graph_families = all(
        _connected(
            model_count,
            frame.arena_first,
            frame.arena_second,
            observed_arena & (frame.arena_family == family),
        )
        for family in range(len(FAMILIES))
    )

    repeat_agreement = np.full(len(frame.repeat_track), np.nan)
    repeat_truth = np.full(len(frame.repeat_track), np.nan)
    for repeat_index, (track, source, slot) in enumerate(
        zip(
            frame.repeat_track,
            frame.repeat_source_track_index,
            frame.repeat_source_rater_slot,
            strict=True,
        )
    ):
        if track == 0:
            original = arena_y[source, slot]
            probabilities = arena_probabilities[source, slot]
            dropped = arena_missing[source, slot]
        else:
            original = uplift_y[source, slot]
            probabilities = uplift_probabilities[source, slot]
            dropped = uplift_missing[source, slot]
        exact_probability = scenario.repeat_copy_probability + (
            1.0 - scenario.repeat_copy_probability
        ) * float(np.square(probabilities).sum())
        repeat_truth[repeat_index] = exact_probability
        if dropped or not np.isfinite(original):
            continue
        if rng.random() < scenario.repeat_copy_probability:
            repeated = original
        else:
            repeated = rng.choice(TIE_VALUES, p=probabilities)
        repeat_missing_probability = scenario.rating_missing_rate
        if scenario.outcome_dependent_missing_slope:
            repeat_missing_probability += scenario.outcome_dependent_missing_slope * (
                1.0 - repeated
            )
        if rng.random() >= min(repeat_missing_probability, 0.95):
            repeat_agreement[repeat_index] = float(repeated == original)

    reliability_task = _task_values(repeat_agreement, frame.repeat_task, frame.task_count)
    reliability_truth = float(np.nanmean(repeat_truth))
    reliability = _stratified_t_interval(reliability_task, frame.task_family, alpha=0.05)
    track_reliability: list[tuple[float, float, float, float, float] | None] = []
    for track in range(2):
        track_task = _task_values(
            np.where(frame.repeat_track == track, repeat_agreement, np.nan),
            frame.repeat_task,
            frame.task_count,
        )
        track_reliability.append(_stratified_t_interval(track_task, frame.task_family, alpha=0.025))

    def interval_metrics(
        interval: tuple[float, float, float, float, float] | None, target: float
    ) -> dict[str, float | bool] | None:
        if interval is None:
            return None
        estimate, _se, _df, low, high = interval
        return {
            "estimate": estimate,
            "covered": low <= target <= high,
            "positive": low > 0.0,
            "rejects_zero": high < 0.0 or low > 0.0,
            "halfwidth": (high - low) / 2.0,
        }

    overall_metrics = interval_metrics(overall, scenario.overall_uplift)
    family_metrics = [
        interval_metrics(interval, scenario.uplift_family_effects[index])
        for index, interval in enumerate(family_intervals)
    ]
    model_metrics = [
        interval_metrics(interval, model_truth[index])
        for index, interval in enumerate(model_intervals)
    ]
    reliability_metrics = interval_metrics(reliability, reliability_truth)
    track_metrics = [
        interval_metrics(interval, reliability_truth) for interval in track_reliability
    ]
    return {
        "overall": overall_metrics,
        "families": family_metrics,
        "models": model_metrics,
        "arena": arena_metrics,
        "graph_full": graph_full,
        "graph_families": graph_families,
        "reliability": reliability_metrics,
        "track_reliability": track_metrics,
        "observed_arena_comparisons": int(np.isfinite(arena_comparison).sum()),
        "observed_uplift_comparisons": int(np.isfinite(uplift_comparison).sum()),
        "observed_primary_ratings": int(np.isfinite(arena_y).sum() + np.isfinite(uplift_y).sum()),
        "observed_repeat_pairs": int(np.isfinite(repeat_agreement).sum()),
    }


def _mcse(rate: float, datasets: int) -> float:
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / datasets)


def _rate(rows: Sequence[Mapping[str, Any]], path: Sequence[str | int]) -> tuple[float, int]:
    values: list[bool] = []
    for row in rows:
        current: Any = row
        for key in path:
            if current is None:
                break
            current = current[key]
        if current is not None:
            values.append(bool(current))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def _mean(rows: Sequence[Mapping[str, Any]], path: Sequence[str | int]) -> float:
    values: list[float] = []
    for row in rows:
        current: Any = row
        for key in path:
            if current is None:
                break
            current = current[key]
        if current is not None and np.isfinite(float(current)):
            values.append(float(current))
    return float(np.mean(values)) if values else float("nan")


def _round(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def evaluate_scenario(
    frame: FrameSpec,
    scenario: Scenario,
    *,
    datasets: int,
    seed: int,
) -> dict[str, Any]:
    """Run a deterministic repeated-sampling validation for one scenario."""

    scenario.validate()
    _require(datasets >= 20, "at least 20 Monte Carlo datasets are required")
    seed_sequence = np.random.SeedSequence(seed)
    arena_assignment = _assignment(frame.arena_comparison_ids, scenario.rater_count)
    uplift_assignment = _assignment(frame.uplift_comparison_ids, scenario.rater_count)
    rows = [
        _simulate_dataset(
            frame,
            scenario,
            np.random.default_rng(child),
            arena_assignment,
            uplift_assignment,
        )
        for child in seed_sequence.spawn(datasets)
    ]

    overall_coverage, overall_n = _rate(rows, ("overall", "covered"))
    overall_reject, _ = _rate(rows, ("overall", "rejects_zero"))
    overall_power, _ = _rate(rows, ("overall", "positive"))
    arena_coverage, arena_n = _rate(rows, ("arena", "simultaneous_coverage"))
    arena_fwer, _ = _rate(rows, ("arena", "null_fwer_rejection"))
    arena_point_rank_one, _ = _rate(rows, ("arena", "shifted_model_point_rank_one"))
    arena_power, _ = _rate(rows, ("arena", "shifted_model_identified_top"))
    family_coverage_values = [
        _rate(rows, ("families", family, "covered"))[0] for family in range(len(FAMILIES))
    ]
    family_power_values = [
        _rate(rows, ("families", family, "positive"))[0] for family in range(len(FAMILIES))
    ]
    model_coverage_values = [
        _rate(rows, ("models", model, "covered"))[0] for model in range(frame.roster_size)
    ]
    model_power_values = [
        _rate(rows, ("models", model, "positive"))[0] for model in range(frame.roster_size)
    ]
    reliability_coverage, reliability_n = _rate(rows, ("reliability", "covered"))
    graph_full, _ = _rate(rows, ("graph_full",))
    graph_families, _ = _rate(rows, ("graph_families",))
    nominal_elo: list[float] = []
    for shift in scenario.arena_family_shifts:
        pair_probability = 0.5 + shift * frame.roster_size / (frame.roster_size - 1)
        pair_probability = min(max(pair_probability, 1e-9), 1.0 - 1e-9)
        nominal_elo.append(400.0 * math.log10(pair_probability / (1.0 - pair_probability)))

    result = {
        "scenario_id": scenario.scenario_id,
        "core": scenario.core,
        "null": scenario.null,
        "data_generating_parameters": {
            "uplift_family_effects_half_win_share": list(scenario.uplift_family_effects),
            "overall_uplift_half_win_share": scenario.overall_uplift,
            "arena_focal_model_family_shifts": list(scenario.arena_family_shifts),
            "arena_focal_vs_other_nominal_elo_by_family": [
                round(value, 6) for value in nominal_elo
            ],
            "tie_probability": scenario.tie_probability,
            "task_sd_half_win_share": scenario.task_sd,
            "comparison_sd_half_win_share": scenario.comparison_sd,
            "rater_sd_half_win_share": scenario.rater_sd,
            "simulated_rater_pool": scenario.rater_count,
            "response_missing_rate": scenario.response_missing_rate,
            "rating_missing_rate": scenario.rating_missing_rate,
            "outcome_dependent_missing_slope": scenario.outcome_dependent_missing_slope,
            "dropout_fraction": scenario.dropout_fraction,
        },
        "monte_carlo": {
            "datasets": datasets,
            "seed": seed,
            "binomial_rate_mcse_formula": "sqrt(p*(1-p)/datasets)",
        },
        "overall_uplift": {
            "coverage": _round(overall_coverage),
            "coverage_mcse": _round(_mcse(overall_coverage, overall_n)),
            "two_sided_type_i_error": _round(overall_reject) if scenario.null else None,
            "type_i_error_mcse": (
                _round(_mcse(overall_reject, overall_n)) if scenario.null else None
            ),
            "one_sided_power": _round(overall_power) if scenario.overall_uplift > 0 else None,
            "power_mcse": (
                _round(_mcse(overall_power, overall_n)) if scenario.overall_uplift > 0 else None
            ),
            "mean_ci_halfwidth": _round(_mean(rows, ("overall", "halfwidth"))),
            "mean_estimate": _round(_mean(rows, ("overall", "estimate"))),
            "bias": _round(_mean(rows, ("overall", "estimate")) - scenario.overall_uplift),
        },
        "arena_ranking": {
            "simultaneous_91_pairwise_coverage": _round(arena_coverage),
            "coverage_mcse": _round(_mcse(arena_coverage, arena_n)),
            "null_pairwise_fwer": _round(arena_fwer) if scenario.null else None,
            "fwer_mcse": _round(_mcse(arena_fwer, arena_n)) if scenario.null else None,
            "focal_shifted_model_identified_top_power": (
                _round(arena_power) if max(scenario.arena_family_shifts) > 0 else None
            ),
            "focal_shifted_model_point_rank_one_rate": (
                _round(arena_point_rank_one) if max(scenario.arena_family_shifts) > 0 else None
            ),
            "power_mcse": (
                _round(_mcse(arena_power, arena_n))
                if max(scenario.arena_family_shifts) > 0
                else None
            ),
            "mean_minimum_focal_pair_lower_bound": _round(
                _mean(rows, ("arena", "minimum_shifted_model_pair_lower_bound"))
            ),
            "mean_median_pairwise_ci_halfwidth": _round(
                _mean(rows, ("arena", "median_pairwise_halfwidth"))
            ),
        },
        "family_uplift": {
            "marginal_coverage_by_family": {
                family: _round(family_coverage_values[index])
                for index, family in enumerate(FAMILIES)
            },
            "minimum_marginal_coverage": _round(min(family_coverage_values)),
            "one_sided_detection_power_by_family": {
                family: _round(family_power_values[index]) for index, family in enumerate(FAMILIES)
            },
            "minimum_one_sided_detection_power": _round(min(family_power_values)),
            "minimum_power_mcse": _round(_mcse(min(family_power_values), datasets)),
            "mean_simultaneous_ci_halfwidth": _round(
                np.mean(
                    [
                        _mean(rows, ("families", family, "halfwidth"))
                        for family in range(len(FAMILIES))
                    ]
                )
            ),
        },
        "per_model_uplift": {
            "minimum_marginal_coverage": _round(min(model_coverage_values)),
            "minimum_one_sided_detection_power": _round(min(model_power_values)),
            "minimum_power_mcse": _round(_mcse(min(model_power_values), datasets)),
            "mean_bonferroni_ci_halfwidth": _round(
                np.mean(
                    [
                        _mean(rows, ("models", model, "halfwidth"))
                        for model in range(frame.roster_size)
                    ]
                )
            ),
        },
        "reliability": {
            "overall_exact_agreement_coverage": _round(reliability_coverage),
            "coverage_mcse": _round(_mcse(reliability_coverage, reliability_n)),
            "mean_ci_halfwidth": _round(_mean(rows, ("reliability", "halfwidth"))),
            "mean_track_simultaneous_ci_halfwidth": _round(
                np.mean(
                    [_mean(rows, ("track_reliability", track, "halfwidth")) for track in range(2)]
                )
            ),
            "mean_observed_repeat_pairs": _round(_mean(rows, ("observed_repeat_pairs",))),
        },
        "missingness_and_connectivity": {
            "mean_observed_arena_comparisons": _round(_mean(rows, ("observed_arena_comparisons",))),
            "mean_observed_uplift_comparisons": _round(
                _mean(rows, ("observed_uplift_comparisons",))
            ),
            "mean_observed_primary_ratings": _round(_mean(rows, ("observed_primary_ratings",))),
            "full_graph_connected_rate": _round(graph_full),
            "all_four_family_graphs_connected_rate": _round(graph_families),
        },
    }
    return result


def analytic_power_curves(frame: FrameSpec) -> dict[str, Any]:
    """Conservative normal approximations used only for planning/remediation."""

    effects = (0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15)
    # Outcome SD 0.40 and task ICC 0.10 are deliberately conservative.  A task
    # contains ten comparisons with two ratings; comparison correlation is not
    # allowed to turn the nominal 1,600/3,200 into an independent sample size.
    outcome_sd = 0.40
    comparisons_per_task = len(frame.uplift_task) / frame.task_count
    ratings_per_comparison = 2
    task_icc = 0.10
    effective_within_task = comparisons_per_task * ratings_per_comparison
    task_sd = outcome_sd * math.sqrt(task_icc + (1.0 - task_icc) / effective_within_task)

    def power(
        effect: float,
        clusters: int,
        alpha: float,
        *,
        cluster_sd: float = task_sd,
    ) -> float:
        se = cluster_sd / math.sqrt(clusters)
        critical = float(stats.t.ppf(1.0 - alpha / 2.0, max(clusters - 1, 1)))
        noncentrality = effect / se
        return float(
            stats.nct.cdf(-critical, clusters - 1, noncentrality)
            + 1.0
            - stats.nct.cdf(critical, clusters - 1, noncentrality)
        )

    clusters_per_family = min(Counter(frame.task_family.tolist()).values())
    model_tasks: dict[tuple[int, int], set[int]] = {
        (model, family): set()
        for model in range(frame.roster_size)
        for family in range(len(FAMILIES))
    }
    model_task_comparisons: Counter[tuple[int, int]] = Counter()
    for model, family, task in zip(
        frame.uplift_model,
        frame.uplift_family,
        frame.uplift_task,
        strict=True,
    ):
        model_tasks[int(model), int(family)].add(int(task))
        model_task_comparisons[int(model), int(task)] += 1
    min_model_clusters = sum(
        min(len(model_tasks[model, family]) for model in range(frame.roster_size))
        for family in range(len(FAMILIES))
    )
    minimum_comparisons_per_model_task = min(model_task_comparisons.values())
    effective_within_model_task = minimum_comparisons_per_model_task * ratings_per_comparison
    model_task_sd = outcome_sd * math.sqrt(
        task_icc + (1.0 - task_icc) / effective_within_model_task
    )
    curves = {
        "overall_uplift_80_task_clusters_alpha_0_05": [
            {"effect": effect, "power": round(power(effect, frame.task_count, 0.05), 6)}
            for effect in effects
        ],
        "one_family_uplift_20_task_clusters_bonferroni_4": [
            {
                "effect": effect,
                "power": round(power(effect, clusters_per_family, 0.05 / 4), 6),
            }
            for effect in effects
        ],
        "one_model_uplift_minimum_task_clusters_bonferroni_roster": [
            {
                "effect": effect,
                "power": round(
                    power(
                        effect,
                        min_model_clusters,
                        0.05 / frame.roster_size,
                        cluster_sd=model_task_sd,
                    ),
                    6,
                ),
            }
            for effect in effects
        ],
    }
    return {
        "status": "planning_approximation_not_a_validation_result",
        "assumptions": {
            "outcome_sd": outcome_sd,
            "task_icc": task_icc,
            "comparisons_per_task": comparisons_per_task,
            "ratings_per_comparison": ratings_per_comparison,
            "task_cluster_sd": round(task_sd, 8),
            "smallest_family_clusters": clusters_per_family,
            "minimum_model_clusters_family_standardized": min_model_clusters,
            "minimum_comparisons_per_model_task": minimum_comparisons_per_model_task,
            "minimum_ratings_per_model_task": effective_within_model_task,
            "model_task_cluster_sd": round(model_task_sd, 8),
        },
        "curves": curves,
    }


def run_validation(
    frame: FrameSpec,
    *,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    datasets: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate all scenarios with separated deterministic seed streams."""

    _require(
        len({scenario.scenario_id for scenario in scenarios}) == len(scenarios),
        "duplicate scenario",
    )
    child_seeds = np.random.SeedSequence(seed).generate_state(len(scenarios), dtype=np.uint64)
    results = [
        evaluate_scenario(frame, scenario, datasets=datasets, seed=int(child_seeds[index]))
        for index, scenario in enumerate(scenarios)
    ]
    return {
        "frame_diagnostics": frame_diagnostics(frame),
        "scenario_results": results,
        "analytic_precision_mde_power_curves": analytic_power_curves(frame),
    }
