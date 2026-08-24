from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .endpoint_contract import UNFROZEN_VALUES, endpoint_contract_sha256
from .models import (
    Battle,
    ResponseArm,
    Season,
    SeasonModel,
    ToolCall,
    ValidatorResult,
    Vote,
)


def _fit_local_bradley_terry(
    comparisons: list[tuple[str, str, float]],
) -> dict[str, tuple[float, float, float]]:
    competitors = sorted({name for comparison in comparisons for name in comparison[:2]})
    if len(competitors) < 2:
        return {}
    index = {name: position for position, name in enumerate(competitors)}
    rows = []
    outcomes = []
    for first, second, outcome in comparisons:
        row = np.zeros(len(competitors) - 1)
        if index[first] < len(competitors) - 1:
            row[index[first]] = 1
        if index[second] < len(competitors) - 1:
            row[index[second]] -= 1
        rows.append(row)
        outcomes.append(outcome)
    design = np.asarray(rows, dtype=float)
    target = np.asarray(outcomes, dtype=float)
    beta = np.zeros(len(competitors) - 1)
    ridge = 1e-6
    for _ in range(100):
        logits = np.clip(design @ beta, -30, 30)
        probability = 1 / (1 + np.exp(-logits))
        weight = np.maximum(probability * (1 - probability), 1e-8)
        hessian = design.T @ (weight[:, None] * design) + ridge * np.eye(len(beta))
        gradient = design.T @ (target - probability) - ridge * beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    covariance = np.linalg.pinv(hessian)
    full_beta = np.append(beta, 0.0)
    full_beta -= full_beta.mean()
    transform = np.full((len(competitors), len(competitors) - 1), -1 / len(competitors))
    transform[: len(competitors) - 1] += np.eye(len(competitors) - 1)
    full_covariance = transform @ covariance @ transform.T
    scale = 400 / math.log(10)
    result = {}
    for name, position in index.items():
        variance = full_covariance[position, position]
        standard_error = math.sqrt(max(float(variance), 0.0)) * scale
        rating = 1000 + full_beta[position] * scale
        result[name] = (rating, rating - 1.96 * standard_error, rating + 1.96 * standard_error)
    return result


def _fit_bradley_terry(
    comparisons: list[tuple[str, str, float]],
    *,
    require_arena_rank: bool = False,
) -> dict[str, tuple[float, float, float]]:
    if len({name for comparison in comparisons for name in comparison[:2]}) < 2:
        return {}
    try:
        import pandas as pd
        from arena_rank.models.bradley_terry import BradleyTerry
        from arena_rank.utils.data_utils import PairDataset

        frame = pd.DataFrame(
            [
                {
                    "model_a": first,
                    "model_b": second,
                    "winner": "model_a" if outcome == 1 else "model_b" if outcome == 0 else "tie",
                }
                for first, second, outcome in comparisons
            ]
        )
        dataset = PairDataset.from_pandas(frame, reweighted=False)
        fitted = BradleyTerry(dataset.n_competitors).compute_ratings_and_cis(
            dataset,
            ci_method="sandwich",
        )
        return {
            str(name): (float(rating), float(low), float(high))
            for name, rating, low, high in zip(
                fitted["competitors"],
                fitted["ratings"],
                fitted["rating_lower"],
                fitted["rating_upper"],
                strict=True,
            )
        }
    except Exception as error:
        if require_arena_rank:
            raise RuntimeError(
                "arena-rank 0.1.1 failed; refusing an undisclosed fallback for official analysis"
            ) from error
        return _fit_local_bradley_terry(comparisons)


def _wilson(success: float, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    z = 1.96
    proportion = success / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _paired_tie_aware_profile(
    wins: int, ties: int, losses: int
) -> tuple[float, float, float]:
    """Estimate tie-half preference with a multinomial profile-likelihood interval.

    The estimand is ``P(win) + 0.5 * P(tie)``. For every candidate value of that
    estimand, the three-outcome multinomial likelihood is maximized over the tie
    probability. Inverting the 95% likelihood-ratio test avoids the zero-width
    and full-range Wald intervals produced by the former numerical Hessian at
    sparse or boundary outcome counts.
    """

    total = wins + ties + losses
    if total == 0:
        return 0.5, 0.0, 1.0

    def log_term(count: int, probability: float) -> float:
        if count == 0:
            return 0.0
        if probability <= 0.0:
            return -math.inf
        return count * math.log(probability)

    def profile_log_likelihood(preference: float) -> float:
        tie_max = 2.0 * min(preference, 1.0 - preference)

        def likelihood(tie_probability: float) -> float:
            win_probability = preference - 0.5 * tie_probability
            loss_probability = 1.0 - preference - 0.5 * tie_probability
            return (
                log_term(wins, win_probability)
                + log_term(ties, tie_probability)
                + log_term(losses, loss_probability)
            )

        if tie_max <= 0.0:
            return likelihood(0.0)

        # This profile is concave. Golden-section maximization plus endpoint
        # checks handles empty outcome cells without a fragile Hessian.
        left = 0.0
        right = tie_max
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        first = right - ratio * (right - left)
        second = left + ratio * (right - left)
        first_value = likelihood(first)
        second_value = likelihood(second)
        for _ in range(160):
            if first_value < second_value:
                left = first
                first = second
                first_value = second_value
                second = left + ratio * (right - left)
                second_value = likelihood(second)
            else:
                right = second
                second = first
                second_value = first_value
                first = right - ratio * (right - left)
                first_value = likelihood(first)
        return max(
            likelihood(0.0),
            likelihood(tie_max),
            likelihood((left + right) / 2.0),
        )

    estimate = (wins + 0.5 * ties) / total
    maximum_log_likelihood = (
        log_term(wins, wins / total)
        + log_term(ties, ties / total)
        + log_term(losses, losses / total)
    )
    # 0.95 quantile of chi-square with one degree of freedom.
    critical_value = 3.841458820694124

    def included(preference: float) -> bool:
        profile = profile_log_likelihood(preference)
        return 2.0 * (maximum_log_likelihood - profile) <= critical_value

    def boundary(low: float, high: float, *, lower: bool) -> float:
        for _ in range(100):
            midpoint = (low + high) / 2.0
            if included(midpoint):
                if lower:
                    high = midpoint
                else:
                    low = midpoint
            elif lower:
                low = midpoint
            else:
                high = midpoint
        return high if lower else low

    interval_low = 0.0 if included(0.0) else boundary(0.0, estimate, lower=True)
    interval_high = 1.0 if included(1.0) else boundary(estimate, 1.0, lower=False)
    return estimate, interval_low, interval_high


def _paired_ordinal(wins: int, ties: int, losses: int) -> tuple[float, float, float]:
    """Retain the original internal entry point for callers and archived tests."""

    return _paired_tie_aware_profile(wins, ties, losses)


def _votes(
    session: Session,
    season: Season,
    track: str,
    cohort: str,
    category: str,
    data_stratum: str,
) -> list[tuple[Vote, Battle]]:
    if data_stratum not in {"public_freeform", "controlled"}:
        raise ValueError(f"unsupported leaderboard data stratum: {data_stratum}")
    statement = (
        select(Vote, Battle)
        .join(Battle, Battle.id == Vote.battle_id)
        .where(
            Battle.season_id == season.id,
            Battle.track == track,
            Battle.status == "complete",
            Battle.rank_eligible.is_(True),
            Battle.run_class == "official",
            Battle.manifest_sha256 == season.manifest_sha256,
            Battle.data_stratum == data_stratum,
        )
    )
    if data_stratum == "controlled":
        statement = statement.where(
            Battle.task_id.is_not(None),
            Battle.task_revision.is_not(None),
        )
    else:
        statement = statement.where(Battle.task_id.is_(None))
    if cohort != "combined":
        statement = statement.where(Vote.cohort == cohort)
    if category != "all":
        statement = statement.where(Battle.category == category)
    return list(session.execute(statement).all())


def _arm_matches_endpoint_contract(
    battle: Battle, arm: ResponseArm, endpoint: SeasonModel | None
) -> bool:
    if endpoint is None:
        return False
    if (
        endpoint.manifest_sha256 != battle.manifest_sha256
        or endpoint.endpoint_contract_sha256 in UNFROZEN_VALUES
        or arm.provider_slug != endpoint.provider_slug
        or arm.actual_model_id != endpoint.expected_actual_model_id
        or arm.actual_provider_slug != endpoint.expected_actual_provider_slug
    ):
        return False
    try:
        computed = endpoint_contract_sha256(
            model_id=endpoint.model_id,
            provider_slug=endpoint.provider_slug,
            expected_actual_model_id=endpoint.expected_actual_model_id,
            expected_actual_provider_slug=endpoint.expected_actual_provider_slug,
            supported_parameters=endpoint.supported_parameters_json,
            decoding=endpoint.decoding_json,
            endpoint_max_completion_tokens=endpoint.endpoint_max_completion_tokens,
            endpoint_document_sha256=endpoint.endpoint_document_sha256,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return computed == endpoint.endpoint_contract_sha256


def _ranked_arms(
    session: Session,
    battle: Battle,
    track: str,
) -> tuple[ResponseArm, ResponseArm] | None:
    left = session.get(ResponseArm, battle.left_arm_id)
    right = session.get(ResponseArm, battle.right_arm_id)
    if not left or not right:
        return None
    arms = (left, right)
    endpoint_contracts = {
        slot.model_id: slot
        for slot in session.scalars(
            select(SeasonModel).where(
                SeasonModel.season_id == battle.season_id,
                SeasonModel.model_id.in_({arm.model_id for arm in arms}),
            )
        ).all()
    }
    for arm in arms:
        endpoint = endpoint_contracts.get(arm.model_id)
        if not _arm_matches_endpoint_contract(battle, arm, endpoint):
            return None
    if any(
        arm.status != "complete"
        or arm.model_id.startswith("flavourbench/mock-")
        or arm.provider_slug == "mock"
        or (arm.actual_provider_slug or "").lower() == "mock"
        or not arm.cost_reconciled
        for arm in arms
    ):
        return None
    if track == "model_arena":
        if any(arm.condition != "epicure_on" for arm in arms):
            return None
    elif (
        left.model_id != right.model_id
        or {left.condition, right.condition} != {"epicure_on", "epicure_off"}
    ):
        return None
    return left, right


def _operational_metrics(
    session: Session,
    season: Season,
    track: str,
    category: str,
    data_stratum: str,
) -> dict[str, dict[str, float]]:
    if data_stratum not in {"public_freeform", "controlled"}:
        raise ValueError(f"unsupported leaderboard data stratum: {data_stratum}")
    statement = (
        select(ResponseArm, Battle)
        .join(Battle, Battle.id == ResponseArm.battle_id)
        .where(
            Battle.season_id == season.id,
            Battle.track == track,
            Battle.rank_eligible.is_(True),
            Battle.run_class == "official",
            Battle.manifest_sha256 == season.manifest_sha256,
            Battle.data_stratum == data_stratum,
            ~ResponseArm.model_id.like("flavourbench/mock-%"),
            ResponseArm.provider_slug != "mock",
        )
    )
    if data_stratum == "controlled":
        statement = statement.where(
            Battle.task_id.is_not(None),
            Battle.task_revision.is_not(None),
        )
    else:
        statement = statement.where(Battle.task_id.is_(None))
    if category != "all":
        statement = statement.where(Battle.category == category)
    endpoint_contracts = {
        slot.model_id: slot
        for slot in session.scalars(
            select(SeasonModel).where(SeasonModel.season_id == season.id)
        ).all()
    }
    metrics: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for arm, battle in session.execute(statement).all():
        record = metrics[arm.model_id]
        record["arms"] += 1
        record["invalid"] += arm.status != "complete" or not _arm_matches_endpoint_contract(
            battle, arm, endpoint_contracts.get(arm.model_id)
        )
        record["cost"] += arm.cost_micros
        record["latency"] += arm.latency_ms
        calls = session.scalars(select(ToolCall).where(ToolCall.arm_id == arm.id)).all()
        record["tool_calls"] += len(calls)
        record["tool_successes"] += sum(not call.is_error for call in calls)
        validators = session.scalars(
            select(ValidatorResult).where(
                ValidatorResult.arm_id == arm.id,
                ValidatorResult.validator_name == "constraint_acknowledgement",
                ValidatorResult.status != "not_applicable",
            )
        ).all()
        record["constraint_checks"] += len(validators)
        record["constraint_passes"] += sum(item.status == "pass" for item in validators)
    return metrics


def _metric_payload(record: dict[str, float]) -> dict[str, float | int | None]:
    arms = int(record["arms"])
    calls = int(record["tool_calls"])
    constraint_checks = int(record["constraint_checks"])
    return {
        "response_arms": arms,
        "invalid_response_rate": round(record["invalid"] / arms, 4) if arms else None,
        "objective_constraint_pass_rate": round(record["constraint_passes"] / constraint_checks, 4)
        if constraint_checks
        else None,
        "tool_success_rate": round(record["tool_successes"] / calls, 4) if calls else None,
        "average_cost_micros": round(record["cost"] / arms) if arms else 0,
        "average_latency_ms": round(record["latency"] / arms) if arms else 0,
    }


def model_leaderboard(
    session: Session,
    season: Season,
    cohort: str,
    category: str,
    data_stratum: str = "controlled",
) -> dict[str, Any]:
    data = _votes(session, season, "model_arena", cohort, category, data_stratum)
    comparisons: list[tuple[str, str, float]] = []
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for vote, battle in data:
        arms = _ranked_arms(session, battle, "model_arena")
        if arms is None:
            continue
        left, right = arms
        counts[left.model_id]["judgments"] += 1
        counts[right.model_id]["judgments"] += 1
        if vote.choice == "both_bad":
            counts[left.model_id]["both_bad"] += 1
            counts[right.model_id]["both_bad"] += 1
            continue
        counts[left.model_id]["appearances"] += 1
        counts[right.model_id]["appearances"] += 1
        outcome = 1.0 if vote.choice == "left" else 0.0 if vote.choice == "right" else 0.5
        comparisons.append((left.model_id, right.model_id, outcome))
        counts[left.model_id]["cost"] += left.cost_micros
        counts[right.model_id]["cost"] += right.cost_micros
        counts[left.model_id]["latency"] += left.latency_ms
        counts[right.model_id]["latency"] += right.latency_ms
    ratings = _fit_bradley_terry(comparisons)
    operational = _operational_metrics(
        session, season, "model_arena", category, data_stratum
    )
    minimum = 40 if cohort.startswith("expert_") else 100
    rows = []
    competitors = set(ratings) | set(operational)
    for model_id in sorted(
        competitors,
        key=lambda item: ratings.get(item, (-math.inf, 0, 0))[0],
        reverse=True,
    ):
        rating_values = ratings.get(model_id)
        rating, low, high = rating_values if rating_values else (None, None, None)
        stats = counts[model_id]
        appearances = int(stats["appearances"])
        judgments = int(stats["judgments"])
        row = {
            "competitor_id": model_id,
            "rating": round(rating, 2) if rating is not None else None,
            "rating_lower": round(low, 2) if low is not None else None,
            "rating_upper": round(high, 2) if high is not None else None,
            "battles": appearances,
            "judgments": judgments,
            "both_bad": int(stats["both_bad"]),
            "provisional": appearances < minimum,
            "both_bad_rate": round(stats["both_bad"] / judgments, 4) if judgments else 0,
        }
        row.update(_metric_payload(operational[model_id]))
        rows.append(row)
    return {
        "track": "model_arena",
        "cohort": cohort,
        "cohort_label": "Combined · secondary" if cohort == "combined" else cohort.title(),
        "category": category,
        "data_stratum": data_stratum,
        "rows": rows,
        "method": "arena-rank 0.1.1 Bradley-Terry with ties as half-wins and 95% intervals",
        "manifest_sha256": season.manifest_sha256,
        "eligibility_filter": {
            "run_class": "official",
            "rank_eligible": True,
            "manifest_match": True,
            "data_stratum": data_stratum,
            "mock_models": "excluded",
            "identity_and_cost_reconciliation": "required_for_preference",
        },
    }


def uplift_leaderboard(
    session: Session,
    season: Season,
    cohort: str,
    category: str,
    data_stratum: str = "controlled",
) -> dict[str, Any]:
    data = _votes(session, season, "epicure_uplift", cohort, category, data_stratum)
    operational = _operational_metrics(
        session, season, "epicure_uplift", category, data_stratum
    )
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for vote, battle in data:
        arms = _ranked_arms(session, battle, "epicure_uplift")
        if arms is None:
            continue
        left, right = arms
        record = stats[left.model_id]
        record["judgments"] += 1
        if vote.choice == "both_bad":
            record["both_bad"] += 1
            continue
        record["total"] += 1
        if vote.choice == "tie":
            record["tie"] += 1
        else:
            winning_condition = left.condition if vote.choice == "left" else right.condition
            record["epicure_win" if winning_condition == "epicure_on" else "unaided_win"] += 1
    minimum = 20 if cohort.startswith("expert_") else 50
    rows = []
    for model_id in set(stats) | set(operational):
        record = stats[model_id]
        estimate, low, high = _paired_tie_aware_profile(
            record["epicure_win"], record["tie"], record["unaided_win"]
        )
        row = {
            "competitor_id": model_id,
            "epicure_win_share": round(estimate, 4),
            "interval_lower": round(low, 4),
            "interval_upper": round(high, 4),
            "epicure_wins": record["epicure_win"],
            "unaided_wins": record["unaided_win"],
            "ties": record["tie"],
            "both_bad": record["both_bad"],
            "battles": record["total"],
            "judgments": record["judgments"],
            "provisional": record["total"] < minimum,
        }
        row.update(_metric_payload(operational[model_id]))
        rows.append(row)
    rows.sort(key=lambda row: row["epicure_win_share"], reverse=True)
    return {
        "track": "epicure_uplift",
        "cohort": cohort,
        "cohort_label": "Combined · secondary" if cohort == "combined" else cohort.title(),
        "category": category,
        "data_stratum": data_stratum,
        "rows": rows,
        "method": (
            "observed tie-half preference with a 95% multinomial "
            "profile-likelihood interval"
        ),
        "manifest_sha256": season.manifest_sha256,
        "eligibility_filter": {
            "run_class": "official",
            "rank_eligible": True,
            "manifest_match": True,
            "data_stratum": data_stratum,
            "mock_models": "excluded",
            "identity_and_cost_reconciliation": "required_for_preference",
        },
    }


def snapshot_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
