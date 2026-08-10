"""Tenant-scoped ranking views for the commercial FlavourBench service.

The retrospective paper binds the byte-exact ``ranking.py`` source. Service-only
access control therefore lives here so product changes cannot silently rewrite a
frozen analysis implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .controlled_integrity import (
    ControlledRunIntegrityError,
    verify_controlled_assignment_battle,
)
from .endpoint_contract import UNFROZEN_VALUES, endpoint_contract_sha256
from .engine import is_complete_finish_reason
from .expert_review import PROTOCOL_SHA256 as EXPERT_REVIEW_PROTOCOL_SHA256
from .expert_review import PROTOCOL_VERSION as EXPERT_REVIEW_PROTOCOL_VERSION
from .models import (
    Battle,
    BedrockBillingCrosscheck,
    BedrockBillingCrosscheckArm,
    ControlledRun,
    ControlledRunAssignment,
    ExpertReviewer,
    ResponseArm,
    RunEvent,
    Season,
    SeasonModel,
    ToolCall,
    ValidatorResult,
    Vote,
)
from .provider import FINAL_SCHEMA_SHA256, system_prompt_sha256
from .reviewer_admission import historical_expert_admission_event
from .reviewer_identity import filter_ranking_vote_rows
from .season1_statistics import (
    ArenaObservation,
    StatisticalContractError,
    UpliftJudgment,
    UpliftScheduledPair,
    analyze_controlled_arena,
    analyze_controlled_uplift,
)
from .task_evidence_registry import (
    TASK_SPECIFIC_VALIDATOR_NAME,
    TASK_SPECIFIC_VALIDATOR_VERSION,
)
from .validators import VALIDATOR_VERSION

VALID_VOTE_CHOICES = frozenset({"left", "right", "tie", "both_bad"})
REQUIRED_PREFERENCE_VALIDATORS = frozenset({"identity_blinding", "semantic_completion"})
RANKING_RESTRICTION_EVENT_TYPES = frozenset(
    {
        "battle_ranking_restricted",
        "expert_review_batch_restricted",
        "expert_review_assignment_submitted",
        "response_arm_non_normal_completion_detected",
        "reviewer_reported_potential_safety_hazard",
        "task_general_track_scope_quarantined",
        "confirmatory_task_retired",
        "battle_general_track_scope_admitted",
    }
)


class RankingEvidenceError(RuntimeError):
    """Stored ranking evidence violates a database-level scientific invariant."""


class InProcessSnapshotAnalysisForbidden(RuntimeError):
    """Snapshot fitting was attempted inside the production API process."""


ANALYSIS_RUNTIME_MODULE_ROOTS = frozenset({"arena_rank", "pandas", "pyarrow"})
SNAPSHOT_ANALYSIS_JOB_ROUTE = "/v1/admin/leaderboards/snapshot-jobs"


def _is_production_api_process() -> bool:
    settings = get_settings()
    return settings.environment == "production" and settings.service_role == "api"


def loaded_analysis_runtime_roots() -> list[str]:
    """Return analysis packages already resident in this interpreter."""

    return sorted(
        ANALYSIS_RUNTIME_MODULE_ROOTS
        & {module_name.partition(".")[0] for module_name in sys.modules}
    )


def assert_api_analysis_runtime_clean() -> None:
    """Fail startup if native analysis dependencies entered the production API."""

    if not _is_production_api_process():
        return
    loaded = loaded_analysis_runtime_roots()
    if loaded:
        raise RuntimeError(
            "production API loaded forbidden snapshot-analysis modules: " + ", ".join(loaded)
        )


def require_snapshot_analysis_process() -> None:
    """Keep pandas, Arrow, and Arena-Rank in the separately supervised worker."""

    if not _is_production_api_process():
        return
    assert_api_analysis_runtime_clean()
    raise InProcessSnapshotAnalysisForbidden(
        "snapshot fitting is disabled in the production API process; submit an "
        f"asynchronous analysis job through {SNAPSHOT_ANALYSIS_JOB_ROUTE}"
    )


def _arm_has_required_preference_validators(
    session: Session,
    arm_id: str,
    *,
    evidence_cutoff_at: datetime | None = None,
) -> bool:
    statement = select(
        ValidatorResult.validator_name,
        ValidatorResult.status,
    ).where(
        ValidatorResult.arm_id == arm_id,
        ValidatorResult.validator_version == VALIDATOR_VERSION,
        ValidatorResult.validator_name.in_(REQUIRED_PREFERENCE_VALIDATORS),
    )
    if evidence_cutoff_at is not None:
        statement = statement.where(ValidatorResult.created_at <= evidence_cutoff_at)
    passed = {str(name) for name, status in session.execute(statement).all() if status == "pass"}
    return passed == REQUIRED_PREFERENCE_VALIDATORS


def _validated_vote_choice(vote: Vote) -> str:
    if vote.choice not in VALID_VOTE_CHOICES:
        raise RankingEvidenceError(f"invalid stored vote choice: {vote.choice!r}")
    return vote.choice


def _cutoff_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return normalized.astimezone(UTC).isoformat()


def _utc_datetime(value: datetime) -> datetime:
    return (value if value.tzinfo is not None else value.replace(tzinfo=UTC)).astimezone(UTC)


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
    require_snapshot_analysis_process()
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


def _comparison_components(
    comparisons: list[tuple[str, str, float]],
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for first, second, _ in comparisons:
        adjacency[first].add(second)
        adjacency[second].add(first)
    components: list[list[str]] = []
    unseen = set(adjacency)
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
    return sorted(components, key=lambda component: (component[0], len(component)))


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


def _paired_tie_aware_profile(wins: int, ties: int, losses: int) -> tuple[float, float, float]:
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
    controlled_run_id: str | None,
    evidence_cutoff_at: datetime | None = None,
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
            Battle.run_class == "official",
            Battle.manifest_sha256 == season.manifest_sha256,
            Battle.data_stratum == data_stratum,
        )
    )
    if data_stratum == "controlled":
        if controlled_run_id is None:
            raise ValueError("controlled leaderboards require a controlled run id")
        statement = statement.where(
            Battle.rank_eligible.is_(True),
            Battle.task_id.is_not(None),
            Battle.task_revision.is_not(None),
            Battle.controlled_run_id == controlled_run_id,
        )
    else:
        if controlled_run_id is not None:
            raise ValueError("public-freeform leaderboards cannot name a controlled run")
        statement = statement.where(
            Battle.task_id.is_(None),
            Battle.controlled_run_id.is_(None),
        )
    if cohort != "combined":
        statement = statement.where(Vote.cohort == cohort)
    if category != "all":
        statement = statement.where(Battle.category == category)
    if evidence_cutoff_at is not None:
        statement = statement.where(
            Battle.completed_at.is_not(None),
            Battle.completed_at <= evidence_cutoff_at,
            Vote.created_at <= evidence_cutoff_at,
        )
    rows = list(session.execute(statement.order_by(Battle.id, Vote.cohort, Vote.id)).all())
    return filter_ranking_vote_rows(
        session,
        rows,
        expert_quorum=get_settings().expert_output_comparison_quorum,
    )


def _task_validity_admissible(vote: Vote) -> bool:
    if not vote.cohort.startswith("expert_"):
        return True
    rubric = vote.rubric_json if isinstance(vote.rubric_json, dict) else {}
    metadata = rubric.get("review_metadata")
    return bool(
        rubric.get("rubric_version") == EXPERT_REVIEW_PROTOCOL_VERSION
        and isinstance(metadata, dict)
        and metadata.get("task_validity") in {"valid", "minor_issue"}
        and metadata.get("general_track_eligible") is True
    )


def analysis_vote_eligibility(
    vote: Vote,
    battle_selection: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve one vote's canonical role in the preference analysis."""

    choice = _validated_vote_choice(vote)
    selection = battle_selection.get(vote.battle_id)
    reasons: list[str] = []
    if selection is None:
        reasons.append("battle_outside_analysis_scope")
    elif not selection.get("preference_included"):
        reasons.extend(str(reason) for reason in selection.get("preference_exclusion_reasons", []))
    if not _task_validity_admissible(vote):
        reasons.append("expert_task_validity_inadmissible")
    if reasons:
        role = "excluded"
    elif choice == "both_bad":
        role = "failure_statistic_only"
        reasons.append("both_bad_excluded_from_preference_fit")
    else:
        role = "included"
    return {
        "preference_role": role,
        "preference_exclusion_reasons": sorted(set(reasons)),
    }


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _verified_public_scope_admissions(
    session: Session,
    events: list[RunEvent],
    battles: dict[str, Battle],
    *,
    evidence_cutoff_at: datetime | None,
) -> set[str]:
    """Validate the full sealed assessment and reviewer-admission chain."""

    verified_reviewers_by_battle: dict[str, set[str]] = defaultdict(set)
    for event in events:
        if (
            event.entity_type != "battle"
            or event.entity_id not in battles
            or event.event_type != "battle_general_track_scope_admitted"
        ):
            continue
        payload = event.payload_json if isinstance(event.payload_json, dict) else {}
        reviewer_id = payload.get("reviewer_id")
        review_session_id = payload.get("review_session_id")
        review_assignment_id = payload.get("review_assignment_id")
        assessment_sha256 = payload.get("assessment_sha256")
        presentation_sha256 = payload.get("presentation_sha256")
        if not (
            payload.get("general_track_eligible") is True
            and payload.get("ranking_use") is True
            and payload.get("scope_protocol_sha256") == EXPERT_REVIEW_PROTOCOL_SHA256
            and payload.get("scope_admission_quorum") == 1
            and payload.get("reviewer_cohort") == "expert_independent"
            and payload.get("reviewer_qualification_verified") is True
            and payload.get("affiliation_class") == "independent_external"
            and isinstance(reviewer_id, str)
            and isinstance(review_session_id, str)
            and isinstance(review_assignment_id, str)
            and _is_sha256(assessment_sha256)
            and _is_sha256(presentation_sha256)
        ):
            continue
        battle = battles[event.entity_id]
        reviewer = session.get(ExpertReviewer, reviewer_id)
        if reviewer is None:
            continue
        admission_event_id = payload.get("reviewer_admission_event_id")
        if not isinstance(admission_event_id, str):
            continue
        admission = historical_expert_admission_event(
            session,
            reviewer_id=reviewer_id,
            event_id=admission_event_id,
            as_of=evidence_cutoff_at or datetime.now(UTC),
        )
        admission_payload = (
            admission.payload_json
            if admission is not None and isinstance(admission.payload_json, dict)
            else {}
        )
        if not (
            admission is not None
            and admission_payload.get("cohort") == "expert_independent"
            and admission_payload.get("affiliation_class") == "independent_external"
            and battle.category in admission_payload.get("qualified_families", [])
            and payload.get("reviewer_admission_event_id") == admission.id
            and payload.get("reviewer_admission_evidence_sha256")
            == _canonical_sha256(admission.payload_json)
        ):
            continue
        assessed = session.scalar(
            select(RunEvent)
            .where(
                RunEvent.entity_type == "expert_review_assignment",
                RunEvent.entity_id == review_assignment_id,
                RunEvent.event_type == "expert_review_task_assessed",
                RunEvent.created_at <= event.created_at,
            )
            .order_by(RunEvent.created_at.desc(), RunEvent.id.desc())
        )
        assessed_payload = (
            assessed.payload_json
            if assessed is not None and isinstance(assessed.payload_json, dict)
            else {}
        )
        assessment = assessed_payload.get("assessment")
        if not (
            isinstance(assessment, dict)
            and assessment.get("general_track_eligible") is True
            and assessed_payload.get("reviewer_id") == reviewer.id
            and assessed_payload.get("review_session_id") == review_session_id
            and assessed_payload.get("battle_id") == battle.id
            and assessed_payload.get("assessment_sha256") == assessment_sha256
            and _canonical_sha256(assessment) == assessment_sha256
            and assessed_payload.get("presentation_sha256") == presentation_sha256
            and assessed_payload.get("protocol_sha256") == EXPERT_REVIEW_PROTOCOL_SHA256
            and _utc_datetime(admission.created_at)
            <= _utc_datetime(assessed.created_at)
            <= _utc_datetime(event.created_at)
        ):
            continue
        verified_reviewers_by_battle[battle.id].add(reviewer.id)
    return {
        battle_id
        for battle_id, reviewer_ids in verified_reviewers_by_battle.items()
        if len(reviewer_ids) >= 1
    }


def _ranking_control_sets(
    session: Session,
    battles: list[Battle],
    *,
    data_stratum: str,
    evidence_cutoff_at: datetime | None,
) -> dict[str, set[str]]:
    """Resolve append-only scope and containment events before any ranking use.

    Preference fitting fails closed on every scope or completion restriction.
    A response-level safety report is measured separately and does not suppress
    the comparison: doing so would condition the preference sample on a model's
    observed answer. Operational metrics keep contained response failures and
    safety reports in view, but still exclude inadmissible tasks, restricted
    human-review batches, and directly restricted battles.
    """

    battle_ids = {battle.id for battle in battles}
    if not battle_ids:
        return {
            "preference_admissible": set(),
            "operational_admissible": set(),
            "restricted_arm_ids": set(),
            "reported_safety_arm_ids": set(),
        }
    arm_to_battle = {
        arm_id: battle.id
        for battle in battles
        for arm_id in (battle.left_arm_id, battle.right_arm_id)
        if arm_id is not None
    }
    task_to_battles: dict[str, set[str]] = defaultdict(set)
    for battle in battles:
        if battle.task_id is not None:
            task_to_battles[battle.task_id].add(battle.id)

    statement = select(RunEvent).where(RunEvent.event_type.in_(RANKING_RESTRICTION_EVENT_TYPES))
    if evidence_cutoff_at is not None:
        statement = statement.where(RunEvent.created_at <= evidence_cutoff_at)
    events = session.scalars(statement.order_by(RunEvent.created_at, RunEvent.id)).all()

    restricted_sessions = {
        event.entity_id
        for event in events
        if event.entity_type == "expert_review_session"
        and event.event_type == "expert_review_batch_restricted"
        and event.payload_json.get("ranking_use") is False
    }
    task_restricted: set[str] = set()
    battle_restricted: set[str] = set()
    batch_restricted: set[str] = set()
    restricted_arm_ids: set[str] = set()
    reported_safety_arm_ids: set[str] = set()
    battles_by_id = {battle.id: battle for battle in battles}
    public_scope_admitted = _verified_public_scope_admissions(
        session,
        events,
        battles_by_id,
        evidence_cutoff_at=evidence_cutoff_at,
    )
    for event in events:
        payload = event.payload_json if isinstance(event.payload_json, dict) else {}
        if (
            event.entity_type == "battle"
            and event.entity_id in battle_ids
            and event.event_type == "battle_ranking_restricted"
            and payload.get("ranking_use") is False
        ):
            battle_restricted.add(event.entity_id)
        if (
            event.entity_type == "task"
            and event.entity_id in task_to_battles
            and event.event_type == "task_general_track_scope_quarantined"
            and payload.get("general_track_eligible") is False
        ):
            task_restricted.update(task_to_battles[event.entity_id])
        if (
            event.entity_type == "task"
            and event.entity_id in task_to_battles
            and event.event_type == "confirmatory_task_retired"
            and payload.get("ranking_use") is False
        ):
            task_restricted.update(task_to_battles[event.entity_id])
        if (
            event.entity_type == "response_arm"
            and event.entity_id in arm_to_battle
            and event.event_type == "response_arm_non_normal_completion_detected"
            and payload.get("ranking_use") is False
        ):
            restricted_arm_ids.add(event.entity_id)
        if (
            event.entity_type == "response_arm"
            and event.entity_id in arm_to_battle
            and event.event_type == "reviewer_reported_potential_safety_hazard"
            and payload.get("status") == "pending_qualified_food_safety_adjudication"
        ):
            reported_safety_arm_ids.add(event.entity_id)
        if (
            event.event_type == "expert_review_assignment_submitted"
            and payload.get("review_session_id") in restricted_sessions
            and payload.get("battle_id") in battle_ids
        ):
            batch_restricted.add(str(payload["battle_id"]))

    public_missing_scope = (
        battle_ids - public_scope_admitted if data_stratum == "public_freeform" else set()
    )
    arm_restricted_battles = {
        arm_to_battle[arm_id] for arm_id in restricted_arm_ids if arm_id in arm_to_battle
    }
    operational_restricted = (
        task_restricted | battle_restricted | batch_restricted | public_missing_scope
    )
    preference_restricted = operational_restricted | arm_restricted_battles
    return {
        "preference_admissible": battle_ids - preference_restricted,
        "operational_admissible": battle_ids - operational_restricted,
        "restricted_arm_ids": restricted_arm_ids,
        "reported_safety_arm_ids": reported_safety_arm_ids,
    }


def analysis_battle_eligibility(
    session: Session,
    season: Season,
    battles: Sequence[Battle],
    *,
    track: str,
    data_stratum: str,
    controlled_run_id: str | None,
    evidence_cutoff_at: datetime | None,
) -> dict[str, dict[str, Any]]:
    """Resolve the canonical as-of eligibility state for analysis evidence.

    This is deliberately shared by ranking and snapshot construction. Public
    battles begin with ``rank_eligible=False`` and become admissible only through
    the append-only scope-review chain; controlled battles remain governed by
    their frozen ``rank_eligible`` bit and controlled-run assignment.
    """

    if data_stratum not in {"public_freeform", "controlled"}:
        raise ValueError(f"unsupported leaderboard data stratum: {data_stratum}")
    if data_stratum == "controlled" and controlled_run_id is None:
        raise ValueError("controlled leaderboards require a controlled run id")
    if data_stratum == "public_freeform" and controlled_run_id is not None:
        raise ValueError("public-freeform leaderboards cannot name a controlled run")

    ordered = sorted({battle.id: battle for battle in battles}.values(), key=lambda row: row.id)
    controls = _ranking_control_sets(
        session,
        ordered,
        data_stratum=data_stratum,
        evidence_cutoff_at=evidence_cutoff_at,
    )
    resolved: dict[str, dict[str, Any]] = {}
    for battle in ordered:
        operational_reasons: list[str] = []
        if battle.track != track:
            operational_reasons.append("track_mismatch")
        if battle.data_stratum != data_stratum:
            operational_reasons.append("data_stratum_mismatch")
        if battle.run_class != "official":
            operational_reasons.append("run_class_not_official")
        if battle.manifest_sha256 != season.manifest_sha256:
            operational_reasons.append("season_manifest_mismatch")
        if battle.status not in {"complete", "failed"}:
            operational_reasons.append("battle_not_terminal")
        if data_stratum == "controlled":
            if not battle.rank_eligible:
                operational_reasons.append("rank_eligible_false")
            if battle.task_id is None or battle.task_revision is None:
                operational_reasons.append("controlled_task_identity_missing")
            if battle.controlled_run_id != controlled_run_id:
                operational_reasons.append("controlled_run_mismatch")
        else:
            if battle.task_id is not None or battle.task_revision is not None:
                operational_reasons.append("public_freeform_contains_registered_task")
            if battle.controlled_run_id is not None:
                operational_reasons.append("public_freeform_contains_controlled_run")
        if battle.id not in controls["operational_admissible"]:
            operational_reasons.append("governance_scope_or_restriction")

        preference_reasons = list(operational_reasons)
        if battle.status != "complete":
            preference_reasons.append("battle_not_complete")
        if battle.protocol_bundle_sha256 != season.protocol_bundle_sha256:
            preference_reasons.append("season_protocol_mismatch")
        if battle.id not in controls["preference_admissible"]:
            preference_reasons.append("preference_restricted")
        if (
            not preference_reasons
            and _ranked_arms(
                session,
                battle,
                track,
                evidence_cutoff_at=evidence_cutoff_at,
            )
            is None
        ):
            preference_reasons.append("arm_contract_status_or_cost_ineligible")

        resolved[battle.id] = {
            "operational_included": not operational_reasons,
            "preference_included": not preference_reasons,
            "operational_exclusion_reasons": sorted(set(operational_reasons)),
            "preference_exclusion_reasons": sorted(set(preference_reasons)),
            "restricted_arm_ids": sorted(
                arm_id
                for arm_id in (battle.left_arm_id, battle.right_arm_id)
                if arm_id in controls["restricted_arm_ids"]
            ),
            "reported_safety_arm_ids": sorted(
                arm_id
                for arm_id in (battle.left_arm_id, battle.right_arm_id)
                if arm_id in controls["reported_safety_arm_ids"]
            ),
        }
    return resolved


def _arm_matches_endpoint_contract(
    season: Season,
    battle: Battle,
    arm: ResponseArm,
    endpoint: SeasonModel | None,
) -> bool:
    if endpoint is None:
        return False
    if (
        season.protocol_bundle_sha256 in UNFROZEN_VALUES
        or hashlib.sha256(
            json.dumps(
                season.protocol_bundle_json,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        != season.protocol_bundle_sha256
        or battle.protocol_bundle_sha256 != season.protocol_bundle_sha256
        or arm.protocol_bundle_sha256 != season.protocol_bundle_sha256
        or endpoint.manifest_sha256 != battle.manifest_sha256
        or endpoint.endpoint_contract_sha256 in UNFROZEN_VALUES
        or (
            endpoint.execution_backend == "bedrock"
            and (
                endpoint.backend_contract_sha256 in UNFROZEN_VALUES
                or hashlib.sha256(
                    json.dumps(
                        endpoint.backend_contract_json,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                != endpoint.backend_contract_sha256
            )
        )
        or arm.execution_backend != endpoint.execution_backend
        or arm.provider_slug != endpoint.provider_slug
        or arm.actual_model_id != endpoint.expected_actual_model_id
        or arm.actual_provider_slug != endpoint.expected_actual_provider_slug
        or arm.system_prompt_sha256 != system_prompt_sha256(arm.condition)
        or arm.schema_sha256 != FINAL_SCHEMA_SHA256
        or arm.tool_schema_sha256 != season.tool_registry_sha256
        or arm.epicure_release_id != season.epicure_release_id
        or arm.epicure_bundle_sha256 != season.epicure_bundle_sha256
        or arm.epicure_application_sha256 != season.epicure_application_sha256
    ):
        return False
    if arm.condition == "epicure_on":
        attestation = arm.epicure_attestation_json
        expected_epicure = {
            "release_id": season.epicure_release_id,
            "bundle_sha256": season.epicure_bundle_sha256,
            "application_sha256": season.epicure_application_sha256,
            "tool_schema_sha256": season.tool_registry_sha256,
        }
        if (
            not isinstance(attestation, dict)
            or any(attestation.get(field) != value for field, value in expected_epicure.items())
            or not arm.epicure_attestation_sha256
            or hashlib.sha256(
                json.dumps(attestation, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            != arm.epicure_attestation_sha256
        ):
            return False
    elif arm.epicure_attestation_json or arm.epicure_attestation_sha256:
        return False
    expected_decoding = {
        name: endpoint.decoding_json.get(name, "provider_fixed_unsupported")
        for name in sorted(("max_tokens", "seed", "temperature", "top_p"))
    }
    if arm.observed_decoding_json != expected_decoding:
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


def _arm_is_analysis_valid(
    session: Session,
    season: Season,
    battle: Battle,
    arm: ResponseArm,
    endpoint: SeasonModel | None,
    *,
    evidence_cutoff_at: datetime | None,
) -> bool:
    return bool(
        _arm_matches_endpoint_contract(season, battle, arm, endpoint)
        and _arm_has_required_preference_validators(
            session,
            arm.id,
            evidence_cutoff_at=evidence_cutoff_at,
        )
        and arm.status == "complete"
        and is_complete_finish_reason(arm.finish_reason)
        and not arm.model_id.startswith("flavourbench/mock-")
        and arm.provider_slug != "mock"
        and (arm.actual_provider_slug or "").lower() != "mock"
        and arm.cost_reconciled
        and (
            arm.execution_backend != "bedrock"
            or _has_active_bedrock_billing_crosscheck(
                session,
                arm.id,
                evidence_cutoff_at=evidence_cutoff_at,
            )
        )
    )


def _ranked_arms(
    session: Session,
    battle: Battle,
    track: str,
    evidence_cutoff_at: datetime | None = None,
) -> tuple[ResponseArm, ResponseArm] | None:
    if (
        battle.left_arm_id is None
        or battle.right_arm_id is None
        or battle.left_arm_id == battle.right_arm_id
    ):
        return None
    season = session.get(Season, battle.season_id)
    left = session.get(ResponseArm, battle.left_arm_id)
    right = session.get(ResponseArm, battle.right_arm_id)
    if not season or not left or not right:
        return None
    if (
        left.battle_id != battle.id
        or right.battle_id != battle.id
        or left.side != "left"
        or right.side != "right"
    ):
        return None
    if battle.controlled_run_id is not None:
        assignment = session.scalar(
            select(ControlledRunAssignment).where(
                ControlledRunAssignment.controlled_run_id == battle.controlled_run_id,
                ControlledRunAssignment.battle_id == battle.id,
            )
        )
        if assignment is None:
            return None
        try:
            verify_controlled_assignment_battle(
                session,
                assignment,
                battle,
                require_terminal=True,
            )
        except ControlledRunIntegrityError:
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
    if any(
        not _arm_is_analysis_valid(
            session,
            season,
            battle,
            arm,
            endpoint_contracts.get(arm.model_id),
            evidence_cutoff_at=evidence_cutoff_at,
        )
        for arm in arms
    ):
        return None
    if track == "model_arena":
        if any(arm.condition != "epicure_on" for arm in arms):
            return None
    elif left.model_id != right.model_id or {left.condition, right.condition} != {
        "epicure_on",
        "epicure_off",
    }:
        return None
    return left, right


def _has_active_bedrock_billing_crosscheck(
    session: Session,
    arm_id: str,
    evidence_cutoff_at: datetime | None = None,
) -> bool:
    statement = (
        select(BedrockBillingCrosscheck.id, BedrockBillingCrosscheck.status)
        .join(
            BedrockBillingCrosscheckArm,
            BedrockBillingCrosscheckArm.crosscheck_id == BedrockBillingCrosscheck.id,
        )
        .where(BedrockBillingCrosscheckArm.arm_id == arm_id)
    )
    if evidence_cutoff_at is not None:
        statement = statement.where(
            BedrockBillingCrosscheck.created_at <= evidence_cutoff_at,
            BedrockBillingCrosscheckArm.created_at <= evidence_cutoff_at,
        )
    rows = session.execute(statement).all()
    if not rows:
        return False
    identifiers = {identifier for identifier, _status in rows}
    superseded_statement = select(BedrockBillingCrosscheck.supersedes_crosscheck_id).where(
        BedrockBillingCrosscheck.supersedes_crosscheck_id.in_(identifiers)
    )
    if evidence_cutoff_at is not None:
        superseded_statement = superseded_statement.where(
            BedrockBillingCrosscheck.created_at <= evidence_cutoff_at
        )
    superseded = {
        value for value in session.scalars(superseded_statement).all() if value is not None
    }
    active = [status for identifier, status in rows if identifier not in superseded]
    return active == ["accepted"]


def _operational_metrics(
    session: Session,
    season: Season,
    track: str,
    category: str,
    data_stratum: str,
    controlled_run_id: str | None,
    evidence_cutoff_at: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    if data_stratum not in {"public_freeform", "controlled"}:
        raise ValueError(f"unsupported leaderboard data stratum: {data_stratum}")
    statement = (
        select(ResponseArm, Battle)
        .join(Battle, Battle.id == ResponseArm.battle_id)
        .where(
            Battle.season_id == season.id,
            Battle.track == track,
            Battle.run_class == "official",
            Battle.manifest_sha256 == season.manifest_sha256,
            Battle.data_stratum == data_stratum,
            Battle.status.in_(("complete", "failed")),
            ~ResponseArm.model_id.like("flavourbench/mock-%"),
            ResponseArm.provider_slug != "mock",
        )
    )
    if data_stratum == "controlled":
        if controlled_run_id is None:
            raise ValueError("controlled leaderboards require a controlled run id")
        statement = statement.where(
            Battle.rank_eligible.is_(True),
            Battle.task_id.is_not(None),
            Battle.task_revision.is_not(None),
            Battle.controlled_run_id == controlled_run_id,
        )
    else:
        if controlled_run_id is not None:
            raise ValueError("public-freeform leaderboards cannot name a controlled run")
        statement = statement.where(
            Battle.task_id.is_(None),
            Battle.controlled_run_id.is_(None),
        )
    if category != "all":
        statement = statement.where(Battle.category == category)
    if evidence_cutoff_at is not None:
        statement = statement.where(
            Battle.completed_at.is_not(None),
            Battle.completed_at <= evidence_cutoff_at,
            ResponseArm.created_at <= evidence_cutoff_at,
        )
    endpoint_contracts = {
        slot.model_id: slot
        for slot in session.scalars(
            select(SeasonModel).where(SeasonModel.season_id == season.id)
        ).all()
    }
    rows = list(session.execute(statement).all())
    eligibility = analysis_battle_eligibility(
        session,
        season,
        list({battle.id: battle for _arm, battle in rows}.values()),
        track=track,
        data_stratum=data_stratum,
        controlled_run_id=controlled_run_id,
        evidence_cutoff_at=evidence_cutoff_at,
    )
    metrics: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    for arm, battle in rows:
        selection = eligibility[battle.id]
        if not selection["operational_included"]:
            continue
        record = metrics[arm.model_id]
        record["arms"] += 1
        record["invalid"] += (
            arm.status != "complete"
            or not is_complete_finish_reason(arm.finish_reason)
            or not _arm_has_required_preference_validators(
                session,
                arm.id,
                evidence_cutoff_at=evidence_cutoff_at,
            )
            or not _arm_matches_endpoint_contract(
                season,
                battle,
                arm,
                endpoint_contracts.get(arm.model_id),
            )
        )
        record["preference_restricted_arms"] += arm.id in selection["restricted_arm_ids"]
        record["reported_potential_safety_arms"] += arm.id in selection["reported_safety_arm_ids"]
        if arm.cost_reconciled:
            record["cost"] += arm.cost_micros
            record["cost_reconciled_arms"] += 1
        else:
            record["cost_unreconciled_arms"] += 1
        if arm.execution_backend == "bedrock" and not _has_active_bedrock_billing_crosscheck(
            session,
            arm.id,
            evidence_cutoff_at=evidence_cutoff_at,
        ):
            record["billing_crosscheck_pending_arms"] += 1
        if arm.latency_ms > 0:
            record.setdefault("latencies", []).append(arm.latency_ms)
        call_statement = select(ToolCall).where(ToolCall.arm_id == arm.id)
        if evidence_cutoff_at is not None:
            call_statement = call_statement.where(ToolCall.created_at <= evidence_cutoff_at)
        calls = session.scalars(call_statement).all()
        record["tool_calls"] += len(calls)
        record["tool_successes"] += sum(not call.is_error for call in calls)
        validator_statement = select(ValidatorResult).where(
            ValidatorResult.arm_id == arm.id,
            ValidatorResult.validator_name == TASK_SPECIFIC_VALIDATOR_NAME,
            ValidatorResult.validator_version == TASK_SPECIFIC_VALIDATOR_VERSION,
            ValidatorResult.status != "not_applicable",
        )
        if evidence_cutoff_at is not None:
            validator_statement = validator_statement.where(
                ValidatorResult.created_at <= evidence_cutoff_at
            )
        validators = session.scalars(validator_statement).all()
        record["constraint_checks"] += len(validators)
        record["constraint_passes"] += sum(item.status == "pass" for item in validators)
    return metrics


def _metric_payload(record: dict[str, Any]) -> dict[str, float | int | bool | None]:
    arms = int(record["arms"])
    calls = int(record["tool_calls"])
    constraint_checks = int(record["constraint_checks"])
    cost_reconciled_arms = int(record["cost_reconciled_arms"])
    cost_unreconciled_arms = int(record["cost_unreconciled_arms"])
    billing_crosscheck_pending_arms = int(record["billing_crosscheck_pending_arms"])
    preference_restricted_arms = int(record["preference_restricted_arms"])
    reported_potential_safety_arms = int(record["reported_potential_safety_arms"])
    latencies = sorted(int(value) for value in record.get("latencies", []))
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1) if latencies else None
    return {
        "response_arms": arms,
        "invalid_response_rate": round(record["invalid"] / arms, 4) if arms else None,
        "preference_restricted_arms": preference_restricted_arms,
        "reported_potential_safety_arms": reported_potential_safety_arms,
        "reported_potential_safety_rate": (
            round(reported_potential_safety_arms / arms, 4) if arms else None
        ),
        "executable_constraint_subset_pass_rate": round(
            record["constraint_passes"] / constraint_checks, 4
        )
        if constraint_checks
        else None,
        "executable_constraint_subset_n": constraint_checks,
        "tool_success_rate": round(record["tool_successes"] / calls, 4) if calls else None,
        "average_cost_micros": (
            round(record["cost"] / cost_reconciled_arms) if cost_reconciled_arms else None
        ),
        "cost_reconciled_arms": cost_reconciled_arms,
        "cost_unreconciled_arms": cost_unreconciled_arms,
        "cost_accounting_complete": arms > 0 and cost_unreconciled_arms == 0,
        "billing_crosscheck_pending_arms": billing_crosscheck_pending_arms,
        "billing_reconciliation_complete": billing_crosscheck_pending_arms == 0,
        "latency_n": len(latencies),
        "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
        "p95_latency_ms": latencies[p95_index] if p95_index is not None else None,
    }


def _accounting_summary(metrics: dict[str, dict[str, Any]]) -> dict[str, int | bool]:
    response_arms = sum(int(record["arms"]) for record in metrics.values())
    reconciled = sum(int(record["cost_reconciled_arms"]) for record in metrics.values())
    unreconciled = sum(int(record["cost_unreconciled_arms"]) for record in metrics.values())
    billing_pending = sum(
        int(record["billing_crosscheck_pending_arms"]) for record in metrics.values()
    )
    return {
        "response_arms": response_arms,
        "cost_reconciled_arms": reconciled,
        "cost_unreconciled_arms": unreconciled,
        "complete": response_arms > 0 and unreconciled == 0 and reconciled == response_arms,
        "billing_crosscheck_pending_arms": billing_pending,
        "billing_reconciliation_complete": billing_pending == 0,
    }


def _controlled_analysis_roster(
    session: Session,
    season: Season,
    controlled_run_id: str,
) -> list[str]:
    run = session.get(ControlledRun, controlled_run_id)
    if run is None or run.season_id != season.id:
        raise RankingEvidenceError("controlled analysis run is unavailable")
    raw = run.model_roster_json
    if not isinstance(raw, list) or not raw:
        raise RankingEvidenceError("controlled analysis roster is missing")
    roster: list[str] = []
    for item in raw:
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict) and isinstance(item.get("model_id"), str):
            model_id = str(item["model_id"])
        else:
            raise RankingEvidenceError("controlled analysis roster is malformed")
        roster.append(model_id)
    if len(roster) != len(set(roster)):
        raise RankingEvidenceError("controlled analysis roster contains duplicates")
    bound = {
        item.model_id: item
        for item in session.scalars(
            select(SeasonModel).where(
                SeasonModel.season_id == season.id,
                SeasonModel.model_id.in_(roster),
            )
        ).all()
    }
    if set(bound) != set(roster) or any(
        not item.eligible or item.manifest_sha256 != season.manifest_sha256
        for item in bound.values()
    ):
        raise RankingEvidenceError(
            "controlled analysis roster is not the eligible frozen endpoint roster"
        )
    return sorted(roster)


def _controlled_uplift_schedule(
    session: Session,
    season: Season,
    controlled_run_id: str,
    *,
    category: str,
    evidence_cutoff_at: datetime | None,
) -> list[UpliftScheduledPair]:
    statement = (
        select(ControlledRunAssignment)
        .where(
            ControlledRunAssignment.controlled_run_id == controlled_run_id,
            ControlledRunAssignment.track == "epicure_uplift",
        )
        .order_by(ControlledRunAssignment.ordinal, ControlledRunAssignment.id)
    )
    if category != "all":
        statement = statement.where(ControlledRunAssignment.task_family == category)
    if evidence_cutoff_at is not None:
        statement = statement.where(ControlledRunAssignment.created_at <= evidence_cutoff_at)
    assignments = session.scalars(statement).all()
    battle_ids = [row.battle_id for row in assignments if row.battle_id is not None]
    battles = (
        session.scalars(select(Battle).where(Battle.id.in_(battle_ids))).all() if battle_ids else []
    )
    battles_by_id = {row.id: row for row in battles}
    eligibility = analysis_battle_eligibility(
        session,
        season,
        battles,
        track="epicure_uplift",
        data_stratum="controlled",
        controlled_run_id=controlled_run_id,
        evidence_cutoff_at=evidence_cutoff_at,
    )
    endpoint_contracts = {
        item.model_id: item
        for item in session.scalars(
            select(SeasonModel).where(SeasonModel.season_id == season.id)
        ).all()
    }
    schedule: list[UpliftScheduledPair] = []
    for assignment in assignments:
        model_ids = assignment.model_ids_json
        if not (
            isinstance(model_ids, list) and len(model_ids) == 1 and isinstance(model_ids[0], str)
        ):
            raise RankingEvidenceError("Season 1 uplift assignment must contain exactly one model")
        model_id = model_ids[0]
        battle = battles_by_id.get(assignment.battle_id or "")
        valid_by_condition = {"epicure_on": False, "epicure_off": False}
        if (
            battle is not None
            and eligibility[battle.id]["operational_included"]
            and battle.left_arm_id is not None
            and battle.right_arm_id is not None
        ):
            arms = session.scalars(
                select(ResponseArm).where(
                    ResponseArm.id.in_([battle.left_arm_id, battle.right_arm_id])
                )
            ).all()
            for arm in arms:
                if arm.model_id != model_id or arm.condition not in valid_by_condition:
                    continue
                valid_by_condition[arm.condition] = _arm_is_analysis_valid(
                    session,
                    season,
                    battle,
                    arm,
                    endpoint_contracts.get(model_id),
                    evidence_cutoff_at=evidence_cutoff_at,
                )
        schedule.append(
            UpliftScheduledPair(
                pair_id=battle.id if battle is not None else assignment.id,
                task_id=assignment.task_id,
                family=assignment.task_family,
                model_id=model_id,
                repetition_index=assignment.repetition_index,
                epicure_valid=valid_by_condition["epicure_on"],
                unaided_valid=valid_by_condition["epicure_off"],
            )
        )
    return schedule


def _controlled_arena_admitted_tasks(
    session: Session,
    controlled_run_id: str,
    *,
    category: str,
) -> dict[str, str]:
    statement = select(ControlledRunAssignment).where(
        ControlledRunAssignment.controlled_run_id == controlled_run_id,
        ControlledRunAssignment.track == "model_arena",
    )
    if category != "all":
        statement = statement.where(ControlledRunAssignment.task_family == category)
    assignments = session.scalars(statement.order_by(ControlledRunAssignment.ordinal)).all()
    admitted: dict[str, str] = {}
    for assignment in assignments:
        prior = admitted.setdefault(assignment.task_id, assignment.task_family)
        if prior != assignment.task_family:
            raise RankingEvidenceError("controlled arena task maps to multiple families")
    return admitted


def _controlled_postcollection_item_audit(
    session: Session,
    controlled_run_id: str,
    *,
    evidence_cutoff_at: datetime | None,
) -> Mapping[str, Any] | None:
    statement = select(RunEvent).where(
        RunEvent.entity_type == "controlled_run",
        RunEvent.entity_id == controlled_run_id,
        RunEvent.event_type == "season1_post_collection_item_audit_verified",
    )
    if evidence_cutoff_at is not None:
        statement = statement.where(RunEvent.created_at <= evidence_cutoff_at)
    events = session.scalars(statement.order_by(RunEvent.created_at, RunEvent.id)).all()
    if not events:
        return None
    identifiers = {event.id for event in events}
    superseded = {
        str(event.payload_json.get("supersedes_event_id"))
        for event in events
        if event.payload_json.get("supersedes_event_id") is not None
    }
    if not superseded.issubset(identifiers):
        return None
    heads = [event for event in events if event.id not in superseded]
    if len(heads) != 1:
        return None
    payload = heads[0].payload_json
    artifact = payload.get("artifact")
    if (
        not isinstance(artifact, Mapping)
        or payload.get("artifact_sha256") != artifact.get("artifact_sha256")
        or payload.get("verification_status") != "verified"
        or payload.get("verifier") != "flavourbench.season1_readiness"
    ):
        return None
    return artifact


def _controlled_arena_method_validation(
    session: Session,
    controlled_run_id: str,
    *,
    evidence_cutoff_at: datetime | None,
) -> Mapping[str, Any] | None:
    statement = select(RunEvent).where(
        RunEvent.entity_type == "controlled_run",
        RunEvent.entity_id == controlled_run_id,
        RunEvent.event_type == "season1_arena_monte_carlo_validation_verified",
    )
    if evidence_cutoff_at is not None:
        statement = statement.where(RunEvent.created_at <= evidence_cutoff_at)
    events = session.scalars(statement.order_by(RunEvent.created_at, RunEvent.id)).all()
    if len(events) != 1:
        return None
    payload = events[0].payload_json
    artifact = payload.get("artifact")
    if (
        not isinstance(artifact, Mapping)
        or payload.get("artifact_sha256") != artifact.get("artifact_sha256")
        or payload.get("verification_status") != "verified"
        or payload.get("verifier") != "flavourbench.season1_arena_monte_carlo"
    ):
        return None
    return artifact


def model_leaderboard(
    session: Session,
    season: Season,
    cohort: str,
    category: str,
    data_stratum: str = "public_freeform",
    controlled_run_id: str | None = None,
    evidence_cutoff_at: datetime | None = None,
) -> dict[str, Any]:
    data = _votes(
        session,
        season,
        "model_arena",
        cohort,
        category,
        data_stratum,
        controlled_run_id,
        evidence_cutoff_at,
    )
    comparisons: list[tuple[str, str, float]] = []
    controlled_observations: list[ArenaObservation] = []
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    excluded_invalid_task_judgments = 0
    eligibility = analysis_battle_eligibility(
        session,
        season,
        list({battle.id: battle for _vote, battle in data}.values()),
        track="model_arena",
        data_stratum=data_stratum,
        controlled_run_id=controlled_run_id,
        evidence_cutoff_at=evidence_cutoff_at,
    )
    excluded_restricted_judgments = 0
    eligible_judgment_ids: list[str] = []
    preference_observation_ids: list[str] = []
    eligible_raters_by_battle: dict[str, set[str]] = defaultdict(set)
    for vote, battle in data:
        vote_selection = analysis_vote_eligibility(vote, eligibility)
        if vote_selection["preference_role"] == "excluded":
            if (
                "expert_task_validity_inadmissible"
                in vote_selection["preference_exclusion_reasons"]
            ):
                excluded_invalid_task_judgments += 1
            else:
                excluded_restricted_judgments += 1
            continue
        eligible_judgment_ids.append(vote.id)
        choice = _validated_vote_choice(vote)
        if vote_selection["preference_role"] == "included":
            preference_observation_ids.append(vote.id)
        arms = _ranked_arms(
            session,
            battle,
            "model_arena",
            evidence_cutoff_at=evidence_cutoff_at,
        )
        if arms is None:
            continue
        left, right = arms
        counts[left.model_id]["judgments"] += 1
        counts[right.model_id]["judgments"] += 1
        if choice == "both_bad":
            counts[left.model_id]["both_bad"] += 1
            counts[right.model_id]["both_bad"] += 1
            continue
        eligible_raters_by_battle[battle.id].add(vote.rater_pseudonym)
        counts[left.model_id]["appearances"] += 1
        counts[right.model_id]["appearances"] += 1
        outcome = 1.0 if choice == "left" else 0.0 if choice == "right" else 0.5
        comparisons.append((left.model_id, right.model_id, outcome))
        if data_stratum == "controlled":
            if battle.task_id is None:
                raise RankingEvidenceError("controlled preference lacks a task identity")
            controlled_observations.append(
                ArenaObservation(
                    observation_id=vote.id,
                    task_id=battle.task_id,
                    family=battle.category,
                    battle_id=battle.id,
                    rater_id=vote.rater_pseudonym,
                    model_a=left.model_id,
                    model_b=right.model_id,
                    response_a_id=left.id,
                    response_b_id=right.id,
                    outcome=outcome,
                )
            )
        counts[left.model_id]["cost"] += left.cost_micros
        counts[right.model_id]["cost"] += right.cost_micros
        counts[left.model_id]["latency"] += left.latency_ms
        counts[right.model_id]["latency"] += right.latency_ms
    operational = _operational_metrics(
        session,
        season,
        "model_arena",
        category,
        data_stratum,
        controlled_run_id,
        evidence_cutoff_at,
    )
    minimum = 40 if cohort.startswith("expert_") else 100
    if data_stratum == "controlled" and season.slug == "season-1":
        if controlled_run_id is None:
            raise RankingEvidenceError("Season 1 controlled analysis lacks a run")
        roster = _controlled_analysis_roster(session, season, controlled_run_id)
        admitted_tasks = _controlled_arena_admitted_tasks(
            session,
            controlled_run_id,
            category=category,
        )
        postcollection_item_audit = _controlled_postcollection_item_audit(
            session,
            controlled_run_id,
            evidence_cutoff_at=evidence_cutoff_at,
        )
        production_layout_method_validation = _controlled_arena_method_validation(
            session,
            controlled_run_id,
            evidence_cutoff_at=evidence_cutoff_at,
        )
        try:
            controlled = analyze_controlled_arena(
                controlled_observations,
                roster,
                view=category,
                admitted_tasks=admitted_tasks,
                comparison_raters={
                    battle_id: sorted(raters)
                    for battle_id, raters in eligible_raters_by_battle.items()
                },
                postcollection_item_audit=postcollection_item_audit,
            )
        except StatisticalContractError as exc:
            raise RankingEvidenceError(f"Season 1 arena contract failed: {exc}") from exc
        controlled_rows = []
        for row in controlled["rows"]:
            model_id = str(row["competitor_id"])
            stats = counts[model_id]
            judgments = int(stats["judgments"])
            enriched = {
                **row,
                "battles": int(row.get("unique_battles", 0)),
                "judgments": judgments,
                "both_bad": int(stats["both_bad"]),
                "both_bad_rate": (round(stats["both_bad"] / judgments, 4) if judgments else 0),
                "provisional": (
                    controlled["statistical_acceptance"]["status"] != "pass"
                    or int(row.get("unique_battles", 0)) < minimum
                ),
            }
            enriched.update(_metric_payload(operational[model_id]))
            controlled_rows.append(enriched)
        return {
            **controlled,
            "track": "model_arena",
            "cohort": cohort,
            "cohort_label": ("Combined · secondary" if cohort == "combined" else cohort.title()),
            "category": category,
            "data_stratum": data_stratum,
            "controlled_run_id": controlled_run_id,
            "evidence_cutoff_at": _cutoff_iso(evidence_cutoff_at),
            "rows": controlled_rows,
            "production_layout_method_validation": production_layout_method_validation,
            "eligible_judgment_ids": sorted(eligible_judgment_ids),
            "preference_observation_ids": sorted(preference_observation_ids),
            "preference_observation_sha256": _canonical_sha256(
                {"vote_ids": sorted(preference_observation_ids)}
            ),
            "rater_coverage": {
                "unique_comparisons": len(eligible_raters_by_battle),
                "minimum_distinct_raters_per_comparison": min(
                    (len(values) for values in eligible_raters_by_battle.values()),
                    default=0,
                ),
                "comparisons_with_two_or_more_distinct_raters": sum(
                    len(values) >= 2 for values in eligible_raters_by_battle.values()
                ),
            },
            "accounting": _accounting_summary(operational),
            "manifest_sha256": season.manifest_sha256,
            "excluded_invalid_task_judgments": excluded_invalid_task_judgments,
            "excluded_restricted_judgments": excluded_restricted_judgments,
            "eligibility_filter": {
                "resolver": "canonical_as_of_analysis_frame_v1",
                "run_class": "official",
                "rank_eligible": True,
                "manifest_match": True,
                "data_stratum": "controlled",
                "controlled_run_id": controlled_run_id,
                "mock_models": "excluded",
                "identity_and_cost_reconciliation": "required_for_preference",
                "expert_task_validity": (
                    "valid_or_minor_issue_and_general_track_eligible_from_sealed_assessment"
                ),
            },
        }
    components = _comparison_components(comparisons)
    graph_connected = len(components) <= 1
    ratings = _fit_bradley_terry(comparisons, require_arena_rank=True) if graph_connected else {}
    rows = []
    competitors = set(ratings) | set(operational)
    for model_id in sorted(
        competitors,
        key=lambda item: (
            -ratings.get(item, (-math.inf, 0, 0))[0],
            item,
        ),
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
        "controlled_run_id": controlled_run_id,
        "evidence_cutoff_at": _cutoff_iso(evidence_cutoff_at),
        "rows": rows,
        "method": (
            "arena-rank 0.1.1 Bradley-Terry with ties as half-wins and 95% intervals"
            if graph_connected and ratings
            else None
        ),
        "ranking_status": (
            "estimated"
            if graph_connected and ratings
            else "withheld_disconnected_graph"
            if not graph_connected
            else "insufficient_comparisons"
        ),
        "comparison_graph_connected": graph_connected,
        "comparison_components": components,
        "eligible_judgment_ids": sorted(eligible_judgment_ids),
        "preference_observation_ids": sorted(preference_observation_ids),
        "preference_observation_sha256": _canonical_sha256(
            {"vote_ids": sorted(preference_observation_ids)}
        ),
        "accounting": _accounting_summary(operational),
        "manifest_sha256": season.manifest_sha256,
        "excluded_invalid_task_judgments": excluded_invalid_task_judgments,
        "excluded_restricted_judgments": excluded_restricted_judgments,
        "eligibility_filter": {
            "run_class": "official",
            "rank_eligible": (
                True
                if data_stratum == "controlled"
                else "append_only_general_track_scope_admission_required"
            ),
            "manifest_match": True,
            "data_stratum": data_stratum,
            "controlled_run_id": controlled_run_id,
            "mock_models": "excluded",
            "identity_and_cost_reconciliation": "required_for_preference",
            "expert_task_validity": (
                "valid_or_minor_issue_and_general_track_eligible_from_sealed_assessment"
            ),
        },
    }


def uplift_leaderboard(
    session: Session,
    season: Season,
    cohort: str,
    category: str,
    data_stratum: str = "public_freeform",
    controlled_run_id: str | None = None,
    evidence_cutoff_at: datetime | None = None,
) -> dict[str, Any]:
    data = _votes(
        session,
        season,
        "epicure_uplift",
        cohort,
        category,
        data_stratum,
        controlled_run_id,
        evidence_cutoff_at,
    )
    operational = _operational_metrics(
        session,
        season,
        "epicure_uplift",
        category,
        data_stratum,
        controlled_run_id,
        evidence_cutoff_at,
    )
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    controlled_judgments: list[UpliftJudgment] = []
    excluded_invalid_task_judgments = 0
    eligibility = analysis_battle_eligibility(
        session,
        season,
        list({battle.id: battle for _vote, battle in data}.values()),
        track="epicure_uplift",
        data_stratum=data_stratum,
        controlled_run_id=controlled_run_id,
        evidence_cutoff_at=evidence_cutoff_at,
    )
    excluded_restricted_judgments = 0
    eligible_judgment_ids: list[str] = []
    preference_observation_ids: list[str] = []
    eligible_raters_by_battle: dict[str, set[str]] = defaultdict(set)
    for vote, battle in data:
        vote_selection = analysis_vote_eligibility(vote, eligibility)
        if vote_selection["preference_role"] == "excluded":
            if (
                "expert_task_validity_inadmissible"
                in vote_selection["preference_exclusion_reasons"]
            ):
                excluded_invalid_task_judgments += 1
            else:
                excluded_restricted_judgments += 1
            continue
        eligible_judgment_ids.append(vote.id)
        choice = _validated_vote_choice(vote)
        if vote_selection["preference_role"] == "included":
            preference_observation_ids.append(vote.id)
        arms = _ranked_arms(
            session,
            battle,
            "epicure_uplift",
            evidence_cutoff_at=evidence_cutoff_at,
        )
        if arms is None:
            continue
        left, right = arms
        record = stats[left.model_id]
        record["judgments"] += 1
        if choice == "both_bad":
            record["both_bad"] += 1
            if data_stratum == "controlled":
                if battle.task_id is None:
                    raise RankingEvidenceError("controlled uplift vote lacks a task identity")
                controlled_judgments.append(
                    UpliftJudgment(
                        judgment_id=vote.id,
                        task_id=battle.task_id,
                        family=battle.category,
                        battle_id=battle.id,
                        rater_id=vote.rater_pseudonym,
                        model_id=left.model_id,
                        choice="both_bad",
                    )
                )
            continue
        eligible_raters_by_battle[battle.id].add(vote.rater_pseudonym)
        record["total"] += 1
        if choice == "tie":
            record["tie"] += 1
            normalized_choice = "tie"
        else:
            winning_condition = left.condition if choice == "left" else right.condition
            record["epicure_win" if winning_condition == "epicure_on" else "unaided_win"] += 1
            normalized_choice = (
                "epicure_win" if winning_condition == "epicure_on" else "unaided_win"
            )
        if data_stratum == "controlled":
            if battle.task_id is None:
                raise RankingEvidenceError("controlled uplift vote lacks a task identity")
            controlled_judgments.append(
                UpliftJudgment(
                    judgment_id=vote.id,
                    task_id=battle.task_id,
                    family=battle.category,
                    battle_id=battle.id,
                    rater_id=vote.rater_pseudonym,
                    model_id=left.model_id,
                    choice=normalized_choice,
                )
            )
    minimum = 20 if cohort.startswith("expert_") else 50
    if data_stratum == "controlled" and season.slug == "season-1" and category == "all":
        if controlled_run_id is None:
            raise RankingEvidenceError("Season 1 controlled uplift lacks a run")
        roster = _controlled_analysis_roster(session, season, controlled_run_id)
        schedule = _controlled_uplift_schedule(
            session,
            season,
            controlled_run_id,
            category=category,
            evidence_cutoff_at=evidence_cutoff_at,
        )
        try:
            controlled = analyze_controlled_uplift(
                controlled_judgments,
                schedule,
                roster,
            )
        except StatisticalContractError as exc:
            raise RankingEvidenceError(f"Season 1 uplift contract failed: {exc}") from exc
        controlled_rows = []
        for row in controlled["rows"]:
            model_id = str(row["competitor_id"])
            record = stats[model_id]
            judgments = int(record["judgments"])
            enriched = {
                **row,
                "epicure_wins": int(record["epicure_win"]),
                "unaided_wins": int(record["unaided_win"]),
                "ties": int(record["tie"]),
                "both_bad": int(record["both_bad"]),
                "both_bad_rate": (round(record["both_bad"] / judgments, 4) if judgments else 0),
                "battles": int(record["total"]),
                "judgments": judgments,
                "provisional": bool(row.get("provisional")) or int(record["total"]) < minimum,
            }
            enriched.update(_metric_payload(operational[model_id]))
            controlled_rows.append(enriched)
        return {
            **controlled,
            "track": "epicure_uplift",
            "cohort": cohort,
            "cohort_label": ("Combined · secondary" if cohort == "combined" else cohort.title()),
            "category": category,
            "data_stratum": data_stratum,
            "controlled_run_id": controlled_run_id,
            "evidence_cutoff_at": _cutoff_iso(evidence_cutoff_at),
            "rows": controlled_rows,
            "eligible_judgment_ids": sorted(eligible_judgment_ids),
            "preference_observation_ids": sorted(preference_observation_ids),
            "preference_observation_sha256": _canonical_sha256(
                {"vote_ids": sorted(preference_observation_ids)}
            ),
            "rater_coverage": {
                "unique_comparisons": len(eligible_raters_by_battle),
                "minimum_distinct_raters_per_comparison": min(
                    (len(values) for values in eligible_raters_by_battle.values()),
                    default=0,
                ),
                "comparisons_with_two_or_more_distinct_raters": sum(
                    len(values) >= 2 for values in eligible_raters_by_battle.values()
                ),
            },
            "accounting": _accounting_summary(operational),
            "manifest_sha256": season.manifest_sha256,
            "excluded_invalid_task_judgments": excluded_invalid_task_judgments,
            "excluded_restricted_judgments": excluded_restricted_judgments,
            "eligibility_filter": {
                "resolver": "canonical_as_of_analysis_frame_v1",
                "run_class": "official",
                "rank_eligible": True,
                "manifest_match": True,
                "data_stratum": "controlled",
                "controlled_run_id": controlled_run_id,
                "mock_models": "excluded",
                "identity_and_cost_reconciliation": "required_for_preference",
                "expert_task_validity": (
                    "valid_or_minor_issue_and_general_track_eligible_from_sealed_assessment"
                ),
            },
        }
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
            "both_bad_rate": (
                round(record["both_bad"] / record["judgments"], 4) if record["judgments"] else 0
            ),
            "battles": record["total"],
            "judgments": record["judgments"],
            "provisional": record["total"] < minimum,
        }
        row.update(_metric_payload(operational[model_id]))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["epicure_win_share"],
            row["competitor_id"],
        )
    )
    return {
        "track": "epicure_uplift",
        "cohort": cohort,
        "cohort_label": "Combined · secondary" if cohort == "combined" else cohort.title(),
        "category": category,
        "data_stratum": data_stratum,
        "controlled_run_id": controlled_run_id,
        "evidence_cutoff_at": _cutoff_iso(evidence_cutoff_at),
        "rows": rows,
        "method": (
            "observed tie-half preference with a 95% multinomial profile-likelihood interval"
        ),
        "eligible_judgment_ids": sorted(eligible_judgment_ids),
        "preference_observation_ids": sorted(preference_observation_ids),
        "preference_observation_sha256": _canonical_sha256(
            {"vote_ids": sorted(preference_observation_ids)}
        ),
        "accounting": _accounting_summary(operational),
        "manifest_sha256": season.manifest_sha256,
        "excluded_invalid_task_judgments": excluded_invalid_task_judgments,
        "excluded_restricted_judgments": excluded_restricted_judgments,
        "eligibility_filter": {
            "run_class": "official",
            "rank_eligible": (
                True
                if data_stratum == "controlled"
                else "append_only_general_track_scope_admission_required"
            ),
            "manifest_match": True,
            "data_stratum": data_stratum,
            "controlled_run_id": controlled_run_id,
            "mock_models": "excluded",
            "identity_and_cost_reconciliation": "required_for_preference",
            "expert_task_validity": (
                "valid_or_minor_issue_and_general_track_eligible_from_sealed_assessment"
            ),
        },
    }


def snapshot_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
