from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ARENA_INFERENCE_POLICY_SHA256 = (
    "bdc0fa93c6365cdcd45694d1d5500d82ccbd622f3be897be9217e252855ffff5"
)
STUDY_DESIGN_SHA256 = "7a63cfd6117338a3af16a422d5ee3458298fdc0ff2fd0abfe45fe851a7e54506"
POLICY_SCHEMA_VERSION = "flavourbench-season1-arena-inference-acceptance-v1"
WITHHOLDING_STATUS = "withheld_insufficient_task_clusters"
FAMILIES = ("composition", "cookability", "evidence", "substitution")


def _default_policy_path() -> Path:
    relative = Path("contracts/season1/season1-arena-inference-acceptance-v1.json")
    candidates = (
        Path.cwd().resolve() / relative,
        Path.cwd().resolve() / "flavourbench" / relative,
        Path(__file__).resolve().parents[2] / relative,
    )
    return next((path for path in candidates if path.is_file()), candidates[-1])


DEFAULT_POLICY_PATH = _default_policy_path()


class ArenaInferenceAcceptanceError(ValueError):
    """The external acceptance policy or supplied evidence is malformed."""


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_arena_inference_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArenaInferenceAcceptanceError("arena inference policy is unreadable") from exc
    if not isinstance(document, dict):
        raise ArenaInferenceAcceptanceError("arena inference policy must be an object")
    embedded = document.get("artifact_sha256")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    observed = canonical_sha256(payload)
    if embedded != observed or observed != ARENA_INFERENCE_POLICY_SHA256:
        raise ArenaInferenceAcceptanceError("arena inference policy content address failed")
    if (
        document.get("schema_version") != POLICY_SCHEMA_VERSION
        or document.get("status") != "frozen_precollection"
        or document.get("study_design_artifact_sha256") != STUDY_DESIGN_SHA256
        or document.get("withholding_status") != WITHHOLDING_STATUS
        or document.get("current_development_pool_is_not_grandfathered") is not True
    ):
        raise ArenaInferenceAcceptanceError("arena inference policy identity is invalid")
    for section_name in ("global_fit", "family_specific_fit", "pairwise_reporting"):
        if not isinstance(document.get(section_name), dict):
            raise ArenaInferenceAcceptanceError(
                f"arena inference policy lacks {section_name}"
            )
    return document


def _components(roster: Sequence[str], edges: Sequence[tuple[str, str]]) -> list[list[str]]:
    adjacency = {model_id: set() for model_id in roster}
    for first, second in edges:
        if first not in adjacency or second not in adjacency:
            raise ArenaInferenceAcceptanceError("comparison contains a model outside the roster")
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
    return sorted(components, key=lambda values: (values[0], len(values)))


def _verified_postcollection_item_audit(value: Mapping[str, Any] | None) -> bool:
    if value is None:
        return False
    # Reuse the release verifier so an inference gate cannot accept a weaker
    # interpretation of the same content-addressed evidence.
    from .season1_readiness import valid_post_collection_item_audit

    return valid_post_collection_item_audit(
        dict(value),
        study_design_sha256=STUDY_DESIGN_SHA256,
    )


def evaluate_arena_inference_acceptance(
    observations: Sequence[Any],
    roster: Sequence[str],
    *,
    view: str,
    admitted_tasks: Mapping[str, str] | None,
    comparison_raters: Mapping[str, Sequence[str]] | None,
    postcollection_item_audit: Mapping[str, Any] | None,
    bootstrap_connected_rate: float | None = None,
    family_bootstrap_connected_rates: Mapping[str, float] | None = None,
    include_bootstrap: bool = False,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen publication gate and enumerate every observed deficit.

    The estimator supplies observations but no thresholds. Thresholds are loaded
    from the frozen policy, so changing a statistical cutoff changes the policy
    digest rather than silently changing estimator code.
    """

    loaded = dict(policy) if policy is not None else load_arena_inference_policy()
    if loaded.get("artifact_sha256") != ARENA_INFERENCE_POLICY_SHA256:
        raise ArenaInferenceAcceptanceError("unverified arena inference policy supplied")
    ordered_roster = tuple(sorted(set(roster)))
    if len(ordered_roster) != len(roster) or len(ordered_roster) < 2:
        raise ArenaInferenceAcceptanceError("arena roster must contain distinct models")
    if view != "all" and view not in FAMILIES:
        raise ArenaInferenceAcceptanceError("arena view must be all or one frozen family")

    observed_task_family: dict[str, str] = {}
    edges_by_family: dict[str, list[tuple[str, str]]] = defaultdict(list)
    model_tasks_by_family: dict[tuple[str, str], set[str]] = defaultdict(set)
    model_comparisons: dict[str, set[str]] = defaultdict(set)
    comparison_raters_from_rows: dict[str, set[str]] = defaultdict(set)
    pair_tasks: dict[tuple[str, str], set[str]] = defaultdict(set)
    comparison_ids: set[str] = set()
    for row in observations:
        task_id = str(row.task_id)
        family = str(row.family)
        battle_id = str(row.battle_id)
        first = str(row.model_a)
        second = str(row.model_b)
        if family not in FAMILIES:
            raise ArenaInferenceAcceptanceError("observation has an unknown family")
        if view != "all" and family != view:
            raise ArenaInferenceAcceptanceError("family view contains another family")
        prior = observed_task_family.setdefault(task_id, family)
        if prior != family:
            raise ArenaInferenceAcceptanceError("task identity maps to multiple families")
        edges_by_family[family].append((first, second))
        model_tasks_by_family[(first, family)].add(task_id)
        model_tasks_by_family[(second, family)].add(task_id)
        model_comparisons[first].add(battle_id)
        model_comparisons[second].add(battle_id)
        comparison_raters_from_rows[battle_id].add(str(row.rater_id))
        comparison_ids.add(battle_id)
        pair_tasks[tuple(sorted((first, second)))].add(task_id)

    admitted = dict(admitted_tasks or observed_task_family)
    if any(family not in FAMILIES for family in admitted.values()):
        raise ArenaInferenceAcceptanceError("admitted task registry has an unknown family")
    if any(
        task_id not in admitted or admitted[task_id] != family
        for task_id, family in observed_task_family.items()
    ):
        raise ArenaInferenceAcceptanceError(
            "observed task is absent from the admitted task registry"
        )
    if view != "all" and any(family != view for family in admitted.values()):
        raise ArenaInferenceAcceptanceError("family view admitted registry crosses families")

    raters_by_comparison = {
        comparison_id: set(comparison_raters_from_rows[comparison_id])
        for comparison_id in comparison_ids
    }
    if comparison_raters is not None:
        supplied_raters = {
            str(comparison_id): {str(rater) for rater in raters}
            for comparison_id, raters in comparison_raters.items()
        }
        if set(supplied_raters) != comparison_ids or any(
            supplied_raters[comparison_id] != raters_by_comparison[comparison_id]
            for comparison_id in comparison_ids
        ):
            raise ArenaInferenceAcceptanceError(
                "comparison-rater evidence does not match the admitted preference rows"
            )

    thresholds = (
        loaded["global_fit"] if view == "all" else loaded["family_specific_fit"]
    )
    deficits: list[dict[str, Any]] = []

    def deficit(code: str, **details: Any) -> None:
        deficits.append({"code": code, **details})

    admitted_by_family = Counter(admitted.values())
    if view == "all":
        required_total = int(thresholds["required_admitted_scored_tasks"])
        if len(admitted) < required_total:
            deficit(
                "admitted_scored_tasks_below_minimum",
                observed=len(admitted),
                required=required_total,
                shortfall=required_total - len(admitted),
            )
        required_family = int(thresholds["required_admitted_scored_tasks_per_family"])
        for family in FAMILIES:
            observed = admitted_by_family[family]
            if observed < required_family:
                deficit(
                    "admitted_scored_tasks_per_family_below_minimum",
                    family=family,
                    observed=observed,
                    required=required_family,
                    shortfall=required_family - observed,
                )
        minimum_model_tasks = int(
            thresholds["minimum_unique_task_clusters_per_model_family"]
        )
        graph_families = FAMILIES
    else:
        required_total = int(thresholds["required_admitted_scored_tasks"])
        if len(admitted) < required_total:
            deficit(
                "admitted_scored_tasks_below_minimum",
                family=view,
                observed=len(admitted),
                required=required_total,
                shortfall=required_total - len(admitted),
            )
        minimum_model_tasks = int(thresholds["minimum_unique_task_clusters_per_model"])
        graph_families = (view,)

    for family in graph_families:
        for model_id in ordered_roster:
            observed = len(model_tasks_by_family[(model_id, family)])
            if observed < minimum_model_tasks:
                deficit(
                    "unique_task_clusters_per_model_family_below_minimum",
                    model_id=model_id,
                    family=family,
                    observed=observed,
                    required=minimum_model_tasks,
                    shortfall=minimum_model_tasks - observed,
                )

    minimum_comparisons = int(thresholds["minimum_unique_comparisons_per_model"])
    for model_id in ordered_roster:
        observed = len(model_comparisons[model_id])
        if observed < minimum_comparisons:
            deficit(
                "unique_comparisons_per_model_below_minimum",
                model_id=model_id,
                observed=observed,
                required=minimum_comparisons,
                shortfall=minimum_comparisons - observed,
            )

    minimum_raters = int(thresholds["minimum_distinct_independent_raters_per_comparison"])
    insufficient_rater_comparisons = sorted(
        comparison_id
        for comparison_id, raters in raters_by_comparison.items()
        if len(raters) < minimum_raters
    )
    if insufficient_rater_comparisons:
        deficit(
            "distinct_independent_raters_per_comparison_below_minimum",
            affected_comparison_count=len(insufficient_rater_comparisons),
            affected_comparison_ids=insufficient_rater_comparisons,
            minimum_observed=min(
                len(raters_by_comparison[value])
                for value in insufficient_rater_comparisons
            ),
            required=minimum_raters,
        )

    family_components = {
        family: _components(ordered_roster, edges_by_family[family])
        for family in graph_families
    }
    require_family_graph = bool(
        thresholds.get("require_connected_graph_per_family")
        if view == "all"
        else thresholds.get("require_connected_graph")
    )
    if require_family_graph:
        for family, components in family_components.items():
            if len(components) != 1:
                deficit(
                    "family_comparison_graph_disconnected",
                    family=family,
                    components=components,
                    required_components=1,
                    observed_components=len(components),
                )
    global_components = _components(
        ordered_roster,
        [edge for family in graph_families for edge in edges_by_family[family]],
    )
    if len(global_components) != 1:
        deficit(
            "comparison_graph_disconnected",
            components=global_components,
            required_components=1,
            observed_components=len(global_components),
        )

    audit_verified = _verified_postcollection_item_audit(postcollection_item_audit)
    unresolved_material_defects = None
    if postcollection_item_audit is not None:
        counts = postcollection_item_audit.get("counts")
        if isinstance(counts, Mapping):
            value = counts.get("unresolved_material_defects")
            if isinstance(value, int) and not isinstance(value, bool):
                unresolved_material_defects = value
    if not audit_verified:
        deficit(
            "postcollection_material_task_defect_evidence_unverified",
            observed=False,
            required=True,
            unresolved_material_defects=unresolved_material_defects,
        )
    elif unresolved_material_defects != int(
        thresholds.get("maximum_unresolved_material_task_defects", 0)
    ):
        deficit(
            "unresolved_material_task_defects_above_maximum",
            observed=unresolved_material_defects,
            required_maximum=int(
                thresholds.get("maximum_unresolved_material_task_defects", 0)
            ),
        )

    if include_bootstrap:
        if bootstrap_connected_rate is None:
            deficit(
                "global_bootstrap_connectivity_not_computed",
                observed=None,
                required=float(
                    thresholds.get(
                        "minimum_global_bootstrap_connected_rate",
                        thresholds.get("minimum_bootstrap_connected_rate"),
                    )
                ),
            )
        else:
            minimum_global_rate = float(
                thresholds.get(
                    "minimum_global_bootstrap_connected_rate",
                    thresholds.get("minimum_bootstrap_connected_rate"),
                )
            )
            if bootstrap_connected_rate < minimum_global_rate:
                deficit(
                    "global_bootstrap_connectivity_below_minimum",
                    observed=round(bootstrap_connected_rate, 6),
                    required=minimum_global_rate,
                    shortfall=round(minimum_global_rate - bootstrap_connected_rate, 6),
                )
        rates = dict(family_bootstrap_connected_rates or {})
        minimum_family_rate = float(
            thresholds.get(
                "minimum_family_bootstrap_connected_rate",
                thresholds.get("minimum_bootstrap_connected_rate"),
            )
        )
        for family in graph_families:
            observed_rate = rates.get(family)
            if observed_rate is None:
                deficit(
                    "family_bootstrap_connectivity_not_computed",
                    family=family,
                    observed=None,
                    required=minimum_family_rate,
                )
            elif observed_rate < minimum_family_rate:
                deficit(
                    "family_bootstrap_connectivity_below_minimum",
                    family=family,
                    observed=round(observed_rate, 6),
                    required=minimum_family_rate,
                    shortfall=round(minimum_family_rate - observed_rate, 6),
                )

    pairwise_floor = int(
        loaded["pairwise_reporting"]["minimum_shared_task_clusters_for_interval"]
    )
    pairwise_support = {
        first: {
            second: {
                "shared_task_clusters": len(pair_tasks.get(tuple(sorted((first, second))), set())),
                "minimum_for_interval": pairwise_floor,
                "interval_reportable": len(
                    pair_tasks.get(tuple(sorted((first, second))), set())
                )
                >= pairwise_floor,
            }
            for second in ordered_roster
            if second != first
        }
        for first in ordered_roster
    }
    return {
        "status": "pass" if not deficits else "fail",
        "policy_sha256": ARENA_INFERENCE_POLICY_SHA256,
        "view": view,
        "withholding_status": WITHHOLDING_STATUS,
        "deficits": deficits,
        "metrics": {
            "admitted_scored_tasks": len(admitted),
            "admitted_scored_tasks_by_family": {
                family: admitted_by_family[family] for family in FAMILIES
            },
            "unique_comparisons": len(comparison_ids),
            "unique_comparisons_by_model": {
                model_id: len(model_comparisons[model_id])
                for model_id in ordered_roster
            },
            "unique_task_clusters_by_model_family": {
                model_id: {
                    family: len(model_tasks_by_family[(model_id, family)])
                    for family in graph_families
                }
                for model_id in ordered_roster
            },
            "minimum_distinct_raters_per_comparison": min(
                (len(value) for value in raters_by_comparison.values()),
                default=0,
            ),
            "comparison_components": global_components,
            "family_comparison_components": family_components,
            "postcollection_item_audit_verified": audit_verified,
            "unresolved_material_task_defects": unresolved_material_defects,
            "bootstrap_connected_rate": (
                round(bootstrap_connected_rate, 6)
                if bootstrap_connected_rate is not None
                else None
            ),
            "family_bootstrap_connected_rates": {
                family: (
                    round(float((family_bootstrap_connected_rates or {}).get(family)), 6)
                    if (family_bootstrap_connected_rates or {}).get(family) is not None
                    else None
                )
                for family in graph_families
            },
        },
        "pairwise_reporting_support": pairwise_support,
    }


def publication_acceptance_deficits(
    payload: Mapping[str, Any],
    *,
    view: str,
) -> list[str]:
    """Recheck the self-describing gate result at the publication boundary."""

    policy = load_arena_inference_policy()
    if view != "all" and view not in FAMILIES:
        return ["invalid_view"]
    thresholds = policy["global_fit"] if view == "all" else policy["family_specific_fit"]
    acceptance = payload.get("statistical_acceptance")
    rows = payload.get("rows")
    errors: list[str] = []
    if not isinstance(acceptance, Mapping):
        return ["statistical_acceptance_missing"]
    metrics = acceptance.get("metrics")
    if acceptance.get("status") != "pass":
        errors.append("acceptance_status_not_pass")
    if acceptance.get("policy_sha256") != ARENA_INFERENCE_POLICY_SHA256:
        errors.append("acceptance_policy_mismatch")
    if acceptance.get("view") != view:
        errors.append("acceptance_view_mismatch")
    if acceptance.get("deficits") != []:
        errors.append("acceptance_deficits_not_empty")
    if not isinstance(metrics, Mapping):
        return [*errors, "acceptance_metrics_missing"]

    admitted_total = metrics.get("admitted_scored_tasks")
    admitted_by_family = metrics.get("admitted_scored_tasks_by_family")
    if not isinstance(admitted_total, int) or isinstance(admitted_total, bool):
        errors.append("admitted_task_count_invalid")
    elif admitted_total < int(thresholds["required_admitted_scored_tasks"]):
        errors.append("admitted_task_count_below_minimum")
    if view == "all":
        required = int(thresholds["required_admitted_scored_tasks_per_family"])
        if not isinstance(admitted_by_family, Mapping) or any(
            not isinstance(admitted_by_family.get(family), int)
            or admitted_by_family.get(family, 0) < required
            for family in FAMILIES
        ):
            errors.append("admitted_family_task_count_below_minimum")

    model_comparisons = metrics.get("unique_comparisons_by_model")
    model_family_tasks = metrics.get("unique_task_clusters_by_model_family")
    minimum_comparisons = int(thresholds["minimum_unique_comparisons_per_model"])
    minimum_model_tasks = int(
        thresholds[
            "minimum_unique_task_clusters_per_model_family"
            if view == "all"
            else "minimum_unique_task_clusters_per_model"
        ]
    )
    model_ids = [
        str(row.get("competitor_id"))
        for row in rows or []
        if isinstance(row, Mapping) and row.get("competitor_id")
    ]
    if not isinstance(rows, list) or not rows or len(model_ids) != len(rows):
        errors.append("model_rows_missing")
    elif not isinstance(model_comparisons, Mapping) or any(
        not isinstance(model_comparisons.get(model_id), int)
        or model_comparisons.get(model_id, 0) < minimum_comparisons
        for model_id in model_ids
    ):
        errors.append("model_comparison_count_below_minimum")
    expected_families = FAMILIES if view == "all" else (view,)
    if not isinstance(model_family_tasks, Mapping) or any(
        not isinstance(model_family_tasks.get(model_id), Mapping)
        or any(
            not isinstance(model_family_tasks[model_id].get(family), int)
            or model_family_tasks[model_id].get(family, 0) < minimum_model_tasks
            for family in expected_families
        )
        for model_id in model_ids
    ):
        errors.append("model_family_task_count_below_minimum")

    if metrics.get("postcollection_item_audit_verified") is not True:
        errors.append("postcollection_audit_unverified")
    if metrics.get("unresolved_material_task_defects") != 0:
        errors.append("unresolved_material_task_defects")
    observed_minimum_raters = metrics.get("minimum_distinct_raters_per_comparison")
    if (
        not isinstance(observed_minimum_raters, int)
        or isinstance(observed_minimum_raters, bool)
        or observed_minimum_raters
        < int(thresholds["minimum_distinct_independent_raters_per_comparison"])
    ):
        errors.append("independent_rater_coverage_below_minimum")

    connected_rate = metrics.get("bootstrap_connected_rate")
    minimum_connected = float(
        thresholds.get(
            "minimum_global_bootstrap_connected_rate",
            thresholds.get("minimum_bootstrap_connected_rate"),
        )
    )
    if not isinstance(connected_rate, (int, float)) or connected_rate < minimum_connected:
        errors.append("bootstrap_connectivity_below_minimum")
    family_rates = metrics.get("family_bootstrap_connected_rates")
    minimum_family_connected = float(
        thresholds.get(
            "minimum_family_bootstrap_connected_rate",
            thresholds.get("minimum_bootstrap_connected_rate"),
        )
    )
    if not isinstance(family_rates, Mapping) or any(
        not isinstance(family_rates.get(family), (int, float))
        or family_rates.get(family, 0) < minimum_family_connected
        for family in expected_families
    ):
        errors.append("family_bootstrap_connectivity_below_minimum")

    if isinstance(rows, list) and any(
        not isinstance(row, Mapping)
        or row.get("rating") is None
        or row.get("rating_lower") is None
        or row.get("rating_upper") is None
        or row.get("provisional") is not False
        for row in rows
    ):
        errors.append("publishable_rating_rows_incomplete")

    support = payload.get("pairwise_reporting_support")
    intervals = payload.get("pairwise_win_probability_interval")
    if not isinstance(support, Mapping) or not isinstance(intervals, Mapping):
        errors.append("pairwise_interval_support_missing")
    else:
        for first, second_rows in support.items():
            if not isinstance(second_rows, Mapping):
                errors.append("pairwise_interval_support_invalid")
                break
            for second, cell in second_rows.items():
                interval = (
                    intervals.get(first, {}).get(second)
                    if isinstance(intervals.get(first), Mapping)
                    else None
                )
                if not isinstance(cell, Mapping):
                    errors.append("pairwise_interval_support_invalid")
                    break
                if cell.get("interval_reportable") is False and interval is not None:
                    errors.append("unsupported_pairwise_interval_not_suppressed")
                    break
                if cell.get("interval_reportable") is True and not (
                    isinstance(interval, Mapping)
                    and isinstance(interval.get("lower"), (int, float))
                    and isinstance(interval.get("upper"), (int, float))
                ):
                    errors.append("supported_pairwise_interval_missing")
                    break
    return sorted(set(errors))
