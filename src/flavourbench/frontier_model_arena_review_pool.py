"""Build a blinded model-arena pool from real Epicure-on frontier responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .current_frontier_task_quarantine import (
    quarantine_binding,
    quarantine_task_ids,
)
from .current_pilot_assets import (
    COHERE_MODEL_ORDER,
    DISPLAY_NAMES,
    EXTENDED_MODEL_ORDER,
    MODEL_ORDER,
)
from .current_pilot_review_import import (
    CurrentPilotReviewImportError,
    ReviewArm,
    ReviewPair,
    ReviewPool,
    _mapping,
    _require_sha256,
    _response_answer,
    _sequence,
    _validate_tool_trace,
    import_review_pool,
)
from .database import session_scope
from .frontier_multirun_assets import (
    BLUE,
    GREY,
    INK,
    LIGHT,
    RunInput,
    _configure_matplotlib,
    _verify_artifact,
    _verify_summary,
    verify_runs,
)
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-frontier-model-arena-review-pool-v1"
RECEIPT_SCHEMA_VERSION = "flavourbench-frontier-model-arena-review-receipt-v1"


@dataclass(frozen=True)
class StratumInput:
    """One execution-policy stratum containing one or more immutable runs."""

    label: str
    runs: tuple[RunInput, ...]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _connected_components(
    edges: Sequence[tuple[str, str]], model_order: Sequence[str]
) -> list[list[str]]:
    adjacency = {model_id: set() for model_id in model_order}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(model_order)
    components: list[list[str]] = []
    while unseen:
        start = min(unseen)
        queue = deque([start])
        component: set[str] = set()
        while queue:
            model_id = queue.popleft()
            if model_id in component:
                continue
            component.add(model_id)
            queue.extend(adjacency[model_id] - component)
        unseen -= component
        components.append(sorted(component))
    return sorted(components, key=lambda component: (-len(component), component))


def _side_layout(
    candidates: Sequence[dict[str, Any]], source_commitment: str
) -> dict[str, tuple[str, str]]:
    """Alternate orientation within each model pair after deterministic hashing."""

    grouped: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for item in candidates:
        pair = tuple(sorted((str(item["model_a"]), str(item["model_b"]))))
        if len(pair) != 2 or pair[0] == pair[1]:
            raise CurrentPilotReviewImportError("arena candidate has no model contrast")
        candidate_id = str(item["candidate_id"])
        score = _sha256_text(f"{source_commitment}:{candidate_id}:side-order")
        grouped[(pair[0], pair[1])].append((score, candidate_id))
    layout: dict[str, tuple[str, str]] = {}
    for pair, rows in grouped.items():
        base = int(_sha256_text(f"{source_commitment}:{pair}:orientation")[:2], 16) % 2
        for ordinal, (_, candidate_id) in enumerate(sorted(rows)):
            layout[candidate_id] = pair if (ordinal + base) % 2 == 0 else pair[::-1]
    return layout


def build_model_arena_review_pool(strata: Sequence[StratumInput]) -> ReviewPool:
    """Create every within-task Epicure-on model comparison across verified strata."""

    if not strata or len({stratum.label for stratum in strata}) != len(strata):
        raise CurrentPilotReviewImportError("arena strata must have unique labels")
    # The content address must not depend on caller argument order. The CLI
    # already groups labels lexicographically; direct library callers must
    # produce the identical artifact for the same immutable inputs.
    ordered_strata = tuple(sorted(strata, key=lambda stratum: stratum.label))

    aggregates: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    task_stratum: dict[str, str] = {}
    response_rows: dict[tuple[str, str], dict[str, Any]] = {}
    source_by_work_item: dict[tuple[str, str], dict[str, Any]] = {}
    work_item_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    response_hashes: set[str] = set()
    generation_ids: set[str] = set()
    epicure_contract: tuple[str, str, str, str] | None = None
    input_records: list[dict[str, Any]] = []

    for stratum in ordered_strata:
        verified = verify_runs(stratum.runs)
        aggregates[stratum.label] = verified.aggregate
        summaries = [_verify_summary(run.summary) for run in stratum.runs]
        for run, summary in zip(stratum.runs, summaries, strict=True):
            run_contracts = {
                str(row["model_id"]): _mapping(row, "model contract")
                for row in summary["manifest"]["models"]
            }
            for model_id, contract in run_contracts.items():
                prior = contracts.setdefault(model_id, contract)
                stable_fields = (
                    "canonical_model_slug",
                    "provider_tag",
                    "execution_backend",
                    "endpoint_execution_sha256",
                )
                if any(prior.get(field) != contract.get(field) for field in stable_fields):
                    raise CurrentPilotReviewImportError(
                        f"model route changed across arena strata: {model_id}"
                    )
            run_work_items = {
                str(item["work_item_id"]): _mapping(item, "work item")
                for item in summary["workload"]["work_items"]
            }
            for work_item_id, item in run_work_items.items():
                model_id = str(item["model_id"])
                key = (stratum.label, work_item_id)
                if key in work_item_by_key:
                    raise CurrentPilotReviewImportError("duplicate arena work item")
                work_item_by_key[key] = item
                task_id = str(item["task_id"])
                prior_stratum = task_stratum.setdefault(task_id, stratum.label)
                if prior_stratum != stratum.label:
                    raise CurrentPilotReviewImportError(
                        "arena protocol strata must use disjoint tasks"
                    )
                if model_id not in run_contracts:
                    raise CurrentPilotReviewImportError("arena work item has an unknown model")
            for path in sorted(run.sources.glob("*.json")):
                source = _verify_artifact(path)
                work_item_id = str(source["dataset_work_item_id"])
                key = (stratum.label, work_item_id)
                if key not in work_item_by_key or key in source_by_work_item:
                    raise CurrentPilotReviewImportError("arena source linkage is invalid")
                source_by_work_item[key] = source
            for path in sorted(run.responses.glob("*.json")):
                document = _verify_artifact(path)
                if document.get("condition") != "epicure_on":
                    continue
                work_item_id = str(document["work_item_id"])
                key = (stratum.label, work_item_id)
                item = work_item_by_key.get(key)
                source = source_by_work_item.get(key)
                if item is None or source is None:
                    raise CurrentPilotReviewImportError("arena response linkage is invalid")
                model_id = str(item["model_id"])
                task_id = str(item["task_id"])
                task_key = (stratum.label, task_id, model_id)
                if task_key in response_rows:
                    raise CurrentPilotReviewImportError("duplicate arena model-task response")
                model = _mapping(document.get("model"), "response model")
                response = _mapping(document.get("response"), "response payload")
                cost = _mapping(document.get("cost"), "response cost")
                task = _mapping(document.get("task"), "response task")
                provenance = _mapping(document.get("provenance"), "response provenance")
                linked_source = _mapping(document.get("source"), "response source")
                contract = contracts[model_id]
                call_count, error_count = _validate_tool_trace(document)
                answer = _response_answer(document)
                cost_accounted = response.get("cost_reconciled") is True or (
                    model.get("execution_backend") in {"kimi_direct", "cohere_direct"}
                    and cost.get("all_generation_usage_accounted") is True
                )
                if (
                    document.get("official") is not False
                    or document.get("rank_eligible") is not False
                    or document.get("research_result") is not False
                    or document.get("research_release_eligible") is not False
                    or model.get("requested_model_id") != model_id
                    or model.get("canonical_model_slug") != contract.get("canonical_model_slug")
                    or model.get("actual_model_id") != contract.get("canonical_model_slug")
                    or task.get("public_id") != task_id
                    or task.get("family") != item.get("task_family")
                    or linked_source.get("artifact_sha256") != source.get("artifact_sha256")
                    or provenance.get("epicure_access") is not True
                    or call_count - error_count <= 0
                    or not cost_accounted
                ):
                    raise CurrentPilotReviewImportError(
                        "response crossed the real Epicure-on arena contract"
                    )
                epicure = _mapping(provenance.get("epicure"), "Epicure provenance")
                observed_epicure = (
                    str(epicure.get("release_id")),
                    _require_sha256(epicure.get("bundle_sha256"), "Epicure bundle digest"),
                    _require_sha256(
                        epicure.get("application_sha256"), "Epicure application digest"
                    ),
                    _require_sha256(
                        provenance.get("epicure_tool_schema_sha256"),
                        "Epicure tool digest",
                    ),
                )
                if epicure_contract is None:
                    epicure_contract = observed_epicure
                elif epicure_contract != observed_epicure:
                    raise CurrentPilotReviewImportError(
                        "Epicure lineage differs across model-arena responses"
                    )
                response_sha256 = str(document["artifact_sha256"])
                if response_sha256 in response_hashes:
                    raise CurrentPilotReviewImportError("duplicate arena response artifact")
                response_hashes.add(response_sha256)
                arm_generation_ids = _sequence(response.get("generation_ids"), "generation IDs")
                if not arm_generation_ids:
                    raise CurrentPilotReviewImportError("arena response has no provider generation")
                for raw_generation_id in arm_generation_ids:
                    generation_id = str(raw_generation_id)
                    if not generation_id or generation_id in generation_ids:
                        raise CurrentPilotReviewImportError("provider generation ID is duplicated")
                    generation_ids.add(generation_id)
                cost_micros = response.get("cost_micros")
                if isinstance(cost_micros, bool) or not isinstance(cost_micros, int):
                    raise CurrentPilotReviewImportError("arena response cost is invalid")
                response_rows[task_key] = {
                    "stratum": stratum.label,
                    "task_id": task_id,
                    "family": str(item["task_family"]),
                    "model_id": model_id,
                    "work_item": item,
                    "response": document,
                    "source": source,
                    "answer_sha256": _sha256_text(answer),
                    "call_count": call_count,
                    "successful_call_count": call_count - error_count,
                    "provider_call_count": len(arm_generation_ids),
                    "generation_ids": tuple(str(value) for value in arm_generation_ids),
                    "cost_micros": cost_micros,
                }
            input_records.append(
                {
                    "stratum": stratum.label,
                    "summary_filename": run.summary.name,
                    "summary_file_sha256": _file_sha256(run.summary),
                    "summary_content_address": summary["content_address"]["digest"],
                }
            )

    if not set(MODEL_ORDER) <= set(contracts):
        raise CurrentPilotReviewImportError("arena inputs do not cover the core frozen panel")
    optional_models = frozenset(set(contracts) - set(MODEL_ORDER))
    if optional_models not in {frozenset(), frozenset(COHERE_MODEL_ORDER)}:
        raise CurrentPilotReviewImportError(
            "arena Cohere extension must contain both frozen direct endpoints"
        )
    active_model_order = tuple(
        model_id for model_id in EXTENDED_MODEL_ORDER if model_id in contracts
    )
    if epicure_contract is None:
        raise CurrentPilotReviewImportError("arena pool has no Epicure contract")
    source_commitment = sha256_json(
        {
            label: {
                "aggregate_sha256": aggregate["artifact_sha256"],
                "execution_policy_sha256": aggregate["execution_policy_sha256"],
                "task_set_sha256": aggregate["task_set_sha256"],
            }
            for label, aggregate in sorted(aggregates.items())
        }
    )
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (stratum_label, task_id, _), row in response_rows.items():
        by_task[(stratum_label, task_id)].append(row)
    source_candidates: list[dict[str, Any]] = []
    for (stratum_label, task_id), rows in sorted(by_task.items()):
        ordered = sorted(
            rows, key=lambda row: active_model_order.index(str(row["model_id"]))
        )
        for left, right in itertools.combinations(ordered, 2):
            model_a = str(left["model_id"])
            model_b = str(right["model_id"])
            candidate_id = _sha256_text(
                f"{source_commitment}:{stratum_label}:{task_id}:{model_a}:{model_b}:arena-candidate"
            )
            source_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "stratum": stratum_label,
                    "task_id": task_id,
                    "family": left["family"],
                    "model_a": model_a,
                    "model_b": model_b,
                }
            )
    held_task_ids = quarantine_task_ids()
    candidates = [
        candidate
        for candidate in source_candidates
        if str(candidate["task_id"]) not in held_task_ids
    ]
    excluded_candidates = [
        candidate
        for candidate in source_candidates
        if str(candidate["task_id"]) in held_task_ids
    ]
    if not candidates:
        raise CurrentPilotReviewImportError("arena pool has no common-task comparisons")
    source_candidate_task_keys = {
        (str(candidate["stratum"]), str(candidate["task_id"]))
        for candidate in source_candidates
    }
    source_participating_rows = [
        row
        for (stratum_label, task_id, _), row in response_rows.items()
        if (stratum_label, task_id) in source_candidate_task_keys
    ]
    candidate_task_keys = {
        (str(candidate["stratum"]), str(candidate["task_id"])) for candidate in candidates
    }
    candidate_task_ids = {task_id for _stratum, task_id in candidate_task_keys}
    participating_rows = [
        row
        for (stratum_label, task_id, _), row in response_rows.items()
        if (stratum_label, task_id) in candidate_task_keys
    ]
    participating_generation_ids = {
        generation_id
        for row in participating_rows
        for generation_id in row["generation_ids"]
    }
    if len(participating_generation_ids) != sum(
        int(row["provider_call_count"]) for row in participating_rows
    ):
        raise CurrentPilotReviewImportError(
            "arena participant generation IDs are not unique"
        )
    layout = _side_layout(candidates, source_commitment)

    pairs: list[ReviewPair] = []
    items: list[dict[str, Any]] = []
    comparison_counts_by_model: Counter[str] = Counter()
    comparison_counts_by_model_pair: Counter[tuple[str, str]] = Counter()
    left_counts_by_model: Counter[str] = Counter()
    comparison_counts_by_family: Counter[str] = Counter()
    comparison_counts_by_stratum: Counter[str] = Counter()
    projected_provider_calls = 0
    projected_tool_calls = 0
    projected_successful_tool_calls = 0
    edges: list[tuple[str, str]] = []
    response_presentation_counts: Counter[str] = Counter()
    comparison_counts_by_model_pair_family: Counter[tuple[str, str, str]] = Counter()

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        stratum_label = str(candidate["stratum"])
        task_id = str(candidate["task_id"])
        left_model, right_model = layout[candidate_id]
        rows = {
            model_id: response_rows[(stratum_label, task_id, model_id)]
            for model_id in (left_model, right_model)
        }
        review_item_id = _sha256_text(f"{source_commitment}:{candidate_id}:review-item")
        review_arms: list[ReviewArm] = []
        side_manifest: dict[str, dict[str, Any]] = {}
        for side, model_id in (("left", left_model), ("right", right_model)):
            row = rows[model_id]
            document = row["response"]
            model = _mapping(document["model"], "response model")
            response = _mapping(document["response"], "response payload")
            side_manifest[side] = {
                "condition": "epicure_on",
                "requested_model_id": model_id,
                "canonical_model_slug": model["canonical_model_slug"],
                "actual_model_id": model["actual_model_id"],
                "actual_provider": model["actual_provider"],
                "response_artifact_sha256": document["artifact_sha256"],
                "source_artifact_sha256": row["source"]["artifact_sha256"],
                "answer_sha256": row["answer_sha256"],
                "generation_ids_sha256": sha256_json(response["generation_ids"]),
            }
            response_presentation_counts[str(document["artifact_sha256"])] += 1
            review_arms.append(
                ReviewArm(
                    side=side,
                    condition="epicure_on",
                    response=document,
                    source=row["source"],
                )
            )
            projected_provider_calls += int(row["provider_call_count"])
            projected_tool_calls += int(row["call_count"])
            projected_successful_tool_calls += int(row["successful_call_count"])
            comparison_counts_by_model[model_id] += 1
            left_counts_by_model[model_id] += int(side == "left")
        work_item = {
            "work_item_id": review_item_id,
            "task_id": task_id,
            "task_family": candidate["family"],
            "stratum": stratum_label,
            "model_ids": [left_model, right_model],
        }
        pairs.append(
            ReviewPair(
                review_item_id=review_item_id,
                work_item=work_item,
                arms=(review_arms[0], review_arms[1]),
            )
        )
        task_document = _mapping(review_arms[0].response["task"], "response task")
        items.append(
            {
                "review_item_id": review_item_id,
                "task_id": task_id,
                "task_family": candidate["family"],
                "prompt_sha256": task_document["prompt_sha256"],
                "execution_stratum": stratum_label,
                "execution_policy_sha256": aggregates[stratum_label]["execution_policy_sha256"],
                "left": side_manifest["left"],
                "right": side_manifest["right"],
            }
        )
        comparison_counts_by_family[str(candidate["family"])] += 1
        comparison_counts_by_stratum[stratum_label] += 1
        model_pair = tuple(sorted((left_model, right_model)))
        comparison_counts_by_model_pair[model_pair] += 1
        comparison_counts_by_model_pair_family[
            (model_pair[0], model_pair[1], str(candidate["family"]))
        ] += 1
        edges.append(model_pair)

    components = _connected_components(edges, active_model_order)
    if components != [sorted(active_model_order)]:
        raise CurrentPilotReviewImportError("arena comparison graph is disconnected")
    release_id, bundle_sha256, application_sha256, tool_schema_sha256 = epicure_contract
    model_contracts = {
        model_id: {
            "canonical_model_slug": contracts[model_id]["canonical_model_slug"],
            "provider_tag": contracts[model_id]["provider_tag"],
            "execution_backend": contracts[model_id]["execution_backend"],
        }
        for model_id in active_model_order
    }
    families = tuple(sorted({str(row["family"]) for row in participating_rows}))
    pair_family_cells = [
        {
            "model_a": model_a,
            "model_b": model_b,
            "task_family": family,
            "candidate_comparisons": comparison_counts_by_model_pair_family[
                (*tuple(sorted((model_a, model_b))), family)
            ],
        }
        for model_a, model_b in itertools.combinations(active_model_order, 2)
        for family in families
    ]
    missing_pair_family_cells = [
        row for row in pair_family_cells if int(row["candidate_comparisons"]) == 0
    ]
    model_family_task_counts = {
        model_id: {
            family: len(
                {
                    str(row["task_id"])
                    for row in participating_rows
                    if str(row["model_id"]) == model_id
                    and str(row["family"]) == family
                }
            )
            for family in families
        }
        for model_id in active_model_order
    }
    response_reuse_values = list(response_presentation_counts.values())
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "source_commitment_sha256": source_commitment,
            "source_run_class": "real_multistratum_development_pilot",
            "inputs": input_records,
            "strata": {
                label: {
                    "aggregate_sha256": aggregate["artifact_sha256"],
                    "execution_policy_sha256": aggregate["execution_policy_sha256"],
                    "task_set_sha256": aggregate["task_set_sha256"],
                }
                for label, aggregate in sorted(aggregates.items())
            },
        },
        "track": "model_arena",
        "selection_policy": {
            "same_task": True,
            "same_execution_stratum": True,
            "epicure_on_only": True,
            "all_available_unordered_model_pairs": True,
            "raw_answers_edited": False,
            "deterministic_model_pair_balanced_side_assignment": True,
            "failed_responses_retained_for_reliability_only": True,
            "task_quarantine": quarantine_binding(),
            "quarantine_pending_qualified_adjudication": True,
        },
        "observed": {
            "source_candidate_comparisons_before_task_quarantine": len(source_candidates),
            "source_response_arms_before_task_quarantine": len(source_participating_rows),
            "source_unique_task_ids_before_task_quarantine": len(
                {task_id for _stratum, task_id in source_candidate_task_keys}
            ),
            "source_task_stratum_clusters_before_task_quarantine": len(
                source_candidate_task_keys
            ),
            "task_quarantined_candidate_comparisons": len(excluded_candidates),
            "task_quarantined_source_response_arms": (
                len(source_participating_rows) - len(participating_rows)
            ),
            "task_quarantined_task_ids_observed": sorted(
                {str(candidate["task_id"]) for candidate in excluded_candidates}
            ),
            "candidate_comparisons": len(pairs),
            "unique_task_ids": len(candidate_task_ids),
            "task_stratum_clusters": len(candidate_task_keys),
            # Backward-compatible alias. The value is the task-ID count, not
            # the task-by-stratum cluster count used by inference.
            "distinct_tasks": len(candidate_task_ids),
            "models": len(active_model_order),
            "source_response_arms": len(participating_rows),
            "projected_response_presentations": 2 * len(pairs),
            "candidate_comparisons_by_family": dict(comparison_counts_by_family),
            "candidate_comparisons_by_stratum": dict(comparison_counts_by_stratum),
            "comparison_exposures_by_model": {
                model_id: comparison_counts_by_model[model_id]
                for model_id in active_model_order
            },
            "candidate_comparisons_by_model_pair": [
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "candidate_comparisons": comparison_counts_by_model_pair[
                        tuple(sorted((model_a, model_b)))
                    ],
                }
                for model_a, model_b in itertools.combinations(active_model_order, 2)
            ],
            "candidate_comparisons_by_model_pair_family": pair_family_cells,
            "model_pair_family_cells": len(pair_family_cells),
            "missing_model_pair_family_cells": len(missing_pair_family_cells),
            "missing_model_pair_family_cell_records": missing_pair_family_cells,
            "model_family_unique_task_counts": model_family_task_counts,
            "left_presentations_by_model": {
                model_id: left_counts_by_model[model_id]
                for model_id in active_model_order
            },
            "unique_provider_generation_ids": len(participating_generation_ids),
            "unique_epicure_calls": sum(
                int(row["call_count"]) for row in participating_rows
            ),
            "unique_successful_epicure_calls": sum(
                int(row["successful_call_count"]) for row in participating_rows
            ),
            "projected_provider_calls": projected_provider_calls,
            "projected_epicure_calls": projected_tool_calls,
            "projected_successful_epicure_calls": projected_successful_tool_calls,
            "unique_source_cost_micros": sum(
                int(row["cost_micros"]) for row in participating_rows
            ),
            "comparison_graph_component_sizes": [len(component) for component in components],
            "evidence_units": {
                "raw_comparison_rows": len(pairs),
                "unique_task_ids": len(candidate_task_ids),
                "task_stratum_clusters": len(candidate_task_keys),
                "unique_response_arms": len(participating_rows),
                "response_arm_presentations": 2 * len(pairs),
                "minimum_comparisons_per_reused_response_arm": min(
                    response_reuse_values
                ),
                "maximum_comparisons_per_reused_response_arm": max(
                    response_reuse_values
                ),
                "median_comparisons_per_reused_response_arm": median(
                    response_reuse_values
                ),
                "independence_unit_for_uncertainty": "task_cluster",
                "response_reuse_policy": (
                    "shared response arms remain locked inside their task cluster during "
                    "resampling; comparison rows are not treated as independent"
                ),
                "scalar_effective_sample_size_claimed": False,
            },
            "synthetic_arms": 0,
        },
        "epicure": {
            "release_id": release_id,
            "bundle_sha256": bundle_sha256,
            "application_sha256": application_sha256,
            "tool_schema_sha256": tool_schema_sha256,
        },
        "model_order": list(active_model_order),
        "model_contracts": model_contracts,
        "items": items,
        "items_commitment_sha256": sha256_json(items),
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "permitted_use": "blinded real-output development-pilot review",
            "prohibited_use": "model-quality ranking before real judgments are collected",
            "quarantined_tasks_may_enter_official_fit": False,
        },
    }
    payload["artifact_sha256"] = sha256_json(payload)
    return ReviewPool(manifest=payload, pairs=tuple(pairs))


def _public_receipt(pool: ReviewPool) -> dict[str, Any]:
    manifest = pool.manifest
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_pool_sha256": manifest["artifact_sha256"],
        "track": manifest["track"],
        "source": manifest["source"],
        "selection_policy": manifest["selection_policy"],
        "observed": manifest["observed"],
        "epicure": manifest["epicure"],
        "model_order": manifest["model_order"],
        "model_contracts": manifest["model_contracts"],
        "items_commitment_sha256": manifest["items_commitment_sha256"],
        "claim_boundary": manifest["claim_boundary"],
    }
    receipt["artifact_sha256"] = sha256_json(receipt)
    return receipt


def _write_coverage_csv(pool: ReviewPool, path: Path) -> None:
    pair_counts: Counter[tuple[str, str]] = Counter()
    exposures: Counter[str] = Counter()
    for item in pool.manifest["items"]:
        model_a = str(item["left"]["requested_model_id"])
        model_b = str(item["right"]["requested_model_id"])
        if model_a == model_b:
            raise CurrentPilotReviewImportError("coverage CSV item has no model contrast")
        pair_counts[tuple(sorted((model_a, model_b)))] += 1
        exposures.update((model_a, model_b))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model_a",
                "model_a_display",
                "model_b",
                "model_b_display",
                "candidate_comparisons",
                "model_a_total_exposure",
                "model_b_total_exposure",
            ),
        )
        writer.writeheader()
        for model_a, model_b in itertools.combinations(pool.manifest["model_order"], 2):
            model_a = str(model_a)
            model_b = str(model_b)
            writer.writerow(
                {
                    "model_a": model_a,
                    "model_a_display": DISPLAY_NAMES[model_a],
                    "model_b": model_b,
                    "model_b_display": DISPLAY_NAMES[model_b],
                    "candidate_comparisons": pair_counts[tuple(sorted((model_a, model_b)))],
                    "model_a_total_exposure": exposures[model_a],
                    "model_b_total_exposure": exposures[model_b],
                }
            )


def _render_coverage_figure(pool: ReviewPool, path: Path) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.ticker import MaxNLocator

    observed = pool.manifest["observed"]
    model_order = tuple(str(value) for value in pool.manifest["model_order"])
    size = len(model_order)
    matrix = np.full((size, size), np.nan)
    index = {model_id: position for position, model_id in enumerate(model_order)}
    pair_counts: Counter[tuple[str, str]] = Counter()
    exposure_counts: Counter[str] = Counter()
    response_reuse: Counter[str] = Counter()
    for item in pool.manifest["items"]:
        left_model = str(item["left"]["requested_model_id"])
        right_model = str(item["right"]["requested_model_id"])
        if left_model not in index or right_model not in index or left_model == right_model:
            raise CurrentPilotReviewImportError("coverage figure item has an invalid model pair")
        pair_counts[tuple(sorted((left_model, right_model)))] += 1
        exposure_counts.update((left_model, right_model))
        for side in ("left", "right"):
            response_reuse[str(item[side]["response_artifact_sha256"])] += 1
    if sum(pair_counts.values()) != int(observed["candidate_comparisons"]):
        raise CurrentPilotReviewImportError("coverage figure item count differs from observed")
    for (model_a, model_b), count in pair_counts.items():
        first = index[model_a]
        second = index[model_b]
        matrix[max(first, second), min(first, second)] = count
    for first, second in itertools.combinations(model_order, 2):
        row = max(index[first], index[second])
        column = min(index[first], index[second])
        if not np.isfinite(matrix[row, column]):
            matrix[row, column] = 0

    finite = matrix[np.isfinite(matrix)]
    maximum = int(finite.max())
    colors = plt.get_cmap("Blues")(np.linspace(0.13, 0.88, maximum + 1))
    colors[0] = np.array([0.94, 0.95, 0.96, 1.0])
    cmap = ListedColormap(colors).with_extremes(bad="white")
    norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)

    figure = plt.figure(figsize=(7.15, max(5.25, 0.34 * size + 0.55)))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.62, 0.76, 0.68),
        wspace=0.48,
    )
    heatmap = figure.add_subplot(grid[0, 0])
    exposure_axis = figure.add_subplot(grid[0, 1])
    reuse_axis = figure.add_subplot(grid[0, 2])

    # Draw the matrix as vector cells. ``imshow`` embeds a raster object in the
    # paper PDF even for this small grid, which weakens print and archive quality.
    image = heatmap.pcolormesh(
        np.arange(size + 1),
        np.arange(size + 1),
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.7,
    )
    short_names = [DISPLAY_NAMES[model_id] for model_id in model_order]
    compact_names = {
        "openai/gpt-5.6-sol-pro": "GPT-5.6",
        "anthropic/claude-fable-5": "Fable 5",
        "anthropic/claude-opus-5": "Opus 5",
        "anthropic/claude-sonnet-5": "Sonnet 5",
        "google/gemini-3.1-pro-preview": "Gemini 3.1",
        "google/gemini-3.6-flash": "Gemini 3.6",
        "x-ai/grok-4.5": "Grok 4.5",
        "moonshotai/kimi-k3": "Kimi K3",
        "z-ai/glm-5.2": "GLM 5.2",
        "deepseek/deepseek-v4-pro": "DS V4 Pro",
        "deepseek/deepseek-v4-flash-0731": "DS V4 Flash",
        "minimax/minimax-m3": "MiniMax M3",
        "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3",
        "mistralai/mistral-medium-3-5": "Mistral Med. 3.5",
        "cohere/command-a-plus-05-2026": "Cmd A+",
        "cohere/command-a-reasoning-08-2025": "Cmd A Reason.",
    }
    heatmap_x_names = tuple(compact_names[model_id] for model_id in model_order)
    cell_centres = np.arange(size) + 0.5
    heatmap.set_xticks(
        cell_centres,
        heatmap_x_names,
        rotation=52,
        ha="right",
        rotation_mode="anchor",
    )
    heatmap.set_yticks(cell_centres, short_names)
    heatmap.set_xlim(0, size)
    heatmap.set_ylim(size, 0)
    heatmap.set_aspect("equal")
    heatmap.set_anchor("N")
    heatmap.tick_params(axis="both", length=0, labelsize=6.7)
    heatmap.set_title("a  Pair coverage", loc="left", fontweight="bold")
    for row_index in range(size):
        for column_index in range(row_index):
            count = int(matrix[row_index, column_index])
            text_color = "white" if count >= max(7, maximum * 0.58) else INK
            heatmap.text(
                column_index + 0.5,
                row_index + 0.5,
                str(count),
                ha="center",
                va="center",
                fontsize=6.2,
                color=text_color,
            )
    colorbar = figure.colorbar(
        image,
        ax=heatmap,
        orientation="horizontal",
        fraction=0.055,
        pad=0.20,
        ticks=range(0, maximum + 1, 2),
    )
    colorbar.set_label("Available blinded comparisons", fontsize=7.2)
    colorbar.ax.tick_params(labelsize=6.5, length=2)

    exposures = [exposure_counts[model] for model in model_order]
    y = np.arange(size)
    colors_by_threshold = [BLUE if value >= 40 else GREY for value in exposures]
    exposure_axis.barh(y, exposures, color=colors_by_threshold, height=0.62)
    exposure_axis.axvline(40, color="#D55E00", linewidth=1.0, linestyle="--")
    exposure_axis.set_yticks(y, heatmap_x_names)
    exposure_axis.invert_yaxis()
    exposure_axis.set_xlabel("Candidate comparisons")
    exposure_axis.set_title("b  Model exposure", loc="left", fontweight="bold")
    exposure_axis.grid(axis="x", color=LIGHT, linewidth=0.7)
    exposure_axis.set_axisbelow(True)
    exposure_axis.spines[["top", "right", "left"]].set_visible(False)
    exposure_axis.tick_params(axis="y", length=0, labelsize=6.7)
    exposure_axis.tick_params(axis="x", labelsize=6.7)
    exposure_axis.set_xlim(0, max(exposures) * 1.16)
    for position, value in enumerate(exposures):
        exposure_axis.text(value + 2, position, str(value), va="center", fontsize=6.5)
    exposure_axis.text(
        40,
        -0.85,
        "provisional floor",
        color="#D55E00",
        fontsize=6.4,
        ha="center",
        va="bottom",
    )

    reuse_frequency = Counter(response_reuse.values())
    reuse_counts = np.asarray(sorted(reuse_frequency), dtype=int)
    unique_answers = np.asarray(
        [reuse_frequency[int(count)] for count in reuse_counts],
        dtype=int,
    )
    reuse_axis.bar(
        reuse_counts,
        unique_answers,
        width=0.72,
        color=GREY,
        edgecolor=INK,
        linewidth=0.45,
    )
    reuse_axis.set_xlabel("Comparisons per answer")
    reuse_axis.set_ylabel("Unique answers")
    reuse_axis.set_title("c  Response reuse", loc="left", fontweight="bold")
    reuse_axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    reuse_axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    reuse_axis.grid(axis="y", color=LIGHT, linewidth=0.7)
    reuse_axis.set_axisbelow(True)
    reuse_axis.spines[["top", "right"]].set_visible(False)
    reuse_axis.tick_params(axis="both", labelsize=6.7)
    reuse_axis.text(
        0.98,
        0.98,
        f"n = {len(response_reuse)} compared answers",
        transform=reuse_axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#5F6B7A",
    )

    figure.suptitle(
        "Coverage of the real-output blinded model-arena workload",
        fontsize=10.5,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.005,
        (
            f"{observed['candidate_comparisons']} comparison rows reuse "
            f"{len(response_reuse)} real Epicure-on answers; all {size} "
            "endpoints form one connected component.\nRows are not independent observations; "
            "counts are workload coverage, not wins or quality scores."
        ),
        ha="center",
        va="bottom",
        multialignment="center",
        fontsize=6.8,
        color="#5F6B7A",
    )
    figure.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"Creator": "FlavourBench", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        path.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"Creator": "FlavourBench", "Date": None},
    )
    plt.close(figure)


def write_model_arena_review_pool(
    pool: ReviewPool,
    output_dir: Path,
    *,
    render_figure: bool = True,
) -> dict[str, Path]:
    """Write the private schedule and a public aggregate receipt."""

    output_dir.mkdir(parents=True, exist_ok=True)
    full = output_dir / f"frontier-model-arena-review-pool-{pool.artifact_sha256}.json"
    full.write_text(
        json.dumps(pool.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = _public_receipt(pool)
    receipt_path = output_dir / (
        f"frontier-model-arena-review-receipt-{receipt['artifact_sha256']}.json"
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stable_receipt = output_dir / "frontier-model-arena-review-receipt.json"
    stable_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    coverage_csv = output_dir / "frontier-model-arena-coverage.csv"
    _write_coverage_csv(pool, coverage_csv)
    observed = pool.manifest["observed"]
    exposures = list(observed["comparison_exposures_by_model"].values())
    evidence_units = observed["evidence_units"]
    below_forty = sum(value < 40 for value in exposures)
    macros = output_dir / "frontier-model-arena-review-macros.tex"
    macros.write_text(
        "\n".join(
            [
                rf"\newcommand{{\FrontierArenaCandidateComparisons}}{{{observed['candidate_comparisons']}}}",
                rf"\newcommand{{\FrontierArenaSourceArms}}{{{observed['source_response_arms']}}}",
                rf"\newcommand{{\FrontierArenaTaskCount}}{{{observed['unique_task_ids']}}}",
                rf"\newcommand{{\FrontierArenaGrossCandidateComparisons}}{{{observed['source_candidate_comparisons_before_task_quarantine']}}}",
                rf"\newcommand{{\FrontierArenaGrossSourceArms}}{{{observed['source_response_arms_before_task_quarantine']}}}",
                rf"\newcommand{{\FrontierArenaGrossTaskCount}}{{{observed['source_unique_task_ids_before_task_quarantine']}}}",
                rf"\newcommand{{\FrontierArenaQuarantinedComparisonCount}}{{{observed['task_quarantined_candidate_comparisons']}}}",
                rf"\newcommand{{\FrontierArenaQuarantinedSourceArmCount}}{{{observed['task_quarantined_source_response_arms']}}}",
                rf"\newcommand{{\FrontierArenaQuarantinedTaskCount}}{{{len(observed['task_quarantined_task_ids_observed'])}}}",
                rf"\newcommand{{\FrontierArenaMissingModelPairFamilyCells}}{{{observed['missing_model_pair_family_cells']}}}",
                rf"\newcommand{{\FrontierArenaModelPairFamilyCells}}{{{observed['model_pair_family_cells']}}}",
                rf"\newcommand{{\FrontierArenaResponseReuseMinimum}}{{{evidence_units['minimum_comparisons_per_reused_response_arm']}}}",
                rf"\newcommand{{\FrontierArenaResponseReuseMedian}}{{{evidence_units['median_comparisons_per_reused_response_arm']}}}",
                rf"\newcommand{{\FrontierArenaResponseReuseMaximum}}{{{evidence_units['maximum_comparisons_per_reused_response_arm']}}}",
                rf"\newcommand{{\FrontierArenaTaskStratumClusters}}{{{evidence_units['task_stratum_clusters']}}}",
                rf"\newcommand{{\FrontierArenaMinimumExposure}}{{{min(exposures)}}}",
                rf"\newcommand{{\FrontierArenaMaximumExposure}}{{{max(exposures)}}}",
                rf"\newcommand{{\FrontierArenaBelowFortyCount}}{{{below_forty}}}",
                rf"\newcommand{{\FrontierArenaPoolHash}}{{{pool.artifact_sha256}}}",
                rf"\newcommand{{\FrontierArenaReceiptHash}}{{{receipt['artifact_sha256']}}}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths = {
        "pool": full,
        "receipt": receipt_path,
        "stable_receipt": stable_receipt,
        "coverage_csv": coverage_csv,
        "macros": macros,
    }
    if render_figure:
        coverage_figure = output_dir / "frontier-model-arena-coverage.pdf"
        _render_coverage_figure(pool, coverage_figure)
        paths["coverage_figure"] = coverage_figure
        paths["coverage_svg"] = coverage_figure.with_suffix(".svg")
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("STRATUM", "SUMMARY", "SOURCE_DIR", "RESPONSE_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--import-db",
        action="store_true",
        help="Project the verified real-output pool into the blinded review database.",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Skip optional Matplotlib outputs, for a minimal production import image.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    grouped: dict[str, list[RunInput]] = defaultdict(list)
    for label, summary, source, response in arguments.run:
        grouped[str(label)].append(RunInput(Path(summary), Path(source), Path(response)))
    strata = [
        StratumInput(label=label, runs=tuple(runs)) for label, runs in sorted(grouped.items())
    ]
    pool = build_model_arena_review_pool(strata)
    paths = write_model_arena_review_pool(
        pool,
        arguments.output_dir,
        render_figure=not arguments.skip_figure,
    )
    database_projection: dict[str, Any] | None = None
    if arguments.import_db:
        with session_scope() as session:
            database_projection = import_review_pool(session, pool)
    print(
        json.dumps(
            {
                "status": "verified",
                "artifact_sha256": pool.artifact_sha256,
                "observed": pool.manifest["observed"],
                "outputs": {key: str(value.resolve()) for key, value in paths.items()},
                "database_projection": database_projection,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
