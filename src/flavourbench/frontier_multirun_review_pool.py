"""Build a blinded Epicure-uplift review pool from verified real frontier runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .current_frontier_task_quarantine import (
    quarantine_binding,
    quarantine_task_ids,
)
from .current_pilot_assets import EXTENDED_MODEL_ORDER, MODEL_ORDER
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
from .frontier_multirun_assets import RunInput, _verify_artifact, _verify_summary, verify_runs
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-frontier-multirun-review-pool-v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _on_left_assignments(
    items: Sequence[Mapping[str, Any]],
    aggregate_sha256: str,
    model_order: Sequence[str],
) -> set[str]:
    """Balance Epicure-on placement within each model, then task family."""

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in items:
        model_id = str(item["model_id"])
        work_item_id = str(item["work_item_id"])
        family = str(item["task_family"])
        score = _sha256_text(
            f"{aggregate_sha256}:{model_id}:{family}:{work_item_id}:left-right"
        )
        grouped[model_id].append((score, work_item_id))
    on_left: set[str] = set()
    for model_id in model_order:
        ordered = sorted(grouped.get(model_id, []))
        on_left.update(work_item_id for _, work_item_id in ordered[: len(ordered) // 2])
        if len(ordered) % 2:
            # Alternate the extra left placement by a precommitted model hash.
            if int(_sha256_text(f"{aggregate_sha256}:{model_id}:odd")[:2], 16) % 2 == 0:
                on_left.add(ordered[len(ordered) // 2][1])
    return on_left


def build_multirun_review_pool(inputs: Sequence[RunInput]) -> ReviewPool:
    """Retain every complete real same-model pair and commit its blinded layout."""

    verified = verify_runs(inputs)
    aggregate_sha256 = str(verified.aggregate["artifact_sha256"])
    summaries = [_verify_summary(item.summary) for item in inputs]
    contracts: dict[str, dict[str, Any]] = {}
    work_items: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    source_by_sha: dict[str, dict[str, Any]] = {}
    responses: dict[tuple[str, str], dict[str, Any]] = {}

    for run_input, summary in zip(inputs, summaries, strict=True):
        for raw in summary["manifest"]["models"]:
            row = _mapping(raw, "model contract")
            model_id = str(row["model_id"])
            prior = contracts.setdefault(model_id, row)
            if (
                prior.get("canonical_model_slug") != row.get("canonical_model_slug")
                or prior.get("provider_tag") != row.get("provider_tag")
                or prior.get("execution_backend") != row.get("execution_backend")
            ):
                raise CurrentPilotReviewImportError("model route changed across review inputs")
        work_items.extend(
            _mapping(item, "work item") for item in summary["workload"]["work_items"]
        )
        for path in sorted(run_input.sources.glob("*.json")):
            source = _verify_artifact(path)
            work_item_id = str(source["dataset_work_item_id"])
            if work_item_id in sources:
                raise CurrentPilotReviewImportError("duplicate review source work item")
            sources[work_item_id] = source
            source_by_sha[str(source["artifact_sha256"])] = source
        for path in sorted(run_input.responses.glob("*.json")):
            document = _verify_artifact(path)
            key = (str(document["work_item_id"]), str(document["condition"]))
            if key in responses:
                raise CurrentPilotReviewImportError("duplicate review response")
            responses[key] = document

    contract_models = set(contracts)
    if contract_models == set(MODEL_ORDER):
        active_model_order = MODEL_ORDER
    elif contract_models == set(EXTENDED_MODEL_ORDER):
        active_model_order = EXTENDED_MODEL_ORDER
    else:
        raise CurrentPilotReviewImportError(
            "review inputs must contain the frozen core panel or its Cohere extension"
        )

    source_complete_items = [
        item
        for item in work_items
        if (str(item["work_item_id"]), "epicure_off") in responses
        and (str(item["work_item_id"]), "epicure_on") in responses
    ]
    expected_complete = int(verified.aggregate["totals"]["complete_pairs"])
    if len(source_complete_items) != expected_complete or not source_complete_items:
        raise CurrentPilotReviewImportError("complete review-pair count does not reconcile")
    held_task_ids = quarantine_task_ids()
    complete_items = [
        item for item in source_complete_items if str(item["task_id"]) not in held_task_ids
    ]
    excluded_items = [
        item for item in source_complete_items if str(item["task_id"]) in held_task_ids
    ]
    if not complete_items:
        raise CurrentPilotReviewImportError(
            "task quarantine removed every complete development pair"
        )
    on_left = _on_left_assignments(
        complete_items, aggregate_sha256, active_model_order
    )

    pairs: list[ReviewPair] = []
    items_manifest: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    generation_ids: set[str] = set()
    provider_calls = 0
    tool_calls = 0
    successful_tool_calls = 0
    reviewed_cost_micros = 0
    identity_rows: list[dict[str, str]] = []
    epicure_contract: tuple[str, str, str, str] | None = None

    for item in complete_items:
        work_item_id = str(item["work_item_id"])
        model_id = str(item["model_id"])
        family = str(item["task_family"])
        contract = contracts[model_id]
        source = sources[work_item_id]
        sides = (
            (("left", "epicure_on"), ("right", "epicure_off"))
            if work_item_id in on_left
            else (("left", "epicure_off"), ("right", "epicure_on"))
        )
        review_item_id = _sha256_text(
            f"{aggregate_sha256}:{work_item_id}:review-item"
        )
        arms: list[ReviewArm] = []
        arm_manifest: dict[str, dict[str, Any]] = {}
        for side, condition in sides:
            document = responses[(work_item_id, condition)]
            response_model = _mapping(document.get("model"), "response model")
            task = _mapping(document.get("task"), "response task")
            response = _mapping(document.get("response"), "response payload")
            cost = _mapping(document.get("cost"), "response cost")
            provenance = _mapping(document.get("provenance"), "response provenance")
            linked_source = _mapping(document.get("source"), "response source")
            if source_by_sha.get(str(linked_source.get("artifact_sha256"))) is not source:
                raise CurrentPilotReviewImportError("response source linkage changed")
            answer = _response_answer(document)
            call_count, error_count = _validate_tool_trace(document)
            cost_accounted = response.get("cost_reconciled") is True or (
                response_model.get("execution_backend")
                in {"kimi_direct", "cohere_direct"}
                and cost.get("all_generation_usage_accounted") is True
            )
            if (
                document.get("official") is not False
                or document.get("rank_eligible") is not False
                or document.get("research_result") is not False
                or document.get("research_release_eligible") is not False
                or response_model.get("requested_model_id") != model_id
                or response_model.get("canonical_model_slug")
                != contract.get("canonical_model_slug")
                or response_model.get("actual_model_id")
                != contract.get("canonical_model_slug")
                or task.get("public_id") != item.get("task_id")
                or task.get("family") != family
                or condition != document.get("condition")
                or bool(provenance.get("epicure_access")) != (condition == "epicure_on")
                or not cost_accounted
            ):
                raise CurrentPilotReviewImportError("response crossed the review-pool contract")
            if condition == "epicure_off" and call_count:
                raise CurrentPilotReviewImportError("Epicure-off response contains tool calls")
            if condition == "epicure_on" and call_count - error_count <= 0:
                raise CurrentPilotReviewImportError("Epicure-on response lacks a successful call")
            epicure = _mapping(provenance.get("epicure"), "Epicure provenance")
            observed_epicure = (
                str(epicure.get("release_id")),
                _require_sha256(epicure.get("bundle_sha256"), "Epicure bundle digest"),
                _require_sha256(
                    epicure.get("application_sha256"), "Epicure application digest"
                ),
                _require_sha256(
                    provenance.get("epicure_tool_schema_sha256"), "Epicure tool digest"
                ),
            )
            if epicure_contract is None:
                epicure_contract = observed_epicure
            elif epicure_contract != observed_epicure:
                raise CurrentPilotReviewImportError("Epicure lineage differs within the pool")
            arm_generation_ids = _sequence(response.get("generation_ids"), "generation IDs")
            if not arm_generation_ids:
                raise CurrentPilotReviewImportError("review response has no provider generation")
            for value in arm_generation_ids:
                generation_id = str(value)
                if not generation_id or generation_id in generation_ids:
                    raise CurrentPilotReviewImportError("provider generation ID is duplicated")
                generation_ids.add(generation_id)
            provider_calls += len(arm_generation_ids)
            tool_calls += call_count
            successful_tool_calls += call_count - error_count
            cost_micros = response.get("cost_micros")
            if not isinstance(cost_micros, int) or isinstance(cost_micros, bool):
                raise CurrentPilotReviewImportError("response cost must be an integer")
            reviewed_cost_micros += cost_micros
            answer_sha256 = _sha256_text(answer)
            arm_manifest[side] = {
                "condition": condition,
                "response_artifact_sha256": document["artifact_sha256"],
                "source_artifact_sha256": source["artifact_sha256"],
                "answer_sha256": answer_sha256,
                "actual_model_id": response_model["actual_model_id"],
                "actual_provider": response_model["actual_provider"],
                "generation_ids_sha256": sha256_json(arm_generation_ids),
            }
            identity_rows.append(
                {
                    "review_item_id": review_item_id,
                    "side": side,
                    "condition": condition,
                    "requested_model_id": model_id,
                    "actual_model_id": str(response_model["actual_model_id"]),
                    "actual_provider": str(response_model["actual_provider"]),
                }
            )
            arms.append(
                ReviewArm(side=side, condition=condition, response=document, source=source)
            )
        pairs.append(
            ReviewPair(
                review_item_id=review_item_id,
                work_item=item,
                arms=(arms[0], arms[1]),
            )
        )
        family_counts[family] += 1
        model_counts[model_id] += 1
        task = _mapping(responses[(work_item_id, "epicure_off")]["task"], "task")
        items_manifest.append(
            {
                "review_item_id": review_item_id,
                "work_item_id": work_item_id,
                "task_id": item["task_id"],
                "task_family": family,
                "prompt_sha256": task["prompt_sha256"],
                "requested_model_id": model_id,
                "canonical_model_slug": contract["canonical_model_slug"],
                "provider_tag": contract["provider_tag"],
                "left": arm_manifest["left"],
                "right": arm_manifest["right"],
            }
        )

    if epicure_contract is None:
        raise CurrentPilotReviewImportError("review pool has no Epicure contract")
    left_on = sum(item["left"]["condition"] == "epicure_on" for item in items_manifest)
    if abs(2 * left_on - len(items_manifest)) > len(active_model_order):
        raise CurrentPilotReviewImportError("left/right treatment placement is not balanced")
    release_id, bundle_sha256, application_sha256, tool_schema_sha256 = epicure_contract
    model_contracts = {
        model_id: {
            "canonical_model_slug": contracts[model_id]["canonical_model_slug"],
            "provider_tag": contracts[model_id]["provider_tag"],
            "execution_backend": contracts[model_id]["execution_backend"],
        }
        for model_id in active_model_order
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "summary_content_address": aggregate_sha256,
            "source_run_class": "real_multirun_development_pilot",
            "input_summaries": [
                {
                    "filename": run_input.summary.name,
                    "content_address": summary["content_address"]["digest"],
                    "file_sha256": hashlib.sha256(run_input.summary.read_bytes()).hexdigest(),
                }
                for run_input, summary in zip(inputs, summaries, strict=True)
            ],
        },
        "selection_policy": {
            "paired_same_model_same_task": True,
            "complete_pairs_only": True,
            "task_quarantine": quarantine_binding(),
            "quarantine_pending_qualified_adjudication": True,
            "raw_answers_edited": False,
            "deterministic_model_balanced_side_assignment": True,
            "execution_policy_sha256": verified.aggregate["execution_policy_sha256"],
            "failed_or_partial_pairs_retained_for_reliability_only": verified.aggregate[
                "totals"
            ]["failed_or_partial_pairs"],
        },
        "observed": {
            "source_candidate_pairs_before_task_quarantine": len(source_complete_items),
            "task_quarantined_candidate_pairs": len(excluded_items),
            "task_quarantined_source_arms": 2 * len(excluded_items),
            "task_quarantined_task_ids_observed": sorted(
                {str(item["task_id"]) for item in excluded_items}
            ),
            "candidate_pairs": len(pairs),
            "source_arms": 2 * len(pairs),
            "unique_task_ids": len({str(item["task_id"]) for item in complete_items}),
            "distinct_tasks": len({str(item["task_id"]) for item in complete_items}),
            "candidate_pairs_by_family": dict(family_counts),
            "candidate_pairs_by_model": {
                model_id: model_counts[model_id] for model_id in active_model_order
            },
            "real_provider_calls": provider_calls,
            "real_epicure_calls": tool_calls,
            "successful_real_epicure_calls": successful_tool_calls,
            "synthetic_arms": 0,
            "reviewed_source_cost_micros": reviewed_cost_micros,
            "left_epicure_on": left_on,
            "right_epicure_on": len(pairs) - left_on,
        },
        "epicure": {
            "release_id": release_id,
            "bundle_sha256": bundle_sha256,
            "application_sha256": application_sha256,
            "tool_schema_sha256": tool_schema_sha256,
        },
        "identity_commitment_sha256": sha256_json(identity_rows),
        "model_order": list(active_model_order),
        "model_contracts": model_contracts,
        "items": items_manifest,
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "permitted_use": "blinded development-pilot review",
            "prohibited_use": "quality leaderboard before real judgments are collected",
            "quarantined_tasks_may_enter_official_fit": False,
        },
    }
    payload["artifact_sha256"] = sha256_json(payload)
    return ReviewPool(manifest=payload, pairs=tuple(pairs))


def write_review_pool(pool: ReviewPool, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"frontier-multirun-review-pool-{pool.artifact_sha256}.json"
    path.write_text(
        json.dumps(pool.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("SUMMARY", "SOURCE_DIR", "RESPONSE_DIR"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    inputs = [RunInput(Path(a), Path(b), Path(c)) for a, b, c in arguments.run]
    pool = build_multirun_review_pool(inputs)
    path = write_review_pool(pool, arguments.output_dir)
    result: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "reviewPoolSha256": pool.artifact_sha256,
        "manifestPath": str(path.resolve()),
        "candidatePairs": len(pool.pairs),
        "syntheticArms": 0,
    }
    if not arguments.manifest_only:
        with session_scope() as session:
            result.update(import_review_pool(session, pool))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
