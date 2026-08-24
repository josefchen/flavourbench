"""Build and optionally import the blinded required-Epicure frontier review pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .current_pilot_assets import DISPLAY_NAMES, FAMILY_ORDER
from .current_pilot_review_import import (
    CurrentPilotReviewImportError,
    ReviewArm,
    ReviewPair,
    ReviewPool,
    _mapping,
    _response_answer,
    _validate_tool_trace,
    import_review_pool,
)
from .database import session_scope
from .real_task_bank import sha256_json
from .required_pilot_assets import (
    EXPECTED_EPICURE_APPLICATION_SHA256,
    EXPECTED_EPICURE_BUNDLE_SHA256,
    EXPECTED_EPICURE_RELEASE,
    EXPECTED_EPICURE_TOOL_SCHEMA_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_POLICY_SHA256,
    EXPECTED_SUMMARY_SHA256,
    VerifiedRequiredPilot,
    verify_required_pilot,
)

SCHEMA_VERSION = "flavourbench-required-frontier-review-pool-v2"

EXPECTED_FAMILY_COUNTS = {
    "substitution": 11,
    "composition": 9,
    "cookability": 13,
    "evidence": 10,
}
EXPECTED_MODEL_COUNTS = {
    "moonshotai/kimi-k3": 4,
    "openai/gpt-5.6-sol-pro": 2,
    "anthropic/claude-fable-5": 3,
    "anthropic/claude-opus-5": 4,
    "anthropic/claude-sonnet-5": 4,
    "google/gemini-3.1-pro-preview": 3,
    "google/gemini-3.6-flash": 4,
    "x-ai/grok-4.5": 4,
    "z-ai/glm-5.2": 4,
    "deepseek/deepseek-v4-pro": 3,
    "deepseek/deepseek-v4-flash-0731": 2,
    "minimax/minimax-m3": 2,
    "nvidia/nemotron-3-ultra-550b-a55b": 4,
    "mistralai/mistral-medium-3-5": 0,
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _balanced_side_assignments(
    work_items: Sequence[Mapping[str, Any]], summary_sha256: str
) -> set[str]:
    """Balance treatment placement globally and within each task family."""

    by_family: dict[str, list[tuple[str, str]]] = {family: [] for family in FAMILY_ORDER}
    for item in work_items:
        work_item_id = str(item["work_item_id"])
        family = str(item["task_family"])
        score = _sha256_text(f"{summary_sha256}:{work_item_id}:left-right-v2")
        by_family[family].append((score, work_item_id))
    ordered_families = sorted(
        (
            _sha256_text(f"{summary_sha256}:{family}:family-left-extra"),
            family,
        )
        for family, items in by_family.items()
        if len(items) % 2
    )
    target_left = (len(work_items) + 1) // 2
    floor_left = sum(len(items) // 2 for items in by_family.values())
    extra_families = {family for _, family in ordered_families[: target_left - floor_left]}
    on_left: set[str] = set()
    for family in FAMILY_ORDER:
        ordered = sorted(by_family[family])
        count = len(ordered) // 2 + int(family in extra_families)
        on_left.update(work_item_id for _, work_item_id in ordered[:count])
    if abs(len(on_left) - (len(work_items) - len(on_left))) > 1:
        raise CurrentPilotReviewImportError("side assignment is not globally balanced")
    return on_left


def _model_contracts(pilot: VerifiedRequiredPilot) -> dict[str, dict[str, Any]]:
    manifest_rows = {
        str(row["model_id"]): _mapping(row, "model contract")
        for row in pilot.summary["manifest"]["models"]
    }
    contracts: dict[str, dict[str, Any]] = {}
    for model_id, row in manifest_rows.items():
        source_response = next(
            (
                document["artifact_sha256"]
                for (work_item_id, _), document in pilot.responses.items()
                if next(
                    item["model_id"]
                    for item in pilot.work_items
                    if item["work_item_id"] == work_item_id
                )
                == model_id
            ),
            None,
        )
        contracts[model_id] = {
            "display_name": DISPLAY_NAMES[model_id],
            "canonical_model_slug": row["canonical_model_slug"],
            "provider_tag": row["provider_tag"],
            "execution_backend": row["execution_backend"],
            "endpoint_execution_sha256": row["endpoint_execution_sha256"],
            "source_response_artifact_sha256": source_response,
        }
    return contracts


def build_required_review_pool(
    *, summary_path: Path, source_dir: Path, response_dir: Path
) -> ReviewPool:
    """Retain every complete real pair after verifying the required treatment."""

    pilot = verify_required_pilot(summary_path, source_dir, response_dir)
    work_items = list(pilot.work_items)
    complete_items = [
        item
        for item in work_items
        if (str(item["work_item_id"]), "epicure_off") in pilot.responses
        and (str(item["work_item_id"]), "epicure_on") in pilot.responses
    ]
    if len(complete_items) != 43:
        raise CurrentPilotReviewImportError("required pilot must yield exactly 43 complete pairs")
    on_left = _balanced_side_assignments(complete_items, EXPECTED_SUMMARY_SHA256)
    contracts = _model_contracts(pilot)

    pairs: list[ReviewPair] = []
    item_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, str]] = []
    family_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    generation_ids: set[str] = set()
    provider_calls = 0
    tool_calls = 0
    successful_tool_calls = 0
    reviewed_cost_micros = 0

    for item in complete_items:
        work_item_id = str(item["work_item_id"])
        model_id = str(item["model_id"])
        family = str(item["task_family"])
        source = pilot.sources[work_item_id]
        by_condition = {
            condition: pilot.responses[(work_item_id, condition)]
            for condition in ("epicure_off", "epicure_on")
        }
        expected_sides = (
            (("left", "epicure_on"), ("right", "epicure_off"))
            if work_item_id in on_left
            else (("left", "epicure_off"), ("right", "epicure_on"))
        )
        review_item_id = _sha256_text(
            f"{EXPECTED_SUMMARY_SHA256}:{work_item_id}:required-review-item-v2"
        )
        arms: list[ReviewArm] = []
        arm_rows: dict[str, dict[str, Any]] = {}
        for side, condition in expected_sides:
            document = by_condition[condition]
            response = _mapping(document.get("response"), "response payload")
            model = _mapping(document.get("model"), "response model")
            provenance = _mapping(document.get("provenance"), "response provenance")
            answer = _response_answer(document)
            call_count, error_count = _validate_tool_trace(document)
            if condition == "epicure_off" and call_count:
                raise CurrentPilotReviewImportError("control response contains a tool call")
            if condition == "epicure_on" and (
                call_count == 0 or call_count - error_count <= 0
            ):
                raise CurrentPilotReviewImportError(
                    "required treatment contains no successful Epicure call"
                )
            if (
                document.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
                or document.get("execution_policy_sha256") != EXPECTED_POLICY_SHA256
                or model.get("requested_model_id") != model_id
                or model.get("canonical_model_slug")
                != contracts[model_id]["canonical_model_slug"]
                or model.get("actual_model_id") != contracts[model_id]["canonical_model_slug"]
                or bool(provenance.get("epicure_access")) != (condition == "epicure_on")
            ):
                raise CurrentPilotReviewImportError("review arm identity changed")
            epicure = _mapping(provenance.get("epicure"), "Epicure provenance")
            if (
                epicure.get("release_id") != EXPECTED_EPICURE_RELEASE
                or epicure.get("bundle_sha256") != EXPECTED_EPICURE_BUNDLE_SHA256
                or epicure.get("application_sha256")
                != EXPECTED_EPICURE_APPLICATION_SHA256
                or provenance.get("epicure_tool_schema_sha256")
                != EXPECTED_EPICURE_TOOL_SCHEMA_SHA256
            ):
                raise CurrentPilotReviewImportError("review Epicure identity changed")
            arm_generation_ids = response.get("generation_ids")
            if not isinstance(arm_generation_ids, list) or not arm_generation_ids:
                raise CurrentPilotReviewImportError("review response has no provider generation")
            for generation_id in map(str, arm_generation_ids):
                if not generation_id or generation_id in generation_ids:
                    raise CurrentPilotReviewImportError("provider generation is duplicated")
                generation_ids.add(generation_id)
            provider_calls += len(arm_generation_ids)
            tool_calls += call_count
            successful_tool_calls += call_count - error_count
            cost_micros = response.get("cost_micros")
            if isinstance(cost_micros, bool) or not isinstance(cost_micros, int):
                raise CurrentPilotReviewImportError("review cost must be an integer")
            reviewed_cost_micros += cost_micros
            answer_sha256 = _sha256_text(answer)
            arm_rows[side] = {
                "condition": condition,
                "response_artifact_sha256": document["artifact_sha256"],
                "source_artifact_sha256": source["artifact_sha256"],
                "answer_sha256": answer_sha256,
                "actual_model_id": model["actual_model_id"],
                "actual_provider": model["actual_provider"],
                "generation_ids_sha256": sha256_json(arm_generation_ids),
            }
            identity_rows.append(
                {
                    "review_item_id": review_item_id,
                    "side": side,
                    "condition": condition,
                    "requested_model_id": model_id,
                    "actual_model_id": str(model["actual_model_id"]),
                    "actual_provider": str(model["actual_provider"]),
                }
            )
            arms.append(
                ReviewArm(
                    side=side,
                    condition=condition,
                    response=document,
                    source=source,
                )
            )
        pairs.append(
            ReviewPair(review_item_id=review_item_id, work_item=item, arms=(arms[0], arms[1]))
        )
        family_counts[family] += 1
        model_counts[model_id] += 1
        task = _mapping(by_condition["epicure_off"].get("task"), "response task")
        item_rows.append(
            {
                "review_item_id": review_item_id,
                "work_item_id": work_item_id,
                "task_id": item["task_id"],
                "task_family": family,
                "prompt_sha256": task["prompt_sha256"],
                "requested_model_id": model_id,
                "canonical_model_slug": contracts[model_id]["canonical_model_slug"],
                "provider_tag": contracts[model_id]["provider_tag"],
                "execution_backend": contracts[model_id]["execution_backend"],
                "left": arm_rows["left"],
                "right": arm_rows["right"],
            }
        )

    normalized_model_counts = {
        model_id: model_counts[model_id] for model_id in EXPECTED_MODEL_COUNTS
    }
    if dict(family_counts) != EXPECTED_FAMILY_COUNTS:
        raise CurrentPilotReviewImportError("review family coverage changed")
    if normalized_model_counts != EXPECTED_MODEL_COUNTS:
        raise CurrentPilotReviewImportError("review model coverage changed")
    if (provider_calls, tool_calls, successful_tool_calls) != (277, 182, 86):
        raise CurrentPilotReviewImportError("review provider or tool totals changed")
    left_on = sum(row["left"]["condition"] == "epicure_on" for row in item_rows)
    if (left_on, len(item_rows) - left_on) not in {(21, 22), (22, 21)}:
        raise CurrentPilotReviewImportError("review side placement is not balanced")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "summary_filename": summary_path.name,
            "summary_content_address": EXPECTED_SUMMARY_SHA256,
            "summary_file_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "source_run_id": pilot.summary["runner_run_id"],
            "source_run_class": "real_required_epicure_development_pilot",
        },
        "selection_policy": {
            "paired_same_model_same_task": True,
            "complete_pairs_only": True,
            "required_successful_epicure_treatment": True,
            "raw_answers_edited": False,
            "deterministic_family_stratified_side_assignment": True,
            "globally_balanced_side_assignment": True,
            "failed_or_partial_pairs_retained_for_reliability_only": 13,
        },
        "observed": {
            "candidate_pairs": 43,
            "source_arms": 86,
            "candidate_pairs_by_family": EXPECTED_FAMILY_COUNTS,
            "candidate_pairs_by_model": normalized_model_counts,
            "real_provider_calls": provider_calls,
            "real_epicure_calls": tool_calls,
            "successful_real_epicure_calls": successful_tool_calls,
            "synthetic_arms": 0,
            "reviewed_source_cost_micros": reviewed_cost_micros,
            "left_epicure_on": left_on,
            "right_epicure_on": 43 - left_on,
            "quality_judgments": 0,
        },
        "epicure": {
            "release_id": EXPECTED_EPICURE_RELEASE,
            "bundle_sha256": EXPECTED_EPICURE_BUNDLE_SHA256,
            "application_sha256": EXPECTED_EPICURE_APPLICATION_SHA256,
            "tool_schema_sha256": EXPECTED_EPICURE_TOOL_SCHEMA_SHA256,
            "lineage_status": "unmatched_exploratory_runtime",
        },
        "identity_commitment_sha256": sha256_json(identity_rows),
        "model_order": [row["model_id"] for row in pilot.summary["manifest"]["models"]],
        "model_contracts": contracts,
        "items": item_rows,
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "permitted_use": "blinded required-Epicure development-pilot review",
            "prohibited_use": "quality leaderboard before real judgments are collected",
        },
    }
    artifact_sha256 = sha256_json(payload)
    return ReviewPool(
        manifest={**payload, "artifact_sha256": artifact_sha256},
        pairs=tuple(pairs),
    )


def write_required_review_pool(pool: ReviewPool, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"required-frontier-review-pool-{pool.artifact_sha256}.json"
    path.write_text(
        json.dumps(pool.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and import the required-Epicure frontier review pool."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    pool = build_required_review_pool(
        summary_path=arguments.summary,
        source_dir=arguments.source,
        response_dir=arguments.responses,
    )
    manifest_path = write_required_review_pool(pool, arguments.output)
    result: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "reviewPoolSha256": pool.artifact_sha256,
        "manifestPath": str(manifest_path),
        "candidatePairs": len(pool.pairs),
        "syntheticArms": 0,
    }
    if not arguments.manifest_only:
        with session_scope() as session:
            result.update(import_review_pool(session, pool))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run()
