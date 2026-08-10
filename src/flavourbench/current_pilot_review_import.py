"""Build and import the blinded review pool for the dated frontier pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from .current_pilot_assets import (
    DISPLAY_NAMES,
    FAMILY_ORDER,
    MODEL_ORDER,
    CurrentPilotAssetError,
    _verify_hashed_artifact,
    verify_pilot,
)
from .database import session_scope
from .engine import is_complete_finish_reason
from .expert_review import canonical_sha256
from .models import Battle, CatalogModel, ResponseArm, RunEvent, Season, Task, ToolCall
from .real_task_bank import sha256_json

SCHEMA_VERSION = "flavourbench-current-frontier-review-pool-v1"
SCHEDULER_VERSION = "current-frontier-real-pilot-review-import-v1"
SEASON_SLUG_PREFIX = "current-frontier-review"
MODEL_ARENA_SEASON_SLUG_PREFIX = "current-frontier-model-arena-review"
UUID_NAMESPACE = uuid.UUID("5ace1895-84fa-49fa-ab96-60ad1befbf49")
OPEN_WEIGHT_ORGANIZATIONS = frozenset(
    {"deepseek", "minimax", "mistralai", "moonshotai", "nvidia", "z-ai"}
)


class CurrentPilotReviewImportError(RuntimeError):
    """The review pool or its database projection failed verification."""


@dataclass(frozen=True)
class ReviewArm:
    side: str
    condition: str
    response: dict[str, Any]
    source: dict[str, Any]


@dataclass(frozen=True)
class ReviewPair:
    review_item_id: str
    work_item: dict[str, Any]
    arms: tuple[ReviewArm, ReviewArm]


@dataclass(frozen=True)
class ReviewPool:
    manifest: dict[str, Any]
    pairs: tuple[ReviewPair, ...]

    @property
    def artifact_sha256(self) -> str:
        return str(self.manifest["artifact_sha256"])


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CurrentPilotReviewImportError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CurrentPilotReviewImportError(f"{label} must be an array")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CurrentPilotReviewImportError(f"{label} must be a sha256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise CurrentPilotReviewImportError(f"{label} must be a sha256 digest") from error
    return value


def _parse_datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise CurrentPilotReviewImportError(f"{label} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CurrentPilotReviewImportError(f"{label} must be an ISO timestamp") from error
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _stable_id(kind: str, value: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, f"{kind}:{value}"))


def _side_assignments(
    work_items: Sequence[Mapping[str, Any]],
    summary_sha256: str,
) -> set[str]:
    """Choose Epicure-on left positions with balance inside each task family."""

    on_left: set[str] = set()
    by_family: dict[str, list[tuple[str, str]]] = {family: [] for family in FAMILY_ORDER}
    for item in work_items:
        work_item_id = str(item["work_item_id"])
        family = str(item["task_family"])
        score = _sha256_text(f"{summary_sha256}:{work_item_id}:left-right")
        by_family[family].append((score, work_item_id))
    for family in FAMILY_ORDER:
        ordered = sorted(by_family[family])
        on_left.update(work_item_id for _, work_item_id in ordered[: (len(ordered) + 1) // 2])
    return on_left


def _response_answer(document: Mapping[str, Any]) -> str:
    response = _mapping(document.get("response"), "response payload")
    answer = response.get("answer_markdown")
    if not isinstance(answer, str) or not answer.strip():
        raise CurrentPilotReviewImportError("a review response has no answer text")
    if not is_complete_finish_reason(str(response.get("finish_reason") or "")):
        raise CurrentPilotReviewImportError("a review response has an incomplete finish reason")
    return answer


def _validate_tool_trace(document: Mapping[str, Any]) -> tuple[int, int]:
    response = _mapping(document.get("response"), "response payload")
    trace = _sequence(response.get("tool_trace"), "tool trace")
    errors = 0
    for event in trace:
        row = _mapping(event, "tool trace event")
        _mapping(row.get("arguments"), "tool arguments")
        result = row.get("result")
        if not isinstance(result, str):
            raise CurrentPilotReviewImportError("tool result must be text")
        if _require_sha256(row.get("result_sha256"), "tool result digest") != _sha256_text(
            result
        ):
            raise CurrentPilotReviewImportError("tool result digest does not verify")
        errors += int(bool(row.get("is_error")))
    return len(trace), errors


def build_review_pool(
    *,
    summary_path: Path,
    source_dir: Path,
    response_dir: Path,
) -> ReviewPool:
    """Verify the full run and retain every complete real same-model pair."""

    try:
        verified = verify_pilot(summary_path, source_dir, response_dir)
    except CurrentPilotAssetError as error:
        raise CurrentPilotReviewImportError(str(error)) from error
    summary = verified.summary
    summary_sha256 = str(summary["content_address"]["digest"])
    manifest_contracts = {
        str(row["model_id"]): row for row in summary["manifest"]["models"]
    }
    work_items = [_mapping(item, "work item") for item in summary["workload"]["work_items"]]

    sources: dict[str, dict[str, Any]] = {}
    source_by_sha: dict[str, dict[str, Any]] = {}
    for path in sorted(source_dir.glob("*.json")):
        source = _verify_hashed_artifact(path)
        work_item_id = str(source["dataset_work_item_id"])
        sources[work_item_id] = source
        source_by_sha[str(source["artifact_sha256"])] = source

    responses: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(response_dir.glob("*.json")):
        document = _verify_hashed_artifact(path)
        key = (str(document["work_item_id"]), str(document["condition"]))
        responses[key] = document

    complete_items = [
        item
        for item in work_items
        if (str(item["work_item_id"]), "epicure_off") in responses
        and (str(item["work_item_id"]), "epicure_on") in responses
    ]
    if len(complete_items) != 47:
        raise CurrentPilotReviewImportError("the final pilot must yield exactly 47 complete pairs")
    on_left = _side_assignments(complete_items, summary_sha256)

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
        requested_model_id = str(item["model_id"])
        family = str(item["task_family"])
        contract = _mapping(manifest_contracts.get(requested_model_id), "model contract")
        source = sources[work_item_id]
        by_condition = {
            condition: responses[(work_item_id, condition)]
            for condition in ("epicure_off", "epicure_on")
        }
        expected_sides = (
            (("left", "epicure_on"), ("right", "epicure_off"))
            if work_item_id in on_left
            else (("left", "epicure_off"), ("right", "epicure_on"))
        )
        review_item_id = _sha256_text(f"{summary_sha256}:{work_item_id}:review-item")
        arms: list[ReviewArm] = []
        arm_manifest: dict[str, dict[str, Any]] = {}
        for side, condition in expected_sides:
            document = by_condition[condition]
            model = _mapping(document.get("model"), "response model")
            task = _mapping(document.get("task"), "response task")
            response = _mapping(document.get("response"), "response payload")
            provenance = _mapping(document.get("provenance"), "response provenance")
            linked_source = _mapping(document.get("source"), "response source")
            source_document = source_by_sha.get(str(linked_source.get("artifact_sha256")))
            if source_document is not source:
                raise CurrentPilotReviewImportError("response source linkage changed")
            answer = _response_answer(document)
            call_count, error_count = _validate_tool_trace(document)
            if (
                document.get("official") is not False
                or document.get("rank_eligible") is not False
                or document.get("research_result") is not False
                or document.get("research_release_eligible") is not False
                or model.get("requested_model_id") != requested_model_id
                or model.get("canonical_model_slug") != contract.get("canonical_model_slug")
                or model.get("actual_model_id") != contract.get("canonical_model_slug")
                or task.get("public_id") != item.get("task_id")
                or task.get("family") != family
                or condition != document.get("condition")
                or bool(provenance.get("epicure_access")) != (condition == "epicure_on")
                or response.get("cost_reconciled") is not True
            ):
                raise CurrentPilotReviewImportError("response crossed the review pool contract")
            if condition == "epicure_off" and call_count:
                raise CurrentPilotReviewImportError("Epicure-off response contains tool calls")
            epicure = _mapping(provenance.get("epicure"), "Epicure provenance")
            observed_epicure_contract = (
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
                epicure_contract = observed_epicure_contract
            elif epicure_contract != observed_epicure_contract:
                raise CurrentPilotReviewImportError("Epicure lineage differs within the pool")
            arm_generation_ids = _sequence(response.get("generation_ids"), "generation ids")
            if not arm_generation_ids:
                raise CurrentPilotReviewImportError("review response has no provider generation")
            for raw_generation_id in arm_generation_ids:
                generation_id = str(raw_generation_id)
                if not generation_id or generation_id in generation_ids:
                    raise CurrentPilotReviewImportError(
                        "provider generation identity is duplicated"
                    )
                generation_ids.add(generation_id)
            provider_calls += len(arm_generation_ids)
            tool_calls += call_count
            successful_tool_calls += call_count - error_count
            cost_micros = response.get("cost_micros")
            if isinstance(cost_micros, bool) or not isinstance(cost_micros, int):
                raise CurrentPilotReviewImportError("response cost must be an integer")
            reviewed_cost_micros += cost_micros
            answer_sha256 = _sha256_text(answer)
            arm_manifest[side] = {
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
                    "requested_model_id": requested_model_id,
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
        if {arm.condition for arm in arms} != {"epicure_off", "epicure_on"}:
            raise CurrentPilotReviewImportError("review item is not an uplift pair")
        pairs.append(
            ReviewPair(
                review_item_id=review_item_id,
                work_item=item,
                arms=(arms[0], arms[1]),
            )
        )
        family_counts[family] += 1
        model_counts[requested_model_id] += 1
        task = _mapping(by_condition["epicure_off"].get("task"), "response task")
        items_manifest.append(
            {
                "review_item_id": review_item_id,
                "work_item_id": work_item_id,
                "task_id": item["task_id"],
                "task_family": family,
                "prompt_sha256": task["prompt_sha256"],
                "requested_model_id": requested_model_id,
                "canonical_model_slug": contract["canonical_model_slug"],
                "provider_tag": contract["provider_tag"],
                "left": arm_manifest["left"],
                "right": arm_manifest["right"],
            }
        )

    if epicure_contract is None:
        raise CurrentPilotReviewImportError("review pool has no Epicure contract")
    expected_family_counts = {
        "substitution": 14,
        "composition": 8,
        "cookability": 12,
        "evidence": 13,
    }
    if dict(family_counts) != expected_family_counts:
        raise CurrentPilotReviewImportError("review pool family coverage changed")
    if set(model_counts) != set(MODEL_ORDER) or sum(model_counts.values()) != 47:
        raise CurrentPilotReviewImportError("review pool model coverage changed")
    if tool_calls != 67 or successful_tool_calls != 25:
        raise CurrentPilotReviewImportError("review pool tool-call totals changed")
    if provider_calls > verified.counts["provider_generations"]:
        raise CurrentPilotReviewImportError("review pool provider calls exceed the source run")
    left_on = sum(item["left"]["condition"] == "epicure_on" for item in items_manifest)
    if left_on not in {23, 24}:
        raise CurrentPilotReviewImportError("left and right placement is not balanced")

    release_id, bundle_sha256, application_sha256, tool_schema_sha256 = epicure_contract
    identity_commitment_sha256 = sha256_json(identity_rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "summary_filename": summary_path.name,
            "summary_content_address": summary_sha256,
            "summary_file_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "source_run_id": summary["runner_run_id"],
            "source_run_class": "real_development_pilot",
        },
        "selection_policy": {
            "paired_same_model_same_task": True,
            "complete_pairs_only": True,
            "raw_answers_edited": False,
            "deterministic_family_balanced_side_assignment": True,
            "failed_or_partial_pairs_retained_for_reliability_only": 9,
        },
        "observed": {
            "candidate_pairs": 47,
            "source_arms": 94,
            "candidate_pairs_by_family": expected_family_counts,
            "candidate_pairs_by_model": {
                model_id: model_counts[model_id] for model_id in MODEL_ORDER
            },
            "real_provider_calls": provider_calls,
            "real_epicure_calls": tool_calls,
            "successful_real_epicure_calls": successful_tool_calls,
            "synthetic_arms": 0,
            "reviewed_source_cost_micros": reviewed_cost_micros,
            "left_epicure_on": left_on,
            "right_epicure_on": 47 - left_on,
        },
        "epicure": {
            "release_id": release_id,
            "bundle_sha256": bundle_sha256,
            "application_sha256": application_sha256,
            "tool_schema_sha256": tool_schema_sha256,
        },
        "identity_commitment_sha256": identity_commitment_sha256,
        "model_order": list(MODEL_ORDER),
        "items": items_manifest,
        "claim_boundary": {
            "official": False,
            "rank_eligible": False,
            "research_result": False,
            "quality_judgments": 0,
            "permitted_use": "blinded development-pilot review",
            "prohibited_use": "quality leaderboard before real judgments are collected",
        },
    }
    artifact_sha256 = sha256_json(payload)
    manifest = {**payload, "artifact_sha256": artifact_sha256}
    return ReviewPool(manifest=manifest, pairs=tuple(pairs))


def write_review_pool_manifest(pool: ReviewPool, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"current-frontier-review-pool-{pool.artifact_sha256}.json"
    path.write_text(
        json.dumps(pool.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _source_output(arm: ReviewArm) -> dict[str, Any]:
    document = arm.response
    response = _mapping(document["response"], "response payload")
    provenance = _mapping(document["provenance"], "response provenance")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sourceResponseArtifactSha256": document["artifact_sha256"],
        "sourceProviderArtifactSha256": document["source"]["artifact_sha256"],
        "sourceRunId": document["source"]["run_id"],
        "sourceManifestSha256": document["manifest_sha256"],
        "sourceTaskRegistrySha256": document["task_registry_sha256"],
        "sourceCost": document["cost"],
        "sourceGenerationMetadata": response.get("generation_metadata", []),
        "sourceOutputJson": response.get("output_json", {}),
        "sourceIntermediateOutputs": response.get("intermediate_outputs", []),
        "sourceMcpTraceEvents": provenance.get("mcp_trace_events", []),
        "sourceProtocolBundleSha256": provenance.get("protocol_bundle_sha256"),
        "sourceEndpointExecutionSha256": document["model"].get(
            "endpoint_execution_sha256"
        ),
        "sourceEndpointManifestSha256": document["model"].get(
            "endpoint_manifest_sha256"
        ),
        "sourceExecutionPolicySha256": document["model"].get(
            "execution_policy_sha256"
        ),
        "sourceRankEligible": False,
        "synthetic": False,
        "archiveUse": "blinded current-frontier development review",
    }


def _pool_track(pool: ReviewPool) -> str:
    track = str(pool.manifest.get("track") or "epicure_uplift")
    if track not in {"epicure_uplift", "model_arena"}:
        raise CurrentPilotReviewImportError(f"unsupported review track: {track}")
    return track


def _source_commitment(pool: ReviewPool) -> str:
    source = _mapping(pool.manifest.get("source"), "pool source")
    value = source.get("summary_content_address") or source.get(
        "source_commitment_sha256"
    )
    return _require_sha256(value, "review source commitment")


def _pool_projection_contract(pool: ReviewPool) -> tuple[int, int]:
    observed = _mapping(pool.manifest.get("observed"), "pool observations")
    if _pool_track(pool) == "model_arena":
        pair_count = int(observed.get("candidate_comparisons") or 0)
        tool_count = int(observed.get("projected_epicure_calls") or 0)
    else:
        pair_count = int(observed.get("candidate_pairs") or 0)
        tool_count = int(observed.get("real_epicure_calls") or 0)
    if pair_count != len(pool.pairs) or pair_count <= 0:
        raise CurrentPilotReviewImportError("review pool pair count does not reconcile")
    if tool_count < 0:
        raise CurrentPilotReviewImportError("review pool tool-call count is invalid")
    return pair_count, tool_count


def _tool_rows(arm_id: str, arm: ReviewArm) -> list[ToolCall]:
    response = _mapping(arm.response["response"], "response payload")
    trace = _sequence(response.get("tool_trace"), "tool trace")
    round_counts: Counter[int] = Counter()
    rows: list[ToolCall] = []
    for raw in trace:
        event = _mapping(raw, "tool trace event")
        round_index = int(event.get("round_index", 0))
        call_index = round_counts[round_index]
        round_counts[round_index] += 1
        arguments = _mapping(event.get("arguments"), "tool arguments")
        result_text = str(event["result"])
        try:
            structured = json.loads(result_text)
        except json.JSONDecodeError:
            structured = {}
        if not isinstance(structured, dict):
            structured = {"value": structured}
        tool_call_id = event.get("tool_call_id")
        rows.append(
            ToolCall(
                id=_stable_id("tool-call", f"{arm_id}:{round_index}:{call_index}"),
                arm_id=arm_id,
                round_index=round_index,
                call_index=call_index,
                tool_call_id=str(tool_call_id) if tool_call_id else None,
                tool_name=str(event.get("name") or ""),
                arguments_json=arguments,
                arguments_sha256=canonical_sha256({"arguments": arguments}),
                result_text=result_text,
                structured_content_json=structured,
                structured_content_sha256=canonical_sha256({"structured": structured}),
                result_sha256=_sha256_text(result_text),
                latency_ms=int(event.get("latency_ms", 0)),
                is_error=bool(event.get("is_error")),
            )
        )
    return rows


def _review_protocol(pool: ReviewPool) -> tuple[dict[str, Any], str]:
    track = _pool_track(pool)
    payload = {
        "schemaVersion": str(pool.manifest.get("schema_version") or SCHEMA_VERSION),
        "reviewPoolSha256": pool.artifact_sha256,
        "sourceSummarySha256": _source_commitment(pool),
        "track": track,
        "blinded": True,
        "official": False,
        "rankEligible": False,
        "syntheticArms": 0,
    }
    return payload, canonical_sha256(payload)


def _projection_counts(session: Session, season_id: str) -> dict[str, int]:
    return {
        "tasks": int(
            session.scalar(
                select(func.count()).select_from(Task).where(Task.season_id == season_id)
            )
            or 0
        ),
        "battles": int(
            session.scalar(
                select(func.count()).select_from(Battle).where(Battle.season_id == season_id)
            )
            or 0
        ),
        "arms": int(
            session.scalar(
                select(func.count())
                .select_from(ResponseArm)
                .join(Battle, Battle.id == ResponseArm.battle_id)
                .where(Battle.season_id == season_id)
            )
            or 0
        ),
        "toolCalls": int(
            session.scalar(
                select(func.count())
                .select_from(ToolCall)
                .join(ResponseArm, ResponseArm.id == ToolCall.arm_id)
                .join(Battle, Battle.id == ResponseArm.battle_id)
                .where(Battle.season_id == season_id)
            )
            or 0
        ),
    }


def _projection_digest(session: Session, season_id: str) -> str:
    season = session.get(Season, season_id)
    if season is None:
        raise CurrentPilotReviewImportError("review season is missing")
    tasks = session.scalars(select(Task).where(Task.season_id == season_id).order_by(Task.id)).all()
    battles = session.scalars(
        select(Battle).where(Battle.season_id == season_id).order_by(Battle.id)
    ).all()
    battle_ids = [battle.id for battle in battles]
    arms = (
        session.scalars(
            select(ResponseArm)
            .where(ResponseArm.battle_id.in_(battle_ids))
            .order_by(ResponseArm.id)
        ).all()
        if battle_ids
        else []
    )
    arm_ids = [arm.id for arm in arms]
    tools = (
        session.scalars(
            select(ToolCall).where(ToolCall.arm_id.in_(arm_ids)).order_by(ToolCall.id)
        ).all()
        if arm_ids
        else []
    )
    payload = {
        "season": {
            "id": season.id,
            "slug": season.slug,
            "status": season.status,
            "official": season.official,
            "manifest_sha256": season.manifest_sha256,
            "protocol_bundle_sha256": season.protocol_bundle_sha256,
            "epicure_release_id": season.epicure_release_id,
            "epicure_bundle_sha256": season.epicure_bundle_sha256,
            "epicure_application_sha256": season.epicure_application_sha256,
        },
        "tasks": [
            {
                "id": task.id,
                "public_id": task.public_id,
                "family": task.family,
                "prompt": task.prompt,
                "prompt_sha256": task.prompt_sha256,
                "split": task.split,
                "review_status": task.review_status,
                "provenance_json": task.provenance_json,
            }
            for task in tasks
        ],
        "battles": [
            {
                "id": battle.id,
                "task_id": battle.task_id,
                "run_class": battle.run_class,
                "rank_eligible": battle.rank_eligible,
                "data_stratum": battle.data_stratum,
                "manifest_sha256": battle.manifest_sha256,
                "protocol_bundle_sha256": battle.protocol_bundle_sha256,
                "track": battle.track,
                "category": battle.category,
                "prompt_sha256": battle.prompt_sha256,
                "status": battle.status,
                "left_arm_id": battle.left_arm_id,
                "right_arm_id": battle.right_arm_id,
            }
            for battle in battles
        ],
        "arms": [
            {
                "id": arm.id,
                "battle_id": arm.battle_id,
                "side": arm.side,
                "condition": arm.condition,
                "model_id": arm.model_id,
                "provider_slug": arm.provider_slug,
                "actual_provider_slug": arm.actual_provider_slug,
                "actual_model_id": arm.actual_model_id,
                "generation_id": arm.generation_id,
                "provider_generation_ids_json": arm.provider_generation_ids_json,
                "status": arm.status,
                "answer_markdown": arm.answer_markdown,
                "answer_markdown_sha256": arm.answer_markdown_sha256,
                "output_json": arm.output_json,
                "output_json_sha256": arm.output_json_sha256,
                "finish_reason": arm.finish_reason,
                "epicure_attestation_json": arm.epicure_attestation_json,
            }
            for arm in arms
        ],
        "tools": [
            {
                "id": tool.id,
                "arm_id": tool.arm_id,
                "round_index": tool.round_index,
                "call_index": tool.call_index,
                "tool_name": tool.tool_name,
                "arguments_json": tool.arguments_json,
                "result_text": tool.result_text,
                "result_sha256": tool.result_sha256,
                "is_error": tool.is_error,
            }
            for tool in tools
        ],
    }
    return canonical_sha256(payload)


def _lock_import(session: Session, pool_sha256: str) -> None:
    if session.get_bind().dialect.name == "postgresql":
        value = int(_sha256_text(f"current-frontier-review:{pool_sha256}")[:16], 16)
        if value >= 2**63:
            value -= 2**64
        session.execute(text("SELECT pg_advisory_xact_lock(:value)"), {"value": value})


def import_review_pool(session: Session, pool: ReviewPool) -> dict[str, Any]:
    """Project the immutable real artifacts into the blinded review tables."""

    pool_sha256 = pool.artifact_sha256
    pool_schema_version = str(pool.manifest.get("schema_version") or SCHEMA_VERSION)
    track = _pool_track(pool)
    source_commitment = _source_commitment(pool)
    season_id = _stable_id("season", pool_sha256)
    _lock_import(session, pool_sha256)
    existing_season = session.get(Season, season_id)
    candidate_pairs, projected_tool_calls = _pool_projection_contract(pool)
    expected_counts = {
        "tasks": len({str(pair.work_item["task_id"]) for pair in pool.pairs}),
        "battles": candidate_pairs,
        "arms": 2 * candidate_pairs,
        "toolCalls": projected_tool_calls,
    }
    event_entity_type = (
        "frontier_model_arena_review_pool"
        if track == "model_arena"
        else "current_frontier_review_pool"
    )
    event_type = (
        "frontier_model_arena_review_pool_imported"
        if track == "model_arena"
        else "current_frontier_review_pool_imported"
    )
    if existing_season is not None:
        observed_counts = _projection_counts(session, season_id)
        event = session.scalar(
            select(RunEvent).where(
                RunEvent.entity_type == event_entity_type,
                RunEvent.entity_id == pool_sha256,
                RunEvent.event_type == event_type,
            )
        )
        if observed_counts != expected_counts or event is None:
            raise CurrentPilotReviewImportError("existing review pool is partial")
        observed_digest = _projection_digest(session, season_id)
        if (
            event.payload_json.get("projection_sha256") != observed_digest
            or event.payload_json.get("review_pool_sha256") != pool_sha256
            or event.payload_json.get("counts") != expected_counts
        ):
            raise CurrentPilotReviewImportError("existing review pool projection has drifted")
        return {
            "schemaVersion": pool_schema_version,
            "reviewPoolSha256": pool_sha256,
            **observed_counts,
            "syntheticArms": 0,
            "rankEligibleBattles": 0,
            "idempotent": True,
            "eventId": event.id,
        }

    protocol_bundle, protocol_sha256 = _review_protocol(pool)
    epicure = pool.manifest["epicure"]
    prompt_rows = sorted(
        {
            (
                str(pair.arms[0].response["task"]["public_id"]),
                str(pair.arms[0].response["task"]["prompt_sha256"]),
            )
            for pair in pool.pairs
        }
    )
    season = Season(
        id=season_id,
        slug=(
            f"{MODEL_ARENA_SEASON_SLUG_PREFIX}-{pool_sha256[:12]}"
            if track == "model_arena"
            else f"{SEASON_SLUG_PREFIX}-{pool_sha256[:12]}"
        ),
        name=(
            "Current frontier real-output blinded model arena"
            if track == "model_arena"
            else "Current frontier real-pilot blinded review pool"
        ),
        status="pilot",
        official=False,
        manifest_sha256=pool_sha256,
        prompt_registry_sha256=canonical_sha256(prompt_rows),
        tool_registry_sha256=str(epicure["tool_schema_sha256"]),
        epicure_release_id=str(epicure["release_id"]),
        epicure_bundle_sha256=str(epicure["bundle_sha256"]),
        epicure_application_sha256=str(epicure["application_sha256"]),
        analysis_plan_sha256=canonical_sha256(
            {
                "track": track,
                "single_rater_reporting": True,
                "quality_ranking_before_judgment": False,
            }
        ),
        protocol_bundle_json=protocol_bundle,
        protocol_bundle_sha256=protocol_sha256,
        budget_cap_micros=0,
        budget_used_micros=0,
        budget_reserved_micros=0,
    )
    session.add(season)

    model_order = tuple(str(value) for value in pool.manifest.get("model_order") or MODEL_ORDER)
    raw_contracts = pool.manifest.get("model_contracts")
    contracts = (
        {
            str(model_id): _mapping(contract, "model contract")
            for model_id, contract in _mapping(raw_contracts, "model contracts").items()
        }
        if raw_contracts is not None
        else {
            str(row["requested_model_id"]): row
            for row in pool.manifest["items"]
            if isinstance(row, dict)
        }
    )
    existing_model_ids = set(
        session.scalars(
            select(CatalogModel.model_id).where(CatalogModel.model_id.in_(model_order))
        ).all()
    )
    for model_id in model_order:
        if model_id in existing_model_ids:
            continue
        contract = contracts[model_id]
        organization = model_id.split("/", 1)[0]
        left_contract = contract.get("left")
        source_response_sha256 = contract.get("source_response_artifact_sha256")
        if source_response_sha256 is None and isinstance(left_contract, Mapping):
            source_response_sha256 = left_contract.get("response_artifact_sha256")
        session.add(
            CatalogModel(
                model_id=model_id,
                canonical_slug=str(contract["canonical_model_slug"]),
                name=DISPLAY_NAMES.get(model_id, model_id.rsplit("/", 1)[-1]),
                family=organization,
                catalog_source="real_pilot_archive",
                open_weight=organization in OPEN_WEIGHT_ORGANIZATIONS,
                open_weight_evidence_json={
                    "source": "frozen current-frontier manifest",
                    "reviewPoolSha256": pool_sha256,
                },
                status="smoke_passed",
                supports_tools=True,
                supports_structured_outputs=True,
                endpoint_json={
                    "providerTag": contract["provider_tag"],
                    "sourceResponseArtifactSha256": source_response_sha256,
                },
            )
        )
    session.flush()

    task_ids: dict[str, str] = {}
    for pair in pool.pairs:
        task_document = _mapping(pair.arms[0].response["task"], "task")
        public_id = str(task_document["public_id"])
        if public_id in task_ids:
            continue
        task_id = _stable_id("task", f"{pool_sha256}:{public_id}")
        task_ids[public_id] = task_id
        session.add(
            Task(
                id=task_id,
                public_id=public_id,
                season_id=season_id,
                family=str(task_document["family"]),
                prompt=str(task_document["prompt"]),
                prompt_sha256=str(task_document["prompt_sha256"]),
                revision=1,
                split="pilot",
                review_status="candidate",
                provenance_json={
                    "reviewPoolSha256": pool_sha256,
                    "sourceSummarySha256": source_commitment,
                    "sourceTaskRegistrySha256": pair.arms[0].response[
                        "task_registry_sha256"
                    ],
                    "synthetic": False,
                },
            )
        )
    session.flush()

    tool_rows: list[ToolCall] = []
    for ordinal, pair in enumerate(pool.pairs):
        execution_stratum = str(pair.work_item.get("stratum") or "development")
        task_document = _mapping(pair.arms[0].response["task"], "task")
        source_started = min(
            _parse_datetime(arm.source["started_at"], "source start") for arm in pair.arms
        )
        source_completed = max(
            _parse_datetime(arm.source["completed_at"], "source completion")
            for arm in pair.arms
        )
        battle_id = _stable_id("battle", f"{pool_sha256}:{pair.review_item_id}")
        battle = Battle(
            id=battle_id,
            season_id=season_id,
            run_class="pilot",
            rank_eligible=False,
            data_stratum="development",
            task_id=task_ids[str(task_document["public_id"])],
            task_revision=1,
            controlled_run_id=None,
            manifest_sha256=pool_sha256,
            protocol_bundle_sha256=protocol_sha256,
            scheduler_version=SCHEDULER_VERSION,
            assignment_seed=_sha256_text(pair.review_item_id),
            track_assignment_probability="archived-deterministic",
            model_assignment_probability="archived-deterministic",
            side_assignment_probability=(
                "deterministic-model-pair-balanced-randomization"
                if track == "model_arena"
                else "deterministic-family-balanced-randomization"
            ),
            track=track,
            category=str(task_document["family"]),
            prompt=str(task_document["prompt"]),
            prompt_sha256=str(task_document["prompt_sha256"]),
            client_nonce_sha256=_sha256_text(f"{pool_sha256}:{ordinal}"),
            prompt_redacted=False,
            research_consent=False,
            retention_basis="development_research",
            release_review_status="not_requested",
            requester_pseudonym=_sha256_text(f"current-frontier-review:{pool_sha256}"),
            status="queued",
            reserved_cost_micros=0,
            provider_reservations_json={"executionStratum": execution_stratum},
            created_at=source_started,
            completed_at=None,
            retention_until=source_completed + timedelta(days=3650),
        )
        session.add(battle)
        session.flush()
        side_ids: dict[str, str] = {}
        for arm in pair.arms:
            document = arm.response
            model = _mapping(document["model"], "response model")
            response = _mapping(document["response"], "response payload")
            provenance = _mapping(document["provenance"], "response provenance")
            output = _source_output(arm)
            answer = _response_answer(document)
            generation_ids = [str(value) for value in response["generation_ids"]]
            attestation = {
                "reviewPoolSha256": pool_sha256,
                "sourceResponseArtifactSha256": document["artifact_sha256"],
                "sourceProviderArtifactSha256": document["source"]["artifact_sha256"],
                "condition": arm.condition,
                "epicureAccess": bool(provenance["epicure_access"]),
                "completeMcpTrace": True,
                "synthetic": False,
            }
            arm_id = _stable_id(
                "response-arm",
                (
                    f"{pool_sha256}:{pair.review_item_id}:{arm.side}:"
                    f"{document['artifact_sha256']}"
                ),
            )
            response_arm = ResponseArm(
                id=arm_id,
                battle_id=battle_id,
                side=arm.side,
                condition=arm.condition,
                model_id=str(model["requested_model_id"]),
                execution_backend=str(model.get("execution_backend") or "openrouter"),
                provider_slug=str(model["provider_tag"]),
                status="queued",
                prompt_sha256=str(task_document["prompt_sha256"]),
                system_prompt_sha256=_require_sha256(
                    provenance.get("system_prompt_sha256"), "system prompt digest"
                ),
                schema_sha256=_require_sha256(
                    response.get("backend_response_schema_sha256"),
                    "response schema digest",
                ),
                tool_schema_sha256=str(epicure["tool_schema_sha256"]),
                decoding_json=_mapping(response.get("decoding"), "decoding configuration"),
                protocol_bundle_sha256=protocol_sha256,
                epicure_release_id=str(epicure["release_id"]),
                epicure_bundle_sha256=str(epicure["bundle_sha256"]),
                epicure_application_sha256=str(epicure["application_sha256"]),
                backend_response_schema_sha256=str(
                    response["backend_response_schema_sha256"]
                ),
                backend_tool_schema_sha256=str(response["backend_tool_schema_sha256"]),
                created_at=_parse_datetime(arm.source["started_at"], "source start"),
            )
            session.add(response_arm)
            session.flush()
            response_arm.actual_provider_slug = str(model["actual_provider"])
            response_arm.actual_model_id = str(model["actual_model_id"])
            response_arm.generation_id = str(response["generation_id"])
            response_arm.provider_generation_ids_json = generation_ids
            response_arm.status = "complete"
            response_arm.answer_markdown = answer
            response_arm.answer_markdown_sha256 = _sha256_text(answer)
            response_arm.output_json = output
            response_arm.output_json_sha256 = canonical_sha256(output)
            response_arm.observed_decoding_json = _mapping(
                response.get("decoding"), "observed decoding configuration"
            )
            response_arm.epicure_attestation_json = attestation
            response_arm.epicure_attestation_sha256 = canonical_sha256(attestation)
            response_arm.prompt_tokens = int(response["prompt_tokens"])
            response_arm.completion_tokens = int(response["completion_tokens"])
            response_arm.reasoning_tokens = int(response["reasoning_tokens"])
            response_arm.cost_micros = 0
            response_arm.cost_reconciled = True
            response_arm.cost_accounting_basis = (
                "archived_review_projection_zero_incremental_cost"
            )
            response_arm.billing_reconciliation_status = (
                "not_applicable_archived_review_projection"
            )
            response_arm.latency_ms = int(response["latency_ms"])
            response_arm.retries = int(response["retries"])
            response_arm.finish_reason = str(response["finish_reason"])
            response_arm.completed_at = _parse_datetime(
                arm.source["completed_at"], "source completion"
            )
            session.add(response_arm)
            side_ids[arm.side] = arm_id
            tool_rows.extend(_tool_rows(arm_id, arm))
        session.flush()
        battle.left_arm_id = side_ids["left"]
        battle.right_arm_id = side_ids["right"]
        battle.status = "complete"
        battle.completed_at = source_completed
        session.add(battle)
    session.add_all(tool_rows)
    session.flush()

    observed_counts = _projection_counts(session, season_id)
    if observed_counts != expected_counts:
        raise CurrentPilotReviewImportError(
            f"imported review pool counts do not reconcile: {observed_counts}"
        )
    projection_sha256 = _projection_digest(session, season_id)
    event = RunEvent(
        id=_stable_id("run-event", pool_sha256),
        entity_type=event_entity_type,
        entity_id=pool_sha256,
        event_type=event_type,
        payload_json={
            "schema_version": pool_schema_version,
            "review_pool_sha256": pool_sha256,
            "source_summary_sha256": source_commitment,
            "track": track,
            "counts": expected_counts,
            "projection_sha256": projection_sha256,
            "synthetic_arm_count": 0,
            "rank_eligible_battle_count": 0,
            "claim_boundary": (
                "Blinded current-frontier development review. Quality results require "
                "real submitted judgments and remain separate from official rankings."
            ),
        },
    )
    session.add(event)
    session.flush()
    return {
        "schemaVersion": pool_schema_version,
        "reviewPoolSha256": pool_sha256,
        **observed_counts,
        "syntheticArms": 0,
        "rankEligibleBattles": 0,
        "idempotent": False,
        "eventId": event.id,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and import the dated real frontier pilot review pool."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    pool = build_review_pool(
        summary_path=arguments.summary,
        source_dir=arguments.source,
        response_dir=arguments.responses,
    )
    manifest_path = write_review_pool_manifest(pool, arguments.output)
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
