from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from .database import session_scope
from .expert_calibration import (
    BLINDING_LEAK_PATTERN_SHA256,
    MANUAL_RESPONSE_QUARANTINE,
    RESPONSE_CONTENT_REVIEW_SHA256,
    TASK_QUALITY_QUARANTINE,
    TASK_QUALITY_REVIEW_SHA256,
    TASK_SCOPE_QUARANTINE,
    TASK_SCOPE_REVIEW_SHA256,
)
from .expert_review import TASK_FAMILIES, canonical_sha256
from .models import (
    Battle,
    CatalogModel,
    ResponseArm,
    RunEvent,
    Season,
    Task,
    ToolCall,
)

SCHEMA_VERSION = "flavourbench-author-evaluator-pool-import-v1"
CANDIDATE_SCHEMA_VERSION = "flavourbench-expert-calibration-candidate-v11"
ACCEPTED_FINAL_FINISH_REASONS = frozenset({"completed", "end_turn", "stop", "stop_sequence"})
SEASON_SLUG_PREFIX = "season-0-author-evaluator"
SCHEDULER_VERSION = "author-evaluator-real-evidence-import-v1"
UUID_NAMESPACE = uuid.UUID("9f453a4d-64aa-41e9-b260-64de9136d49a")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AuthorEvaluatorImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImportArm:
    side: str
    answer_sha256: str
    source: dict[str, Any]
    comparison_reference: dict[str, Any]


@dataclass(frozen=True)
class ImportPair:
    item: dict[str, Any]
    comparison: dict[str, Any]
    arms: tuple[ImportArm, ImportArm]


@dataclass(frozen=True)
class ImportBundle:
    candidate: dict[str, Any]
    comparison_manifest: dict[str, Any]
    model_manifest: dict[str, Any]
    pairs: tuple[ImportPair, ...]

    @property
    def candidate_sha256(self) -> str:
        return str(self.candidate["artifact_sha256"])


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorEvaluatorImportError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise AuthorEvaluatorImportError(f"expected JSON object: {path}")
    return value


def _verify_artifact(path: Path, value: Mapping[str, Any]) -> str:
    claimed = value.get("artifact_sha256")
    if not isinstance(claimed, str) or SHA256_PATTERN.fullmatch(claimed) is None:
        raise AuthorEvaluatorImportError(f"artifact has no valid digest: {path}")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    artifact_digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if artifact_digest != claimed:
        raise AuthorEvaluatorImportError(f"artifact digest mismatch: {path}")
    return claimed


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuthorEvaluatorImportError(f"{label} must be an object")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise AuthorEvaluatorImportError(f"{label} must be a sha256 digest")
    return value


def _arm_inventory(arm_directory: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    inventory: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(arm_directory.glob("*.json")):
        arm = _object(path)
        arm_id = arm.get("arm_id")
        if not isinstance(arm_id, str) or not arm_id:
            raise AuthorEvaluatorImportError(f"arm has no arm_id: {path}")
        if arm_id in inventory:
            raise AuthorEvaluatorImportError(f"duplicate source arm id: {arm_id}")
        inventory[arm_id] = (path, arm)
    return inventory


def _answer_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise AuthorEvaluatorImportError("unsupported candidate schema")
    observed = _require_mapping(candidate.get("observed"), "candidate observed")
    use_policy = _require_mapping(candidate.get("use_policy"), "candidate use policy")
    selection = _require_mapping(candidate.get("selection_policy"), "candidate selection policy")
    items = candidate.get("items")
    if not isinstance(items, list) or len(items) != 32:
        raise AuthorEvaluatorImportError("author review pool requires exactly 32 candidate pairs")
    if (
        observed.get("candidate_pairs") != 32
        or observed.get("source_arms") != 64
        or observed.get("synthetic_arms") != 0
        or observed.get("real_provider_calls", 0) < 64
        or observed.get("successful_real_epicure_calls", 0) < 32
    ):
        raise AuthorEvaluatorImportError("candidate real-output counts are inadmissible")
    if (
        use_policy.get("calibration_only") is not True
        or use_policy.get("rank_eligible") is not False
        or use_policy.get("official_season_items") is not False
        or use_policy.get("benchmark_result_use") != "prohibited"
    ):
        raise AuthorEvaluatorImportError("candidate quarantine policy is not intact")
    if (
        selection.get("paired_same_model_same_task") is not True
        or selection.get("raw_answers_edited") is not False
        or selection.get("real_provider_evidence_required") is not True
        or selection.get("successful_real_epicure_call_required_in_on_arm") is not True
        or selection.get("normal_final_completion_required") is not True
        or selection.get("accepted_final_finish_reasons") != sorted(ACCEPTED_FINAL_FINISH_REASONS)
        or selection.get("specialist_scope_quarantine_required") is not True
        or selection.get("specialist_scope_quarantine_task_ids") != sorted(TASK_SCOPE_QUARANTINE)
        or selection.get("specialist_scope_review_sha256") != TASK_SCOPE_REVIEW_SHA256
        or selection.get("task_quality_quarantine_required") is not True
        or selection.get("task_quality_quarantine_task_ids") != sorted(TASK_QUALITY_QUARANTINE)
        or selection.get("task_quality_review_sha256") != TASK_QUALITY_REVIEW_SHA256
        or selection.get("manual_response_content_quarantine_required") is not True
        or selection.get("manual_response_quarantine_answer_sha256s")
        != sorted(MANUAL_RESPONSE_QUARANTINE)
        or selection.get("response_content_review_sha256") != RESPONSE_CONTENT_REVIEW_SHA256
        or selection.get("blinding_leak_pattern_sha256") != BLINDING_LEAK_PATTERN_SHA256
    ):
        raise AuthorEvaluatorImportError("candidate selection evidence is incomplete")
    candidate_task_ids = {
        str(_require_mapping(item, "candidate item").get("task_id")) for item in items
    }
    overlap = candidate_task_ids.intersection(TASK_SCOPE_QUARANTINE)
    if overlap:
        raise AuthorEvaluatorImportError(
            "candidate contains specialist-scope quarantined tasks: " + ", ".join(sorted(overlap))
        )
    quality_overlap = candidate_task_ids.intersection(TASK_QUALITY_QUARANTINE)
    if quality_overlap:
        raise AuthorEvaluatorImportError(
            "candidate contains task-quality quarantined tasks: "
            + ", ".join(sorted(quality_overlap))
        )
    candidate_answer_sha256s: set[str] = set()
    for raw_item in items:
        item = _require_mapping(raw_item, "candidate item")
        for side in ("left", "right"):
            answer = _require_mapping(item.get(side), f"candidate {side} answer")
            candidate_answer_sha256s.add(
                _require_sha256(answer.get("answer_sha256"), f"candidate {side} answer digest")
            )
    response_overlap = candidate_answer_sha256s.intersection(MANUAL_RESPONSE_QUARANTINE)
    if response_overlap:
        raise AuthorEvaluatorImportError(
            "candidate contains response-content quarantined answers: "
            + ", ".join(sorted(response_overlap))
        )
    family_counts = Counter(
        str(_require_mapping(item, "candidate item").get("family")) for item in items
    )
    if family_counts != Counter({family: 8 for family in TASK_FAMILIES}):
        raise AuthorEvaluatorImportError("candidate family balance is not 8/8/8/8")


def load_bundle(
    *,
    candidate_path: Path,
    comparison_manifest_path: Path,
    model_manifest_path: Path,
    arm_directory: Path,
) -> ImportBundle:
    candidate = _object(candidate_path)
    comparison_manifest = _object(comparison_manifest_path)
    model_manifest = _object(model_manifest_path)
    _verify_artifact(candidate_path, candidate)
    _verify_artifact(comparison_manifest_path, comparison_manifest)
    _verify_artifact(model_manifest_path, model_manifest)
    _validate_candidate(candidate)

    models = model_manifest.get("models")
    if not isinstance(models, list):
        raise AuthorEvaluatorImportError("model manifest has no models")
    models_by_season_id = {
        str(model["season_model_id"]): _require_mapping(model, "manifest model")
        for model in models
        if isinstance(model, dict) and isinstance(model.get("season_model_id"), str)
    }
    comparisons = comparison_manifest.get("comparisons")
    if not isinstance(comparisons, list):
        raise AuthorEvaluatorImportError("comparison manifest has no comparisons")
    comparisons_by_key: dict[tuple[str, frozenset[str]], list[dict[str, Any]]] = {}
    for raw in comparisons:
        comparison = _require_mapping(raw, "comparison")
        left = _require_mapping(comparison.get("left"), "comparison left arm")
        right = _require_mapping(comparison.get("right"), "comparison right arm")
        left_answer_sha = left.get("answer_sha256")
        right_answer_sha = right.get("answer_sha256")
        if (
            comparison.get("judgable") is not True
            or not isinstance(left_answer_sha, str)
            or SHA256_PATTERN.fullmatch(left_answer_sha) is None
            or not isinstance(right_answer_sha, str)
            or SHA256_PATTERN.fullmatch(right_answer_sha) is None
        ):
            continue
        task_id = str(comparison.get("task_id", ""))
        key = (
            task_id,
            frozenset({left_answer_sha, right_answer_sha}),
        )
        comparisons_by_key.setdefault(key, []).append(comparison)

    inventory = _arm_inventory(arm_directory)
    used_arm_ids: set[str] = set()
    import_pairs: list[ImportPair] = []
    for raw_item in candidate["items"]:
        item = _require_mapping(raw_item, "candidate item")
        task_id = str(item.get("task_id", ""))
        left_candidate = _require_mapping(item.get("left"), "candidate left answer")
        right_candidate = _require_mapping(item.get("right"), "candidate right answer")
        candidate_answers = {
            "left": left_candidate,
            "right": right_candidate,
        }
        key = (
            task_id,
            frozenset(
                _require_sha256(answer.get("answer_sha256"), "candidate answer digest")
                for answer in candidate_answers.values()
            ),
        )
        matches = comparisons_by_key.get(key, [])
        if len(matches) != 1:
            raise AuthorEvaluatorImportError(
                f"{task_id} resolves to {len(matches)} comparison records"
            )
        comparison = matches[0]
        if (
            comparison.get("track") != "epicure_uplift"
            or comparison.get("judgable") is not True
            or comparison.get("task_family") != item.get("family")
            or comparison.get("task_sha256") != item.get("task_sha256")
        ):
            raise AuthorEvaluatorImportError(f"{task_id} comparison contract mismatch")
        comparison_by_answer: dict[str, dict[str, Any]] = {}
        for comparison_side in ("left", "right"):
            reference = _require_mapping(
                comparison.get(comparison_side),
                f"comparison {comparison_side}",
            )
            comparison_by_answer[
                _require_sha256(
                    reference.get("answer_sha256"),
                    f"comparison {comparison_side} answer digest",
                )
            ] = reference
        imported_arms: list[ImportArm] = []
        for presented_side, candidate_answer in candidate_answers.items():
            answer = candidate_answer.get("answer_markdown")
            answer_sha = _require_sha256(
                candidate_answer.get("answer_sha256"),
                f"{task_id} {presented_side} answer digest",
            )
            if not isinstance(answer, str) or _answer_sha256(answer) != answer_sha:
                raise AuthorEvaluatorImportError(
                    f"{task_id} {presented_side} answer content mismatch"
                )
            reference = comparison_by_answer.get(answer_sha)
            if reference is None:
                raise AuthorEvaluatorImportError(f"{task_id} {presented_side} has no source arm")
            arm_id = str(reference.get("arm_id", ""))
            if arm_id in used_arm_ids:
                raise AuthorEvaluatorImportError(f"source arm reused: {arm_id}")
            source_entry = inventory.get(arm_id)
            if source_entry is None:
                raise AuthorEvaluatorImportError(f"source arm file missing: {arm_id}")
            source_path, source = source_entry
            source_sha = _verify_artifact(source_path, source)
            source_result = _require_mapping(source.get("result"), f"{arm_id} result")
            source_model = _require_mapping(source.get("model"), f"{arm_id} model")
            source_task = _require_mapping(source.get("task"), f"{arm_id} task")
            source_contracts = _require_mapping(source.get("contracts"), f"{arm_id} contracts")
            season_model_id = str(reference.get("season_model_id", ""))
            manifest_model = models_by_season_id.get(season_model_id)
            if (
                source.get("status") != "success"
                or source.get("synthetic") is not False
                or source.get("rank_eligible") is not True
                or str(source_result.get("finish_reason") or "").strip().lower()
                not in ACCEPTED_FINAL_FINISH_REASONS
                or source.get("artifact_sha256") != reference.get("arm_artifact_sha256")
                or source.get("condition") != reference.get("condition")
                or source_task.get("task_id") != task_id
                or source_task.get("task_sha256") != item.get("task_sha256")
                or source_task.get("prompt") != item.get("prompt")
                or source_task.get("prompt_sha256") != item.get("prompt_sha256")
                or source_result.get("answer_markdown") != answer
                or source_model.get("season_model_id") != season_model_id
                or manifest_model is None
                or source_model.get("canonical_model_id")
                != manifest_model.get("canonical_model_id")
                or source_contracts.get("model_manifest_artifact_sha256")
                != model_manifest.get("artifact_sha256")
                or source_sha != reference.get("arm_artifact_sha256")
            ):
                raise AuthorEvaluatorImportError(f"source arm contract mismatch: {arm_id}")
            if source.get("condition") == "epicure_on" and (
                source_result.get("real_epicure_calls", 0) < 1
                or not any(
                    isinstance(trace, dict) and trace.get("is_error") is False
                    for trace in source_result.get("tool_trace", [])
                )
            ):
                raise AuthorEvaluatorImportError(
                    f"Epicure-on arm lacks a successful real tool call: {arm_id}"
                )
            used_arm_ids.add(arm_id)
            imported_arms.append(
                ImportArm(
                    side=presented_side,
                    answer_sha256=answer_sha,
                    source=source,
                    comparison_reference=reference,
                )
            )
        if {arm.source["condition"] for arm in imported_arms} != {
            "epicure_on",
            "epicure_off",
        }:
            raise AuthorEvaluatorImportError(f"{task_id} is not a paired uplift item")
        if len({arm.source["model"]["canonical_model_id"] for arm in imported_arms}) != 1:
            raise AuthorEvaluatorImportError(f"{task_id} does not use the same model")
        import_pairs.append(
            ImportPair(
                item=item,
                comparison=comparison,
                arms=(imported_arms[0], imported_arms[1]),
            )
        )
    if len(import_pairs) != 32 or len(used_arm_ids) != 64:
        raise AuthorEvaluatorImportError("validated author review pool is incomplete")
    return ImportBundle(
        candidate=candidate,
        comparison_manifest=comparison_manifest,
        model_manifest=model_manifest,
        pairs=tuple(import_pairs),
    )


def _stable_id(kind: str, value: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, f"{kind}:{value}"))


def _lock_candidate_import(session: Session, candidate_sha256: str) -> None:
    """Serialize first import and idempotent retries for one candidate."""

    if session.get_bind().dialect.name == "postgresql":
        material = hashlib.sha256(
            f"author-evaluator-import:{candidate_sha256}".encode()
        ).hexdigest()
        lock_key = int(material[:16], 16)
        if lock_key >= 2**63:
            lock_key -= 2**64
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )
        return
    session.scalar(
        select(Season.id).where(Season.manifest_sha256 == candidate_sha256).with_for_update()
    )


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _same_datetime(observed: datetime | None, expected: datetime | None) -> bool:
    if observed is None or expected is None:
        return observed is expected
    observed_utc = (
        observed.replace(tzinfo=UTC) if observed.tzinfo is None else observed.astimezone(UTC)
    )
    expected_utc = (
        expected.replace(tzinfo=UTC) if expected.tzinfo is None else expected.astimezone(UTC)
    )
    return observed_utc == expected_utc


def _provider_slug(source: Mapping[str, Any]) -> str:
    model = _require_mapping(source.get("model"), "source model")
    explicit = model.get("provider_slug")
    return str(explicit) if isinstance(explicit, str) and explicit else str(model["provider"])


def _actual_model_id(source: Mapping[str, Any]) -> str:
    result = _require_mapping(source.get("result"), "source result")
    returned = result.get("returned_model_ids")
    if isinstance(returned, list) and len(returned) == 1 and isinstance(returned[0], str):
        return returned[0]
    model = _require_mapping(source.get("model"), "source model")
    return str(model["canonical_model_id"])


def _source_output(source: Mapping[str, Any]) -> dict[str, Any]:
    result = _require_mapping(source.get("result"), "source result")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sourceArtifactSha256": source["artifact_sha256"],
        "sourceArmId": source["arm_id"],
        "sourcePhase": source.get("phase"),
        "sourceRankEligible": source.get("rank_eligible"),
        "sourceDeliveryState": source.get("delivery_state"),
        "providerCalls": result.get("provider_calls"),
        "realEpicureCalls": result.get("real_epicure_calls"),
        "requestIdSha256s": result.get("request_id_sha256s", []),
        "requestPayloadSha256s": result.get("request_payload_sha256s", []),
        "responseLatenciesMs": result.get("response_latencies_ms", []),
        "returnedModelIds": result.get("returned_model_ids", []),
        "actualProviderNames": result.get("actual_provider_names", []),
        "generationIds": result.get("generation_ids", []),
        "generationAccounting": result.get("generation_accounting", []),
        "usage": result.get("usage", {}),
        "toolTrace": result.get("tool_trace", []),
        "sourceActualCostUsd": result.get("actual_cost_usd"),
        "costStatus": result.get("cost_status"),
        "archiveUse": (
            "blinded author-evaluator calibration case study; development stratum; "
            "never official or rank eligible"
        ),
    }


def _tool_rows(response_arm_id: str, source: Mapping[str, Any]) -> list[ToolCall]:
    result = _require_mapping(source.get("result"), "source result")
    traces = result.get("tool_trace", [])
    if not isinstance(traces, list):
        raise AuthorEvaluatorImportError("source tool trace must be a list")
    per_round: Counter[int] = Counter()
    rows: list[ToolCall] = []
    for trace_value in traces:
        trace = _require_mapping(trace_value, "tool trace entry")
        round_index = int(trace.get("round_index", 0))
        call_index = per_round[round_index]
        per_round[round_index] += 1
        arguments = _require_mapping(trace.get("arguments"), "tool arguments")
        result_text = trace.get("model_visible_result", trace.get("result"))
        if not isinstance(result_text, str):
            raise AuthorEvaluatorImportError("tool result must be text")
        try:
            structured = json.loads(result_text)
        except json.JSONDecodeError:
            structured = {}
        if not isinstance(structured, dict):
            structured = {"value": structured}
        rows.append(
            ToolCall(
                id=_stable_id(
                    "tool-call",
                    f"{response_arm_id}:{round_index}:{call_index}",
                ),
                arm_id=response_arm_id,
                round_index=round_index,
                call_index=call_index,
                tool_name=str(trace.get("name", "")),
                arguments_json=arguments,
                arguments_sha256=canonical_sha256({"arguments": arguments}),
                result_text=result_text,
                structured_content_json=structured,
                structured_content_sha256=canonical_sha256({"structured": structured}),
                result_sha256=hashlib.sha256(result_text.encode()).hexdigest(),
                latency_ms=int(trace.get("latency_ms", 0)),
                is_error=bool(trace.get("is_error")),
            )
        )
    return rows


def _existing_pool_summary(session: Session, candidate_sha256: str) -> dict[str, int]:
    season_id = _stable_id("season", candidate_sha256)
    battles = session.scalar(
        select(func.count()).select_from(Battle).where(Battle.season_id == season_id)
    )
    arms = session.scalar(
        select(func.count())
        .select_from(ResponseArm)
        .join(Battle, Battle.id == ResponseArm.battle_id)
        .where(Battle.season_id == season_id)
    )
    tools = session.scalar(
        select(func.count())
        .select_from(ToolCall)
        .join(ResponseArm, ResponseArm.id == ToolCall.arm_id)
        .join(Battle, Battle.id == ResponseArm.battle_id)
        .where(Battle.season_id == season_id)
    )
    return {
        "battles": int(battles or 0),
        "arms": int(arms or 0),
        "toolCalls": int(tools or 0),
    }


def _projection_contract(bundle: ImportBundle) -> dict[str, Any]:
    candidate_sha = bundle.candidate_sha256
    used_sources = [arm.source for pair in bundle.pairs for arm in pair.arms]
    contracts = [
        _require_mapping(source.get("contracts"), "source contracts") for source in used_sources
    ]
    release_ids = {str(contract.get("epicure_release_id")) for contract in contracts}
    bundle_hashes = {str(contract.get("epicure_bundle_sha256")) for contract in contracts}
    application_hashes = {str(contract.get("epicure_application_sha256")) for contract in contracts}
    tool_hashes = {str(contract.get("epicure_tool_catalog_sha256")) for contract in contracts}
    if not (
        len(release_ids) == len(bundle_hashes) == len(application_hashes) == len(tool_hashes) == 1
    ):
        raise AuthorEvaluatorImportError("source arms do not share one Epicure contract")
    protocol_bundle = {
        "schemaVersion": SCHEMA_VERSION,
        "candidatePackSha256": candidate_sha,
        "comparisonManifestSha256": bundle.comparison_manifest["artifact_sha256"],
        "modelManifestSha256": bundle.model_manifest["artifact_sha256"],
        "use": "author_evaluator_calibration_case_study",
        "official": False,
        "rankEligible": False,
        "syntheticArms": 0,
    }
    return {
        "used_sources": used_sources,
        "release_id": next(iter(release_ids)),
        "epicure_bundle_sha256": _require_sha256(
            next(iter(bundle_hashes)), "Epicure bundle digest"
        ),
        "epicure_application_sha256": _require_sha256(
            next(iter(application_hashes)), "Epicure application digest"
        ),
        "tool_schema_sha256": _require_sha256(
            next(iter(tool_hashes)), "Epicure tool catalog digest"
        ),
        "prompt_registry_sha256": canonical_sha256(
            [pair.item["prompt_sha256"] for pair in bundle.pairs]
        ),
        "analysis_plan_sha256": canonical_sha256(
            {
                "cohort": "author_evaluator",
                "independent_validation": False,
                "ranking": False,
            }
        ),
        "protocol_bundle": protocol_bundle,
        "protocol_sha256": canonical_sha256(protocol_bundle),
    }


def _validate_existing_projection(session: Session, bundle: ImportBundle) -> None:
    candidate_sha = bundle.candidate_sha256
    projection = _projection_contract(bundle)
    season_id = _stable_id("season", candidate_sha)
    season = session.get(Season, season_id)
    if season is None or (
        season.slug != f"{SEASON_SLUG_PREFIX}-{candidate_sha[:12]}"
        or season.name != "Season 0 author-evaluator calibration reserve"
        or season.status != "pilot"
        or season.official
        or season.manifest_sha256 != candidate_sha
        or season.prompt_registry_sha256 != projection["prompt_registry_sha256"]
        or season.tool_registry_sha256 != projection["tool_schema_sha256"]
        or season.epicure_release_id != projection["release_id"]
        or season.epicure_bundle_sha256 != projection["epicure_bundle_sha256"]
        or season.epicure_application_sha256 != projection["epicure_application_sha256"]
        or season.analysis_plan_sha256 != projection["analysis_plan_sha256"]
        or season.protocol_bundle_json != projection["protocol_bundle"]
        or season.protocol_bundle_sha256 != projection["protocol_sha256"]
        or season.budget_cap_micros != 0
        or season.budget_used_micros != 0
        or season.budget_reserved_micros != 0
        or season.frozen_at is not None
    ):
        raise AuthorEvaluatorImportError("existing season projection has drifted")

    expected_tasks = {
        _stable_id("task", f"{candidate_sha}:{pair.item['task_id']}"): pair for pair in bundle.pairs
    }
    stored_tasks = session.scalars(select(Task).where(Task.season_id == season_id)).all()
    if {task.id for task in stored_tasks} != set(expected_tasks):
        raise AuthorEvaluatorImportError("existing projection has unexpected task identities")
    source_class = _require_mapping(
        bundle.candidate.get("created_from"), "candidate creation evidence"
    ).get("source_class")
    for task in stored_tasks:
        item = expected_tasks[task.id].item
        expected_provenance = {
            "taskSha256": item["task_sha256"],
            "candidatePackSha256": candidate_sha,
            "calibrationItemId": item["calibration_item_id"],
            "sourceClass": source_class,
            "synthetic": False,
        }
        if (
            task.public_id != item["task_id"]
            or task.family != item["family"]
            or task.prompt != item["prompt"]
            or task.prompt_sha256 != item["prompt_sha256"]
            or task.revision != 1
            or task.split != "calibration"
            or task.review_status != "reviewed"
            or task.provenance_json != expected_provenance
        ):
            raise AuthorEvaluatorImportError(f"existing task projection has drifted: {task.id}")

    expected_battles = {
        _stable_id("battle", f"{candidate_sha}:{pair.item['calibration_item_id']}"): (
            ordinal,
            pair,
        )
        for ordinal, pair in enumerate(bundle.pairs)
    }
    stored_battles = session.scalars(select(Battle).where(Battle.season_id == season_id)).all()
    if {battle.id for battle in stored_battles} != set(expected_battles):
        raise AuthorEvaluatorImportError("existing projection has unexpected battle identities")
    for battle in stored_battles:
        ordinal, pair = expected_battles[battle.id]
        item = pair.item
        expected_arm_ids = {
            imported.side: _stable_id(
                "response-arm",
                f"{candidate_sha}:{imported.source['arm_id']}",
            )
            for imported in pair.arms
        }
        if (
            battle.run_class != "pilot"
            or battle.rank_eligible
            or battle.data_stratum != "development"
            or battle.task_id != _stable_id("task", f"{candidate_sha}:{item['task_id']}")
            or battle.task_revision != 1
            or battle.controlled_run_id is not None
            or battle.manifest_sha256 != candidate_sha
            or battle.protocol_bundle_sha256 != projection["protocol_sha256"]
            or battle.scheduler_version != SCHEDULER_VERSION
            or battle.assignment_seed
            != hashlib.sha256(str(item["calibration_item_id"]).encode()).hexdigest()
            or battle.track_assignment_probability != "archived-deterministic"
            or battle.model_assignment_probability != "archived-deterministic"
            or battle.side_assignment_probability != "deterministic-blinded-randomization"
            or battle.track != "epicure_uplift"
            or battle.category != item["family"]
            or battle.prompt != item["prompt"]
            or battle.prompt_sha256 != item["prompt_sha256"]
            or battle.client_nonce_sha256
            != hashlib.sha256(f"{candidate_sha}:{ordinal}".encode()).hexdigest()
            or battle.prompt_redacted
            or battle.research_consent
            or battle.release_review_status != "not_requested"
            or battle.release_reviewed_at is not None
            or battle.requester_pseudonym
            != hashlib.sha256(f"author-evaluator-pool:{candidate_sha}".encode()).hexdigest()
            or battle.status != "complete"
            or battle.left_arm_id != expected_arm_ids["left"]
            or battle.right_arm_id != expected_arm_ids["right"]
            or battle.reserved_cost_micros != 0
            or battle.provider_reservations_json != {}
        ):
            raise AuthorEvaluatorImportError(f"existing battle projection has drifted: {battle.id}")

    expected_arms: dict[str, tuple[ImportArm, ImportPair, str]] = {}
    for pair in bundle.pairs:
        battle_id = _stable_id(
            "battle",
            f"{candidate_sha}:{pair.item['calibration_item_id']}",
        )
        for imported in pair.arms:
            arm_id = _stable_id(
                "response-arm",
                f"{candidate_sha}:{imported.source['arm_id']}",
            )
            expected_arms[arm_id] = (imported, pair, battle_id)
    stored_arms = session.scalars(
        select(ResponseArm)
        .join(Battle, Battle.id == ResponseArm.battle_id)
        .where(Battle.season_id == season_id)
    ).all()
    if {arm.id for arm in stored_arms} != set(expected_arms):
        raise AuthorEvaluatorImportError(
            "existing projection has unexpected response-arm identities"
        )
    for arm in stored_arms:
        imported, pair, battle_id = expected_arms[arm.id]
        source = imported.source
        result = _require_mapping(source.get("result"), "source result")
        model = _require_mapping(source.get("model"), "source model")
        contracts = _require_mapping(source.get("contracts"), "source contracts")
        usage = _require_mapping(result.get("usage"), "source usage")
        expected_output = _source_output(source)
        generation_ids = result.get("generation_ids", [])
        if not isinstance(generation_ids, list):
            generation_ids = []
        recorded_generation_ids = [
            str(value) for value in generation_ids if isinstance(value, str)
        ] or [f"archive:{source['arm_id']}"]
        expected_attestation = {
            "sourceArtifactSha256": source["artifact_sha256"],
            "sourceArmId": source["arm_id"],
            "condition": source["condition"],
            "realEpicureCalls": result.get("real_epicure_calls", 0),
            "completeToolTrace": True,
            "synthetic": False,
        }
        if (
            arm.battle_id != battle_id
            or arm.side != imported.side
            or arm.condition != source["condition"]
            or arm.model_id != model["canonical_model_id"]
            or arm.execution_backend != model["provider"]
            or arm.route_revision_id is not None
            or arm.endpoint_descriptor_sha256 is not None
            or arm.provider_slug != _provider_slug(source)
            or arm.actual_provider_slug
            != str(result.get("actual_provider_name") or model.get("provider"))
            or arm.actual_model_id != _actual_model_id(source)
            or arm.generation_id != recorded_generation_ids[0]
            or arm.provider_generation_ids_json != recorded_generation_ids
            or arm.status != "complete"
            or arm.answer_markdown != result["answer_markdown"]
            or str(arm.finish_reason or "").strip().lower() not in ACCEPTED_FINAL_FINISH_REASONS
            or str(arm.finish_reason or "").strip().lower()
            != str(result.get("finish_reason") or "").strip().lower()
            or arm.answer_markdown_sha256 != imported.answer_sha256
            or arm.output_json != expected_output
            or arm.output_json_sha256 != canonical_sha256(expected_output)
            or arm.prompt_sha256 != pair.item["prompt_sha256"]
            or arm.system_prompt_sha256 != contracts["system_prompt_sha256"]
            or arm.schema_sha256 != contracts["execution_contract_sha256"]
            or arm.tool_schema_sha256 != projection["tool_schema_sha256"]
            or arm.decoding_json != {"source": "frozen season0 collector contract"}
            or arm.observed_decoding_json != {"source": "immutable paid real-output artifact"}
            or arm.protocol_bundle_sha256 != projection["protocol_sha256"]
            or arm.epicure_release_id != projection["release_id"]
            or arm.epicure_bundle_sha256 != projection["epicure_bundle_sha256"]
            or arm.epicure_application_sha256 != projection["epicure_application_sha256"]
            or arm.epicure_attestation_json != expected_attestation
            or arm.epicure_attestation_sha256 != canonical_sha256(expected_attestation)
            or arm.prompt_tokens != int(usage.get("input_tokens", 0))
            or arm.completion_tokens != int(usage.get("output_tokens", 0))
            or arm.reasoning_tokens != int(usage.get("reasoning_tokens", 0))
            or arm.cost_micros != 0
            or not arm.cost_reconciled
            or arm.cost_accounting_basis != "archived_review_projection_zero_incremental_cost"
            or arm.billing_reconciliation_status != "not_applicable_archived_review_projection"
            or arm.backend_response_schema_sha256 != contracts["execution_contract_sha256"]
            or arm.backend_tool_schema_sha256 != projection["tool_schema_sha256"]
            or arm.latency_ms != int(result.get("wall_clock_latency_ms", 0))
            or arm.retries != 0
            or arm.error_code is not None
            or arm.error_detail is not None
            or not _same_datetime(arm.created_at, _parse_datetime(source.get("started_at")))
            or not _same_datetime(arm.completed_at, _parse_datetime(source.get("completed_at")))
        ):
            raise AuthorEvaluatorImportError(
                f"existing response-arm projection has drifted: {arm.id}"
            )

    expected_tools: dict[tuple[str, int, int], ToolCall] = {}
    for arm_id, (imported, _pair, _battle_id) in expected_arms.items():
        for row in _tool_rows(arm_id, imported.source):
            expected_tools[(arm_id, row.round_index, row.call_index)] = row
    stored_tools = session.scalars(
        select(ToolCall)
        .join(ResponseArm, ResponseArm.id == ToolCall.arm_id)
        .join(Battle, Battle.id == ResponseArm.battle_id)
        .where(Battle.season_id == season_id)
    ).all()
    observed_keys = {(row.arm_id, row.round_index, row.call_index) for row in stored_tools}
    if observed_keys != set(expected_tools):
        raise AuthorEvaluatorImportError("existing tool-call projection has drifted")
    for row in stored_tools:
        expected = expected_tools[(row.arm_id, row.round_index, row.call_index)]
        if (
            row.id != expected.id
            or row.tool_call_id is not None
            or row.tool_name != expected.tool_name
            or row.arguments_json != expected.arguments_json
            or row.arguments_sha256 != expected.arguments_sha256
            or row.result_text != expected.result_text
            or row.structured_content_json != expected.structured_content_json
            or row.structured_content_sha256 != expected.structured_content_sha256
            or row.result_sha256 != expected.result_sha256
            or row.latency_ms != expected.latency_ms
            or row.is_error != expected.is_error
        ):
            raise AuthorEvaluatorImportError(f"existing tool-call projection has drifted: {row.id}")

    import_events = session.scalars(
        select(RunEvent).where(
            RunEvent.entity_type == "author_evaluator_pool",
            RunEvent.entity_id == candidate_sha,
            RunEvent.event_type == "author_evaluator_pool_imported",
        )
    ).all()
    expected_event_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_pack_sha256": candidate_sha,
        "comparison_manifest_sha256": bundle.comparison_manifest["artifact_sha256"],
        "model_manifest_sha256": bundle.model_manifest["artifact_sha256"],
        "battle_count": 32,
        "source_arm_count": 64,
        "tool_call_count": len(expected_tools),
        "synthetic_arm_count": 0,
        "rank_eligible_battle_count": 0,
        "data_stratum": "development",
        "run_class": "pilot",
        "claim_boundary": (
            "Single-rater blinded author-evaluator calibration case study; "
            "not an official ranking and not independent expert validation."
        ),
    }
    if (
        len(import_events) != 1
        or import_events[0].id != _stable_id("run-event", candidate_sha)
        or import_events[0].payload_json != expected_event_payload
    ):
        raise AuthorEvaluatorImportError("existing import-event projection has drifted")


def import_bundle(session: Session, bundle: ImportBundle) -> dict[str, Any]:
    candidate_sha = bundle.candidate_sha256
    _lock_candidate_import(session, candidate_sha)
    existing = _existing_pool_summary(session, candidate_sha)
    expected_tool_calls = sum(
        len(arm.source["result"].get("tool_trace", []))
        for pair in bundle.pairs
        for arm in pair.arms
    )
    if any(existing.values()):
        expected = {
            "battles": 32,
            "arms": 64,
            "toolCalls": expected_tool_calls,
        }
        if existing != expected:
            raise AuthorEvaluatorImportError(
                f"existing author review pool is partial or inconsistent: {existing}"
            )
        event = session.scalar(
            select(RunEvent).where(
                RunEvent.entity_type == "author_evaluator_pool",
                RunEvent.entity_id == candidate_sha,
                RunEvent.event_type == "author_evaluator_pool_imported",
            )
        )
        if event is None:
            raise AuthorEvaluatorImportError("existing pool has no immutable import event")
        _validate_existing_projection(session, bundle)
        return {
            "schemaVersion": SCHEMA_VERSION,
            "candidatePackSha256": candidate_sha,
            **existing,
            "syntheticArms": 0,
            "rankEligibleBattles": 0,
            "idempotent": True,
            "eventId": event.id,
        }

    manifest_models = {
        str(model["season_model_id"]): model
        for model in bundle.model_manifest["models"]
        if isinstance(model, dict)
    }
    projection = _projection_contract(bundle)
    used_sources = projection["used_sources"]
    release_id = projection["release_id"]
    epicure_bundle_sha = projection["epicure_bundle_sha256"]
    epicure_application_sha = projection["epicure_application_sha256"]
    tool_schema_sha = projection["tool_schema_sha256"]
    protocol_bundle = projection["protocol_bundle"]
    protocol_sha = projection["protocol_sha256"]
    season = Season(
        id=_stable_id("season", candidate_sha),
        slug=f"{SEASON_SLUG_PREFIX}-{candidate_sha[:12]}",
        name="Season 0 author-evaluator calibration reserve",
        status="pilot",
        official=False,
        manifest_sha256=candidate_sha,
        prompt_registry_sha256=projection["prompt_registry_sha256"],
        tool_registry_sha256=tool_schema_sha,
        epicure_release_id=release_id,
        epicure_bundle_sha256=epicure_bundle_sha,
        epicure_application_sha256=epicure_application_sha,
        analysis_plan_sha256=projection["analysis_plan_sha256"],
        protocol_bundle_json=protocol_bundle,
        protocol_bundle_sha256=protocol_sha,
        budget_cap_micros=0,
        budget_used_micros=0,
        budget_reserved_micros=0,
    )
    session.add(season)

    catalog_ids = {str(source["model"]["canonical_model_id"]) for source in used_sources}
    existing_catalog = set(
        session.scalars(
            select(CatalogModel.model_id).where(CatalogModel.model_id.in_(catalog_ids))
        ).all()
    )
    for source in used_sources:
        model = _require_mapping(source.get("model"), "source model")
        canonical_id = str(model["canonical_model_id"])
        if canonical_id in existing_catalog:
            continue
        manifest_model = manifest_models[str(model["season_model_id"])]
        slot_role = str(manifest_model.get("slot_role", "unknown"))
        session.add(
            CatalogModel(
                model_id=canonical_id,
                canonical_slug=str(manifest_model.get("canonical_slug") or canonical_id),
                name=str(model.get("display_name") or canonical_id),
                family=canonical_id.split("/", 1)[0].split(".", 1)[0],
                catalog_source=str(model.get("provider") or "archive"),
                open_weight=slot_role == "open_weight",
                open_weight_evidence_json={
                    "slotRole": slot_role,
                    "modelManifestSha256": bundle.model_manifest["artifact_sha256"],
                },
                status="smoke_passed",
                supports_tools=True,
                supports_structured_outputs=True,
                endpoint_json=_require_mapping(manifest_model.get("endpoint"), "manifest endpoint"),
            )
        )
        existing_catalog.add(canonical_id)
    session.flush()

    tool_rows: list[ToolCall] = []
    for ordinal, pair in enumerate(bundle.pairs):
        item = pair.item
        task_id = _stable_id("task", f"{candidate_sha}:{item['task_id']}")
        task = Task(
            id=task_id,
            public_id=str(item["task_id"]),
            season_id=season.id,
            family=str(item["family"]),
            prompt=str(item["prompt"]),
            prompt_sha256=str(item["prompt_sha256"]),
            revision=1,
            split="calibration",
            review_status="reviewed",
            provenance_json={
                "taskSha256": item["task_sha256"],
                "candidatePackSha256": candidate_sha,
                "calibrationItemId": item["calibration_item_id"],
                "sourceClass": bundle.candidate["created_from"]["source_class"],
                "synthetic": False,
            },
        )
        session.add(task)
        session.flush()
        battle_id = _stable_id(
            "battle",
            f"{candidate_sha}:{item['calibration_item_id']}",
        )
        source_completed = max(_parse_datetime(arm.source.get("completed_at")) for arm in pair.arms)
        battle = Battle(
            id=battle_id,
            season_id=season.id,
            run_class="pilot",
            rank_eligible=False,
            data_stratum="development",
            task_id=task_id,
            task_revision=1,
            controlled_run_id=None,
            manifest_sha256=candidate_sha,
            protocol_bundle_sha256=protocol_sha,
            scheduler_version=SCHEDULER_VERSION,
            assignment_seed=hashlib.sha256(str(item["calibration_item_id"]).encode()).hexdigest(),
            track_assignment_probability="archived-deterministic",
            model_assignment_probability="archived-deterministic",
            side_assignment_probability="deterministic-blinded-randomization",
            track="epicure_uplift",
            category=str(item["family"]),
            prompt=str(item["prompt"]),
            prompt_sha256=str(item["prompt_sha256"]),
            client_nonce_sha256=hashlib.sha256(f"{candidate_sha}:{ordinal}".encode()).hexdigest(),
            prompt_redacted=False,
            research_consent=False,
            retention_basis="development_research",
            release_review_status="not_requested",
            requester_pseudonym=hashlib.sha256(
                f"author-evaluator-pool:{candidate_sha}".encode()
            ).hexdigest(),
            status="queued",
            reserved_cost_micros=0,
            provider_reservations_json={},
            created_at=min(_parse_datetime(arm.source.get("started_at")) for arm in pair.arms),
            completed_at=None,
            retention_until=datetime.now(UTC) + timedelta(days=3650),
        )
        session.add(battle)
        session.flush()
        arm_ids: dict[str, str] = {}
        for imported in pair.arms:
            source = imported.source
            source_result = _require_mapping(source.get("result"), "source result")
            source_model = _require_mapping(source.get("model"), "source model")
            source_contract = _require_mapping(source.get("contracts"), "source contracts")
            usage = _require_mapping(source_result.get("usage"), "source usage")
            output = _source_output(source)
            epicure_attestation = {
                "sourceArtifactSha256": source["artifact_sha256"],
                "sourceArmId": source["arm_id"],
                "condition": source["condition"],
                "realEpicureCalls": source_result.get("real_epicure_calls", 0),
                "completeToolTrace": True,
                "synthetic": False,
            }
            arm_id = _stable_id(
                "response-arm",
                f"{candidate_sha}:{source['arm_id']}",
            )
            generation_ids = source_result.get("generation_ids", [])
            if not isinstance(generation_ids, list):
                generation_ids = []
            archived_generation_id = f"archive:{source['arm_id']}"
            recorded_generation_ids = [
                str(value) for value in generation_ids if isinstance(value, str)
            ] or [archived_generation_id]
            response_arm = ResponseArm(
                id=arm_id,
                battle_id=battle_id,
                side=imported.side,
                condition=str(source["condition"]),
                model_id=str(source_model["canonical_model_id"]),
                execution_backend=str(source_model["provider"]),
                provider_slug=_provider_slug(source),
                status="queued",
                prompt_sha256=str(item["prompt_sha256"]),
                system_prompt_sha256=_require_sha256(
                    source_contract.get("system_prompt_sha256"),
                    "system prompt digest",
                ),
                schema_sha256=_require_sha256(
                    source_contract.get("execution_contract_sha256"),
                    "execution contract digest",
                ),
                tool_schema_sha256=tool_schema_sha,
                decoding_json={"source": "frozen season0 collector contract"},
                protocol_bundle_sha256=protocol_sha,
                epicure_release_id=release_id,
                epicure_bundle_sha256=epicure_bundle_sha,
                epicure_application_sha256=epicure_application_sha,
                created_at=_parse_datetime(source.get("started_at")),
            )
            session.add(response_arm)
            session.flush()
            response_arm.actual_provider_slug = str(
                source_result.get("actual_provider_name") or source_model.get("provider")
            )
            response_arm.actual_model_id = _actual_model_id(source)
            response_arm.generation_id = recorded_generation_ids[0]
            response_arm.provider_generation_ids_json = recorded_generation_ids
            response_arm.status = "complete"
            response_arm.answer_markdown = str(source_result["answer_markdown"])
            response_arm.answer_markdown_sha256 = imported.answer_sha256
            response_arm.output_json = output
            response_arm.output_json_sha256 = canonical_sha256(output)
            response_arm.observed_decoding_json = {"source": "immutable paid real-output artifact"}
            response_arm.epicure_attestation_json = epicure_attestation
            response_arm.epicure_attestation_sha256 = canonical_sha256(epicure_attestation)
            response_arm.prompt_tokens = int(usage.get("input_tokens", 0))
            response_arm.completion_tokens = int(usage.get("output_tokens", 0))
            response_arm.reasoning_tokens = int(usage.get("reasoning_tokens", 0))
            response_arm.cost_micros = 0
            response_arm.cost_reconciled = True
            response_arm.cost_accounting_basis = "archived_review_projection_zero_incremental_cost"
            response_arm.billing_reconciliation_status = "not_applicable_archived_review_projection"
            response_arm.backend_response_schema_sha256 = _require_sha256(
                source_contract.get("execution_contract_sha256"),
                "backend response contract digest",
            )
            response_arm.backend_tool_schema_sha256 = tool_schema_sha
            response_arm.latency_ms = int(source_result.get("wall_clock_latency_ms", 0))
            response_arm.retries = 0
            response_arm.finish_reason = str(source_result.get("finish_reason") or "unknown")
            response_arm.completed_at = _parse_datetime(source.get("completed_at"))
            session.add(response_arm)
            arm_ids[imported.side] = arm_id
            tool_rows.extend(_tool_rows(arm_id, source))
        session.flush()
        battle.left_arm_id = arm_ids["left"]
        battle.right_arm_id = arm_ids["right"]
        battle.status = "complete"
        battle.completed_at = source_completed
        session.add(battle)
        session.flush()
    session.add_all(tool_rows)
    event = RunEvent(
        id=_stable_id("run-event", candidate_sha),
        entity_type="author_evaluator_pool",
        entity_id=candidate_sha,
        event_type="author_evaluator_pool_imported",
        payload_json={
            "schema_version": SCHEMA_VERSION,
            "candidate_pack_sha256": candidate_sha,
            "comparison_manifest_sha256": bundle.comparison_manifest["artifact_sha256"],
            "model_manifest_sha256": bundle.model_manifest["artifact_sha256"],
            "battle_count": 32,
            "source_arm_count": 64,
            "tool_call_count": len(tool_rows),
            "synthetic_arm_count": 0,
            "rank_eligible_battle_count": 0,
            "data_stratum": "development",
            "run_class": "pilot",
            "claim_boundary": (
                "Single-rater blinded author-evaluator calibration case study; "
                "not an official ranking and not independent expert validation."
            ),
        },
    )
    session.add(event)
    session.flush()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "candidatePackSha256": candidate_sha,
        "battles": 32,
        "arms": 64,
        "toolCalls": len(tool_rows),
        "syntheticArms": 0,
        "rankEligibleBattles": 0,
        "idempotent": False,
        "eventId": event.id,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import the verified real-output author-evaluator review pool."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    bundle = load_bundle(
        candidate_path=arguments.candidate,
        comparison_manifest_path=arguments.comparisons,
        model_manifest_path=arguments.models,
        arm_directory=arguments.arms,
    )
    with session_scope() as session:
        result = import_bundle(session, bundle)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run()
