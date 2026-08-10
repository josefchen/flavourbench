"""Append-only rotation of a contained anonymous-review response pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .anonymous_reviewer_control import (
    anonymous_pool_reconsented,
    repaired_activation_sha256,
    rotation_activation_sha256,
)
from .database import SessionLocal
from .expert_calibration import (
    BLINDING_LEAK_PATTERN_SHA256,
    MANUAL_RESPONSE_QUARANTINE,
    RESPONSE_CONTENT_REVIEW_SHA256,
    TASK_QUALITY_QUARANTINE,
    TASK_QUALITY_REVIEW_SHA256,
    TASK_SCOPE_QUARANTINE,
    TASK_SCOPE_REVIEW_SHA256,
)
from .expert_review import (
    author_evaluator_workload_cell_targets,
    canonical_sha256,
    isolated_uplift_workload_cell_targets,
)
from .human_review_containment import (
    HumanReviewContainmentError,
    _latest_anonymous_reviewer,
    _latest_review_session_for_pool,
)
from .models import ExpertReviewer, RunEvent

LEGACY_CANDIDATE_SCHEMA_VERSION = "flavourbench-expert-calibration-candidate-v11"
REQUIRED_FRONTIER_SCHEMA_VERSION = "flavourbench-required-frontier-review-pool-v2"
SUPPORTED_CANDIDATE_SCHEMA_VERSIONS = frozenset(
    {LEGACY_CANDIDATE_SCHEMA_VERSION, REQUIRED_FRONTIER_SCHEMA_VERSION}
)
CANDIDATE_SCHEMA_VERSION = LEGACY_CANDIDATE_SCHEMA_VERSION
REQUIRED_FRONTIER_SOURCE_CLASS = "real_required_epicure_development_pilot"
REQUIRED_FRONTIER_SUMMARY_SHA256 = (
    "f1a38e30042b9614fa82f5c38b43b98c7c9a18916c4541f21bb12d3bcce8ba70"
)
REQUIRED_FRONTIER_FAMILY_COUNTS = {
    "composition": 9,
    "cookability": 13,
    "evidence": 10,
    "substitution": 11,
}
REQUIRED_FRONTIER_MODEL_IDS = frozenset(
    {
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "deepseek/deepseek-v4-flash-0731",
        "deepseek/deepseek-v4-pro",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.6-flash",
        "minimax/minimax-m3",
        "mistralai/mistral-medium-3-5",
        "moonshotai/kimi-k3",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "openai/gpt-5.6-sol-pro",
        "x-ai/grok-4.5",
        "z-ai/glm-5.2",
    }
)
ROTATION_EVENT_TYPE = "expert_anonymous_external_pool_superseded"
ROTATION_CORRECTION_EVENT_TYPE = "expert_anonymous_external_pool_supersession_corrected"
ROTATION_CORRECTION_SCHEMA_VERSION = (
    "flavourbench-anonymous-external-pool-supersession-correction-v1"
)
ROTATION_NAMESPACE = uuid.UUID("d4784775-01ba-42ec-8c80-1bb99c37672f")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_candidate(path: Path) -> dict[str, Any]:
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanReviewContainmentError(f"cannot read replacement candidate: {path}") from exc
    if not isinstance(candidate, dict):
        raise HumanReviewContainmentError("replacement candidate must be an object")
    claimed = candidate.get("artifact_sha256")
    payload = {key: value for key, value in candidate.items() if key != "artifact_sha256"}
    actual = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if not isinstance(claimed, str) or claimed != actual:
        raise HumanReviewContainmentError("replacement candidate digest mismatch")
    if candidate.get("schema_version") not in SUPPORTED_CANDIDATE_SCHEMA_VERSIONS:
        raise HumanReviewContainmentError("replacement candidate schema is unsupported")
    return candidate


def _legacy_candidate_record(candidate: dict[str, Any], path: Path) -> dict[str, Any]:
    observed = candidate.get("observed")
    selection = candidate.get("selection_policy")
    use_policy = candidate.get("use_policy")
    blinding = candidate.get("blinding")
    created_from = candidate.get("created_from")
    if not all(
        isinstance(value, dict)
        for value in (observed, selection, use_policy, blinding, created_from)
    ):
        raise HumanReviewContainmentError("replacement candidate evidence is incomplete")
    assert isinstance(observed, dict)
    assert isinstance(selection, dict)
    assert isinstance(use_policy, dict)
    assert isinstance(blinding, dict)
    assert isinstance(created_from, dict)
    quarantined = set(selection.get("specialist_scope_quarantine_task_ids", []))
    quality_quarantined = set(selection.get("task_quality_quarantine_task_ids", []))
    response_quarantined = set(selection.get("manual_response_quarantine_answer_sha256s", []))
    if (
        observed.get("candidate_pairs") != 32
        or observed.get("source_arms") != 64
        or observed.get("synthetic_arms") != 0
        or selection.get("normal_final_completion_required") is not True
        or selection.get("specialist_scope_quarantine_required") is not True
        or quarantined != set(TASK_SCOPE_QUARANTINE)
        or selection.get("specialist_scope_review_sha256") != TASK_SCOPE_REVIEW_SHA256
        or selection.get("task_quality_quarantine_required") is not True
        or quality_quarantined != set(TASK_QUALITY_QUARANTINE)
        or selection.get("task_quality_review_sha256") != TASK_QUALITY_REVIEW_SHA256
        or selection.get("manual_response_content_quarantine_required") is not True
        or response_quarantined != set(MANUAL_RESPONSE_QUARANTINE)
        or selection.get("response_content_review_sha256") != RESPONSE_CONTENT_REVIEW_SHA256
        or selection.get("blinding_leak_pattern_sha256") != BLINDING_LEAK_PATTERN_SHA256
        or use_policy.get("rank_eligible") is not False
        or use_policy.get("benchmark_result_use") != "prohibited"
        or created_from.get("source_class") != "paid_real_legacy_pilot_quarantined_from_season1"
    ):
        raise HumanReviewContainmentError("replacement candidate policy is inadmissible")
    family_counts = observed.get("candidate_pairs_by_family")
    if not isinstance(family_counts, dict) or family_counts != {
        "composition": 8,
        "cookability": 8,
        "evidence": 8,
        "substitution": 8,
    }:
        raise HumanReviewContainmentError("replacement candidate is not family balanced")
    identity_commitment = blinding.get("identity_commitment_sha256")
    if not isinstance(identity_commitment, str) or len(identity_commitment) != 64:
        raise HumanReviewContainmentError("replacement identity commitment is unavailable")
    return {
        "candidate_pack_sha256": candidate["artifact_sha256"],
        "identity_commitment_sha256": identity_commitment,
        "source_class": created_from["source_class"],
        "candidate_pack_reference": (
            f"artifacts/expert-calibration/{path.parent.name}/{path.name}"
        ),
        "candidate_pairs": observed["candidate_pairs"],
        "candidate_pairs_by_family": family_counts,
        "source_arms": observed["source_arms"],
        "real_provider_calls": observed["real_provider_calls"],
        "real_epicure_calls": observed["real_epicure_calls"],
        "successful_real_epicure_calls": observed["successful_real_epicure_calls"],
        "synthetic_arms": observed["synthetic_arms"],
        "rank_eligible": use_policy["rank_eligible"],
        "status": candidate["status"],
    }


def _required_frontier_candidate_record(
    candidate: dict[str, Any], path: Path
) -> dict[str, Any]:
    observed = candidate.get("observed")
    selection = candidate.get("selection_policy")
    claim_boundary = candidate.get("claim_boundary")
    source = candidate.get("source")
    epicure = candidate.get("epicure")
    model_contracts = candidate.get("model_contracts")
    model_order = candidate.get("model_order")
    if not all(
        isinstance(value, dict)
        for value in (observed, selection, claim_boundary, source, epicure, model_contracts)
    ) or not isinstance(model_order, list):
        raise HumanReviewContainmentError(
            "required-frontier replacement evidence is incomplete"
        )
    assert isinstance(observed, dict)
    assert isinstance(selection, dict)
    assert isinstance(claim_boundary, dict)
    assert isinstance(source, dict)
    assert isinstance(epicure, dict)
    assert isinstance(model_contracts, dict)

    family_counts = observed.get("candidate_pairs_by_family")
    model_ids = set(map(str, model_order))
    contract_ids = set(map(str, model_contracts))
    identity_commitment = candidate.get("identity_commitment_sha256")
    if (
        observed.get("candidate_pairs") != 43
        or observed.get("source_arms") != 86
        or observed.get("synthetic_arms") != 0
        or observed.get("real_provider_calls") != 277
        or observed.get("real_epicure_calls") != 182
        or observed.get("successful_real_epicure_calls") != 86
        or observed.get("quality_judgments") != 0
        or family_counts != REQUIRED_FRONTIER_FAMILY_COUNTS
        or len(model_order) != len(REQUIRED_FRONTIER_MODEL_IDS)
        or model_ids != REQUIRED_FRONTIER_MODEL_IDS
        or contract_ids != REQUIRED_FRONTIER_MODEL_IDS
        or selection.get("paired_same_model_same_task") is not True
        or selection.get("complete_pairs_only") is not True
        or selection.get("required_successful_epicure_treatment") is not True
        or selection.get("raw_answers_edited") is not False
        or selection.get("deterministic_family_stratified_side_assignment") is not True
        or selection.get("globally_balanced_side_assignment") is not True
        or claim_boundary.get("official") is not False
        or claim_boundary.get("rank_eligible") is not False
        or claim_boundary.get("research_result") is not False
        or claim_boundary.get("quality_judgments") != 0
        or claim_boundary.get("prohibited_use")
        != "quality leaderboard before real judgments are collected"
        or source.get("source_run_class") != REQUIRED_FRONTIER_SOURCE_CLASS
        or source.get("summary_content_address") != REQUIRED_FRONTIER_SUMMARY_SHA256
        or epicure.get("lineage_status") != "unmatched_exploratory_runtime"
    ):
        raise HumanReviewContainmentError(
            "required-frontier replacement policy is inadmissible"
        )
    if not isinstance(identity_commitment, str) or not SHA256_PATTERN.fullmatch(
        identity_commitment
    ):
        raise HumanReviewContainmentError(
            "required-frontier identity commitment is unavailable"
        )
    for model_id, contract in model_contracts.items():
        if not isinstance(contract, dict):
            raise HumanReviewContainmentError("required-frontier model contract is invalid")
        canonical_slug = contract.get("canonical_model_slug")
        endpoint_sha256 = contract.get("endpoint_execution_sha256")
        source_response_sha256 = contract.get("source_response_artifact_sha256")
        if (
            not isinstance(canonical_slug, str)
            or not canonical_slug
            or not isinstance(endpoint_sha256, str)
            or not SHA256_PATTERN.fullmatch(endpoint_sha256)
            or (
                source_response_sha256 is not None
                and (
                    not isinstance(source_response_sha256, str)
                    or not SHA256_PATTERN.fullmatch(source_response_sha256)
                )
            )
        ):
            raise HumanReviewContainmentError(
                f"required-frontier model contract is incomplete: {model_id}"
            )
    return {
        "candidate_pack_sha256": candidate["artifact_sha256"],
        "identity_commitment_sha256": identity_commitment,
        "source_class": REQUIRED_FRONTIER_SOURCE_CLASS,
        "candidate_pack_reference": (
            "artifacts/season1/current-quality-run/pilot-v24-required-epicure/"
            f"review-pool/{path.name}"
        ),
        "candidate_pairs": observed["candidate_pairs"],
        "candidate_pairs_by_family": dict(family_counts),
        "source_arms": observed["source_arms"],
        "real_provider_calls": observed["real_provider_calls"],
        "real_epicure_calls": observed["real_epicure_calls"],
        "successful_real_epicure_calls": observed[
            "successful_real_epicure_calls"
        ],
        "synthetic_arms": observed["synthetic_arms"],
        "rank_eligible": claim_boundary["rank_eligible"],
        "status": "candidate_pending_blinded_development_review",
    }


def _candidate_record(candidate: dict[str, Any], path: Path) -> dict[str, Any]:
    schema_version = candidate.get("schema_version")
    if schema_version == LEGACY_CANDIDATE_SCHEMA_VERSION:
        return _legacy_candidate_record(candidate, path)
    if schema_version == REQUIRED_FRONTIER_SCHEMA_VERSION:
        return _required_frontier_candidate_record(candidate, path)
    raise HumanReviewContainmentError("replacement candidate schema is unsupported")


def _replacement_pool_imported(
    session: Session,
    *,
    replacement: dict[str, Any],
) -> bool:
    replacement_sha256 = str(replacement["candidate_pack_sha256"])
    if replacement["source_class"] == REQUIRED_FRONTIER_SOURCE_CLASS:
        imported = session.scalar(
            select(RunEvent).where(
                RunEvent.entity_type == "current_frontier_review_pool",
                RunEvent.entity_id == replacement_sha256,
                RunEvent.event_type == "current_frontier_review_pool_imported",
            )
        )
        if imported is None:
            return False
        expected_counts = {
            "tasks": 4,
            "battles": replacement["candidate_pairs"],
            "arms": replacement["source_arms"],
            "toolCalls": replacement["real_epicure_calls"],
        }
        return bool(
            imported.payload_json.get("review_pool_sha256") == replacement_sha256
            and imported.payload_json.get("counts") == expected_counts
            and imported.payload_json.get("synthetic_arm_count") == 0
            and imported.payload_json.get("rank_eligible_battle_count") == 0
        )
    imported = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "author_evaluator_pool",
            RunEvent.entity_id == replacement_sha256,
            RunEvent.event_type == "author_evaluator_pool_imported",
        )
    )
    return imported is not None


def _stable_event_id(kind: str, material: str) -> str:
    return str(uuid.uuid5(ROTATION_NAMESPACE, f"{kind}:{material}"))


def _restricted_session_for_pool(
    session: Session,
    *,
    reviewer_id: str,
    pool_sha256: str,
) -> tuple[RunEvent | None, str, str]:
    review_session = _latest_review_session_for_pool(
        session,
        reviewer_id,
        pool_sha256,
    )
    if review_session is None:
        return None, "none_for_prior_pool", "no_review_session_for_prior_pool"
    restriction = session.scalar(
        select(RunEvent).where(
            RunEvent.entity_type == "expert_review_session",
            RunEvent.entity_id == review_session.entity_id,
            RunEvent.event_type == "expert_review_batch_restricted",
        )
    )
    if restriction is None or restriction.payload_json.get("research_use") is not False:
        raise HumanReviewContainmentError(
            "the review session bound to the prior pool is not fail-closed"
        )
    return review_session, "matched_prior_pool", "restricted_operational_qa"


def _rotation_event_for_replacement(
    session: Session,
    *,
    reviewer_id: str,
    replacement_sha256: str,
) -> RunEvent | None:
    return session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_reviewer",
            RunEvent.entity_id == reviewer_id,
            RunEvent.event_type == ROTATION_EVENT_TYPE,
            RunEvent.payload_json["replacement_pool_sha256"].as_string() == replacement_sha256,
        )
        .order_by(RunEvent.created_at.desc(), RunEvent.id.desc())
    )


def _repair_existing_rotation(
    session: Session,
    *,
    reviewer: ExpertReviewer,
    rotation_event: RunEvent,
) -> tuple[RunEvent | None, bool]:
    """Append, never mutate, a correction for a legacy false session binding."""

    payload = rotation_event.payload_json
    prior_sha256 = str(payload.get("prior_pool_sha256") or "")
    replacement_sha256 = str(payload.get("replacement_pool_sha256") or "")
    if not (
        SHA256_PATTERN.fullmatch(prior_sha256) and SHA256_PATTERN.fullmatch(replacement_sha256)
    ):
        raise HumanReviewContainmentError("existing rotation has invalid pool evidence")
    matched_session, binding, batch_status = _restricted_session_for_pool(
        session,
        reviewer_id=reviewer.id,
        pool_sha256=prior_sha256,
    )
    original_session_id = payload.get("prior_review_session_id")
    corrected_session_id = matched_session.entity_id if matched_session is not None else None
    existing_correction = session.scalar(
        select(RunEvent)
        .where(
            RunEvent.entity_type == "expert_reviewer",
            RunEvent.entity_id == reviewer.id,
            RunEvent.event_type == ROTATION_CORRECTION_EVENT_TYPE,
            RunEvent.payload_json["supersedes_rotation_event_id"].as_string() == rotation_event.id,
        )
        .order_by(RunEvent.created_at.desc(), RunEvent.id.desc())
    )
    if existing_correction is not None:
        correction_payload = existing_correction.payload_json
        evidence_payload = {
            key: value for key, value in correction_payload.items() if key != "evidence_sha256"
        }
        activation_sha256 = correction_payload.get("replacement_pool_activation_sha256")
        if not (
            correction_payload.get("prior_pool_sha256") == prior_sha256
            and correction_payload.get("replacement_pool_sha256") == replacement_sha256
            and correction_payload.get("corrected_prior_review_session_id") == corrected_session_id
            and correction_payload.get("corrected_prior_session_binding") == binding
            and correction_payload.get("evidence_sha256") == canonical_sha256(evidence_payload)
            and isinstance(activation_sha256, str)
            and SHA256_PATTERN.fullmatch(activation_sha256)
        ):
            raise HumanReviewContainmentError("rotation correction evidence has drifted")
        reviewer.profile_json = {
            **reviewer.profile_json,
            "anonymous_external_pool_activation_sha256": activation_sha256,
            "pool_rotation_status": (
                "replacement_ready_pending_active_consent_and_pool_specific_reconsent"
            ),
            "pool_rotation_correction_event_id": existing_correction.id,
        }
        session.add(reviewer)
        return existing_correction, False
    binding_mismatch = original_session_id != corrected_session_id
    activation_sha256 = reviewer.profile_json.get("anonymous_external_pool_activation_sha256")
    activation_missing = not (
        isinstance(activation_sha256, str) and SHA256_PATTERN.fullmatch(activation_sha256)
    )
    if activation_missing:
        activation_sha256 = repaired_activation_sha256(
            reviewer_id=reviewer.id,
            pool_sha256=replacement_sha256,
            rotation_event_id=rotation_event.id,
        )
    if not binding_mismatch and not activation_missing:
        return None, False
    reasons = []
    if binding_mismatch:
        reasons.append("prior_review_session_pool_mismatch")
    if activation_missing:
        reasons.append("missing_pool_activation_epoch")
    correction_base = {
        "schema_version": ROTATION_CORRECTION_SCHEMA_VERSION,
        "supersedes_rotation_event_id": rotation_event.id,
        "prior_pool_sha256": prior_sha256,
        "replacement_pool_sha256": replacement_sha256,
        "original_prior_review_session_id": original_session_id,
        "corrected_prior_review_session_id": corrected_session_id,
        "corrected_prior_session_binding": binding,
        "corrected_prior_batch_status": batch_status,
        "replacement_pool_activation_sha256": activation_sha256,
        "correction_reasons": reasons,
        "review_enabled": False,
        "activation_requirement": (
            "active expert consent and immutable pool-specific reviewer re-consent"
        ),
        "raw_prior_records_preserved": True,
    }
    correction_payload = {
        **correction_base,
        "evidence_sha256": canonical_sha256(correction_base),
    }
    correction_id = _stable_event_id(
        "rotation-correction",
        f"{rotation_event.id}:{correction_payload['evidence_sha256']}",
    )
    correction = session.get(RunEvent, correction_id)
    appended = correction is None
    if correction is None:
        correction = RunEvent(
            id=correction_id,
            entity_type="expert_reviewer",
            entity_id=reviewer.id,
            event_type=ROTATION_CORRECTION_EVENT_TYPE,
            payload_json=correction_payload,
        )
        session.add(correction)
    elif not (
        correction.entity_type == "expert_reviewer"
        and correction.entity_id == reviewer.id
        and correction.event_type == ROTATION_CORRECTION_EVENT_TYPE
        and correction.payload_json == correction_payload
    ):
        raise HumanReviewContainmentError("rotation correction evidence has drifted")
    reviewer.profile_json = {
        **reviewer.profile_json,
        "anonymous_external_pool_activation_sha256": activation_sha256,
        "pool_rotation_status": (
            "replacement_ready_pending_active_consent_and_pool_specific_reconsent"
        ),
        "pool_rotation_correction_event_id": correction.id,
    }
    session.add_all([reviewer, correction])
    return correction, appended


def rotate_anonymous_pool(session: Session, *, candidate_path: Path) -> dict[str, Any]:
    candidate = _load_candidate(candidate_path)
    replacement = _candidate_record(candidate, candidate_path)
    replacement_sha256 = str(replacement["candidate_pack_sha256"])
    reviewer = _latest_anonymous_reviewer(session, for_update=True)
    current_sha256 = str(reviewer.profile_json.get("anonymous_external_pool_sha256") or "")
    if current_sha256 == replacement_sha256:
        event = _rotation_event_for_replacement(
            session,
            reviewer_id=reviewer.id,
            replacement_sha256=replacement_sha256,
        )
        if event is None:
            raise HumanReviewContainmentError(
                "reviewer references the replacement pool without a rotation event"
            )
        correction, correction_appended = _repair_existing_rotation(
            session,
            reviewer=reviewer,
            rotation_event=event,
        )
        session.commit()
        return {
            "reviewerCode": reviewer.reviewer_code,
            "priorPoolSha256": event.payload_json.get("prior_pool_sha256"),
            "replacementPoolSha256": replacement_sha256,
            "reviewEnabled": anonymous_pool_reconsented(session, reviewer),
            "idempotent": True,
            "eventId": event.id,
            "correctionEventId": correction.id if correction is not None else None,
            "correctionAppended": correction_appended,
        }
    if SHA256_PATTERN.fullmatch(current_sha256) is None:
        raise HumanReviewContainmentError("reviewer has no current anonymous response pool")
    review_session, prior_binding, prior_batch_status = _restricted_session_for_pool(
        session,
        reviewer_id=reviewer.id,
        pool_sha256=current_sha256,
    )
    if not _replacement_pool_imported(session, replacement=replacement):
        raise HumanReviewContainmentError("replacement pool has not been imported")
    family_counts = replacement["candidate_pairs_by_family"]
    targets = (
        isolated_uplift_workload_cell_targets(family_counts)
        if replacement["source_class"] == REQUIRED_FRONTIER_SOURCE_CLASS
        else author_evaluator_workload_cell_targets(int(replacement["candidate_pairs"]))
    )
    prior_activation_sha256 = reviewer.profile_json.get("anonymous_external_pool_activation_sha256")
    if not (
        isinstance(prior_activation_sha256, str)
        and SHA256_PATTERN.fullmatch(prior_activation_sha256)
    ):
        prior_activation_sha256 = canonical_sha256(
            {
                "kind": "legacy_prior_pool_without_activation_epoch",
                "reviewer_id": reviewer.id,
                "pool_sha256": current_sha256,
            }
        )
    event_id = _stable_event_id(
        "rotation",
        (f"{reviewer.id}:{current_sha256}:{prior_activation_sha256}:{replacement_sha256}"),
    )
    replacement_activation_sha256 = rotation_activation_sha256(
        reviewer_id=reviewer.id,
        prior_activation_sha256=prior_activation_sha256,
        prior_pool_sha256=current_sha256,
        replacement_pool_sha256=replacement_sha256,
        rotation_event_id=event_id,
    )
    event_payload = {
        "reviewer_code": reviewer.reviewer_code,
        "prior_pool_sha256": current_sha256,
        "replacement_pool_sha256": replacement_sha256,
        "prior_pool_activation_sha256": prior_activation_sha256,
        "replacement_pool_activation_sha256": replacement_activation_sha256,
        "prior_review_session_id": (
            review_session.entity_id if review_session is not None else None
        ),
        "prior_review_session_binding": prior_binding,
        "prior_batch_status": prior_batch_status,
        "replacement_candidate_record_sha256": canonical_sha256(replacement),
        "replacement_source_arms": replacement["source_arms"],
        "replacement_real_provider_calls": replacement["real_provider_calls"],
        "replacement_real_epicure_calls": replacement["real_epicure_calls"],
        "replacement_successful_real_epicure_calls": replacement[
            "successful_real_epicure_calls"
        ],
        "replacement_candidate_pairs_by_family": family_counts,
        "replacement_synthetic_arms": replacement["synthetic_arms"],
        "review_enabled": False,
        "activation_requirement": (
            "active expert consent and immutable pool-specific reviewer re-consent"
        ),
        "raw_prior_records_preserved": True,
    }
    event = RunEvent(
        id=event_id,
        entity_type="expert_reviewer",
        entity_id=reviewer.id,
        event_type=ROTATION_EVENT_TYPE,
        payload_json={
            **event_payload,
            "evidence_sha256": canonical_sha256(event_payload),
        },
    )
    updated_profile = {
        **reviewer.profile_json,
        "calibration_candidate": replacement,
        "anonymous_external_pool_sha256": replacement_sha256,
        "anonymous_external_pool_activation_sha256": replacement_activation_sha256,
        "anonymous_external_primary_judgments": targets["primary_judgments"],
        "anonymous_external_reliability_repeats": targets["reliability_repeats"],
        "anonymous_external_target_judgments": targets["total_presentations"],
        "prior_restricted_pool_sha256": current_sha256,
        "prior_restricted_review_session_id": (
            review_session.entity_id if review_session is not None else None
        ),
        "pool_rotation_status": (
            "replacement_ready_pending_active_consent_and_pool_specific_reconsent"
        ),
    }
    if replacement["source_class"] == REQUIRED_FRONTIER_SOURCE_CLASS:
        updated_profile["anonymous_external_primary_by_family"] = family_counts
    else:
        updated_profile.pop("anonymous_external_primary_by_family", None)
    reviewer.profile_json = updated_profile
    session.add_all([reviewer, event])
    session.commit()
    return {
        "reviewerCode": reviewer.reviewer_code,
        "priorPoolSha256": current_sha256,
        "replacementPoolSha256": replacement_sha256,
        "reviewEnabled": False,
        "idempotent": False,
        "eventId": event.id,
    }


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    with SessionLocal() as session:
        result = rotate_anonymous_pool(session, candidate_path=args.candidate)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run()
