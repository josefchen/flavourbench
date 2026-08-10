from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .contamination_calibration import (
    ContaminationCalibrationArtifact,
    load_contamination_calibration,
    verify_contamination_calibration,
)
from .models import Battle, RunEvent, Task, TaskEvidenceArtifact
from .task_evidence import (
    VALIDATOR_DSL_VERSION,
    ContaminationScanBundle,
    TaskContaminationAuditArtifact,
    TaskEvidenceError,
    TaskValidatorContractArtifact,
    canonical_sha256,
    evaluate_contract,
    load_contamination_scan_bundle,
    task_evidence_review_sha256,
    task_evidence_root_sha256,
    verify_contamination_audit,
    verify_validator_contract,
)
from .task_lifecycle import verify_task_lifecycle
from .validator_calibration import (
    ValidatorCalibrationArtifact,
    load_validator_calibration,
    verify_validator_calibration,
)

TASK_SPECIFIC_VALIDATOR_NAME = "task_specific_constraint_dsl"
TASK_SPECIFIC_VALIDATOR_VERSION = VALIDATOR_DSL_VERSION


@dataclass(frozen=True)
class VerifiedTaskEvidence:
    validator_contract: TaskValidatorContractArtifact
    validator_receipt: dict[str, object]
    contamination_audit: TaskContaminationAuditArtifact
    contamination_receipt: dict[str, object]
    validator_calibration: ValidatorCalibrationArtifact
    validator_calibration_receipt: dict[str, object]
    contamination_calibration: ContaminationCalibrationArtifact
    contamination_calibration_receipt: dict[str, object]
    evidence_root_sha256: str


def _provenance(task: Task) -> dict[str, Any]:
    if not isinstance(task.provenance_json, dict):
        raise TaskEvidenceError("task provenance is not an object")
    return task.provenance_json


def verify_task_evidence_registry(
    session: Session,
    task: Task,
    *,
    expected_container_image_digest: str,
    contamination_scan_bundle: ContaminationScanBundle | None = None,
    validator_calibration: (tuple[ValidatorCalibrationArtifact, dict[str, object]] | None) = None,
    contamination_calibration: (
        tuple[ContaminationCalibrationArtifact, dict[str, object]] | None
    ) = None,
) -> VerifiedTaskEvidence:
    """Recompute every frozen task-evidence binding from stored bytes.

    The function is shared by season admission and the generation worker so a
    task cannot be admitted under one verifier and scored under another.
    """

    provenance = _provenance(task)
    rows = list(
        session.scalars(
            select(TaskEvidenceArtifact).where(TaskEvidenceArtifact.task_id == task.id)
        ).all()
    )
    by_type = {row.evidence_type: row for row in rows}
    if len(rows) != 2 or set(by_type) != {"validator_contract", "contamination_audit"}:
        raise TaskEvidenceError("task does not have exactly one validator and contamination record")
    if (
        session.scalar(
            select(TaskEvidenceArtifact.id).where(
                TaskEvidenceArtifact.supersedes_artifact_id.in_([row.id for row in rows])
            )
        )
        is not None
    ):
        raise TaskEvidenceError("task evidence was superseded after this task revision was sealed")

    validator_row = by_type["validator_contract"]
    contamination_row = by_type["contamination_audit"]
    if validator_row.artifact_sha256 != provenance.get("validator_contract_sha256"):
        raise TaskEvidenceError("validator record differs from task provenance")
    if contamination_row.artifact_sha256 != provenance.get("contamination_audit_sha256"):
        raise TaskEvidenceError("contamination record differs from task provenance")

    try:
        validator_contract = TaskValidatorContractArtifact.model_validate(
            validator_row.artifact_json
        )
        contamination_audit = TaskContaminationAuditArtifact.model_validate(
            contamination_row.artifact_json
        )
    except ValueError as exc:
        raise TaskEvidenceError("stored task evidence is not schema-valid") from exc

    validator_receipt = verify_validator_contract(
        validator_contract,
        task_public_id=task.public_id,
        task_family=task.family,
        task_revision=task.revision,
        prompt_sha256=task.prompt_sha256,
        objective_validator_possible=provenance.get("objective_validator_possible") is True,
        expected_container_image_digest=expected_container_image_digest,
    )
    if validator_calibration is None:
        settings = get_settings()
        validator_calibration = load_validator_calibration(
            settings.validator_calibration_artifact_path,
            expected_sha256=settings.validator_calibration_artifact_sha256,
            expected_container_image_digest=expected_container_image_digest,
        )
    calibration_artifact, calibration_receipt = validator_calibration
    reproduced_calibration_receipt = verify_validator_calibration(
        calibration_artifact,
        expected_container_image_digest=expected_container_image_digest,
    )
    if calibration_receipt != reproduced_calibration_receipt:
        raise TaskEvidenceError("validator calibration receipt does not reproduce")
    if provenance.get(
        "validator_calibration_artifact_sha256"
    ) != calibration_artifact.artifact_sha256 or provenance.get(
        "validator_calibration_receipt_sha256"
    ) != calibration_receipt.get("receipt_sha256"):
        raise TaskEvidenceError("task provenance points at another validator calibration")
    approving_ids = {
        str(review.get("reviewer_id"))
        for review in provenance.get("independent_reviews") or []
        if isinstance(review, dict)
    }
    adjudication = provenance.get("adjudication")
    adjudicator_id = (
        str(adjudication.get("adjudicator_reviewer_id", ""))
        if isinstance(adjudication, dict)
        else ""
    )
    if contamination_scan_bundle is None:
        settings = get_settings()
        contamination_scan_bundle = load_contamination_scan_bundle(
            settings.contamination_scan_bundle_path,
            expected_sha256=settings.contamination_scan_bundle_sha256,
        )
    if provenance.get("contamination_scan_bundle_sha256") != (
        contamination_scan_bundle.artifact_sha256
    ):
        raise TaskEvidenceError("task provenance points at a different contamination corpus")
    if contamination_calibration is None:
        settings = get_settings()
        contamination_calibration = load_contamination_calibration(
            settings.contamination_calibration_artifact_path,
            expected_sha256=settings.contamination_calibration_artifact_sha256,
            scan_bundle=contamination_scan_bundle,
            expected_container_image_digest=expected_container_image_digest,
        )
    contamination_calibration_artifact, contamination_calibration_receipt = (
        contamination_calibration
    )
    reproduced_contamination_calibration_receipt = verify_contamination_calibration(
        contamination_calibration_artifact,
        scan_bundle=contamination_scan_bundle,
        expected_container_image_digest=expected_container_image_digest,
    )
    if contamination_calibration_receipt != reproduced_contamination_calibration_receipt:
        raise TaskEvidenceError("contamination calibration receipt does not reproduce")
    if provenance.get(
        "contamination_calibration_artifact_sha256"
    ) != contamination_calibration_artifact.artifact_sha256 or provenance.get(
        "contamination_calibration_receipt_sha256"
    ) != contamination_calibration_receipt.get("receipt_sha256"):
        raise TaskEvidenceError("task provenance points at another contamination calibration")
    contamination_receipt = verify_contamination_audit(
        contamination_audit,
        scan_bundle=contamination_scan_bundle,
        prompt=task.prompt,
        task_public_id=task.public_id,
        task_family=task.family,
        task_revision=task.revision,
        prompt_sha256=task.prompt_sha256,
        expected_container_image_digest=expected_container_image_digest,
        forbidden_reviewer_ids={
            str(provenance.get("human_author_id", "")),
            adjudicator_id,
            *approving_ids,
        },
    )

    source_candidate_id = str(provenance.get("source_candidate_id", ""))
    validator_review = provenance.get("validator_contract_review")
    contamination_review = provenance.get("contamination_audit_review")
    if not isinstance(validator_review, dict) or not isinstance(
        contamination_review,
        dict,
    ):
        raise TaskEvidenceError("task provenance lacks human evidence-review bindings")
    review_bindings = {
        "validator_contract": (
            validator_review,
            validator_contract.artifact_sha256,
            str(validator_receipt["receipt_sha256"]),
            validator_contract.verifier_reviewer_id,
            "task_candidate_validator_contract_verified",
        ),
        "contamination_audit": (
            contamination_review,
            contamination_audit.artifact_sha256,
            str(contamination_receipt["receipt_sha256"]),
            contamination_audit.auditor_reviewer_id,
            "task_candidate_contamination_audit_verified",
        ),
    }
    candidate_review_events = list(
        session.scalars(
            select(RunEvent).where(
                RunEvent.entity_type == "task_candidate",
                RunEvent.entity_id == source_candidate_id,
                RunEvent.event_type.in_(
                    {
                        "task_candidate_validator_contract_verified",
                        "task_candidate_contamination_audit_verified",
                    }
                ),
            )
        ).all()
    )
    task_review_events = list(
        session.scalars(
            select(RunEvent).where(
                RunEvent.entity_type == "task",
                RunEvent.entity_id == task.id,
                RunEvent.event_type == "confirmatory_task_evidence_review_recorded",
            )
        ).all()
    )
    if len(candidate_review_events) != 2 or len(task_review_events) != 2:
        raise TaskEvidenceError("task does not have exactly two sealed evidence reviews")
    candidate_events_by_type = {
        str(event.payload_json.get("evidence_type")): event for event in candidate_review_events
    }
    task_events_by_type = {
        str(event.payload_json.get("evidence_type")): event for event in task_review_events
    }
    if set(candidate_events_by_type) != set(review_bindings) or set(task_events_by_type) != set(
        review_bindings
    ):
        raise TaskEvidenceError("task evidence-review types do not reproduce")
    evidence_reviewer_ids: set[str] = set()
    for evidence_type, (
        binding,
        artifact_sha256,
        receipt_sha256,
        artifact_reviewer_id,
        expected_event_type,
    ) in review_bindings.items():
        candidate_event = candidate_events_by_type[evidence_type]
        task_event = task_events_by_type[evidence_type]
        payload = candidate_event.payload_json
        review = payload.get("review")
        if not isinstance(review, dict):
            raise TaskEvidenceError("sealed evidence review has no typed review payload")
        reviewer_id = str(binding.get("reviewer_id", ""))
        expected_review_sha256 = task_evidence_review_sha256(
            candidate_id=source_candidate_id,
            candidate_record_sha256=str(provenance.get("candidate_record_sha256", "")),
            task_public_id=task.public_id,
            reviewer_id=reviewer_id,
            evidence_type=evidence_type,  # type: ignore[arg-type]
            artifact_sha256=artifact_sha256,
            verification_receipt_sha256=receipt_sha256,
            review=review,
        )
        required_review_checks = {
            "decision": "approve",
            "task_binding_checked": True,
            "model_outputs_not_consulted": True,
            "independent_of_task_roles": True,
        }
        if evidence_type == "validator_contract":
            required_review_checks.update(
                {
                    "rules_and_fixtures_inspected": True,
                    "verification_receipt_reproduced": True,
                }
            )
        else:
            required_review_checks.update(
                {
                    "replay_receipt_reproduced": True,
                    "hit_dispositions_inspected": True,
                }
            )
        task_event_expected = {
            **binding,
            "public_id": task.public_id,
            "family": task.family,
            "split": task.split,
            "source_candidate_id": source_candidate_id,
            "candidate_record_sha256": provenance.get("candidate_record_sha256"),
            "source_event_id": candidate_event.id,
            "task_record_sha256": provenance.get("task_record_sha256"),
            "task_evidence_root_sha256": provenance.get("task_evidence_root_sha256"),
        }
        if (
            candidate_event.event_type != expected_event_type
            or reviewer_id != artifact_reviewer_id
            or reviewer_id in evidence_reviewer_ids
            or reviewer_id
            in {
                str(provenance.get("human_author_id", "")),
                adjudicator_id,
                *approving_ids,
            }
            or binding.get("evidence_type") != evidence_type
            or binding.get("artifact_sha256") != artifact_sha256
            or binding.get("verification_receipt_sha256") != receipt_sha256
            or binding.get("review_event_sha256") != expected_review_sha256
            or binding.get("decision") != "approve"
            or binding.get("independent_of_task_roles") is not True
            or payload.get("candidate_record_sha256") != provenance.get("candidate_record_sha256")
            or payload.get("task_public_id") != task.public_id
            or payload.get("reviewer_id") != reviewer_id
            or payload.get("evidence_type") != evidence_type
            or payload.get("artifact_sha256") != artifact_sha256
            or payload.get("verification_receipt_sha256") != receipt_sha256
            or payload.get("review_event_sha256") != expected_review_sha256
            or payload.get("artifact_visible") is not True
            or payload.get("model_outputs_visible") is not False
            or payload.get("independent_of_task_roles") is not True
            or any(review.get(key) != value for key, value in required_review_checks.items())
            or not isinstance(review.get("note"), str)
            or len(str(review.get("note", ""))) < 40
            or task_event.payload_json != task_event_expected
        ):
            raise TaskEvidenceError("human evidence-review binding does not reproduce")
        evidence_reviewer_ids.add(reviewer_id)

    for row, receipt in (
        (validator_row, validator_receipt),
        (contamination_row, contamination_receipt),
    ):
        receipt_sha256 = str(receipt["receipt_sha256"])
        expected_binding_sha256 = canonical_sha256(
            {
                "artifact_sha256": row.artifact_sha256,
                "evidence_type": row.evidence_type,
                "revision_ordinal": row.revision_ordinal,
                "supersedes_artifact_id": row.supersedes_artifact_id,
                "task_id": row.task_id,
                "verification_receipt_sha256": receipt_sha256,
            }
        )
        if row.revision_ordinal != 1:
            raise TaskEvidenceError("frozen task points at a non-initial evidence revision")
        if row.verification_receipt_json != receipt:
            raise TaskEvidenceError("stored verification receipt does not reproduce")
        if row.verification_receipt_sha256 != receipt_sha256:
            raise TaskEvidenceError("verification receipt digest does not reproduce")
        if row.task_binding_sha256 != expected_binding_sha256:
            raise TaskEvidenceError("database task-evidence binding does not reproduce")

    evidence_root_sha256 = task_evidence_root_sha256(
        task_record_sha256=str(provenance.get("task_record_sha256", "")),
        candidate_record_sha256=str(provenance.get("candidate_record_sha256", "")),
        review_history_sha256=str(provenance.get("review_history_sha256", "")),
        validator_contract_sha256=validator_row.artifact_sha256,
        contamination_audit_sha256=contamination_row.artifact_sha256,
        validator_receipt_sha256=validator_row.verification_receipt_sha256,
        contamination_receipt_sha256=contamination_row.verification_receipt_sha256,
        validator_review_event_sha256=str(validator_review.get("review_event_sha256", "")),
        contamination_review_event_sha256=str(contamination_review.get("review_event_sha256", "")),
    )
    if provenance.get("evidence_registry_status") != "verified" or not hmac.compare_digest(
        str(provenance.get("task_evidence_root_sha256", "")), evidence_root_sha256
    ):
        raise TaskEvidenceError("task evidence root does not reproduce")
    try:
        verify_task_lifecycle(session, task)
    except ValueError as exc:
        raise TaskEvidenceError("task lifecycle does not reproduce") from exc
    return VerifiedTaskEvidence(
        validator_contract=validator_contract,
        validator_receipt=validator_receipt,
        contamination_audit=contamination_audit,
        contamination_receipt=contamination_receipt,
        validator_calibration=calibration_artifact,
        validator_calibration_receipt=calibration_receipt,
        contamination_calibration=contamination_calibration_artifact,
        contamination_calibration_receipt=contamination_calibration_receipt,
        evidence_root_sha256=evidence_root_sha256,
    )


def verified_task_evidence_for_battle(
    session: Session,
    battle: Battle,
    *,
    expected_container_image_digest: str,
    contamination_scan_bundle: ContaminationScanBundle | None = None,
    validator_calibration: (tuple[ValidatorCalibrationArtifact, dict[str, object]] | None) = None,
    contamination_calibration: (
        tuple[ContaminationCalibrationArtifact, dict[str, object]] | None
    ) = None,
) -> VerifiedTaskEvidence | None:
    """Resolve the immutable evidence for a task-bound battle.

    Development tasks without confirmatory evidence remain unscored. Official
    task-bound battles fail closed rather than silently falling back to a regex
    acknowledgement check.
    """

    if battle.task_id is None:
        return None
    task = session.get(Task, battle.task_id)
    if task is None:
        raise TaskEvidenceError("battle task no longer exists")
    prompt_sha256 = (
        hashlib.sha256(battle.prompt.encode("utf-8")).hexdigest()
        if isinstance(battle.prompt, str)
        else None
    )
    if (
        task.season_id != battle.season_id
        or battle.task_revision != task.revision
        or prompt_sha256 != task.prompt_sha256
    ):
        raise TaskEvidenceError("battle input differs from its frozen task revision")

    provenance = _provenance(task)
    if provenance.get("confirmatory_eligible") is not True:
        if battle.rank_eligible or battle.run_class == "official":
            raise TaskEvidenceError("official task-bound battle lacks confirmatory evidence")
        return None

    return verify_task_evidence_registry(
        session,
        task,
        expected_container_image_digest=expected_container_image_digest,
        contamination_scan_bundle=contamination_scan_bundle,
        validator_calibration=validator_calibration,
        contamination_calibration=contamination_calibration,
    )


def task_validator_receipt_for_battle(
    session: Session,
    battle: Battle,
    response_text: str,
    *,
    expected_container_image_digest: str,
    contamination_scan_bundle: ContaminationScanBundle | None = None,
    validator_calibration: (tuple[ValidatorCalibrationArtifact, dict[str, object]] | None) = None,
    contamination_calibration: (
        tuple[ContaminationCalibrationArtifact, dict[str, object]] | None
    ) = None,
) -> dict[str, object] | None:
    """Evaluate the sealed executable subset for a task-bound response."""

    verified = verified_task_evidence_for_battle(
        session,
        battle,
        expected_container_image_digest=expected_container_image_digest,
        contamination_scan_bundle=contamination_scan_bundle,
        validator_calibration=validator_calibration,
        contamination_calibration=contamination_calibration,
    )
    if verified is None:
        return None
    receipt = evaluate_contract(verified.validator_contract, response_text)
    rule_results = receipt.get("rule_results")
    passed = (
        sum(result.get("status") == "pass" for result in rule_results)
        if isinstance(rule_results, list)
        else 0
    )
    rule_count = len(rule_results) if isinstance(rule_results, list) else 0
    score_milli = (
        round(1000 * passed / rule_count)
        if receipt.get("status") != "not_applicable" and rule_count
        else None
    )
    return {
        "status": receipt["status"],
        "score_milli": score_milli,
        "receipt": receipt,
        "task_evidence_root_sha256": verified.evidence_root_sha256,
        "validator_container_image_digest": (
            verified.validator_contract.validator_container_image_digest
        ),
    }
