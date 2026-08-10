from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .task_evidence import (
    OCI_DIGEST_PATTERN,
    SHA256_PATTERN,
    TASK_EVIDENCE_IMPLEMENTATION_SHA256,
    TaskEvidenceError,
    TaskValidatorRule,
    artifact_sha256,
    canonical_sha256,
    evaluate_rules,
)

VALIDATOR_CALIBRATION_SCHEMA_VERSION = "flavourbench-validator-calibration-v1"
VALIDATOR_CALIBRATION_RECEIPT_SCHEMA_VERSION = (
    "flavourbench-validator-calibration-receipt-v1"
)
VALIDATOR_RULE_KINDS = frozenset(
    {
        "required_entity",
        "prohibited_claim",
        "numeric_range",
        "ratio_range",
        "ordered_steps",
        "evidence_calibration",
    }
)
MINIMUM_CASES_PER_RULE_KIND = 20
MINIMUM_PASS_CASES_PER_RULE_KIND = 10
MINIMUM_FAIL_CASES_PER_RULE_KIND = 10
MINIMUM_VIOLATION_PRECISION = 0.95
MINIMUM_VIOLATION_RECALL = 0.90


class CalibrationLabelBinding(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    labeler_reviewer_id: str = Field(
        min_length=3,
        max_length=160,
        alias="labelerReviewerId",
    )
    decision: Literal["pass", "fail"]
    label_event_sha256: str = Field(pattern=SHA256_PATTERN, alias="labelEventSha256")
    blind_to_evaluator_output: Literal[True] = Field(alias="blindToEvaluatorOutput")
    independent_of_case_author: Literal[True] = Field(alias="independentOfCaseAuthor")


class ValidatorCalibrationCase(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    case_id: str = Field(
        pattern=r"^[a-z][a-z0-9_-]{2,79}$",
        alias="caseId",
    )
    case_author_reviewer_id: str = Field(
        min_length=3,
        max_length=160,
        alias="caseAuthorReviewerId",
    )
    mutation_class: Literal[
        "valid_control",
        "omission",
        "negation",
        "lexical_adversary",
        "numeric_boundary",
        "unit_conversion",
        "ratio_boundary",
        "step_reversal",
        "causal_overclaim",
    ] = Field(alias="mutationClass")
    rule: TaskValidatorRule
    response_text: str = Field(min_length=1, max_length=20000, alias="responseText")
    response_sha256: str = Field(pattern=SHA256_PATTERN, alias="responseSha256")
    expected_status: Literal["pass", "fail"] = Field(alias="expectedStatus")
    labels: list[CalibrationLabelBinding] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_case(self) -> ValidatorCalibrationCase:
        expected_response_sha256 = hashlib.sha256(self.response_text.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(self.response_sha256, expected_response_sha256):
            raise ValueError("responseSha256 does not match responseText")
        labeler_ids = [label.labeler_reviewer_id for label in self.labels]
        if len(set(labeler_ids)) != 2:
            raise ValueError("calibration cases require two distinct labelers")
        if self.case_author_reviewer_id in labeler_ids:
            raise ValueError("the calibration case author cannot label their own case")
        if {label.decision for label in self.labels} != {self.expected_status}:
            raise ValueError("both blind labels must agree with expectedStatus")
        for label in self.labels:
            expected_event_sha256 = canonical_sha256(
                {
                    "case_id": self.case_id,
                    "response_sha256": self.response_sha256,
                    "rule_id": self.rule.rule_id,
                    "rule_kind": self.rule.kind,
                    "labeler_reviewer_id": label.labeler_reviewer_id,
                    "decision": label.decision,
                    "blind_to_evaluator_output": True,
                    "independent_of_case_author": True,
                }
            )
            if not hmac.compare_digest(label.label_event_sha256, expected_event_sha256):
                raise ValueError("labelEventSha256 does not match the blind label payload")
        return self


class ValidatorCalibrationArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_version: Literal[VALIDATOR_CALIBRATION_SCHEMA_VERSION] = Field(
        alias="schemaVersion"
    )
    artifact_sha256: str = Field(pattern=SHA256_PATTERN, alias="artifactSha256")
    created_at: datetime = Field(alias="createdAt")
    evaluator_implementation_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="evaluatorImplementationSha256",
    )
    validator_container_image_digest: str = Field(
        pattern=OCI_DIGEST_PATTERN,
        alias="validatorContainerImageDigest",
    )
    minimum_violation_precision: float = Field(
        ge=0,
        le=1,
        alias="minimumViolationPrecision",
    )
    minimum_violation_recall: float = Field(
        ge=0,
        le=1,
        alias="minimumViolationRecall",
    )
    cases: list[ValidatorCalibrationCase] = Field(min_length=120, max_length=2000)
    case_set_sha256: str = Field(pattern=SHA256_PATTERN, alias="caseSetSha256")
    label_set_sha256: str = Field(pattern=SHA256_PATTERN, alias="labelSetSha256")
    label_ledger_root_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="labelLedgerRootSha256",
    )
    status: Literal["sealed"]

    @model_validator(mode="after")
    def validate_artifact_shape(self) -> ValidatorCalibrationArtifact:
        case_ids = [case.case_id for case in self.cases]
        response_sha256s = [case.response_sha256 for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("calibration case IDs must be unique")
        if len(set(response_sha256s)) != len(response_sha256s):
            raise ValueError("calibration response texts must be unique")
        kind_counts = Counter(case.rule.kind for case in self.cases)
        if set(kind_counts) != VALIDATOR_RULE_KINDS:
            raise ValueError("calibration artifact must cover every validator rule kind")
        for kind in VALIDATOR_RULE_KINDS:
            cases = [case for case in self.cases if case.rule.kind == kind]
            status_counts = Counter(case.expected_status for case in cases)
            if len(cases) < MINIMUM_CASES_PER_RULE_KIND:
                raise ValueError(f"{kind} has too few calibration cases")
            if status_counts["pass"] < MINIMUM_PASS_CASES_PER_RULE_KIND:
                raise ValueError(f"{kind} has too few passing controls")
            if status_counts["fail"] < MINIMUM_FAIL_CASES_PER_RULE_KIND:
                raise ValueError(f"{kind} has too few violation cases")
        if self.minimum_violation_precision != MINIMUM_VIOLATION_PRECISION:
            raise ValueError("minimumViolationPrecision differs from the frozen policy")
        if self.minimum_violation_recall != MINIMUM_VIOLATION_RECALL:
            raise ValueError("minimumViolationRecall differs from the frozen policy")
        case_payloads = [
            case.model_dump(mode="json", by_alias=True, exclude={"labels"})
            for case in sorted(self.cases, key=lambda item: item.case_id)
        ]
        label_payloads = [
            {
                "caseId": case.case_id,
                "labels": [
                    label.model_dump(mode="json", by_alias=True)
                    for label in sorted(
                        case.labels,
                        key=lambda item: item.labeler_reviewer_id,
                    )
                ],
            }
            for case in sorted(self.cases, key=lambda item: item.case_id)
        ]
        label_event_sha256s = sorted(
            label.label_event_sha256 for case in self.cases for label in case.labels
        )
        if self.case_set_sha256 != canonical_sha256(case_payloads):
            raise ValueError("caseSetSha256 does not match the calibration cases")
        if self.label_set_sha256 != canonical_sha256(label_payloads):
            raise ValueError("labelSetSha256 does not match the blind labels")
        if self.label_ledger_root_sha256 != canonical_sha256(label_event_sha256s):
            raise ValueError("labelLedgerRootSha256 does not match the label events")
        return self


def _confusion_metrics(
    expected: list[str],
    predicted: list[str],
) -> dict[str, int | float]:
    true_positive = sum(
        expected_status == "fail" and predicted_status == "fail"
        for expected_status, predicted_status in zip(expected, predicted, strict=True)
    )
    false_positive = sum(
        expected_status == "pass" and predicted_status == "fail"
        for expected_status, predicted_status in zip(expected, predicted, strict=True)
    )
    false_negative = sum(
        expected_status == "fail" and predicted_status == "pass"
        for expected_status, predicted_status in zip(expected, predicted, strict=True)
    )
    true_negative = sum(
        expected_status == "pass" and predicted_status == "pass"
        for expected_status, predicted_status in zip(expected, predicted, strict=True)
    )
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "violation_precision": precision,
        "violation_recall": recall,
    }


def verify_validator_calibration(
    artifact: ValidatorCalibrationArtifact,
    *,
    expected_container_image_digest: str,
) -> dict[str, object]:
    if artifact.artifact_sha256 != artifact_sha256(artifact):
        raise TaskEvidenceError("validator calibration artifact digest mismatch")
    if artifact.evaluator_implementation_sha256 != TASK_EVIDENCE_IMPLEMENTATION_SHA256:
        raise TaskEvidenceError("validator calibration uses another evaluator implementation")
    if (
        not re.fullmatch(OCI_DIGEST_PATTERN, expected_container_image_digest)
        or artifact.validator_container_image_digest != expected_container_image_digest
    ):
        raise TaskEvidenceError(
            "validator calibration container digest is unresolved or mismatched"
        )

    by_kind: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    case_receipts: list[dict[str, str]] = []
    for case in sorted(artifact.cases, key=lambda item: item.case_id):
        result = evaluate_rules([case.rule], case.response_text)
        if len(result) != 1:
            raise TaskEvidenceError("validator calibration case did not produce one rule result")
        predicted_status = str(result[0]["status"])
        by_kind[case.rule.kind].append(
            (case.case_id, case.expected_status, predicted_status)
        )
        case_receipts.append(
            {
                "case_id": case.case_id,
                "rule_kind": case.rule.kind,
                "expected_status": case.expected_status,
                "predicted_status": predicted_status,
                "response_sha256": case.response_sha256,
            }
        )

    metrics_by_kind: dict[str, dict[str, int | float]] = {}
    all_expected: list[str] = []
    all_predicted: list[str] = []
    for kind in sorted(VALIDATOR_RULE_KINDS):
        rows = by_kind[kind]
        expected = [row[1] for row in rows]
        predicted = [row[2] for row in rows]
        metrics = _confusion_metrics(expected, predicted)
        metrics_by_kind[kind] = metrics
        all_expected.extend(expected)
        all_predicted.extend(predicted)
        if (
            float(metrics["violation_precision"])
            < artifact.minimum_violation_precision
            or float(metrics["violation_recall"]) < artifact.minimum_violation_recall
        ):
            raise TaskEvidenceError(
                f"validator calibration threshold failed for {kind}"
            )

    overall = _confusion_metrics(all_expected, all_predicted)
    if (
        float(overall["violation_precision"]) < artifact.minimum_violation_precision
        or float(overall["violation_recall"]) < artifact.minimum_violation_recall
    ):
        raise TaskEvidenceError("overall validator calibration threshold failed")
    payload: dict[str, object] = {
        "schema_version": VALIDATOR_CALIBRATION_RECEIPT_SCHEMA_VERSION,
        "artifact_sha256": artifact.artifact_sha256,
        "case_set_sha256": artifact.case_set_sha256,
        "label_set_sha256": artifact.label_set_sha256,
        "label_ledger_root_sha256": artifact.label_ledger_root_sha256,
        "evaluator_implementation_sha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "validator_container_image_digest": expected_container_image_digest,
        "case_count": len(artifact.cases),
        "metrics_by_rule_kind": metrics_by_kind,
        "overall": overall,
        "case_receipts_sha256": canonical_sha256(case_receipts),
        "status": "verified",
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def load_validator_calibration(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_container_image_digest: str,
) -> tuple[ValidatorCalibrationArtifact, dict[str, object]]:
    artifact_path = Path(path)
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise TaskEvidenceError("validator calibration artifact is unavailable")
    try:
        document = json.loads(artifact_path.read_bytes())
        artifact = ValidatorCalibrationArtifact.model_validate(document)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TaskEvidenceError("validator calibration artifact is not schema-valid") from exc
    if not re.fullmatch(SHA256_PATTERN, expected_sha256) or not hmac.compare_digest(
        artifact.artifact_sha256,
        expected_sha256,
    ):
        raise TaskEvidenceError("validator calibration configured digest mismatch")
    receipt = verify_validator_calibration(
        artifact,
        expected_container_image_digest=expected_container_image_digest,
    )
    return artifact, receipt
