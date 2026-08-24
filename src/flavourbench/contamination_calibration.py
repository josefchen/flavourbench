from __future__ import annotations

import hashlib
import hmac
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .task_evidence import (
    CONTAMINATION_AUTOREJECT_THRESHOLDS,
    TASK_EVIDENCE_IMPLEMENTATION_SHA256,
    ContaminationScanBundle,
    artifact_sha256,
    canonical_sha256,
    normalize_text,
    replay_contamination_scan,
)

CONTAMINATION_CALIBRATION_SCHEMA_VERSION = "flavourbench-contamination-calibration-v1"
CONTAMINATION_CALIBRATION_RECEIPT_SCHEMA_VERSION = (
    "flavourbench-contamination-calibration-receipt-v1"
)
MINIMUM_CASES_PER_CLASS = 50
MINIMUM_TOTAL_CASES = 150
MINIMUM_PRECISION = 0.95
MINIMUM_RECALL = 0.90
MINIMUM_PARAPHRASE_RECALL = 0.85
SHA256_PATTERN = r"^[0-9a-f]{64}$"
OCI_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_VERIFIED_RECEIPT_CACHE: dict[tuple[str, str, str], dict[str, object]] = {}


class ContaminationCalibrationError(ValueError):
    """The leak detector or its labeled calibration artifact does not reproduce."""


class ContaminationCalibrationCase(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    case_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,95}$", alias="caseId")
    prompt: str = Field(min_length=20, max_length=4000)
    source_reference_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="sourceReferenceSha256",
    )
    relation: Literal["exact", "paraphrase", "unrelated"]
    expected_leak: bool = Field(alias="expectedLeak")
    adjudicator_person_commitment_sha256s: list[str] = Field(
        min_length=2,
        max_length=2,
        alias="adjudicatorPersonCommitmentSha256s",
    )
    label_event_sha256s: list[str] = Field(
        min_length=2,
        max_length=2,
        alias="labelEventSha256s",
    )
    model_outputs_not_consulted: Literal[True] = Field(alias="modelOutputsNotConsulted")

    @field_validator(
        "adjudicator_person_commitment_sha256s",
        "label_event_sha256s",
    )
    @classmethod
    def distinct_sha256_rows(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value) or any(
            len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
            for item in value
        ):
            raise ValueError("calibration labels require two distinct SHA-256 commitments")
        return value

    @model_validator(mode="after")
    def validate_relation_label(self) -> ContaminationCalibrationCase:
        if self.expected_leak != (self.relation != "unrelated"):
            raise ValueError("calibration relation and expectedLeak disagree")
        return self


class ContaminationCalibrationArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_version: Literal[CONTAMINATION_CALIBRATION_SCHEMA_VERSION] = Field(alias="schemaVersion")
    artifact_sha256: str = Field(pattern=SHA256_PATTERN, alias="artifactSha256")
    scan_bundle_sha256: str = Field(pattern=SHA256_PATTERN, alias="scanBundleSha256")
    implementation_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="implementationSha256",
    )
    container_image_digest: str = Field(
        pattern=OCI_DIGEST_PATTERN,
        alias="containerImageDigest",
    )
    cases: list[ContaminationCalibrationCase] = Field(
        min_length=MINIMUM_TOTAL_CASES,
        max_length=3000,
    )
    case_set_sha256: str = Field(pattern=SHA256_PATTERN, alias="caseSetSha256")
    label_ledger_root_sha256: str = Field(
        pattern=SHA256_PATTERN,
        alias="labelLedgerRootSha256",
    )
    minimum_precision: float = Field(alias="minimumPrecision")
    minimum_recall: float = Field(alias="minimumRecall")
    minimum_paraphrase_recall: float = Field(alias="minimumParaphraseRecall")
    calibrated_at: datetime = Field(alias="calibratedAt")
    status: Literal["verified"]

    @model_validator(mode="after")
    def validate_calibration_shape(self) -> ContaminationCalibrationArtifact:
        case_ids = [case.case_id for case in self.cases]
        pair_keys = [
            (hashlib.sha256(case.prompt.encode()).hexdigest(), case.source_reference_sha256)
            for case in self.cases
        ]
        if len(set(case_ids)) != len(case_ids) or len(set(pair_keys)) != len(pair_keys):
            raise ValueError("contamination calibration cases must be unique")
        relation_counts = Counter(case.relation for case in self.cases)
        if any(
            relation_counts[relation] < MINIMUM_CASES_PER_CLASS
            for relation in ("exact", "paraphrase", "unrelated")
        ):
            raise ValueError("calibration requires at least 50 cases in every relation class")
        if (
            self.minimum_precision != MINIMUM_PRECISION
            or self.minimum_recall != MINIMUM_RECALL
            or self.minimum_paraphrase_recall != MINIMUM_PARAPHRASE_RECALL
        ):
            raise ValueError("contamination calibration thresholds differ from policy")
        case_payloads = [
            case.model_dump(mode="json", by_alias=True)
            for case in sorted(self.cases, key=lambda row: row.case_id)
        ]
        if self.case_set_sha256 != canonical_sha256(case_payloads):
            raise ValueError("caseSetSha256 does not match the labeled cases")
        label_ledger = [
            {
                "case_id": case.case_id,
                "adjudicator_person_commitment_sha256s": sorted(
                    case.adjudicator_person_commitment_sha256s
                ),
                "label_event_sha256s": sorted(case.label_event_sha256s),
                "relation": case.relation,
                "expected_leak": case.expected_leak,
            }
            for case in sorted(self.cases, key=lambda row: row.case_id)
        ]
        if self.label_ledger_root_sha256 != canonical_sha256(label_ledger):
            raise ValueError("labelLedgerRootSha256 does not match the human labels")
        return self


def _source_texts(bundle: ContaminationScanBundle) -> dict[str, str]:
    return {
        record.source_reference_sha256: record.text
        for record in [*bundle.benchmark_records, *bundle.web_records]
    }


def _prediction(
    case: ContaminationCalibrationCase,
    *,
    bundle: ContaminationScanBundle,
    completed_at: datetime,
) -> tuple[bool, dict[str, int]]:
    _, hits = replay_contamination_scan(case.prompt, bundle, completed_at=completed_at)
    designated = [
        hit for hit in hits if hit["source_reference_sha256"] == case.source_reference_sha256
    ]
    scores: dict[str, int] = {}
    for hit in designated:
        method = str(hit["method"])
        scores[method] = max(scores.get(method, 0), int(hit["similarity_milli"]))
    predicted = any(
        score >= round(1000 * CONTAMINATION_AUTOREJECT_THRESHOLDS[method])
        for method, score in scores.items()
    )
    return predicted, dict(sorted(scores.items()))


def verify_contamination_calibration(
    artifact: ContaminationCalibrationArtifact,
    *,
    scan_bundle: ContaminationScanBundle,
    expected_container_image_digest: str,
) -> dict[str, object]:
    if artifact.artifact_sha256 != artifact_sha256(artifact):
        raise ContaminationCalibrationError("contamination calibration digest mismatch")
    if artifact.scan_bundle_sha256 != scan_bundle.artifact_sha256:
        raise ContaminationCalibrationError("calibration is bound to another scan bundle")
    if artifact.implementation_sha256 != TASK_EVIDENCE_IMPLEMENTATION_SHA256:
        raise ContaminationCalibrationError("contamination implementation digest drifted")
    if artifact.container_image_digest != expected_container_image_digest:
        raise ContaminationCalibrationError("contamination calibration container drifted")
    cache_key = (
        artifact.artifact_sha256,
        scan_bundle.artifact_sha256,
        expected_container_image_digest,
    )
    cached = _VERIFIED_RECEIPT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    source_texts = _source_texts(scan_bundle)
    case_results: list[dict[str, object]] = []
    for case in sorted(artifact.cases, key=lambda row: row.case_id):
        source_text = source_texts.get(case.source_reference_sha256)
        if source_text is None:
            raise ContaminationCalibrationError("calibration case source is absent from bundle")
        if case.relation == "exact" and normalize_text(case.prompt) != normalize_text(source_text):
            raise ContaminationCalibrationError("an exact calibration case is not exact")
        predicted, method_scores = _prediction(
            case,
            bundle=scan_bundle,
            completed_at=artifact.calibrated_at,
        )
        case_results.append(
            {
                "case_id": case.case_id,
                "relation": case.relation,
                "expected_leak": case.expected_leak,
                "predicted_leak": predicted,
                "method_scores_milli": method_scores,
            }
        )
    true_positives = sum(
        bool(row["expected_leak"]) and bool(row["predicted_leak"]) for row in case_results
    )
    false_positives = sum(
        not bool(row["expected_leak"]) and bool(row["predicted_leak"]) for row in case_results
    )
    false_negatives = sum(
        bool(row["expected_leak"]) and not bool(row["predicted_leak"]) for row in case_results
    )
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    paraphrase_rows = [row for row in case_results if row["relation"] == "paraphrase"]
    paraphrase_recall = sum(bool(row["predicted_leak"]) for row in paraphrase_rows) / len(
        paraphrase_rows
    )
    if (
        precision < artifact.minimum_precision
        or recall < artifact.minimum_recall
        or paraphrase_recall < artifact.minimum_paraphrase_recall
    ):
        raise ContaminationCalibrationError(
            "contamination detector misses its precision or recall threshold"
        )
    result_set_sha256 = canonical_sha256(case_results)
    receipt: dict[str, object] = {
        "schema_version": CONTAMINATION_CALIBRATION_RECEIPT_SCHEMA_VERSION,
        "artifact_sha256": artifact.artifact_sha256,
        "scan_bundle_sha256": scan_bundle.artifact_sha256,
        "implementation_sha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "container_image_digest": expected_container_image_digest,
        "case_count": len(case_results),
        "relation_counts": dict(sorted(Counter(row["relation"] for row in case_results).items())),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision_milli": round(1000 * precision),
        "recall_milli": round(1000 * recall),
        "paraphrase_recall_milli": round(1000 * paraphrase_recall),
        "case_result_set_sha256": result_set_sha256,
        "status": "verified",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    if len(_VERIFIED_RECEIPT_CACHE) >= 16:
        _VERIFIED_RECEIPT_CACHE.pop(next(iter(_VERIFIED_RECEIPT_CACHE)))
    _VERIFIED_RECEIPT_CACHE[cache_key] = dict(receipt)
    return receipt


def load_contamination_calibration(
    path: str | Path,
    *,
    expected_sha256: str,
    scan_bundle: ContaminationScanBundle,
    expected_container_image_digest: str,
) -> tuple[ContaminationCalibrationArtifact, dict[str, object]]:
    if not path or not expected_sha256:
        raise ContaminationCalibrationError("contamination calibration is not configured")
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        artifact = ContaminationCalibrationArtifact.model_validate(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ContaminationCalibrationError("contamination calibration is invalid") from exc
    if not hmac.compare_digest(artifact.artifact_sha256, expected_sha256):
        raise ContaminationCalibrationError("configured contamination calibration hash differs")
    receipt = verify_contamination_calibration(
        artifact,
        scan_bundle=scan_bundle,
        expected_container_image_digest=expected_container_image_digest,
    )
    return artifact, receipt
