from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from flavourbench.contamination_calibration import (
    CONTAMINATION_CALIBRATION_SCHEMA_VERSION,
    MINIMUM_PARAPHRASE_RECALL,
    MINIMUM_PRECISION,
    MINIMUM_RECALL,
    ContaminationCalibrationArtifact,
    ContaminationCalibrationError,
    verify_contamination_calibration,
)
from flavourbench.task_evidence import (
    CONTAMINATION_AUTOREJECT_THRESHOLDS,
    CONTAMINATION_REPORT_THRESHOLDS,
    CONTAMINATION_SCAN_BUNDLE_SCHEMA_VERSION,
    TASK_EVIDENCE_IMPLEMENTATION_SHA256,
    ContaminationCorpusRecord,
    ContaminationScanBundle,
    artifact_sha256,
    canonical_sha256,
    normalized_prompt_sha256,
)

CONTAINER_DIGEST = "sha256:" + "c" * 64
OBSERVED_AT = datetime(2026, 8, 1, tzinfo=UTC)


def _record(
    *,
    reference: str,
    source_class: str,
    text: str,
    query_sha256: str | None,
) -> dict[str, object]:
    payload = {
        "sourceReferenceSha256": hashlib.sha256(reference.encode()).hexdigest(),
        "sourceClass": source_class,
        "text": text,
        "textSha256": hashlib.sha256(text.encode()).hexdigest(),
        "querySha256": query_sha256,
        "firstPublishedAt": "2025-01-01T00:00:00+00:00",
        "capturedAt": OBSERVED_AT.isoformat(),
    }
    return ContaminationCorpusRecord.model_validate(payload).model_dump(
        mode="json",
        by_alias=True,
    )


def _calibration_fixture() -> tuple[
    ContaminationCalibrationArtifact,
    ContaminationScanBundle,
]:
    benchmark_records: list[dict[str, object]] = []
    web_records: list[dict[str, object]] = []
    web_receipts: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    case_index = 0
    for relation in ("exact", "paraphrase", "unrelated"):
        for ordinal in range(50):
            case_index += 1
            if relation == "exact":
                prompt = (
                    f"Prepare calibration plate {ordinal} with roasted carrot, white beans, "
                    "lemon, cumin, and a crisp herb finish."
                )
                source_text = prompt
            elif relation == "paraphrase":
                prompt = (
                    f"Prepare calibration bowl {ordinal} with roasted squash, chickpeas, "
                    "lime, coriander, and toasted seeds."
                )
                source_text = (
                    f"Prepare calibration bowl {ordinal} using roasted squash, chickpeas, "
                    "lime, coriander, and toasted seeds."
                )
            else:
                prompt = (
                    f"Design calibration supper {ordinal} around coconut rice, aubergine, "
                    "tamarind, basil, and peanuts."
                )
                source_text = (
                    f"Astronomy field note {ordinal} describes telescope alignment, stellar "
                    "parallax, detector cooling, and cloud cover."
                )
            source = _record(
                reference=f"benchmark:{relation}:{ordinal}",
                source_class="benchmark_corpus",
                text=source_text,
                query_sha256=None,
            )
            benchmark_records.append(source)
            query_sha256 = normalized_prompt_sha256(prompt)
            web = _record(
                reference=f"web:{relation}:{ordinal}",
                source_class="web_snapshot",
                text=(
                    f"Garden maintenance bulletin {case_index} covers pruning, irrigation, "
                    "mulching, and seasonal tool storage."
                ),
                query_sha256=query_sha256,
            )
            web_records.append(web)
            web_receipts.append(
                {
                    "querySha256": query_sha256,
                    "provider": "frozen-test-search",
                    "providerContractSha256": hashlib.sha256(b"provider-contract").hexdigest(),
                    "rawResponseSha256": hashlib.sha256(
                        f"raw:{relation}:{ordinal}".encode()
                    ).hexdigest(),
                    "resultRecordSetSha256": canonical_sha256([web["sourceReferenceSha256"]]),
                    "collectedAt": OBSERVED_AT.isoformat(),
                }
            )
            cases.append(
                {
                    "caseId": f"{relation}-case-{ordinal:03d}",
                    "prompt": prompt,
                    "sourceReferenceSha256": source["sourceReferenceSha256"],
                    "relation": relation,
                    "expectedLeak": relation != "unrelated",
                    "adjudicatorPersonCommitmentSha256s": [
                        hashlib.sha256(f"adjudicator-a:{case_index}".encode()).hexdigest(),
                        hashlib.sha256(f"adjudicator-b:{case_index}".encode()).hexdigest(),
                    ],
                    "labelEventSha256s": [
                        hashlib.sha256(f"label-a:{case_index}".encode()).hexdigest(),
                        hashlib.sha256(f"label-b:{case_index}".encode()).hexdigest(),
                    ],
                    "modelOutputsNotConsulted": True,
                }
            )
    bundle_payload = {
        "schemaVersion": CONTAMINATION_SCAN_BUNDLE_SCHEMA_VERSION,
        "artifactSha256": "0" * 64,
        "createdAt": OBSERVED_AT.isoformat(),
        "benchmarkSnapshotSha256": canonical_sha256(
            sorted(
                benchmark_records,
                key=lambda row: str(row["sourceReferenceSha256"]),
            )
        ),
        "webSnapshotSha256": canonical_sha256(
            sorted(web_records, key=lambda row: str(row["sourceReferenceSha256"]))
        ),
        "benchmarkRecords": benchmark_records,
        "webRecords": web_records,
        "webCollectionReceipts": web_receipts,
        "semanticMethod": "distributional-random-indexing-v1",
        "reportThresholds": CONTAMINATION_REPORT_THRESHOLDS,
        "autoRejectThresholds": CONTAMINATION_AUTOREJECT_THRESHOLDS,
    }
    provisional_bundle = ContaminationScanBundle.model_validate(bundle_payload)
    bundle_payload["artifactSha256"] = artifact_sha256(provisional_bundle)
    bundle = ContaminationScanBundle.model_validate(bundle_payload)
    case_rows = sorted(cases, key=lambda row: str(row["caseId"]))
    label_ledger = [
        {
            "case_id": row["caseId"],
            "adjudicator_person_commitment_sha256s": sorted(
                row["adjudicatorPersonCommitmentSha256s"]
            ),
            "label_event_sha256s": sorted(row["labelEventSha256s"]),
            "relation": row["relation"],
            "expected_leak": row["expectedLeak"],
        }
        for row in case_rows
    ]
    artifact_payload = {
        "schemaVersion": CONTAMINATION_CALIBRATION_SCHEMA_VERSION,
        "artifactSha256": "0" * 64,
        "scanBundleSha256": bundle.artifact_sha256,
        "implementationSha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "containerImageDigest": CONTAINER_DIGEST,
        "cases": cases,
        "caseSetSha256": canonical_sha256(case_rows),
        "labelLedgerRootSha256": canonical_sha256(label_ledger),
        "minimumPrecision": MINIMUM_PRECISION,
        "minimumRecall": MINIMUM_RECALL,
        "minimumParaphraseRecall": MINIMUM_PARAPHRASE_RECALL,
        "calibratedAt": OBSERVED_AT.isoformat(),
        "status": "verified",
    }
    provisional = ContaminationCalibrationArtifact.model_validate(artifact_payload)
    artifact_payload["artifactSha256"] = artifact_sha256(provisional)
    return ContaminationCalibrationArtifact.model_validate(artifact_payload), bundle


def test_labeled_contamination_calibration_recovers_exact_paraphrase_and_negative_cases() -> None:
    artifact, bundle = _calibration_fixture()

    receipt = verify_contamination_calibration(
        artifact,
        scan_bundle=bundle,
        expected_container_image_digest=CONTAINER_DIGEST,
    )

    assert receipt["status"] == "verified"
    assert receipt["case_count"] == 150
    assert receipt["precision_milli"] >= 950
    assert receipt["recall_milli"] >= 900
    assert receipt["paraphrase_recall_milli"] >= 850


def test_calibration_fails_closed_on_container_or_bundle_drift() -> None:
    artifact, bundle = _calibration_fixture()

    with pytest.raises(ContaminationCalibrationError, match="container drifted"):
        verify_contamination_calibration(
            artifact,
            scan_bundle=bundle,
            expected_container_image_digest="sha256:" + "f" * 64,
        )
    drifted = artifact.model_copy(
        update={"artifact_sha256": "0" * 64, "scan_bundle_sha256": "e" * 64}
    )
    drifted = drifted.model_copy(update={"artifact_sha256": artifact_sha256(drifted)})
    with pytest.raises(ContaminationCalibrationError, match="another scan bundle"):
        verify_contamination_calibration(
            drifted,
            scan_bundle=bundle,
            expected_container_image_digest=CONTAINER_DIGEST,
        )
