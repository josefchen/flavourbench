from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from flavourbench.task_evidence import (
    TASK_EVIDENCE_IMPLEMENTATION_SHA256,
    TaskEvidenceError,
    artifact_sha256,
    canonical_sha256,
)
from flavourbench.validator_calibration import (
    MINIMUM_VIOLATION_PRECISION,
    MINIMUM_VIOLATION_RECALL,
    VALIDATOR_CALIBRATION_SCHEMA_VERSION,
    ValidatorCalibrationArtifact,
    load_validator_calibration,
    verify_validator_calibration,
)

CONTAINER_DIGEST = f"sha256:{'9' * 64}"
RULE_KINDS = (
    "required_entity",
    "prohibited_claim",
    "numeric_range",
    "ratio_range",
    "ordered_steps",
    "evidence_calibration",
)


def _rule(kind: str, ordinal: int) -> dict:
    base = {
        "kind": kind,
        "ruleId": f"{kind.replace('_', '-')}-{ordinal:02d}",
        "description": f"Calibration rule for {kind} case {ordinal}",
    }
    if kind == "required_entity":
        return {**base, "aliases": ["tomato"], "minimumMentions": 1}
    if kind == "prohibited_claim":
        return {
            **base,
            "phrases": ["guaranteed safe"],
            "negationAware": True,
        }
    if kind == "numeric_range":
        return {
            **base,
            "anchorAliases": ["salt"],
            "unitGroup": "mass_g",
            "minimum": "1",
            "maximum": "2",
            "maxDistanceChars": 40,
        }
    if kind == "ratio_range":
        return {
            **base,
            "numeratorAliases": ["acid"],
            "denominatorAliases": ["oil"],
            "minimumRatio": "0.2",
            "maximumRatio": "0.5",
        }
    if kind == "ordered_steps":
        return {**base, "steps": [["toast"], ["simmer"]]}
    if kind == "evidence_calibration":
        return {
            **base,
            "qualifierPhrases": ["may"],
            "overclaimPhrases": ["proves"],
        }
    raise AssertionError(kind)


def _response(kind: str, expected: str, ordinal: int) -> str:
    suffix = f" Calibration example {ordinal}."
    if kind == "required_entity":
        return ("Use tomato in the sauce." if expected == "pass" else "Use potato.") + suffix
    if kind == "prohibited_claim":
        return (
            "The available evidence remains uncertain."
            if expected == "pass"
            else "This is guaranteed safe."
        ) + suffix
    if kind == "numeric_range":
        return ("Use 1.5 g salt." if expected == "pass" else "Use 5 g salt.") + suffix
    if kind == "ratio_range":
        return (
            "Use 1 part acid to 3 parts oil."
            if expected == "pass"
            else "Use 3 parts acid to 1 part oil."
        ) + suffix
    if kind == "ordered_steps":
        return (
            "Toast the spices, then simmer the sauce."
            if expected == "pass"
            else "Simmer the sauce, then toast the spices."
        ) + suffix
    if kind == "evidence_calibration":
        return (
            "This association may support the hypothesis."
            if expected == "pass"
            else "This association proves the mechanism."
        ) + suffix
    raise AssertionError(kind)


def _case(kind: str, ordinal: int, expected: str) -> dict:
    case_id = f"{kind.replace('_', '-')}-{ordinal:02d}"
    response_text = _response(kind, expected, ordinal)
    response_sha256 = hashlib.sha256(response_text.encode()).hexdigest()
    labels = []
    for labeler in (f"labeler-a-{kind}", f"labeler-b-{kind}"):
        label_payload = {
            "case_id": case_id,
            "response_sha256": response_sha256,
            "rule_id": _rule(kind, ordinal)["ruleId"],
            "rule_kind": kind,
            "labeler_reviewer_id": labeler,
            "decision": expected,
            "blind_to_evaluator_output": True,
            "independent_of_case_author": True,
        }
        labels.append(
            {
                "labelerReviewerId": labeler,
                "decision": expected,
                "labelEventSha256": canonical_sha256(label_payload),
                "blindToEvaluatorOutput": True,
                "independentOfCaseAuthor": True,
            }
        )
    return {
        "caseId": case_id,
        "caseAuthorReviewerId": f"author-{kind}",
        "mutationClass": "valid_control" if expected == "pass" else _mutation(kind),
        "rule": _rule(kind, ordinal),
        "responseText": response_text,
        "responseSha256": response_sha256,
        "expectedStatus": expected,
        "labels": labels,
    }


def _mutation(kind: str) -> str:
    return {
        "required_entity": "omission",
        "prohibited_claim": "lexical_adversary",
        "numeric_range": "numeric_boundary",
        "ratio_range": "ratio_boundary",
        "ordered_steps": "step_reversal",
        "evidence_calibration": "causal_overclaim",
    }[kind]


def _artifact(*, override_cases: list[dict] | None = None) -> ValidatorCalibrationArtifact:
    cases = override_cases or [
        _case(kind, ordinal, "pass" if ordinal < 10 else "fail")
        for kind in RULE_KINDS
        for ordinal in range(20)
    ]
    case_set = [
        {key: value for key, value in case.items() if key != "labels"}
        for case in sorted(cases, key=lambda row: row["caseId"])
    ]
    label_set = [
        {
            "caseId": case["caseId"],
            "labels": sorted(
                case["labels"],
                key=lambda row: row["labelerReviewerId"],
            ),
        }
        for case in sorted(cases, key=lambda row: row["caseId"])
    ]
    label_events = sorted(
        label["labelEventSha256"] for case in cases for label in case["labels"]
    )
    value = ValidatorCalibrationArtifact.model_validate(
        {
            "schemaVersion": VALIDATOR_CALIBRATION_SCHEMA_VERSION,
            "artifactSha256": "0" * 64,
            "createdAt": datetime(2026, 8, 1, tzinfo=UTC).isoformat(),
            "evaluatorImplementationSha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
            "validatorContainerImageDigest": CONTAINER_DIGEST,
            "minimumViolationPrecision": MINIMUM_VIOLATION_PRECISION,
            "minimumViolationRecall": MINIMUM_VIOLATION_RECALL,
            "cases": cases,
            "caseSetSha256": canonical_sha256(case_set),
            "labelSetSha256": canonical_sha256(label_set),
            "labelLedgerRootSha256": canonical_sha256(label_events),
            "status": "sealed",
        }
    )
    return value.model_copy(update={"artifact_sha256": artifact_sha256(value)})


def test_blind_mutation_calibration_passes_every_rule_family() -> None:
    artifact = _artifact()

    receipt = verify_validator_calibration(
        artifact,
        expected_container_image_digest=CONTAINER_DIGEST,
    )

    assert receipt["status"] == "verified"
    assert receipt["case_count"] == 120
    assert len(receipt["metrics_by_rule_kind"]) == 6
    for metrics in receipt["metrics_by_rule_kind"].values():
        assert metrics["violation_precision"] == 1.0
        assert metrics["violation_recall"] == 1.0


def test_per_rule_family_recall_failure_blocks_calibration() -> None:
    cases = [
        _case(kind, ordinal, "pass" if ordinal < 10 else "fail")
        for kind in RULE_KINDS
        for ordinal in range(20)
    ]
    for ordinal in (10, 11):
        index = next(
            index
            for index, case in enumerate(cases)
            if case["caseId"] == f"required-entity-{ordinal:02d}"
        )
        case = _case("required_entity", ordinal, "fail")
        case["responseText"] = f"Use tomato despite the violation label. Case {ordinal}."
        case["responseSha256"] = hashlib.sha256(case["responseText"].encode()).hexdigest()
        for label in case["labels"]:
            label["labelEventSha256"] = canonical_sha256(
                {
                    "case_id": case["caseId"],
                    "response_sha256": case["responseSha256"],
                    "rule_id": case["rule"]["ruleId"],
                    "rule_kind": case["rule"]["kind"],
                    "labeler_reviewer_id": label["labelerReviewerId"],
                    "decision": "fail",
                    "blind_to_evaluator_output": True,
                    "independent_of_case_author": True,
                }
            )
        cases[index] = case
    artifact = _artifact(override_cases=cases)

    with pytest.raises(TaskEvidenceError, match="threshold failed for required_entity"):
        verify_validator_calibration(
            artifact,
            expected_container_image_digest=CONTAINER_DIGEST,
        )


def test_label_event_hash_and_reviewer_independence_are_schema_enforced() -> None:
    artifact = _artifact()
    payload = artifact.model_dump(mode="json", by_alias=True)
    payload["cases"][0]["labels"][0]["labelEventSha256"] = "f" * 64

    with pytest.raises(ValidationError, match="labelEventSha256"):
        ValidatorCalibrationArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json", by_alias=True)
    payload["cases"][0]["labels"][0]["labelerReviewerId"] = payload["cases"][0][
        "caseAuthorReviewerId"
    ]
    with pytest.raises(ValidationError, match="author cannot label"):
        ValidatorCalibrationArtifact.model_validate(payload)


def test_content_addressed_calibration_load_fails_closed(tmp_path: Path) -> None:
    artifact = _artifact()
    path = tmp_path / "validator-calibration.json"
    path.write_text(
        json.dumps(artifact.model_dump(mode="json", by_alias=True)),
        encoding="utf-8",
    )

    loaded, receipt = load_validator_calibration(
        path,
        expected_sha256=artifact.artifact_sha256,
        expected_container_image_digest=CONTAINER_DIGEST,
    )
    assert loaded.artifact_sha256 == artifact.artifact_sha256
    assert receipt["status"] == "verified"

    with pytest.raises(TaskEvidenceError, match="configured digest mismatch"):
        load_validator_calibration(
            path,
            expected_sha256="a" * 64,
            expected_container_image_digest=CONTAINER_DIGEST,
        )
