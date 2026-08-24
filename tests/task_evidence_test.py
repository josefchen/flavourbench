from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime

import pytest

from flavourbench.task_evidence import (
    CONTAMINATION_AUDIT_SCHEMA_VERSION,
    CONTAMINATION_AUTOREJECT_THRESHOLDS,
    CONTAMINATION_REPORT_THRESHOLDS,
    CONTAMINATION_SCAN_BUNDLE_SCHEMA_VERSION,
    TASK_EVIDENCE_IMPLEMENTATION_SHA256,
    VALIDATOR_CONTRACT_SCHEMA_VERSION,
    VALIDATOR_DSL_VERSION,
    ContaminationScanBundle,
    TaskContaminationAuditArtifact,
    TaskEvidenceError,
    TaskValidatorContractArtifact,
    artifact_sha256,
    canonical_sha256,
    evaluate_contract,
    normalize_text,
    normalized_prompt_sha256,
    replay_contamination_scan,
    verify_contamination_audit,
    verify_validator_contract,
)

PROMPT = "Design a pear and miso dish using 8–12 g salt; toast buckwheat before mixing."
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()
CONTAINER_DIGEST = "sha256:" + "a" * 64


def _contract_payload() -> dict:
    fixtures = [
        {
            "fixtureId": "fixture_good",
            "responseText": (
                "Brown 20 g butter. Use 10 g salt. Toast buckwheat, then mix it into the "
                "pear and miso. The pairing evidence is suggestive and not causal."
            ),
            "expectedRuleStatus": {
                "bridge_butter": "pass",
                "salt_range": "pass",
                "step_order": "pass",
                "evidence_bounds": "pass",
                "no_causal_claim": "pass",
            },
        },
        {
            "fixtureId": "fixture_bad",
            "responseText": (
                "Mix the pear and miso first, then toast buckwheat. Use 30 g salt without "
                "butter. The data proves this combination will work."
            ),
            "expectedRuleStatus": {
                "bridge_butter": "fail",
                "salt_range": "fail",
                "step_order": "fail",
                "evidence_bounds": "fail",
                "no_causal_claim": "fail",
            },
        },
    ]
    payload = {
        "schemaVersion": VALIDATOR_CONTRACT_SCHEMA_VERSION,
        "artifactSha256": "0" * 64,
        "taskPublicId": "fb-s1-001",
        "taskFamily": "composition",
        "taskRevision": 1,
        "promptSha256": PROMPT_SHA256,
        "objectiveScope": "executable_subset",
        "validatorDslVersion": VALIDATOR_DSL_VERSION,
        "evaluatorImplementationSha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "validatorContainerImageDigest": CONTAINER_DIGEST,
        "rules": [
            {
                "kind": "required_entity",
                "ruleId": "bridge_butter",
                "description": "Use butter as the declared bridge ingredient.",
                "aliases": ["butter", "brown butter"],
            },
            {
                "kind": "numeric_range",
                "ruleId": "salt_range",
                "description": "Keep salt within the task range.",
                "anchorAliases": ["salt"],
                "unitGroup": "mass_g",
                "minimum": "8",
                "maximum": "12",
            },
            {
                "kind": "ordered_steps",
                "ruleId": "step_order",
                "description": "Toast buckwheat before mixing it into the dish.",
                "steps": [["toast buckwheat"], ["mix"]],
            },
            {
                "kind": "evidence_calibration",
                "ruleId": "evidence_bounds",
                "description": "Qualify suggestive evidence and avoid causal overclaiming.",
                "qualifierPhrases": ["suggestive", "not causal", "does not prove"],
                "overclaimPhrases": ["data proves", "data confirms"],
            },
            {
                "kind": "prohibited_claim",
                "ruleId": "no_causal_claim",
                "description": "Do not present pairing evidence as proof.",
                "phrases": ["data proves", "data confirms"],
            },
        ],
        "fixtures": fixtures,
        "fixtureSetSha256": canonical_sha256(fixtures),
        "status": "verified",
        "verifiedAt": datetime(2026, 8, 1, tzinfo=UTC).isoformat(),
        "verifierReviewerId": "reviewer-three",
    }
    provisional = TaskValidatorContractArtifact.model_validate(payload)
    payload["artifactSha256"] = artifact_sha256(provisional)
    return payload


def _contract() -> TaskValidatorContractArtifact:
    return TaskValidatorContractArtifact.model_validate(_contract_payload())


def _scan_bundle(*, leaked_prompt: bool = False) -> ContaminationScanBundle:
    observed_at = "2026-08-01T00:00:00Z"
    benchmark_text = (
        PROMPT
        if leaked_prompt
        else "A technical note on sourdough fermentation, rye starters, and oven spring."
    )
    web_text = (
        PROMPT
        if leaked_prompt
        else "A captured result about pruning pear orchards and monitoring tree health."
    )
    query_sha256 = normalized_prompt_sha256(PROMPT)
    benchmark_record = {
        "sourceReferenceSha256": "1" * 64,
        "sourceClass": "benchmark_corpus",
        "text": benchmark_text,
        "textSha256": hashlib.sha256(benchmark_text.encode()).hexdigest(),
        "querySha256": None,
        "firstPublishedAt": "2025-01-01T00:00:00Z",
        "capturedAt": observed_at,
    }
    web_record = {
        "sourceReferenceSha256": "2" * 64,
        "sourceClass": "web_snapshot",
        "text": web_text,
        "textSha256": hashlib.sha256(web_text.encode()).hexdigest(),
        "querySha256": query_sha256,
        "firstPublishedAt": "2025-01-01T00:00:00Z",
        "capturedAt": observed_at,
    }
    payload = {
        "schemaVersion": CONTAMINATION_SCAN_BUNDLE_SCHEMA_VERSION,
        "artifactSha256": "0" * 64,
        "createdAt": observed_at,
        "benchmarkSnapshotSha256": canonical_sha256([benchmark_record]),
        "webSnapshotSha256": canonical_sha256([web_record]),
        "benchmarkRecords": [benchmark_record],
        "webRecords": [web_record],
        "webCollectionReceipts": [
            {
                "querySha256": query_sha256,
                "provider": "test-search-provider",
                "providerContractSha256": "3" * 64,
                "rawResponseSha256": "4" * 64,
                "resultRecordSetSha256": canonical_sha256(["2" * 64]),
                "collectedAt": observed_at,
            }
        ],
        "semanticMethod": "distributional-random-indexing-v1",
        "reportThresholds": CONTAMINATION_REPORT_THRESHOLDS,
        "autoRejectThresholds": CONTAMINATION_AUTOREJECT_THRESHOLDS,
    }
    provisional = ContaminationScanBundle.model_validate(payload)
    payload["artifactSha256"] = artifact_sha256(provisional)
    return ContaminationScanBundle.model_validate(payload)


def _audit_payload(
    *,
    bundle: ContaminationScanBundle | None = None,
    conclusion: str = "pass",
) -> dict:
    bundle = bundle or _scan_bundle()
    query_sha256 = normalized_prompt_sha256(PROMPT)
    observed_at = datetime(2026, 8, 1, tzinfo=UTC)
    methods, raw_hits = replay_contamination_scan(
        PROMPT,
        bundle,
        completed_at=observed_at,
    )
    hits = [
        {
            "method": hit["method"],
            "sourceReferenceSha256": hit["source_reference_sha256"],
            "matchedTextSha256": hit["matched_text_sha256"],
            "similarityMilli": hit["similarity_milli"],
            "disposition": "confirmed_leak" if conclusion == "reject" else "unrelated",
            "dispositionReviewerId": "reviewer-four",
        }
        for hit in raw_hits
    ]
    payload = {
        "schemaVersion": CONTAMINATION_AUDIT_SCHEMA_VERSION,
        "artifactSha256": "0" * 64,
        "taskPublicId": "fb-s1-001",
        "taskFamily": "composition",
        "taskRevision": 1,
        "promptSha256": PROMPT_SHA256,
        "normalizedPromptSha256": query_sha256,
        "auditImplementationSha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "scanBundleSha256": bundle.artifact_sha256,
        "auditContainerImageDigest": CONTAINER_DIGEST,
        "methods": methods,
        "hits": hits,
        "conclusion": conclusion,
        "auditorReviewerId": "reviewer-four",
        "observedAt": observed_at.isoformat(),
    }
    provisional = TaskContaminationAuditArtifact.model_validate(payload)
    payload["artifactSha256"] = artifact_sha256(provisional)
    return payload


def test_validator_contract_reproduces_golden_positive_and_negative_fixtures() -> None:
    contract = _contract()

    receipt = verify_validator_contract(
        contract,
        task_public_id="fb-s1-001",
        task_family="composition",
        task_revision=1,
        prompt_sha256=PROMPT_SHA256,
        objective_validator_possible=True,
        expected_container_image_digest=CONTAINER_DIGEST,
    )

    assert receipt["status"] == "verified"
    assert receipt["fixture_count"] == 2
    assert receipt["rule_count"] == 5
    assert len(receipt["receipt_sha256"]) == 64


def test_validator_mutations_detect_units_negation_order_omission_and_overclaim() -> None:
    contract = _contract()
    good = evaluate_contract(
        contract,
        (
            "Use brown butter and 0.01 kg salt. Toast buckwheat before you mix the dish. "
            "The evidence is suggestive; it does not prove the result."
        ),
    )
    bad = evaluate_contract(
        contract,
        (
            "Mix first and toast buckwheat later. Add 0.03 kg salt without butter. "
            "The data confirms the dish will succeed."
        ),
    )

    assert good["status"] == "pass"
    assert bad["status"] == "fail"
    assert {row["rule_id"] for row in bad["rule_results"] if row["status"] == "fail"} == {
        "bridge_butter",
        "salt_range",
        "step_order",
        "evidence_bounds",
        "no_causal_claim",
    }


def test_ratio_and_temperature_rules_use_canonical_units() -> None:
    payload = _contract_payload()
    fixtures = [
        {
            "fixtureId": "fixture_good",
            "responseText": "Use 2 parts miso to 1 part water and roast at 400 °F.",
            "expectedRuleStatus": {"miso_ratio": "pass", "oven_range": "pass"},
        },
        {
            "fixtureId": "fixture_bad",
            "responseText": "Use 1 part miso to 4 parts water and roast at 120 °C.",
            "expectedRuleStatus": {"miso_ratio": "fail", "oven_range": "fail"},
        },
    ]
    payload["rules"] = [
        {
            "kind": "ratio_range",
            "ruleId": "miso_ratio",
            "description": "Keep the miso-to-water ratio between 1.5 and 2.5.",
            "numeratorAliases": ["miso"],
            "denominatorAliases": ["water"],
            "minimumRatio": "1.5",
            "maximumRatio": "2.5",
        },
        {
            "kind": "numeric_range",
            "ruleId": "oven_range",
            "description": "Roast between 195 and 210 degrees Celsius.",
            "anchorAliases": ["roast"],
            "unitGroup": "temperature_c",
            "minimum": "195",
            "maximum": "210",
        },
    ]
    payload["fixtures"] = fixtures
    payload["fixtureSetSha256"] = canonical_sha256(fixtures)
    provisional = TaskValidatorContractArtifact.model_validate(payload)
    payload["artifactSha256"] = artifact_sha256(provisional)
    contract = TaskValidatorContractArtifact.model_validate(payload)

    receipt = verify_validator_contract(
        contract,
        task_public_id="fb-s1-001",
        task_family="composition",
        task_revision=1,
        prompt_sha256=PROMPT_SHA256,
        objective_validator_possible=True,
        expected_container_image_digest=CONTAINER_DIGEST,
    )

    assert receipt["status"] == "verified"


def test_random_or_cross_task_validator_hashes_fail_closed() -> None:
    payload = _contract_payload()
    payload["artifactSha256"] = "f" * 64
    contract = TaskValidatorContractArtifact.model_validate(payload)

    with pytest.raises(TaskEvidenceError, match="digest mismatch"):
        verify_validator_contract(
            contract,
            task_public_id="fb-s1-001",
            task_family="composition",
            task_revision=1,
            prompt_sha256=PROMPT_SHA256,
            objective_validator_possible=True,
            expected_container_image_digest=CONTAINER_DIGEST,
        )

    contract = _contract()
    with pytest.raises(TaskEvidenceError, match="different task revision"):
        verify_validator_contract(
            contract,
            task_public_id="fb-s1-002",
            task_family="composition",
            task_revision=1,
            prompt_sha256=PROMPT_SHA256,
            objective_validator_possible=True,
            expected_container_image_digest=CONTAINER_DIGEST,
        )


def test_contamination_audit_resolves_all_methods_and_rejects_reuse_or_leaks() -> None:
    scan_bundle = _scan_bundle()
    audit = TaskContaminationAuditArtifact.model_validate(
        _audit_payload(bundle=scan_bundle)
    )
    receipt = verify_contamination_audit(
        audit,
        scan_bundle=scan_bundle,
        prompt=PROMPT,
        task_public_id="fb-s1-001",
        task_family="composition",
        task_revision=1,
        prompt_sha256=PROMPT_SHA256,
        expected_container_image_digest=CONTAINER_DIGEST,
        forbidden_reviewer_ids={"author-one", "reviewer-one", "reviewer-two"},
    )

    assert receipt["status"] == "verified"
    assert receipt["methods"] == ["exact", "fuzzy", "ngram", "semantic", "web"]

    with pytest.raises(TaskEvidenceError, match="different task revision"):
        verify_contamination_audit(
            audit,
            scan_bundle=scan_bundle,
            prompt=PROMPT,
            task_public_id="fb-s1-002",
            task_family="composition",
            task_revision=1,
            prompt_sha256=PROMPT_SHA256,
            expected_container_image_digest=CONTAINER_DIGEST,
            forbidden_reviewer_ids=set(),
        )

    leaked_bundle = _scan_bundle(leaked_prompt=True)
    rejected = TaskContaminationAuditArtifact.model_validate(
        _audit_payload(bundle=leaked_bundle, conclusion="reject")
    )
    with pytest.raises(TaskEvidenceError, match="high-similarity contamination"):
        verify_contamination_audit(
            rejected,
            scan_bundle=leaked_bundle,
            prompt=PROMPT,
            task_public_id="fb-s1-001",
            task_family="composition",
            task_revision=1,
            prompt_sha256=PROMPT_SHA256,
            expected_container_image_digest=CONTAINER_DIGEST,
            forbidden_reviewer_ids=set(),
        )


def test_artifact_hash_changes_after_any_evidence_byte_changes() -> None:
    original = _contract_payload()
    changed = copy.deepcopy(original)
    changed["rules"][0]["description"] += " Precisely."
    changed["artifactSha256"] = "0" * 64
    changed_model = TaskValidatorContractArtifact.model_validate(changed)

    assert artifact_sha256(changed_model) != original["artifactSha256"]
    assert normalize_text("  Pear\nMISO  ") == "pear miso"
