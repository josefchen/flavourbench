from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import select, text

import flavourbench.main as main_module
import flavourbench.task_evidence_registry as task_evidence_registry_module
from flavourbench.config import Settings
from flavourbench.construct_blueprint import BLUEPRINT_SHA256
from flavourbench.database import init_database, session_scope
from flavourbench.models import (
    Battle,
    ExpertReviewer,
    LeaderboardSnapshot,
    RunEvent,
    Season,
    Task,
    TaskEvidenceArtifact,
)
from flavourbench.schemas import (
    ConfirmatoryTaskCreate,
    TaskCandidateAdjudicationCreate,
    TaskCandidateBlindValidityCreate,
    TaskCandidateReconciliationCreate,
    TaskChallengeAdjudicationCreate,
    TaskChallengeCreate,
    TaskContaminationAuditReviewCreate,
    TaskValidatorContractReviewCreate,
)
from flavourbench.task_contributor_protocol import (
    PROTOCOL_SCOPE as TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
)
from flavourbench.task_contributor_protocol import (
    PROTOCOL_SHA256 as TASK_CONTRIBUTOR_PROTOCOL_SHA256,
)
from flavourbench.task_contributor_protocol import (
    PROTOCOL_VERSION as TASK_CONTRIBUTOR_PROTOCOL_VERSION,
)
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
    TaskValidatorContractArtifact,
    artifact_sha256,
    canonical_sha256,
    normalized_prompt_sha256,
    replay_contamination_scan,
    task_evidence_review_sha256,
    task_evidence_root_sha256,
    verify_contamination_audit,
    verify_validator_contract,
)
from flavourbench.task_evidence_registry import task_validator_receipt_for_battle
from flavourbench.task_lifecycle import (
    TaskLifecycleError,
    record_task_first_use,
    verify_task_lifecycle,
)

CONTAINER_DIGEST = "sha256:" + "d" * 64
CALIBRATION_SHA256 = "8" * 64
CALIBRATION_RECEIPT_SHA256 = "9" * 64
CONTAMINATION_CALIBRATION_SHA256 = "6" * 64
CONTAMINATION_CALIBRATION_RECEIPT_SHA256 = "7" * 64
CONSTRUCT_CELL = "bridge_ingredient_reasoning"


@pytest.fixture(autouse=True)
def _verified_validator_calibration(monkeypatch) -> None:
    evidence = (
        SimpleNamespace(artifact_sha256=CALIBRATION_SHA256),
        {
            "receipt_sha256": CALIBRATION_RECEIPT_SHA256,
            "case_count": 120,
            "status": "verified",
        },
    )
    monkeypatch.setattr(
        main_module,
        "_validator_calibration",
        lambda: evidence,
    )
    monkeypatch.setattr(
        task_evidence_registry_module,
        "verify_validator_calibration",
        lambda *_args, **_kwargs: evidence[1],
    )
    contamination_evidence = (
        SimpleNamespace(artifact_sha256=CONTAMINATION_CALIBRATION_SHA256),
        {
            "receipt_sha256": CONTAMINATION_CALIBRATION_RECEIPT_SHA256,
            "case_count": 150,
            "precision_milli": 960,
            "recall_milli": 920,
            "paraphrase_recall_milli": 880,
            "status": "verified",
        },
    )
    monkeypatch.setattr(
        main_module,
        "_contamination_calibration",
        lambda *_args, **_kwargs: contamination_evidence,
    )
    monkeypatch.setattr(
        task_evidence_registry_module,
        "load_contamination_calibration",
        lambda *_args, **_kwargs: contamination_evidence,
    )
    monkeypatch.setattr(
        task_evidence_registry_module,
        "verify_contamination_calibration",
        lambda *_args, **_kwargs: contamination_evidence[1],
    )


def _main_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _blind_review_payload() -> dict[str, object]:
    return TaskCandidateBlindValidityCreate.model_validate(
        {
            "decision": "valid",
            "constructFit": True,
            "contextComplete": True,
            "coherentQuestion": True,
            "generalTrackScope": True,
            "answerLeakageAbsent": True,
            "discriminationValue": True,
            "familyClassification": "composition",
            "constructCellClassification": CONSTRUCT_CELL,
            "difficultyTierClassification": "foundation",
            "independentSolutionOutline": (
                "Build the preparation around pear sweetness, white-miso savouriness, and "
                "buckwheat toast while preserving a deliberate texture contrast."
            ),
            "successCriteria": [
                "Uses pear, white miso, and buckwheat in a coherent preparation.",
                "Explains the culinary bridge and a practical order of operations.",
            ],
            "disqualifyingErrors": [
                "Omits a named ingredient or offers no executable cooking method."
            ],
            "issueTags": [],
            "criteriaAuthoredByReviewer": True,
            "authorPackNotSeen": True,
            "modelOutputsNotConsulted": True,
            "note": "The prompt is self-contained and tests practical composition reasoning.",
        }
    ).model_dump(mode="json")


def _reconciliation_payload() -> dict[str, object]:
    return TaskCandidateReconciliationCreate.model_validate(
        {
            "decision": "approve",
            "authorPackAdequacy": "adequate",
            "constructLabelAgreement": True,
            "difficultyLabelAgreement": True,
            "constraintSetAdequate": True,
            "solutionOutlineAdequate": True,
            "validatorPlanAdequate": True,
            "rightsBasisCredible": True,
            "successCriteria": _blind_review_payload()["success_criteria"],
            "permittedVariations": [
                "Equivalent techniques are acceptable when their practical tradeoffs are stated."
            ],
            "disqualifyingErrors": _blind_review_payload()["disqualifying_errors"],
            "objectiveChecks": ["Every explicitly named ingredient is addressed."],
            "issueTags": [],
            "criteriaAuthoredByReviewer": True,
            "independentOfAuthor": True,
            "modelOutputsNotConsulted": True,
            "note": "The author pack agrees with the independently sealed task interpretation.",
        }
    ).model_dump(mode="json")


def _adjudication_payload() -> dict[str, object]:
    return TaskCandidateAdjudicationCreate.model_validate(
        {
            "decision": "approve",
            "family": "composition",
            "constructCellId": CONSTRUCT_CELL,
            "difficultyTier": "foundation",
            "successCriteria": _blind_review_payload()["success_criteria"],
            "permittedVariations": _reconciliation_payload()["permitted_variations"],
            "disqualifyingErrors": _blind_review_payload()["disqualifying_errors"],
            "objectiveChecks": _reconciliation_payload()["objective_checks"],
            "criteriaAuthoredByAdjudicator": True,
            "independentOfAuthorAndReviewers": True,
            "modelOutputsNotConsulted": True,
            "note": (
                "The two independent solutions converge, and the merged criterion pack "
                "preserves their shared mechanism and execution requirements."
            ),
        }
    ).model_dump(mode="json")


def _sealed_validator(
    *,
    task_id: str,
    family: str,
    prompt_sha256: str,
    verifier_id: str,
    objective_validator_possible: bool = False,
) -> TaskValidatorContractArtifact:
    rules = (
        [
            {
                "kind": "required_entity",
                "ruleId": "include-pear",
                "description": "The response must use pear.",
                "aliases": ["pear", "pears"],
                "minimumMentions": 1,
            }
        ]
        if objective_validator_possible
        else []
    )
    fixtures = (
        [
            {
                "fixtureId": "pear-positive",
                "responseText": "Roast the pear before plating.",
                "expectedRuleStatus": {"include-pear": "pass"},
            },
            {
                "fixtureId": "pear-negative",
                "responseText": "Use only apple and oats.",
                "expectedRuleStatus": {"include-pear": "fail"},
            },
        ]
        if objective_validator_possible
        else []
    )
    payload = {
        "schemaVersion": VALIDATOR_CONTRACT_SCHEMA_VERSION,
        "artifactSha256": "0" * 64,
        "taskPublicId": task_id,
        "taskFamily": family,
        "taskRevision": 1,
        "promptSha256": prompt_sha256,
        "objectiveScope": ("executable_subset" if objective_validator_possible else "human_only"),
        "humanOnlyReason": (
            None
            if objective_validator_possible
            else "This task has no deterministic sensory target; expert review remains primary."
        ),
        "validatorDslVersion": VALIDATOR_DSL_VERSION,
        "evaluatorImplementationSha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "validatorContainerImageDigest": CONTAINER_DIGEST,
        "rules": rules,
        "fixtures": fixtures,
        "fixtureSetSha256": canonical_sha256(fixtures),
        "status": "verified",
        "verifiedAt": datetime(2026, 8, 1, tzinfo=UTC).isoformat(),
        "verifierReviewerId": verifier_id,
    }
    provisional = TaskValidatorContractArtifact.model_validate(payload)
    payload["artifactSha256"] = artifact_sha256(provisional)
    return TaskValidatorContractArtifact.model_validate(payload)


def _scan_bundle(prompt: str) -> ContaminationScanBundle:
    observed_at = "2026-08-01T00:00:00Z"
    query_sha256 = normalized_prompt_sha256(prompt)
    benchmark_text = "A benchmark item about sourdough fermentation and rye starter maintenance."
    web_text = "A captured web result about pruning pear orchards and monitoring tree health."
    benchmark_record = {
        "sourceReferenceSha256": hashlib.sha256(b"benchmark-unrelated-source").hexdigest(),
        "sourceClass": "benchmark_corpus",
        "text": benchmark_text,
        "textSha256": hashlib.sha256(benchmark_text.encode()).hexdigest(),
        "querySha256": None,
        "firstPublishedAt": "2025-01-01T00:00:00Z",
        "capturedAt": observed_at,
    }
    web_record = {
        "sourceReferenceSha256": hashlib.sha256(
            f"web-unrelated-source:{query_sha256}".encode()
        ).hexdigest(),
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
                "resultRecordSetSha256": canonical_sha256([web_record["sourceReferenceSha256"]]),
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


def _sealed_audit(
    *,
    task_id: str,
    family: str,
    prompt: str,
    prompt_sha256: str,
    auditor_id: str,
    scan_bundle: ContaminationScanBundle,
) -> TaskContaminationAuditArtifact:
    query_sha256 = normalized_prompt_sha256(prompt)
    observed_at = datetime(2026, 8, 1, tzinfo=UTC)
    methods, raw_hits = replay_contamination_scan(
        prompt,
        scan_bundle,
        completed_at=observed_at,
    )
    hits = [
        {
            "method": hit["method"],
            "sourceReferenceSha256": hit["source_reference_sha256"],
            "matchedTextSha256": hit["matched_text_sha256"],
            "similarityMilli": hit["similarity_milli"],
            "disposition": "unrelated",
            "dispositionReviewerId": auditor_id,
        }
        for hit in raw_hits
    ]
    payload = {
        "schemaVersion": CONTAMINATION_AUDIT_SCHEMA_VERSION,
        "artifactSha256": "0" * 64,
        "taskPublicId": task_id,
        "taskFamily": family,
        "taskRevision": 1,
        "promptSha256": prompt_sha256,
        "normalizedPromptSha256": query_sha256,
        "auditImplementationSha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
        "scanBundleSha256": scan_bundle.artifact_sha256,
        "auditContainerImageDigest": CONTAINER_DIGEST,
        "methods": methods,
        "hits": hits,
        "conclusion": "pass",
        "auditorReviewerId": auditor_id,
        "observedAt": observed_at.isoformat(),
    }
    provisional = TaskContaminationAuditArtifact.model_validate(payload)
    payload["artifactSha256"] = artifact_sha256(provisional)
    return TaskContaminationAuditArtifact.model_validate(payload)


def _seed_task_evidence_context(
    suffix: str,
    *,
    objective_validator_possible: bool = False,
    include_evidence_review_events: bool = True,
) -> tuple[str, ConfirmatoryTaskCreate, ContaminationScanBundle]:
    init_database()
    season_slug = f"evidence-import-{suffix}"
    task_public_id = f"fb-{suffix}"
    family = "composition"
    prompt = f"Create an original pear, white miso, and buckwheat preparation for case {suffix}."
    scan_bundle = _scan_bundle(prompt)
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    candidate_id = str(uuid.uuid4())
    candidate_record_sha256 = hashlib.sha256(f"candidate-{suffix}".encode()).hexdigest()
    author_id = f"author-{suffix}"
    acceptance_event_id = str(uuid.uuid4())
    reviewer_ids = [f"{role}-{suffix}" for role in ("reviewer-one", "reviewer-two")]
    adjudicator_id = f"adjudicator-{suffix}"
    verifier_id = f"verifier-{suffix}"
    auditor_id = f"auditor-{suffix}"
    blind_payload = _blind_review_payload()
    reconciliation_payload = _reconciliation_payload()
    blind_review_sha256s = [
        _main_sha256(
            {
                "candidate_id": candidate_id,
                "candidate_record_sha256": candidate_record_sha256,
                "reviewer_id": reviewer_id,
                "blind_review": blind_payload,
            }
        )
        for reviewer_id in reviewer_ids
    ]
    reconciliation_sha256s = [
        _main_sha256(
            {
                "candidate_id": candidate_id,
                "candidate_record_sha256": candidate_record_sha256,
                "blind_review_sha256": blind_sha256,
                "reviewer_id": reviewer_id,
                "reconciliation": reconciliation_payload,
            }
        )
        for reviewer_id, blind_sha256 in zip(
            reviewer_ids,
            blind_review_sha256s,
            strict=True,
        )
    ]
    adjudication_payload = _adjudication_payload()
    criterion_pack = {
        key: adjudication_payload[key]
        for key in (
            "family",
            "construct_cell_id",
            "difficulty_tier",
            "success_criteria",
            "permitted_variations",
            "disqualifying_errors",
            "objective_checks",
        )
    }
    criterion_pack_sha256 = _main_sha256(criterion_pack)
    adjudication_sha256 = _main_sha256(
        {
            "candidate_id": candidate_id,
            "candidate_record_sha256": candidate_record_sha256,
            "source_blind_sha256s": sorted(blind_review_sha256s),
            "source_reconciliation_sha256s": sorted(reconciliation_sha256s),
            "adjudicator_id": adjudicator_id,
            "adjudication": adjudication_payload,
            "criterion_pack_sha256": criterion_pack_sha256,
        }
    )
    with session_scope() as session:
        session.add(
            Season(
                slug=season_slug,
                name=f"Evidence import {suffix}",
                status="draft",
                official=False,
                epicure_release_id="development-epicure",
            )
        )
        session.add(
            ExpertReviewer(
                id=author_id,
                reviewer_code=f"code-{author_id}",
                invitation_sha256=hashlib.sha256(f"invite-author-{suffix}".encode()).hexdigest(),
                qualification_json=[family],
                qualification_verified=False,
                cohort="expert_independent",
                profile_json={
                    "admission_pathway": "task_contributor",
                    "task_contributor_status": "active",
                    "raw_identity_retention_prohibited": True,
                    "person_uniqueness_verified": True,
                    "person_uniqueness_method": "admin-witnessed-season-hmac-v1",
                    "person_uniqueness_commitment_sha256": hashlib.sha256(
                        f"person:{author_id}".encode()
                    ).hexdigest(),
                    "person_uniqueness_evidence_sha256": hashlib.sha256(
                        f"person-evidence:{author_id}".encode()
                    ).hexdigest(),
                    "task_contributor_protocol_version": (
                        TASK_CONTRIBUTOR_PROTOCOL_VERSION
                    ),
                    "task_contributor_protocol_sha256": (
                        TASK_CONTRIBUTOR_PROTOCOL_SHA256
                    ),
                    "task_contributor_protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
                    "task_contributor_protocol_accepted": True,
                    "task_contributor_protocol_acceptance_event_id": acceptance_event_id,
                },
                active=True,
            )
        )
        for ordinal, reviewer_id in enumerate(
            [*reviewer_ids, adjudicator_id, verifier_id, auditor_id],
            1,
        ):
            identity_commitment_sha256 = hashlib.sha256(
                f"identity:{reviewer_id}".encode()
            ).hexdigest()
            qualification_evidence_sha256 = hashlib.sha256(
                f"qualification:{reviewer_id}".encode()
            ).hexdigest()
            session.add(
                ExpertReviewer(
                    id=reviewer_id,
                    reviewer_code=f"code-{reviewer_id}",
                    invitation_sha256=hashlib.sha256(
                        f"invite-{ordinal}-{suffix}".encode()
                    ).hexdigest(),
                    qualification_json=[family],
                    qualification_verified=True,
                    cohort="expert_independent",
                    profile_json={
                        "admission_pathway": "development_task_validator",
                        "task_validation_status": "active",
                        "task_adjudication_authorized": reviewer_id == adjudicator_id,
                        "identity_commitment_sha256": identity_commitment_sha256,
                        "qualification_evidence_sha256": qualification_evidence_sha256,
                        "independence_attestation_sha256": hashlib.sha256(
                            f"independence:{reviewer_id}".encode()
                        ).hexdigest(),
                        "verification_record_sha256": hashlib.sha256(
                            f"verification:{reviewer_id}".encode()
                        ).hexdigest(),
                        "identity_commitment_algorithm": "HMAC-SHA256",
                        "person_uniqueness_verified": True,
                        "person_uniqueness_method": "admin-witnessed-season-hmac-v1",
                        "person_uniqueness_commitment_sha256": hashlib.sha256(
                            f"person:{reviewer_id}".encode()
                        ).hexdigest(),
                        "person_uniqueness_evidence_sha256": hashlib.sha256(
                            f"person-evidence:{reviewer_id}".encode()
                        ).hexdigest(),
                        "affiliation_class": "independent_external",
                        "independent_validation_claim": True,
                        "evidence_verified_by_admin": True,
                    },
                    active=True,
                )
            )
        session.add(
            RunEvent(
                id=acceptance_event_id,
                entity_type="task_contributor",
                entity_id=author_id,
                event_type="task_contributor_protocol_accepted",
                payload_json={
                    "schema_version": "flavourbench-task-contributor-protocol-acceptance-v1",
                    "protocol_version": TASK_CONTRIBUTOR_PROTOCOL_VERSION,
                    "protocol_sha256": TASK_CONTRIBUTOR_PROTOCOL_SHA256,
                    "protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
                    "voluntary_participation_accepted": True,
                    "task_contribution_agreement_accepted": True,
                    "human_only_methods_acknowledged": True,
                },
            )
        )
        session.add(
            RunEvent(
                entity_type="task_candidate",
                entity_id=candidate_id,
                event_type="task_candidate_submitted",
                payload_json={
                    "candidate_record_sha256": candidate_record_sha256,
                    "author_reviewer_id": author_id,
                    "family": family,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "construct_blueprint_sha256": BLUEPRINT_SHA256,
                    "construct_cell_id": CONSTRUCT_CELL,
                    "difficulty_tier": "foundation",
                    "subskills": [CONSTRUCT_CELL],
                    "objective_validator_possible": objective_validator_possible,
                    "task_contributor_protocol_version": (
                        TASK_CONTRIBUTOR_PROTOCOL_VERSION
                    ),
                    "task_contributor_protocol_sha256": (
                        TASK_CONTRIBUTOR_PROTOCOL_SHA256
                    ),
                    "task_contributor_protocol_scope": TASK_CONTRIBUTOR_PROTOCOL_SCOPE,
                    "task_contributor_protocol_acceptance_event_id": acceptance_event_id,
                },
            )
        )
        for reviewer_id, blind_sha256, reconciliation_sha256 in zip(
            reviewer_ids,
            blind_review_sha256s,
            reconciliation_sha256s,
            strict=True,
        ):
            identity_commitment_sha256 = hashlib.sha256(
                f"identity:{reviewer_id}".encode()
            ).hexdigest()
            qualification_evidence_sha256 = hashlib.sha256(
                f"qualification:{reviewer_id}".encode()
            ).hexdigest()
            session.add(
                RunEvent(
                    entity_type="task_candidate",
                    entity_id=candidate_id,
                    event_type="task_candidate_blind_validity_recorded",
                    payload_json={
                        **blind_payload,
                        "reviewer_id": reviewer_id,
                        "candidate_record_sha256": candidate_record_sha256,
                        "prompt_sha256": prompt_sha256,
                        "blind_review_sha256": blind_sha256,
                        "idempotency_key_sha256": hashlib.sha256(
                            f"blind-idempotency:{reviewer_id}".encode()
                        ).hexdigest(),
                        "author_pack_visible": False,
                        "model_outputs_visible": False,
                        "independent_review": True,
                        "identity_commitment_sha256": identity_commitment_sha256,
                        "qualification_evidence_sha256": qualification_evidence_sha256,
                        "independence_attestation_sha256": hashlib.sha256(
                            f"independence:{reviewer_id}".encode()
                        ).hexdigest(),
                    },
                )
            )
            session.add(
                RunEvent(
                    entity_type="task_candidate",
                    entity_id=candidate_id,
                    event_type="task_candidate_reconciliation_recorded",
                    payload_json={
                        **reconciliation_payload,
                        "reviewer_id": reviewer_id,
                        "candidate_record_sha256": candidate_record_sha256,
                        "prompt_sha256": prompt_sha256,
                        "blind_review_sha256": blind_sha256,
                        "reconciliation_sha256": reconciliation_sha256,
                        "idempotency_key_sha256": hashlib.sha256(
                            f"reconciliation-idempotency:{reviewer_id}".encode()
                        ).hexdigest(),
                        "author_pack_visible": True,
                        "model_outputs_visible": False,
                        "independent_review": True,
                        "identity_commitment_sha256": identity_commitment_sha256,
                        "qualification_evidence_sha256": qualification_evidence_sha256,
                        "independence_attestation_sha256": hashlib.sha256(
                            f"independence:{reviewer_id}".encode()
                        ).hexdigest(),
                    },
                )
            )
        session.add(
            RunEvent(
                entity_type="task_candidate",
                entity_id=candidate_id,
                event_type="task_candidate_adjudication_recorded",
                payload_json={
                    **adjudication_payload,
                    "reviewer_id": adjudicator_id,
                    "candidate_record_sha256": candidate_record_sha256,
                    "prompt_sha256": prompt_sha256,
                    "source_blind_sha256s": sorted(blind_review_sha256s),
                    "source_reconciliation_sha256s": sorted(reconciliation_sha256s),
                    "criterion_pack_sha256": criterion_pack_sha256,
                    "adjudication_sha256": adjudication_sha256,
                    "idempotency_key_sha256": hashlib.sha256(
                        f"adjudication-idempotency:{adjudicator_id}".encode()
                    ).hexdigest(),
                    "source_reviewer": False,
                    "model_outputs_visible": False,
                    "identity_commitment_sha256": hashlib.sha256(
                        f"identity:{adjudicator_id}".encode()
                    ).hexdigest(),
                    "qualification_evidence_sha256": hashlib.sha256(
                        f"qualification:{adjudicator_id}".encode()
                    ).hexdigest(),
                    "independence_attestation_sha256": hashlib.sha256(
                        f"independence:{adjudicator_id}".encode()
                    ).hexdigest(),
                },
            )
        )

    validator = _sealed_validator(
        task_id=task_public_id,
        family=family,
        prompt_sha256=prompt_sha256,
        verifier_id=verifier_id,
        objective_validator_possible=objective_validator_possible,
    )
    audit = _sealed_audit(
        task_id=task_public_id,
        family=family,
        prompt=prompt,
        prompt_sha256=prompt_sha256,
        auditor_id=auditor_id,
        scan_bundle=scan_bundle,
    )
    validator_receipt = verify_validator_contract(
        validator,
        task_public_id=task_public_id,
        task_family=family,
        task_revision=1,
        prompt_sha256=prompt_sha256,
        objective_validator_possible=objective_validator_possible,
        expected_container_image_digest=CONTAINER_DIGEST,
    )
    contamination_receipt = verify_contamination_audit(
        audit,
        scan_bundle=scan_bundle,
        prompt=prompt,
        task_public_id=task_public_id,
        task_family=family,
        task_revision=1,
        prompt_sha256=prompt_sha256,
        expected_container_image_digest=CONTAINER_DIGEST,
        forbidden_reviewer_ids={author_id, adjudicator_id, *reviewer_ids},
    )
    validator_review_payload = {
        "decision": "approve",
        "task_binding_checked": True,
        "rules_and_fixtures_inspected": True,
        "verification_receipt_reproduced": True,
        "model_outputs_not_consulted": True,
        "independent_of_task_roles": True,
        "note": (
            "I reproduced the contract receipt and inspected its scope, rules, and "
            "positive and negative mutation fixtures without consulting model outputs."
        ),
    }
    contamination_review_payload = {
        "decision": "approve",
        "task_binding_checked": True,
        "replay_receipt_reproduced": True,
        "hit_dispositions_inspected": True,
        "model_outputs_not_consulted": True,
        "independent_of_task_roles": True,
        "note": (
            "I replayed all frozen contamination methods and inspected each hit disposition "
            "without consulting any model response."
        ),
    }
    validator_review_sha256 = task_evidence_review_sha256(
        candidate_id=candidate_id,
        candidate_record_sha256=candidate_record_sha256,
        task_public_id=task_public_id,
        reviewer_id=verifier_id,
        evidence_type="validator_contract",
        artifact_sha256=validator.artifact_sha256,
        verification_receipt_sha256=str(validator_receipt["receipt_sha256"]),
        review=validator_review_payload,
    )
    contamination_review_sha256 = task_evidence_review_sha256(
        candidate_id=candidate_id,
        candidate_record_sha256=candidate_record_sha256,
        task_public_id=task_public_id,
        reviewer_id=auditor_id,
        evidence_type="contamination_audit",
        artifact_sha256=audit.artifact_sha256,
        verification_receipt_sha256=str(contamination_receipt["receipt_sha256"]),
        review=contamination_review_payload,
    )
    if include_evidence_review_events:
        with session_scope() as session:
            for (
                reviewer_id,
                evidence_type,
                artifact_sha256_value,
                receipt_sha256,
                review_payload,
                review_sha256,
                event_type,
            ) in (
                (
                    verifier_id,
                    "validator_contract",
                    validator.artifact_sha256,
                    str(validator_receipt["receipt_sha256"]),
                    validator_review_payload,
                    validator_review_sha256,
                    "task_candidate_validator_contract_verified",
                ),
                (
                    auditor_id,
                    "contamination_audit",
                    audit.artifact_sha256,
                    str(contamination_receipt["receipt_sha256"]),
                    contamination_review_payload,
                    contamination_review_sha256,
                    "task_candidate_contamination_audit_verified",
                ),
            ):
                session.add(
                    RunEvent(
                        entity_type="task_candidate",
                        entity_id=candidate_id,
                        event_type=event_type,
                        payload_json={
                            "candidate_record_sha256": candidate_record_sha256,
                            "task_public_id": task_public_id,
                            "reviewer_id": reviewer_id,
                            "evidence_type": evidence_type,
                            "artifact_sha256": artifact_sha256_value,
                            "verification_receipt_sha256": receipt_sha256,
                            "review": review_payload,
                            "review_event_sha256": review_sha256,
                            "idempotency_key_sha256": hashlib.sha256(
                                f"evidence-review:{evidence_type}:{suffix}".encode()
                            ).hexdigest(),
                            "artifact_visible": True,
                            "model_outputs_visible": False,
                            "independent_of_task_roles": True,
                            "identity_commitment_sha256": hashlib.sha256(
                                f"identity:{reviewer_id}".encode()
                            ).hexdigest(),
                            "qualification_evidence_sha256": hashlib.sha256(
                                f"qualification:{reviewer_id}".encode()
                            ).hexdigest(),
                            "independence_attestation_sha256": hashlib.sha256(
                                f"independence:{reviewer_id}".encode()
                            ).hexdigest(),
                        },
                    )
                )
    validator_review = {
        "reviewerId": verifier_id,
        "evidenceType": "validator_contract",
        "artifactSha256": validator.artifact_sha256,
        "verificationReceiptSha256": validator_receipt["receipt_sha256"],
        "reviewEventSha256": validator_review_sha256,
        "decision": "approve",
        "independentOfTaskRoles": True,
    }
    contamination_review = {
        "reviewerId": auditor_id,
        "evidenceType": "contamination_audit",
        "artifactSha256": audit.artifact_sha256,
        "verificationReceiptSha256": contamination_receipt["receipt_sha256"],
        "reviewEventSha256": contamination_review_sha256,
        "decision": "approve",
        "independentOfTaskRoles": True,
    }
    independent_reviews = [
        {
            "reviewerId": reviewer_id,
            "blindReviewEventSha256": blind_sha256,
            "reconciliationEventSha256": reconciliation_sha256,
            "decision": "approve",
            "independentOfAuthor": True,
        }
        for reviewer_id, blind_sha256, reconciliation_sha256 in zip(
            reviewer_ids,
            blind_review_sha256s,
            reconciliation_sha256s,
            strict=True,
        )
    ]
    adjudication = {
        "adjudicatorReviewerId": adjudicator_id,
        "adjudicationEventSha256": adjudication_sha256,
        "criterionPackSha256": criterion_pack_sha256,
        "decision": "approve",
        "independentOfAuthorAndReviewers": True,
    }
    review_history_sha256 = _main_sha256(
        {
            "candidate_id": candidate_id,
            "candidate_record_sha256": candidate_record_sha256,
            "blind_review_sha256s": sorted(blind_review_sha256s),
            "reconciliation_sha256s": sorted(reconciliation_sha256s),
            "adjudication_sha256": adjudication_sha256,
            "criterion_pack_sha256": criterion_pack_sha256,
        }
    )
    task_record = {
        "public_id": task_public_id,
        "family": family,
        "split": "scored",
        "prompt_sha256": prompt_sha256,
        "revision": 1,
        "construct_blueprint_sha256": BLUEPRINT_SHA256,
        "construct_cell_id": CONSTRUCT_CELL,
        "difficulty_tier": "foundation",
        "human_author_id": author_id,
        "source_candidate_id": candidate_id,
        "candidate_record_sha256": candidate_record_sha256,
        "independent_reviews": [
            {
                "reviewer_id": review["reviewerId"],
                "blind_review_event_sha256": review["blindReviewEventSha256"],
                "reconciliation_event_sha256": review["reconciliationEventSha256"],
                "decision": "approve",
                "independent_of_author": True,
            }
            for review in independent_reviews
        ],
        "adjudication": {
            "adjudicator_reviewer_id": adjudicator_id,
            "adjudication_event_sha256": adjudication_sha256,
            "criterion_pack_sha256": criterion_pack_sha256,
            "decision": "approve",
            "independent_of_author_and_reviewers": True,
        },
        "validator_contract_sha256": validator.artifact_sha256,
        "validator_contract_review": {
            "reviewer_id": verifier_id,
            "evidence_type": "validator_contract",
            "artifact_sha256": validator.artifact_sha256,
            "verification_receipt_sha256": validator_receipt["receipt_sha256"],
            "review_event_sha256": validator_review_sha256,
            "decision": "approve",
            "independent_of_task_roles": True,
        },
        "review_history_sha256": review_history_sha256,
        "contamination_audit_sha256": audit.artifact_sha256,
        "contamination_audit_review": {
            "reviewer_id": auditor_id,
            "evidence_type": "contamination_audit",
            "artifact_sha256": audit.artifact_sha256,
            "verification_receipt_sha256": contamination_receipt["receipt_sha256"],
            "review_event_sha256": contamination_review_sha256,
            "decision": "approve",
            "independent_of_task_roles": True,
        },
    }
    task_record_sha256 = _main_sha256(task_record)
    evidence_root = task_evidence_root_sha256(
        task_record_sha256=task_record_sha256,
        candidate_record_sha256=candidate_record_sha256,
        review_history_sha256=review_history_sha256,
        validator_contract_sha256=validator.artifact_sha256,
        contamination_audit_sha256=audit.artifact_sha256,
        validator_receipt_sha256=str(validator_receipt["receipt_sha256"]),
        contamination_receipt_sha256=str(contamination_receipt["receipt_sha256"]),
        validator_review_event_sha256=validator_review_sha256,
        contamination_review_event_sha256=contamination_review_sha256,
    )
    item = ConfirmatoryTaskCreate.model_validate(
        {
            "publicId": task_public_id,
            "family": family,
            "split": "scored",
            "prompt": prompt,
            "revision": 1,
            "constructBlueprintSha256": BLUEPRINT_SHA256,
            "constructCellId": CONSTRUCT_CELL,
            "difficultyTier": "foundation",
            "humanAuthorId": author_id,
            "sourceCandidateId": candidate_id,
            "candidateRecordSha256": candidate_record_sha256,
            "independentReviews": independent_reviews,
            "adjudication": adjudication,
            "validatorContract": validator.model_dump(mode="json", by_alias=True),
            "validatorContractReview": validator_review,
            "reviewHistorySha256": review_history_sha256,
            "taskRecordSha256": task_record_sha256,
            "contaminationAudit": audit.model_dump(mode="json", by_alias=True),
            "contaminationAuditReview": contamination_review,
            "taskEvidenceRootSha256": evidence_root,
        }
    )
    return season_slug, item, scan_bundle


def _request(item: ConfirmatoryTaskCreate) -> SimpleNamespace:
    manifest_sha256 = _main_sha256(
        {
            "construct_blueprint_sha256": BLUEPRINT_SHA256,
            "validator_calibration_artifact_sha256": CALIBRATION_SHA256,
            "contamination_calibration_artifact_sha256": (CONTAMINATION_CALIBRATION_SHA256),
            "tasks": [
                {
                    "public_id": item.public_id,
                    "task_record_sha256": item.task_record_sha256,
                    "task_evidence_root_sha256": item.task_evidence_root_sha256,
                }
            ],
        }
    )
    return SimpleNamespace(
        tasks=[item],
        bank_manifest_sha256=manifest_sha256,
        validator_calibration_artifact_sha256=CALIBRATION_SHA256,
        contamination_calibration_artifact_sha256=(CONTAMINATION_CALIBRATION_SHA256),
        import_reference="test content-addressed import",
    )


def test_evidence_review_endpoints_seal_two_independent_reproducible_events(
    monkeypatch,
) -> None:
    season_slug, item, scan_bundle = _seed_task_evidence_context(
        "review-endpoints",
        include_evidence_review_events=False,
    )
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(build_image_digest=CONTAINER_DIGEST),
    )
    monkeypatch.setattr(main_module, "_contamination_scan_bundle", lambda: scan_bundle)
    validator_request = TaskValidatorContractReviewCreate.model_validate(
        {
            "validatorContract": item.validator_contract.model_dump(
                mode="json",
                by_alias=True,
            ),
            "decision": "approve",
            "taskBindingChecked": True,
            "rulesAndFixturesInspected": True,
            "verificationReceiptReproduced": True,
            "modelOutputsNotConsulted": True,
            "independentOfTaskRoles": True,
            "note": (
                "I reproduced the contract receipt and inspected its scope, rules, and "
                "positive and negative mutation fixtures without consulting model outputs."
            ),
        }
    )
    contamination_request = TaskContaminationAuditReviewCreate.model_validate(
        {
            "contaminationAudit": item.contamination_audit.model_dump(
                mode="json",
                by_alias=True,
            ),
            "decision": "approve",
            "taskBindingChecked": True,
            "replayReceiptReproduced": True,
            "hitDispositionsInspected": True,
            "modelOutputsNotConsulted": True,
            "independentOfTaskRoles": True,
            "note": (
                "I replayed all frozen contamination methods and inspected each hit "
                "disposition without consulting any model response."
            ),
        }
    )

    with session_scope() as session:
        verifier = session.get(
            ExpertReviewer,
            item.validator_contract_review.reviewer_id,
        )
        assert verifier is not None
        monkeypatch.setattr(
            main_module,
            "_development_task_reviewer",
            lambda _session, _authorization: verifier,
        )
        validator_result = main_module.record_task_candidate_validator_contract_review(
            item.source_candidate_id,
            validator_request,
            session,
            "Bearer test",
            "validator-endpoint-idempotency",
        )
        assert validator_result["reviewEventSha256"] == (
            item.validator_contract_review.review_event_sha256
        )

        auditor = session.get(
            ExpertReviewer,
            item.contamination_audit_review.reviewer_id,
        )
        assert auditor is not None
        monkeypatch.setattr(
            main_module,
            "_development_task_reviewer",
            lambda _session, _authorization: auditor,
        )
        contamination_result = main_module.record_task_candidate_contamination_audit_review(
            item.source_candidate_id,
            contamination_request,
            session,
            "Bearer test",
            "contamination-endpoint-idempotency",
        )
        assert contamination_result["reviewEventSha256"] == (
            item.contamination_audit_review.review_event_sha256
        )
        assert (
            len(
                main_module._task_candidate_evidence_review_events(
                    session,
                    item.source_candidate_id,
                )
            )
            == 2
        )

        result = main_module.admin_import_confirmatory_tasks(
            season_slug,
            _request(item),
            session,
        )
        assert result["tasksImported"] == 1


def test_tampered_human_evidence_review_blocks_task_import(monkeypatch) -> None:
    season_slug, item, scan_bundle = _seed_task_evidence_context("tampered-review")
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(build_image_digest=CONTAINER_DIGEST),
    )
    monkeypatch.setattr(main_module, "_contamination_scan_bundle", lambda: scan_bundle)

    with session_scope() as session:
        event = session.scalar(
            select(RunEvent).where(
                RunEvent.entity_type == "task_candidate",
                RunEvent.entity_id == item.source_candidate_id,
                RunEvent.event_type == "task_candidate_validator_contract_verified",
            )
        )
        assert event is not None
        session.execute(
            text("UPDATE run_events SET payload_json = :payload_json WHERE id = :event_id"),
            {
                "payload_json": json.dumps({**event.payload_json, "model_outputs_visible": True}),
                "event_id": event.id,
            },
        )
        session.expire_all()
        with pytest.raises(HTTPException, match="append-only ledger"):
            main_module.admin_import_confirmatory_tasks(
                season_slug,
                _request(item),
                session,
            )


def test_withdrawn_candidate_cannot_enter_confirmatory_task_bank(monkeypatch) -> None:
    season_slug, item, scan_bundle = _seed_task_evidence_context("withdrawn-candidate")
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(build_image_digest=CONTAINER_DIGEST),
    )
    monkeypatch.setattr(main_module, "_contamination_scan_bundle", lambda: scan_bundle)

    with session_scope() as session:
        session.add(
            RunEvent(
                entity_type="task_candidate",
                entity_id=item.source_candidate_id,
                event_type="task_candidate_withdrawal_recorded",
                payload_json={
                    "schema_version": "flavourbench-task-candidate-withdrawal-v1",
                    "candidate_record_sha256": item.candidate_record_sha256,
                    "withdrawal_record_sha256": hashlib.sha256(
                        b"withdrawn-candidate-record"
                    ).hexdigest(),
                    "reason_category": "voluntary_withdrawal",
                    "withdrawal_confirmed": True,
                    "rank_eligible": False,
                },
            )
        )
        session.flush()
        with pytest.raises(HTTPException, match="approved human task candidate"):
            main_module.admin_import_confirmatory_tasks(
                season_slug,
                _request(item),
                session,
            )


def test_random_hash_is_rejected_before_any_task_or_evidence_row(monkeypatch) -> None:
    season_slug, item, scan_bundle = _seed_task_evidence_context("reject-random-hash")
    tampered_contract = item.validator_contract.model_copy(update={"artifact_sha256": "f" * 64})
    tampered_item = item.model_copy(update={"validator_contract": tampered_contract})
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(build_image_digest=CONTAINER_DIGEST),
    )
    monkeypatch.setattr(main_module, "_contamination_scan_bundle", lambda: scan_bundle)

    with session_scope() as session, pytest.raises(HTTPException, match="digest mismatch"):
        main_module.admin_import_confirmatory_tasks(
            season_slug,
            _request(tampered_item),
            session,
        )

    with session_scope() as session:
        season = session.scalar(select(Season).where(Season.slug == season_slug))
        assert season is not None
        assert session.scalars(select(Task).where(Task.season_id == season.id)).all() == []
        assert (
            session.scalars(
                select(TaskEvidenceArtifact)
                .join(Task, Task.id == TaskEvidenceArtifact.task_id)
                .where(Task.season_id == season.id)
            ).all()
            == []
        )


def test_resolved_artifacts_are_stored_and_tampering_blocks_verification(monkeypatch) -> None:
    season_slug, item, scan_bundle = _seed_task_evidence_context("resolved-evidence")
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(build_image_digest=CONTAINER_DIGEST),
    )
    monkeypatch.setattr(main_module, "_contamination_scan_bundle", lambda: scan_bundle)

    with session_scope() as session:
        result = main_module.admin_import_confirmatory_tasks(
            season_slug,
            _request(item),
            session,
        )
        assert result["tasksImported"] == 1
        task = session.scalar(select(Task).where(Task.public_id == item.public_id))
        assert task is not None
        assert main_module._verified_task_evidence_registry(session, task) is True
        rows = session.scalars(
            select(TaskEvidenceArtifact).where(TaskEvidenceArtifact.task_id == task.id)
        ).all()
        assert {row.evidence_type for row in rows} == {
            "validator_contract",
            "contamination_audit",
        }
        assert task.provenance_json["task_evidence_root_sha256"] == (item.task_evidence_root_sha256)
        lifecycle = verify_task_lifecycle(session, task)
        assert lifecycle.authored_at <= lifecycle.sealed_at
        assert lifecycle.first_used_at is None
        validator = next(row for row in rows if row.evidence_type == "validator_contract")
        session.execute(
            text(
                "UPDATE task_evidence_artifacts SET artifact_json = :artifact_json "
                "WHERE id = :artifact_id"
            ),
            {
                "artifact_json": json.dumps(
                    {**validator.artifact_json, "humanOnlyReason": "tampered evidence bytes"}
                ),
                "artifact_id": validator.id,
            },
        )
        session.flush()
        session.expire_all()
        task = session.scalar(select(Task).where(Task.public_id == item.public_id))
        assert task is not None
        assert main_module._verified_task_evidence_registry(session, task) is False


def test_first_task_use_is_append_only_and_idempotent(monkeypatch) -> None:
    season_slug, item, scan_bundle = _seed_task_evidence_context("first-use")
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(build_image_digest=CONTAINER_DIGEST),
    )
    monkeypatch.setattr(main_module, "_contamination_scan_bundle", lambda: scan_bundle)

    with session_scope() as session:
        main_module.admin_import_confirmatory_tasks(
            season_slug,
            _request(item),
            session,
        )
        task = session.scalar(select(Task).where(Task.public_id == item.public_id))
        assert task is not None
        battle = Battle(
            season_id=task.season_id,
            run_class="pilot",
            rank_eligible=False,
            data_stratum="development",
            task_id=task.id,
            task_revision=task.revision,
            track="model_arena",
            category=task.family,
            prompt=task.prompt,
            prompt_sha256=task.prompt_sha256,
            client_nonce_sha256=hashlib.sha256(b"first-use").hexdigest(),
            requester_pseudonym=hashlib.sha256(b"first-use-reviewer").hexdigest(),
            retention_until=datetime.now(UTC) + timedelta(days=30),
        )
        session.add(battle)
        session.flush()

        first = record_task_first_use(session, task=task, battle=battle)
        repeated = record_task_first_use(session, task=task, battle=battle)
        lifecycle = verify_task_lifecycle(session, task)

        assert repeated.id == first.id
        assert lifecycle.first_used_at is not None
        assert lifecycle.first_used_at >= lifecycle.sealed_at


def test_confirmed_independent_challenge_retires_task_and_blocks_future_use(
    monkeypatch,
) -> None:
    season_slug, item, scan_bundle = _seed_task_evidence_context("confirmed-challenge")
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(build_image_digest=CONTAINER_DIGEST),
    )
    monkeypatch.setattr(main_module, "_contamination_scan_bundle", lambda: scan_bundle)

    with session_scope() as session:
        main_module.admin_import_confirmatory_tasks(
            season_slug,
            _request(item),
            session,
        )
        season = session.scalar(select(Season).where(Season.slug == season_slug))
        task = session.scalar(select(Task).where(Task.public_id == item.public_id))
        assert season is not None and task is not None
        season.official = True
        published_snapshot = LeaderboardSnapshot(
            season_id=season.id,
            track="model_arena",
            cohort="public",
            category="all",
            data_stratum="controlled",
            publication_status="draft",
            input_sha256=hashlib.sha256(b"pre-correction input").hexdigest(),
            input_evidence_sha256=hashlib.sha256(b"pre-correction evidence").hexdigest(),
            input_evidence_json={"status": "pre-correction"},
            payload_sha256=hashlib.sha256(b"pre-correction payload").hexdigest(),
            payload_json={"rows": []},
            evidence_cutoff_at=datetime.now(UTC),
        )
        session.add(published_snapshot)
        session.flush()
        published_snapshot.publication_status = "published"
        published_snapshot.publication_reference_sha256 = hashlib.sha256(
            b"pre-correction publication"
        ).hexdigest()
        published_snapshot.published_at = datetime.now(UTC)
        for ordinal in (1, 2):
            reviewer_id = f"challenge-reviewer-{ordinal}"
            session.add(
                ExpertReviewer(
                    id=reviewer_id,
                    reviewer_code=f"code-{reviewer_id}",
                    invitation_sha256=hashlib.sha256(f"invite:{reviewer_id}".encode()).hexdigest(),
                    qualification_json=[task.family],
                    qualification_verified=True,
                    cohort="expert_independent",
                    profile_json={
                        "admission_pathway": "development_task_validator",
                        "task_validation_status": "active",
                        "identity_commitment_sha256": hashlib.sha256(
                            f"identity:{reviewer_id}".encode()
                        ).hexdigest(),
                        "qualification_evidence_sha256": hashlib.sha256(
                            f"qualification:{reviewer_id}".encode()
                        ).hexdigest(),
                        "independence_attestation_sha256": hashlib.sha256(
                            f"independence:{reviewer_id}".encode()
                        ).hexdigest(),
                        "verification_record_sha256": hashlib.sha256(
                            f"verification:{reviewer_id}".encode()
                        ).hexdigest(),
                        "identity_commitment_algorithm": "HMAC-SHA256",
                        "person_uniqueness_verified": True,
                        "person_uniqueness_method": "admin-witnessed-season-hmac-v1",
                        "person_uniqueness_commitment_sha256": hashlib.sha256(
                            f"person:{reviewer_id}".encode()
                        ).hexdigest(),
                        "affiliation_class": "independent_external",
                        "independent_validation_claim": True,
                        "evidence_verified_by_admin": True,
                    },
                    active=True,
                )
            )
        session.flush()
        challenge_request = TaskChallengeCreate.model_validate(
            {
                "issueType": "criterion_pack_error",
                "description": (
                    "The sealed criterion pack treats a merely optional garnish as mandatory, "
                    "which changes the construct and can reverse otherwise valid judgments."
                ),
                "evidenceReference": "sealed-review-note-criterion-pack-001",
                "clientNonce": "confirmed-challenge-client-nonce",
                "noPersonalDataAttestation": True,
            }
        )
        submitted = main_module.submit_task_challenge(
            task.public_id,
            challenge_request,
            session,
            "confirmed-challenge-idempotency",
            "a" * 64,
        )
        adjudication_request = TaskChallengeAdjudicationCreate.model_validate(
            {
                "decision": "confirmed",
                "adjudicatorReviewerIds": [
                    "challenge-reviewer-1",
                    "challenge-reviewer-2",
                ],
                "rationale": (
                    "Both independent reviewers reproduced the criterion mismatch from the "
                    "sealed record. The optional garnish is outside the prompt constraints, "
                    "so retaining this task would produce construct-irrelevant exclusions."
                ),
                "evidenceBundleSha256": hashlib.sha256(b"confirmed-challenge-evidence").hexdigest(),
                "correctionReference": "season-1-correction-ledger-entry-001",
                "independentOfOriginalTaskRoles": True,
                "modelOutputsConsultedOnlyIfMaterial": True,
            }
        )
        decision = main_module.adjudicate_task_challenge(
            submitted["challengeId"],
            adjudication_request,
            session,
            "confirmed-challenge-decision-idempotency",
        )
        replay = main_module.adjudicate_task_challenge(
            submitted["challengeId"],
            adjudication_request,
            session,
            "confirmed-challenge-decision-idempotency",
        )
        lifecycle = verify_task_lifecycle(session, task)

        assert decision["taskRetired"] is True
        assert decision["leaderboardRecomputationRequired"] is True
        assert decision["withdrawnSnapshotIds"] == [published_snapshot.id]
        assert replay["idempotent"] is True
        assert replay["leaderboardRecomputationRequired"] is True
        assert replay["withdrawnSnapshotIds"] == [published_snapshot.id]
        assert published_snapshot.publication_status == "withdrawn"
        withdrawal_event = session.scalar(
            select(RunEvent).where(
                RunEvent.entity_type == "leaderboard_snapshot",
                RunEvent.entity_id == published_snapshot.id,
                RunEvent.event_type == "leaderboard_snapshot_automatically_withdrawn",
            )
        )
        assert withdrawal_event is not None
        assert (
            withdrawal_event.payload_json["reason_code"]
            == "confirmatory_task_retired_after_confirmed_challenge"
        )
        assert lifecycle.retired_at is not None

        battle = Battle(
            season_id=task.season_id,
            run_class="pilot",
            rank_eligible=False,
            data_stratum="development",
            task_id=task.id,
            task_revision=task.revision,
            track="model_arena",
            category=task.family,
            prompt=task.prompt,
            prompt_sha256=task.prompt_sha256,
            client_nonce_sha256=hashlib.sha256(b"retired-use").hexdigest(),
            requester_pseudonym=hashlib.sha256(b"retired-use-reviewer").hexdigest(),
            retention_until=datetime.now(UTC) + timedelta(days=30),
        )
        session.add(battle)
        session.flush()
        with pytest.raises(TaskLifecycleError, match="retired task"):
            record_task_first_use(session, task=task, battle=battle)


def test_task_bound_runtime_receipt_scores_only_the_frozen_executable_subset(
    monkeypatch,
) -> None:
    season_slug, item, scan_bundle = _seed_task_evidence_context(
        "runtime-validator",
        objective_validator_possible=True,
    )
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(build_image_digest=CONTAINER_DIGEST),
    )
    monkeypatch.setattr(main_module, "_contamination_scan_bundle", lambda: scan_bundle)

    with session_scope() as session:
        main_module.admin_import_confirmatory_tasks(
            season_slug,
            _request(item),
            session,
        )
        task = session.scalar(select(Task).where(Task.public_id == item.public_id))
        assert task is not None
        battle = Battle(
            season_id=task.season_id,
            run_class="official",
            rank_eligible=True,
            # This is a local validator-receipt fixture, not a scheduled
            # controlled collection.  Controlled rows require a bound
            # ControlledRun so they cannot masquerade as official evidence.
            data_stratum="development",
            task_id=task.id,
            task_revision=task.revision,
            track="model_arena",
            category=task.family,
            prompt=task.prompt,
            prompt_sha256=task.prompt_sha256,
            client_nonce_sha256=hashlib.sha256(b"runtime-validator").hexdigest(),
            requester_pseudonym=hashlib.sha256(b"runtime-reviewer").hexdigest(),
            retention_until=datetime.now(UTC) + timedelta(days=30),
        )
        session.add(battle)
        session.flush()

        passing = task_validator_receipt_for_battle(
            session,
            battle,
            "Roast the pear, then fold it through the buckwheat.",
            expected_container_image_digest=CONTAINER_DIGEST,
            contamination_scan_bundle=scan_bundle,
            validator_calibration=main_module._validator_calibration(),
        )
        failing = task_validator_receipt_for_battle(
            session,
            battle,
            "Use only apple and buckwheat.",
            expected_container_image_digest=CONTAINER_DIGEST,
            contamination_scan_bundle=scan_bundle,
            validator_calibration=main_module._validator_calibration(),
        )

        assert passing is not None and passing["status"] == "pass"
        assert passing["score_milli"] == 1000
        assert failing is not None and failing["status"] == "fail"
        assert failing["score_milli"] == 0
        assert passing["task_evidence_root_sha256"] == item.task_evidence_root_sha256
