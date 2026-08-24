from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient
from sqlalchemy import select

from flavourbench.config import get_settings
from flavourbench.construct_blueprint import BLUEPRINT_SHA256
from flavourbench.database import SessionLocal
from flavourbench.main import app
from flavourbench.models import RunEvent
from flavourbench.task_contributor_protocol import PROTOCOL_SHA256, PROTOCOL_VERSION

SERVICE_HEADERS = {"X-FlavourBench-Service-Token": "test-service-token"}
ADMIN_HEADERS = {
    **SERVICE_HEADERS,
    "X-FlavourBench-Admin-Token": "test-admin-token",
}
ACTIVE_EXPERT_CONSENT_SHA256 = get_settings().active_expert_consent_sha256s[0]


def _expert_invitation(
    client: TestClient,
    code: str,
    *,
    adjudication_authorized: bool = False,
) -> str:
    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    response = client.post(
        "/v1/admin/task-validators",
        headers=ADMIN_HEADERS,
        json={
            "validatorCode": code,
            "qualifiedFamilies": ["composition"],
            "qualificationReference": f"verified-culinary-practice-{code}",
            "verifiedIdentityHandle": f"verified-private-identity:{code}",
            "qualificationEvidenceSha256": digest(f"qualification:{code}"),
            "independenceAttestationSha256": digest(f"independence:{code}"),
            "verificationRecordSha256": digest(f"verification:{code}"),
            "affiliationClass": "independent_external",
            "conflictDisclosureReference": "fixture-no-conflict",
            "consentDocumentSha256": ACTIVE_EXPERT_CONSENT_SHA256,
            "adjudicationAuthorized": adjudication_authorized,
            "evidenceVerifiedByAdmin": True,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["invitation"]


def _blind_validity() -> dict[str, object]:
    return {
        "decision": "valid",
        "constructFit": True,
        "contextComplete": True,
        "coherentQuestion": True,
        "generalTrackScope": True,
        "answerLeakageAbsent": True,
        "discriminationValue": True,
        "familyClassification": "composition",
        "constructCellClassification": "bridge_ingredient_reasoning",
        "difficultyTierClassification": "integrative",
        "independentSolutionOutline": (
            "Use the beans as the creamy base, preserved lemon as an acidic bridge, and "
            "hazelnuts as the final toasted contrast while roasting the cauliflower hard."
        ),
        "successCriteria": [
            "Uses every named ingredient in a coherent main-dish composition.",
            "Explains both the flavour bridges and the texture-preserving workflow.",
        ],
        "disqualifyingErrors": [
            "Omits a named ingredient or proposes a workflow exceeding the equipment limit."
        ],
        "issueTags": [],
        "criteriaAuthoredByReviewer": True,
        "authorPackNotSeen": True,
        "modelOutputsNotConsulted": True,
        "note": "The prompt is answerable and discriminates practical composition reasoning.",
    }


def _reconciliation() -> dict[str, object]:
    return {
        "decision": "approve",
        "authorPackAdequacy": "adequate",
        "constructLabelAgreement": True,
        "difficultyLabelAgreement": True,
        "constraintSetAdequate": True,
        "solutionOutlineAdequate": True,
        "validatorPlanAdequate": True,
        "rightsBasisCredible": True,
        "successCriteria": _blind_validity()["successCriteria"],
        "permittedVariations": [
            "Equivalent parallel workflows are acceptable when the time limit is respected."
        ],
        "disqualifyingErrors": _blind_validity()["disqualifyingErrors"],
        "objectiveChecks": [
            "Every named ingredient is used and stated time and equipment limits are respected."
        ],
        "issueTags": [],
        "criteriaAuthoredByReviewer": True,
        "independentOfAuthor": True,
        "modelOutputsNotConsulted": True,
        "note": "The author pack agrees with the independently sealed construct and criteria.",
    }


def _adjudication() -> dict[str, object]:
    return {
        "decision": "approve",
        "family": "composition",
        "constructCellId": "bridge_ingredient_reasoning",
        "difficultyTier": "integrative",
        "successCriteria": _blind_validity()["successCriteria"],
        "permittedVariations": _reconciliation()["permittedVariations"],
        "disqualifyingErrors": _blind_validity()["disqualifyingErrors"],
        "objectiveChecks": _reconciliation()["objectiveChecks"],
        "criteriaAuthoredByAdjudicator": True,
        "independentOfAuthorAndReviewers": True,
        "modelOutputsNotConsulted": True,
        "note": (
            "Both prompt-only solutions independently recover the intended construct; this "
            "merged criterion pack preserves their shared requirements."
        ),
    }


def test_anonymous_human_task_requires_blind_review_reconciliation_and_adjudication() -> None:
    with TestClient(app) as client:
        obsolete_invite = client.post(
            "/v1/admin/task-contributors",
            headers=ADMIN_HEADERS,
            json={
                "contributorCode": "anonymous-task-author-obsolete-protocol",
                "qualifiedFamilies": ["composition"],
                "protocolVersion": "flavourbench-human-task-contributor-v1",
                "protocolSha256": "1" * 64,
                "verifiedIdentityHandle": "verified-private-person:obsolete-protocol",
                "personUniquenessEvidenceSha256": hashlib.sha256(
                    b"person-uniqueness:obsolete-protocol"
                ).hexdigest(),
                "personUniquenessVerifiedByAdmin": True,
            },
        )
        assert obsolete_invite.status_code == 422
        assert "current frozen protocol" in obsolete_invite.text

        invite_response = client.post(
            "/v1/admin/task-contributors",
            headers=ADMIN_HEADERS,
            json={
                "contributorCode": "anonymous-task-author-fixture-01",
                "qualifiedFamilies": ["composition"],
                "protocolVersion": PROTOCOL_VERSION,
                "protocolSha256": PROTOCOL_SHA256,
                "verifiedIdentityHandle": "verified-private-person:task-author-fixture-01",
                "personUniquenessEvidenceSha256": hashlib.sha256(
                    b"person-uniqueness:task-author-fixture-01"
                ).hexdigest(),
                "personUniquenessVerifiedByAdmin": True,
            },
        )
        assert invite_response.status_code == 200, invite_response.text
        contributor_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {invite_response.json()['invitation']}",
        }

        blocked_before_acceptance = client.get(
            "/v1/task-contributions/onboarding",
            headers=contributor_headers,
        )
        assert blocked_before_acceptance.status_code == 403

        protocol = client.get(
            "/v1/task-contributions/protocol",
            headers=contributor_headers,
        )
        assert protocol.status_code == 200, protocol.text
        protocol_contract = protocol.json()
        assert protocol_contract["protocolVersion"] == PROTOCOL_VERSION
        assert protocol_contract["protocolSha256"] == PROTOCOL_SHA256
        assert hashlib.sha256(protocol_contract["protocolText"].encode()).hexdigest() == (
            PROTOCOL_SHA256
        )
        assert protocol_contract["accepted"] is False

        acceptance_body = {
            "protocolVersion": PROTOCOL_VERSION,
            "protocolSha256": PROTOCOL_SHA256,
            "voluntaryParticipationAccepted": True,
            "taskContributionAgreementAccepted": True,
            "humanOnlyMethodsAcknowledged": True,
        }
        wrong_acceptance = client.post(
            "/v1/task-contributions/protocol-acceptance",
            headers=contributor_headers,
            json={**acceptance_body, "protocolSha256": "f" * 64},
        )
        assert wrong_acceptance.status_code == 422
        incomplete_acceptance = client.post(
            "/v1/task-contributions/protocol-acceptance",
            headers=contributor_headers,
            json={**acceptance_body, "humanOnlyMethodsAcknowledged": False},
        )
        assert incomplete_acceptance.status_code == 422

        accepted = client.post(
            "/v1/task-contributions/protocol-acceptance",
            headers=contributor_headers,
            json=acceptance_body,
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["contributionEnabled"] is True
        assert accepted.json()["idempotent"] is False
        acceptance_event_id = accepted.json()["eventId"]

        replayed_acceptance = client.post(
            "/v1/task-contributions/protocol-acceptance",
            headers=contributor_headers,
            json=acceptance_body,
        )
        assert replayed_acceptance.status_code == 200, replayed_acceptance.text
        assert replayed_acceptance.json()["idempotent"] is True
        assert replayed_acceptance.json()["eventId"] == acceptance_event_id

        onboarding = client.get(
            "/v1/task-contributions/onboarding",
            headers=contributor_headers,
        )
        assert onboarding.status_code == 200, onboarding.text
        assert onboarding.json()["identityCollectionProhibited"] is False
        assert onboarding.json()["rawIdentityRetentionProhibited"] is True
        assert onboarding.json()["personUniquenessVerified"] is True
        assert onboarding.json()["syntheticTasksAccepted"] is False
        assert onboarding.json()["protocolVersion"] == PROTOCOL_VERSION
        assert onboarding.json()["protocolSha256"] == PROTOCOL_SHA256
        assert onboarding.json()["protocolAcceptanceEventId"] == acceptance_event_id

        body = {
            "family": "composition",
            "constructBlueprintSha256": BLUEPRINT_SHA256,
            "constructCellId": "bridge_ingredient_reasoning",
            "difficultyTier": "integrative",
            "prompt": (
                "Build a weeknight main from roasted cauliflower, canned white beans, "
                "preserved lemon, and toasted hazelnuts. Explain the bridge ingredients, "
                "texture plan, and order of operations for a conventional home oven."
            ),
            "subskills": [
                "bridge_ingredient_reasoning",
                "texture contrast",
                "workflow planning",
            ],
            "explicitConstraints": [
                "Use every named ingredient",
                "Finish within 45 minutes",
                "Use one oven and two stovetop burners at most",
            ],
            "unacceptableOutcomes": [
                "Invented specialist equipment",
                "An ingredient list without a cooking method",
            ],
            "acceptableSolutionOutline": (
                "A credible answer connects acidity, fat, toasted aroma, and creamy beans; "
                "preserves texture contrast; and gives an executable parallel workflow."
            ),
            "objectiveValidatorPossible": True,
            "validatorNotes": "Check named ingredient use, elapsed time, and equipment limits.",
            "rightsBasis": "original_personal_authorship",
            "humanAuthorshipAttestation": True,
            "noPersonalDataAttestation": True,
            "researchUseConsent": True,
            "clientNonce": "task-author-fixture-nonce-0001",
        }
        submission_headers = {
            **contributor_headers,
            "Idempotency-Key": "task-author-fixture-idempotency-0001",
        }
        submitted = client.post(
            "/v1/task-contributions",
            headers=submission_headers,
            json=body,
        )
        assert submitted.status_code == 201, submitted.text
        candidate_id = submitted.json()["candidateId"]
        assert submitted.json()["status"] == "awaiting_independent_review"
        with SessionLocal() as session:
            candidate_event = session.scalar(
                select(RunEvent).where(
                    RunEvent.entity_type == "task_candidate",
                    RunEvent.entity_id == candidate_id,
                    RunEvent.event_type == "task_candidate_submitted",
                )
            )
            assert candidate_event is not None
            assert candidate_event.payload_json["task_contributor_protocol_version"] == (
                PROTOCOL_VERSION
            )
            assert candidate_event.payload_json["task_contributor_protocol_sha256"] == (
                PROTOCOL_SHA256
            )
            assert candidate_event.payload_json[
                "task_contributor_protocol_acceptance_event_id"
            ] == acceptance_event_id

        replayed = client.post(
            "/v1/task-contributions",
            headers=submission_headers,
            json=body,
        )
        assert replayed.status_code == 201, replayed.text
        assert replayed.json()["candidateId"] == candidate_id
        assert replayed.json()["idempotent"] is True

        own_records = client.get(
            "/v1/task-contributions",
            headers=contributor_headers,
        )
        assert own_records.status_code == 200, own_records.text
        assert len(own_records.json()["candidates"]) == 1
        assert "prompt" not in own_records.json()["candidates"][0]

        invitations = [
            _expert_invitation(client, "task-reviewer-fixture-01"),
            _expert_invitation(client, "task-reviewer-fixture-02"),
        ]
        for index, invitation in enumerate(invitations, start=1):
            reviewer_headers = {
                **SERVICE_HEADERS,
                "Authorization": f"Bearer {invitation}",
            }
            assignment = client.get(
                "/v1/expert/task-candidates/next",
                headers=reviewer_headers,
            )
            assert assignment.status_code == 200, assignment.text
            candidate = assignment.json()["candidate"]
            assert candidate["candidateId"] == candidate_id
            assert candidate["phase"] == "blind_validity"
            assert candidate["authorIdentity"] is None
            assert candidate["authorPackVisible"] is False
            assert "acceptableSolutionOutline" not in candidate
            assert "family" not in candidate
            assert "authorReviewerId" not in assignment.text

            blind = client.post(
                f"/v1/expert/task-candidates/{candidate_id}/blind-validity",
                headers={
                    **reviewer_headers,
                    "Idempotency-Key": f"task-blind-review-fixture-{index}",
                },
                json=_blind_validity(),
            )
            assert blind.status_code == 200, blind.text
            assert blind.json()["nextPhase"] == "reconciliation"

            revealed = client.get(
                "/v1/expert/task-candidates/next",
                headers=reviewer_headers,
            )
            assert revealed.status_code == 200, revealed.text
            assert revealed.json()["candidate"]["candidateId"] == candidate_id
            assert revealed.json()["candidate"]["phase"] == "reconciliation"
            assert revealed.json()["candidate"]["authorPackVisible"] is True
            assert revealed.json()["candidate"]["acceptableSolutionOutline"]

            reconciled = client.post(
                f"/v1/expert/task-candidates/{candidate_id}/reconciliation",
                headers={
                    **reviewer_headers,
                    "Idempotency-Key": f"task-reconciliation-fixture-{index}",
                },
                json=_reconciliation(),
            )
            assert reconciled.status_code == 200, reconciled.text
            assert reconciled.json()["complete"] is True

        awaiting = client.get(
            "/v1/task-contributions",
            headers=contributor_headers,
        ).json()["candidates"][0]
        assert awaiting["status"] == "awaiting_independent_adjudication"
        assert awaiting["approvals"] == 2

        legacy = client.post(
            f"/v1/expert/task-candidates/{candidate_id}/reviews",
            headers={
                **SERVICE_HEADERS,
                "Authorization": f"Bearer {invitations[0]}",
                "Idempotency-Key": "legacy-task-review-fixture",
            },
            json={
                "decision": "approve",
                "constructFit": True,
                "contextComplete": True,
                "specialistScopeClear": True,
                "answerLeakageAbsent": True,
                "rightsBasisCredible": True,
                "validatorPlanAdequate": True,
                "issueTags": [],
                "note": "Legacy review should be rejected.",
            },
        )
        assert legacy.status_code == 410

        adjudicator_invitation = _expert_invitation(
            client,
            "task-adjudicator-fixture-03",
            adjudication_authorized=True,
        )
        adjudicator_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {adjudicator_invitation}",
        }
        adjudication_assignment = client.get(
            "/v1/expert/task-candidates/adjudication/next",
            headers=adjudicator_headers,
        )
        assert adjudication_assignment.status_code == 200, adjudication_assignment.text
        adjudication_candidate = adjudication_assignment.json()["candidate"]
        assert adjudication_candidate["candidateId"] == candidate_id
        assert adjudication_candidate["phase"] == "adjudication"
        assert adjudication_candidate["modelOutputsVisible"] is False
        assert len(adjudication_candidate["independentReviews"]) == 2

        adjudicated = client.post(
            f"/v1/expert/task-candidates/{candidate_id}/adjudication",
            headers={
                **adjudicator_headers,
                "Idempotency-Key": "task-adjudication-fixture-03",
            },
            json=_adjudication(),
        )
        assert adjudicated.status_code == 200, adjudicated.text
        assert adjudicated.json()["status"] == "approved_for_bank_assembly"

        final_records = client.get(
            "/v1/task-contributions",
            headers=contributor_headers,
        )
        final_candidate = final_records.json()["candidates"][0]
        assert final_candidate["status"] == "approved_for_bank_assembly"
        assert final_candidate["approvals"] == 2


def test_task_candidate_withdrawal_is_append_only_idempotent_and_fail_closed() -> None:
    with TestClient(app) as client:
        invite_response = client.post(
            "/v1/admin/task-contributors",
            headers=ADMIN_HEADERS,
            json={
                "contributorCode": "anonymous-task-author-withdrawal-fixture",
                "qualifiedFamilies": ["composition"],
                "protocolVersion": PROTOCOL_VERSION,
                "protocolSha256": PROTOCOL_SHA256,
                "verifiedIdentityHandle": "verified-private-person:withdrawal-fixture",
                "personUniquenessEvidenceSha256": hashlib.sha256(
                    b"person-uniqueness:withdrawal-fixture"
                ).hexdigest(),
                "personUniquenessVerifiedByAdmin": True,
            },
        )
        assert invite_response.status_code == 200, invite_response.text
        contributor_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {invite_response.json()['invitation']}",
        }
        acceptance = client.post(
            "/v1/task-contributions/protocol-acceptance",
            headers=contributor_headers,
            json={
                "protocolVersion": PROTOCOL_VERSION,
                "protocolSha256": PROTOCOL_SHA256,
                "voluntaryParticipationAccepted": True,
                "taskContributionAgreementAccepted": True,
                "humanOnlyMethodsAcknowledged": True,
            },
        )
        assert acceptance.status_code == 200, acceptance.text

        candidate_body = {
            "family": "composition",
            "constructBlueprintSha256": BLUEPRINT_SHA256,
            "constructCellId": "bridge_ingredient_reasoning",
            "difficultyTier": "integrative",
            "prompt": (
                "Design a coherent warm salad from roasted squash, rye croutons, cultured "
                "cream, and dill, with a forty-minute limit and one oven tray. Explain the "
                "flavour bridge, texture contrast, and executable order of operations."
            ),
            "subskills": ["bridge_ingredient_reasoning", "workflow planning"],
            "explicitConstraints": ["Use every named ingredient", "Finish in 40 minutes"],
            "unacceptableOutcomes": ["Omitting a named ingredient"],
            "acceptableSolutionOutline": (
                "A credible solution links the cultured acidity and dill, preserves crisp "
                "rye texture, and sequences the single-tray roast within the time limit."
            ),
            "objectiveValidatorPossible": True,
            "validatorNotes": "Check ingredient coverage, stated time, and one-tray use.",
            "rightsBasis": "original_personal_authorship",
            "humanAuthorshipAttestation": True,
            "noPersonalDataAttestation": True,
            "researchUseConsent": True,
            "clientNonce": "withdrawal-candidate-client-nonce-0001",
        }
        submitted = client.post(
            "/v1/task-contributions",
            headers={
                **contributor_headers,
                "Idempotency-Key": "withdrawal-candidate-submit-0001",
            },
            json=candidate_body,
        )
        assert submitted.status_code == 201, submitted.text
        candidate_id = submitted.json()["candidateId"]
        record_sha256 = submitted.json()["recordSha256"]
        withdrawal_body = {
            "candidateRecordSha256": record_sha256,
            "reasonCategory": "voluntary_withdrawal",
            "note": "Withdrawing before curation.",
            "withdrawalConfirmed": True,
            "clientNonce": "withdrawal-candidate-client-nonce-0002",
        }
        withdrawal_headers = {
            **contributor_headers,
            "Idempotency-Key": "withdrawal-candidate-request-0001",
        }

        wrong_receipt = client.post(
            f"/v1/task-contributions/{candidate_id}/withdrawal",
            headers=withdrawal_headers,
            json={**withdrawal_body, "candidateRecordSha256": "f" * 64},
        )
        assert wrong_receipt.status_code == 409
        unconfirmed = client.post(
            f"/v1/task-contributions/{candidate_id}/withdrawal",
            headers=withdrawal_headers,
            json={**withdrawal_body, "withdrawalConfirmed": False},
        )
        assert unconfirmed.status_code == 422

        withdrawn = client.post(
            f"/v1/task-contributions/{candidate_id}/withdrawal",
            headers=withdrawal_headers,
            json=withdrawal_body,
        )
        assert withdrawn.status_code == 200, withdrawn.text
        assert withdrawn.json()["status"] == "withdrawn"
        assert withdrawn.json()["idempotent"] is False
        assert "prompt" not in withdrawn.text
        replay = client.post(
            f"/v1/task-contributions/{candidate_id}/withdrawal",
            headers=withdrawal_headers,
            json=withdrawal_body,
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["idempotent"] is True
        assert replay.json()["withdrawalSha256"] == withdrawn.json()["withdrawalSha256"]
        conflict = client.post(
            f"/v1/task-contributions/{candidate_id}/withdrawal",
            headers={
                **contributor_headers,
                "Idempotency-Key": "withdrawal-candidate-request-0002",
            },
            json={**withdrawal_body, "clientNonce": "withdrawal-client-conflict-0002"},
        )
        assert conflict.status_code == 409

        records = client.get("/v1/task-contributions", headers=contributor_headers)
        own = next(
            item for item in records.json()["candidates"] if item["candidateId"] == candidate_id
        )
        assert own["status"] == "withdrawn"
        assert own["withdrawalCount"] == 1
        assert own["withdrawalEligible"] is False

        reviewer_invitation = _expert_invitation(client, "withdrawal-reviewer-fixture")
        reviewer_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {reviewer_invitation}",
        }
        blocked_review = client.post(
            f"/v1/expert/task-candidates/{candidate_id}/blind-validity",
            headers={
                **reviewer_headers,
                "Idempotency-Key": "withdrawn-task-blind-review",
            },
            json=_blind_validity(),
        )
        assert blocked_review.status_code == 409
        assert "withdrawn" in blocked_review.text
        assignment = client.get(
            "/v1/expert/task-candidates/next",
            headers=reviewer_headers,
        )
        assert assignment.status_code == 200, assignment.text
        assigned = assignment.json()["candidate"]
        assert assigned is None or assigned["candidateId"] != candidate_id

        with SessionLocal() as session:
            submitted_events = session.scalars(
                select(RunEvent).where(
                    RunEvent.entity_type == "task_candidate",
                    RunEvent.entity_id == candidate_id,
                    RunEvent.event_type == "task_candidate_submitted",
                )
            ).all()
            withdrawal_events = session.scalars(
                select(RunEvent).where(
                    RunEvent.entity_type == "task_candidate",
                    RunEvent.entity_id == candidate_id,
                    RunEvent.event_type == "task_candidate_withdrawal_recorded",
                )
            ).all()
            assert len(submitted_events) == 1
            assert len(withdrawal_events) == 1
            assert withdrawal_events[0].payload_json["rank_eligible"] is False
