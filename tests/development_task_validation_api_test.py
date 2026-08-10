from __future__ import annotations

import copy
import hashlib

import pytest
from fastapi.testclient import TestClient

from flavourbench.config import get_settings
from flavourbench.development_task_status import (
    DevelopmentTaskStatusError,
    verify_status_artifact,
)
from flavourbench.expert_review import canonical_sha256
from flavourbench.main import _development_task_blind_assignment_key, app

SERVICE_HEADERS = {"X-FlavourBench-Service-Token": "test-service-token"}
ADMIN_HEADERS = {
    **SERVICE_HEADERS,
    "X-FlavourBench-Admin-Token": "test-admin-token",
}
ACTIVE_EXPERT_CONSENT_SHA256 = get_settings().active_expert_consent_sha256s[0]


def test_blind_assignment_prioritizes_coverage_before_stable_tiebreak() -> None:
    arguments = {
        "packet_sha256": "a" * 64,
        "reviewer_id": "reviewer-fixed",
        "task_id": "task-fixed",
    }
    empty = _development_task_blind_assignment_key(
        **arguments,
        complete_independent_reviews=0,
    )
    reviewed = _development_task_blind_assignment_key(
        **arguments,
        complete_independent_reviews=1,
    )

    assert empty < reviewed
    assert empty == _development_task_blind_assignment_key(
        **arguments,
        complete_independent_reviews=0,
    )
    assert empty[1] != _development_task_blind_assignment_key(
        packet_sha256="a" * 64,
        reviewer_id="another-reviewer",
        task_id="task-fixed",
        complete_independent_reviews=0,
    )[1]


def test_live_status_artifact_verifier_rejects_imputation_and_identity_fields() -> None:
    with TestClient(app) as client:
        response = client.get("/v1/admin/development-tasks/status", headers=ADMIN_HEADERS)
        assert response.status_code == 200, response.text
        document = response.json()
        verify_status_artifact(document, expected_packet_sha256=document["packetSha256"])

        imputed = copy.deepcopy(document)
        if imputed["statistics"]["agreement"]["taskCount"] == 0:
            imputed["statistics"]["agreement"]["fleissKappa"] = 1.0
            imputed["artifactSha256"] = canonical_sha256(
                {key: value for key, value in imputed.items() if key != "artifactSha256"}
            )
            with pytest.raises(DevelopmentTaskStatusError, match="must remain null"):
                verify_status_artifact(imputed)

        leaked = copy.deepcopy(document)
        leaked["tasks"][0]["reviewerId"] = "prohibited"
        leaked["artifactSha256"] = canonical_sha256(
            {key: value for key, value in leaked.items() if key != "artifactSha256"}
        )
        with pytest.raises(DevelopmentTaskStatusError, match="prohibited review material"):
            verify_status_artifact(leaked)


def _expert_invitation(
    client: TestClient,
    code: str,
    *,
    adjudication_authorized: bool = False,
    identity_key: str | None = None,
) -> str:
    identity_key = identity_key or code

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    response = client.post(
        "/v1/admin/task-validators",
        headers=ADMIN_HEADERS,
        json={
            "validatorCode": code,
            "qualifiedFamilies": ["composition"],
            "qualificationReference": f"verified-culinary-practice-{code}",
            "verifiedIdentityHandle": f"verified-private-identity:{identity_key}",
            "qualificationEvidenceSha256": digest(f"qualification-evidence:{identity_key}"),
            "independenceAttestationSha256": digest(f"independence-attestation:{identity_key}"),
            "verificationRecordSha256": digest(f"verification-record:{identity_key}"),
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
        "issueTags": [],
        "note": "The prompt is answerable and discriminates practical culinary reasoning.",
    }


def _criterion_pack() -> dict[str, object]:
    return {
        "referenceAdequacy": "partial",
        "successCriteria": [
            "Explains the central culinary mechanism in terms applicable to the prompt.",
            "Provides an executable recommendation with relevant practical constraints.",
        ],
        "permittedVariations": [
            "Equivalent methods are acceptable when their tradeoffs are stated."
        ],
        "disqualifyingErrors": ["A recommendation that contradicts the stated physical mechanism."],
        "objectiveChecks": ["Every explicitly named ingredient is addressed."],
        "criteriaAuthoredByReviewer": True,
        "note": (
            "The accepted answer is useful but not exhaustive, so the criteria allow "
            "other technically sound approaches."
        ),
    }


def test_development_task_review_is_answer_blind_then_collects_human_criteria() -> None:
    with TestClient(app) as client:
        invitation = _expert_invitation(client, "development-task-reviewer-fixture-01")
        headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {invitation}",
        }

        assignment = client.get("/v1/expert/development-tasks/next", headers=headers)
        assert assignment.status_code == 200, assignment.text
        first = assignment.json()["task"]
        assert first["phase"] == "blind_validity"
        assert first["family"] == "composition"
        assert first["sourceUrl"] is None
        assert first["humanReference"] is None
        assert assignment.json()["independentClaim"] is True

        invalid = client.post(
            f"/v1/expert/development-tasks/{first['taskId']}/blind-validity",
            headers={**headers, "Idempotency-Key": "invalid-blind-review"},
            json={**_blind_validity(), "constructFit": False},
        )
        assert invalid.status_code == 422

        validity_headers = {**headers, "Idempotency-Key": "blind-validity-fixture-01"}
        validity = client.post(
            f"/v1/expert/development-tasks/{first['taskId']}/blind-validity",
            headers=validity_headers,
            json=_blind_validity(),
        )
        assert validity.status_code == 200, validity.text
        assert validity.json()["nextPhase"] == "criteria"
        replay = client.post(
            f"/v1/expert/development-tasks/{first['taskId']}/blind-validity",
            headers=validity_headers,
            json=_blind_validity(),
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["idempotent"] is True

        unlocked = client.get("/v1/expert/development-tasks/next", headers=headers)
        assert unlocked.status_code == 200, unlocked.text
        criterion_task = unlocked.json()["task"]
        assert criterion_task["taskId"] == first["taskId"]
        assert criterion_task["phase"] == "criteria"
        assert criterion_task["sourceUrl"].startswith("https://cooking.stackexchange.com/")
        assert criterion_task["sourceAuthor"]["display_name"]
        assert criterion_task["humanReference"]["text"]
        assert criterion_task["humanReference"]["author"]["display_name"]
        assert criterion_task["humanReference"]["use"] == ("review_aid_not_automatic_ground_truth")

        criteria = client.post(
            f"/v1/expert/development-tasks/{first['taskId']}/criteria",
            headers={**headers, "Idempotency-Key": "criteria-fixture-01"},
            json=_criterion_pack(),
        )
        assert criteria.status_code == 200, criteria.text
        assert criteria.json()["complete"] is True

        next_assignment = client.get("/v1/expert/development-tasks/next", headers=headers)
        assert next_assignment.status_code == 200, next_assignment.text
        assert next_assignment.json()["task"]["taskId"] != first["taskId"]
        assert next_assignment.json()["progress"] == {
            "eligible": 10,
            "blindDecisions": 1,
            "criterionPacks": 1,
        }


def test_three_unanimous_independent_reviews_complete_without_adjudication() -> None:
    with TestClient(app) as client:
        invitations = [
            _expert_invitation(client, f"development-task-unanimous-reviewer-{ordinal}")
            for ordinal in range(1, 5)
        ]
        adjudicator_invitation = _expert_invitation(
            client,
            "development-task-unanimous-adjudicator",
            adjudication_authorized=True,
        )
        reviewer_headers = [
            {**SERVICE_HEADERS, "Authorization": f"Bearer {invitation}"}
            for invitation in invitations
        ]
        adjudicator_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {adjudicator_invitation}",
        }

        initial_status = client.get("/v1/admin/development-tasks/status", headers=ADMIN_HEADERS)
        assert initial_status.json()["taskCount"] == 40
        assert initial_status.json()["requiredIndependentReviewsPerTask"] == 3
        initial_statistics = initial_status.json()["statistics"]
        assert initial_statistics["coverage"]["requiredIndependentReviews"] == 120
        assert initial_statistics["coverage"]["completeIndependentReviews"] >= 0
        assert initial_statistics["claimBoundary"]["packetRowsCountAsHumanEvidence"] is False
        baseline_complete_tasks = initial_statistics["agreement"]["taskCount"]
        baseline_unanimous_tasks = initial_statistics["agreement"][
            "unanimousDecisionTasks"
        ]
        status_payload = {
            key: value
            for key, value in initial_status.json().items()
            if key != "artifactSha256"
        }
        assert initial_status.json()["artifactSha256"] == canonical_sha256(status_payload)
        task_id = next(
            row["taskId"]
            for row in initial_status.json()["tasks"]
            if row["family"] == "composition" and row["completeIndependentReviews"] == 0
        )
        for ordinal, headers in enumerate(reviewer_headers[:3], start=1):
            validity = client.post(
                f"/v1/expert/development-tasks/{task_id}/blind-validity",
                headers={**headers, "Idempotency-Key": f"blind-validity-unanimous-{ordinal}"},
                json=_blind_validity(),
            )
            assert validity.status_code == 200, validity.text
            criteria = client.post(
                f"/v1/expert/development-tasks/{task_id}/criteria",
                headers={**headers, "Idempotency-Key": f"criteria-unanimous-{ordinal}"},
                json=_criterion_pack(),
            )
            assert criteria.status_code == 200, criteria.text

        status = client.get("/v1/admin/development-tasks/status", headers=ADMIN_HEADERS)
        assert status.status_code == 200, status.text
        task_status = next(row for row in status.json()["tasks"] if row["taskId"] == task_id)
        assert task_status["status"] == "validated_unanimous"
        assert task_status["completeIndependentReviews"] == 3
        assert task_status["decisionCounts"] == {"valid": 3}
        assert len(task_status["consensusSha256"]) == 64
        assert status.json()["independentlyValidatedTasks"] >= 1
        agreement = status.json()["statistics"]["agreement"]
        assert agreement["taskCount"] == baseline_complete_tasks + 1
        assert agreement["unanimousDecisionTasks"] == baseline_unanimous_tasks + 1

        assignment = client.get(
            "/v1/expert/development-tasks/adjudication/next",
            headers=adjudicator_headers,
        )
        assert assignment.status_code == 200, assignment.text
        assert assignment.json()["task"] is None

        excess = client.post(
            f"/v1/expert/development-tasks/{task_id}/blind-validity",
            headers={
                **reviewer_headers[3],
                "Idempotency-Key": "blind-validity-unanimous-excess",
            },
            json=_blind_validity(),
        )
        assert excess.status_code == 409
        assert excess.json()["detail"] == "independent source-review slate is full"


def test_nonunanimous_three_label_record_requires_independent_adjudication() -> None:
    with TestClient(app) as client:
        invitations = [
            _expert_invitation(client, f"development-task-disagreement-reviewer-{ordinal}")
            for ordinal in range(1, 4)
        ]
        adjudicator_invitation = _expert_invitation(
            client,
            "development-task-disagreement-adjudicator",
            adjudication_authorized=True,
        )
        reviewer_headers = [
            {**SERVICE_HEADERS, "Authorization": f"Bearer {invitation}"}
            for invitation in invitations
        ]
        adjudicator_headers = {
            **SERVICE_HEADERS,
            "Authorization": f"Bearer {adjudicator_invitation}",
        }

        initial_status = client.get("/v1/admin/development-tasks/status", headers=ADMIN_HEADERS)
        baseline_statistics = initial_status.json()["statistics"]
        baseline_complete_tasks = baseline_statistics["agreement"]["taskCount"]
        baseline_unanimous_tasks = baseline_statistics["agreement"][
            "unanimousDecisionTasks"
        ]
        baseline_missing_context = baseline_statistics["observedDefects"][
            "issueTagCounts"
        ]["missing_context"]
        task_id = next(
            row["taskId"]
            for row in initial_status.json()["tasks"]
            if row["family"] == "composition" and row["completeIndependentReviews"] == 0
        )
        for ordinal, headers in enumerate(reviewer_headers[:2], start=1):
            validity = client.post(
                f"/v1/expert/development-tasks/{task_id}/blind-validity",
                headers={**headers, "Idempotency-Key": f"blind-validity-agree-{ordinal}"},
                json=_blind_validity(),
            )
            assert validity.status_code == 200, validity.text
            criteria = client.post(
                f"/v1/expert/development-tasks/{task_id}/criteria",
                headers={**headers, "Idempotency-Key": f"criteria-agree-{ordinal}"},
                json=_criterion_pack(),
            )
            assert criteria.status_code == 200, criteria.text

        revision = client.post(
            f"/v1/expert/development-tasks/{task_id}/blind-validity",
            headers={
                **reviewer_headers[2],
                "Idempotency-Key": "blind-validity-disagreement",
            },
            json={
                **_blind_validity(),
                "decision": "revise",
                "contextComplete": False,
                "issueTags": ["missing_context"],
                "note": (
                    "The prompt omits context needed to choose between materially different "
                    "methods."
                ),
            },
        )
        assert revision.status_code == 200, revision.text
        assert revision.json()["nextPhase"] == "complete"

        status = client.get("/v1/admin/development-tasks/status", headers=ADMIN_HEADERS)
        task_status = next(row for row in status.json()["tasks"] if row["taskId"] == task_id)
        assert task_status["status"] == "awaiting_independent_adjudication"
        assert task_status["completeIndependentReviews"] == 3
        assert task_status["decisionCounts"] == {"revise": 1, "valid": 2}
        assert task_status["consensusSha256"] is None
        agreement = status.json()["statistics"]["agreement"]
        assert agreement["taskCount"] == baseline_complete_tasks + 1
        assert agreement["unanimousDecisionTasks"] == baseline_unanimous_tasks
        assert status.json()["statistics"]["observedDefects"]["issueTagCounts"][
            "missing_context"
        ] == (baseline_missing_context + 1)

        assignment = client.get(
            "/v1/expert/development-tasks/adjudication/next",
            headers=adjudicator_headers,
        )
        assert assignment.status_code == 200, assignment.text
        adjudication_task = assignment.json()["task"]
        assert adjudication_task["taskId"] == task_id
        assert adjudication_task["modelOutputsVisible"] is False
        assert len(adjudication_task["independentReviews"]) == 3

        adjudication_headers = {
            **adjudicator_headers,
            "Idempotency-Key": "development-task-disagreement-adjudication",
        }
        adjudication_payload = {
            "decision": "valid",
            "referenceAdequacy": "partial",
            "successCriteria": _criterion_pack()["successCriteria"],
            "permittedVariations": _criterion_pack()["permittedVariations"],
            "disqualifyingErrors": _criterion_pack()["disqualifyingErrors"],
            "objectiveChecks": _criterion_pack()["objectiveChecks"],
            "criteriaAuthoredByAdjudicator": True,
            "independentOfSourceReviewers": True,
            "modelOutputsNotConsulted": True,
            "note": (
                "The missing-context concern does not prevent a discriminating answer under the "
                "prompt's explicit constraints. The merged pack preserves valid alternatives."
            ),
        }
        adjudication = client.post(
            f"/v1/expert/development-tasks/{task_id}/adjudication",
            headers=adjudication_headers,
            json=adjudication_payload,
        )
        assert adjudication.status_code == 200, adjudication.text
        assert adjudication.json()["decision"] == "valid"
        replay = client.post(
            f"/v1/expert/development-tasks/{task_id}/adjudication",
            headers=adjudication_headers,
            json=adjudication_payload,
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["idempotent"] is True

        final_status = client.get("/v1/admin/development-tasks/status", headers=ADMIN_HEADERS)
        final_task_status = next(
            row for row in final_status.json()["tasks"] if row["taskId"] == task_id
        )
        assert final_task_status["status"] == "adjudicated_valid"
        assert len(final_task_status["consensusSha256"]) == 64

        frozen_intake = client.post(
            f"/v1/expert/development-tasks/{task_id}/blind-validity",
            headers={
                **adjudicator_headers,
                "Idempotency-Key": "post-adjudication-source-review",
            },
            json=_blind_validity(),
        )
        assert frozen_intake.status_code == 409
        assert frozen_intake.json()["detail"] == "task adjudication is already sealed"


def test_task_validator_identity_commitment_cannot_receive_multiple_codes() -> None:
    with TestClient(app) as client:
        _expert_invitation(
            client,
            "development-task-reviewer-fixture-21",
            identity_key="same-verified-person",
        )
        duplicate = client.post(
            "/v1/admin/task-validators",
            headers=ADMIN_HEADERS,
            json={
                "validatorCode": "development-task-reviewer-fixture-22",
                "qualifiedFamilies": ["composition"],
                "qualificationReference": "verified-culinary-practice-fixture-22",
                "verifiedIdentityHandle": "verified-private-identity:same-verified-person",
                "qualificationEvidenceSha256": hashlib.sha256(
                    b"qualification-evidence:same-verified-person"
                ).hexdigest(),
                "independenceAttestationSha256": hashlib.sha256(
                    b"independence-attestation:same-verified-person"
                ).hexdigest(),
                "verificationRecordSha256": hashlib.sha256(
                    b"verification-record:same-verified-person"
                ).hexdigest(),
                "affiliationClass": "independent_external",
                "conflictDisclosureReference": "fixture-no-conflict",
                "consentDocumentSha256": ACTIVE_EXPERT_CONSENT_SHA256,
                "adjudicationAuthorized": False,
                "evidenceVerifiedByAdmin": True,
            },
        )
        assert duplicate.status_code == 409
        assert "verified identity" in duplicate.json()["detail"]
