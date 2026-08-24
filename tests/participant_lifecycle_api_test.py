from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import flavourbench.main as main
import flavourbench.participant_lifecycle as participant_lifecycle
from flavourbench.models import Base, ExpertReviewer, Season
from flavourbench.participant_lifecycle import (
    ActiveHumanStudyBinding,
    ParticipantLifecycleError,
)

CONSENT_SHA256 = "a" * 64
ACTIVATION_SHA256 = "b" * 64
RETENTION_SHA256 = "c" * 64
SERVICE_HEADERS = {"X-FlavourBench-Service-Token": "test-service-token"}
ADMIN_HEADERS = {
    **SERVICE_HEADERS,
    "X-FlavourBench-Admin-Token": "test-admin-token",
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.fixture
def participant_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, Path, dict[str, bool]]:
    database_path = tmp_path / "participant-api.sqlite3"
    engine = create_engine(f"sqlite+pysqlite:///{database_path}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as session:
        session.add(
            Season(
                id="participant-api-season",
                slug="participant-api-season",
                name="Participant API season",
                epicure_release_id="epicure-test",
            )
        )
        session.commit()

    governance = {"active": True}

    def active_gate(**_kwargs: object) -> ActiveHumanStudyBinding:
        if not governance["active"]:
            raise ParticipantLifecycleError("human-study activation is suspended")
        return ActiveHumanStudyBinding(
            consent_document_sha256=CONSENT_SHA256,
            activation_manifest_sha256=ACTIVATION_SHA256,
            retention_policy_sha256=RETENTION_SHA256,
            consent_text="# Participant API test consent",
        )

    monkeypatch.setattr(
        participant_lifecycle,
        "require_active_human_study",
        active_gate,
    )
    monkeypatch.setattr(
        participant_lifecycle,
        "require_original_accepted_manifest",
        lambda _acceptance, **_kwargs: None,
    )

    def override_database():
        with sessions() as session:
            yield session

    prior_overrides = dict(main.app.dependency_overrides)
    main.app.dependency_overrides[main.get_db] = override_database
    client = TestClient(main.app)
    try:
        yield client, database_path, governance
    finally:
        client.close()
        main.app.dependency_overrides.clear()
        main.app.dependency_overrides.update(prior_overrides)
        engine.dispose()


def test_participant_api_is_consent_first_privacy_safe_and_rights_preserving(
    participant_api: tuple[TestClient, Path, dict[str, bool]],
) -> None:
    client, database_path, governance = participant_api
    issued = client.post(
        "/v1/admin/seasons/participant-api-season/participant-enrollment-offers",
        headers=ADMIN_HEADERS,
        json={
            "consentDocumentSha256": CONSENT_SHA256,
            "ttlSeconds": 3600,
        },
    )
    assert issued.status_code == 200, issued.text
    enrollment_credential = issued.json()["enrollmentCredential"]
    assert issued.json()["identityCollected"] is False

    missing_scheme = client.get(
        "/v1/participant/enrollment/consent",
        headers={**SERVICE_HEADERS, "Authorization": enrollment_credential},
    )
    assert missing_scheme.status_code == 401

    enrollment_headers = {
        **SERVICE_HEADERS,
        "Authorization": f"Bearer {enrollment_credential}",
    }
    consent = client.get(
        "/v1/participant/enrollment/consent",
        headers=enrollment_headers,
    )
    assert consent.status_code == 200, consent.text
    assert consent.json()["consentDocumentSha256"] == CONSENT_SHA256
    assert consent.json()["activationManifestSha256"] == ACTIVATION_SHA256

    acceptance_body = {
        "consentDocumentSha256": CONSENT_SHA256,
        "activationManifestSha256": ACTIVATION_SHA256,
        "confirmations": list(participant_lifecycle.CONSENT_CONFIRMATIONS),
        "idempotencyKey": "participant-api-consent-once",
    }
    accepted = client.post(
        "/v1/participant/enrollment/consent-acceptance",
        headers=enrollment_headers,
        json=acceptance_body,
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["identityCollected"] is False
    receipt_credential = accepted.json()["consentReceiptCredential"]
    receipt_headers = {
        **SERVICE_HEADERS,
        "Authorization": f"Bearer {receipt_credential}",
    }

    raw_subject = "raw-participant-subject-must-never-persist"
    identity_body = {
        "identityIssuer": "https://identity.example.test",
        "issuerSubject": raw_subject,
        "identityEvidenceSha256": _sha("participant-api-identity"),
        "roles": ["output_rater"],
        "qualifiedFamilies": ["cookability"],
        "affiliationClass": "independent_external",
    }
    rejected_contact = client.post(
        "/v1/participant/enrollment/identity",
        headers=receipt_headers,
        json={**identity_body, "contactEmail": "prohibited@example.test"},
    )
    assert rejected_contact.status_code == 422

    identity = client.post(
        "/v1/participant/enrollment/identity",
        headers=receipt_headers,
        json=identity_body,
    )
    assert identity.status_code == 200, identity.text
    assert identity.json()["participationStatus"] == "active"
    assert identity.json()["rawIdentityPersisted"] is False
    assert identity.json()["contactDataPersisted"] is False
    assert raw_subject not in identity.text
    assert "reviewerId" not in identity.text
    assert "identityBinding" not in identity.text

    with Session(create_engine(f"sqlite+pysqlite:///{database_path}")) as session:
        reviewer_id = session.scalar(select(ExpertReviewer.id))
        assert reviewer_id is not None

    schedule = client.post(
        f"/v1/admin/reviewers/{reviewer_id}/retention-schedule",
        headers=ADMIN_HEADERS,
        json={
            "analysisFreezeAt": "2026-08-09T12:00:00+00:00",
            "firstPublicReleaseAt": "2026-08-09T12:00:00+00:00",
        },
    )
    assert schedule.status_code == 200, schedule.text

    governance["active"] = False
    blocked_forward_identity = client.post(
        "/v1/participant/enrollment/identity",
        headers=receipt_headers,
        json=identity_body,
    )
    assert blocked_forward_identity.status_code == 409

    withdrawal = client.post(
        "/v1/participant/withdrawal",
        headers=receipt_headers,
        json={
            "idempotencyKey": "participant-api-withdrawal-once",
            "reasonCode": "privacy_request",
        },
    )
    assert withdrawal.status_code == 200, withdrawal.text
    assert withdrawal.json()["participationStatus"] == "withdrawn"
    assert withdrawal.json()["priorJudgmentsPreserved"] is True

    deletion = client.post(
        "/v1/participant/private-payload-deletion",
        headers=receipt_headers,
        json={"idempotencyKey": "participant-api-deletion-once"},
    )
    assert deletion.status_code == 200, deletion.text
    assert deletion.json()["participationStatus"] == "redacted"
    assert deletion.json()["priorJudgmentsPreserved"] is True

    status = client.get("/v1/participant/status", headers=receipt_headers)
    assert status.status_code == 200, status.text
    assert status.json()["participationStatus"] == "redacted"
    prohibited_public_fields = (
        "reviewerId",
        "identityBinding",
        "personCommitment",
        "contact",
        "issuerSubject",
        "credentialPrefix",
    )
    assert all(field not in status.text for field in prohibited_public_fields)
    assert raw_subject.encode() not in database_path.read_bytes()
