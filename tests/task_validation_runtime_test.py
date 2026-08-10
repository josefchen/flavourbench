from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import delete, select

import flavourbench.task_validation_runtime as task_validation_runtime
from flavourbench.config import Settings, get_settings
from flavourbench.database import SessionLocal, init_database
from flavourbench.main import app
from flavourbench.models import (
    ExpertReviewer,
    Season,
    TaskValidationAuditAuthorization,
    TaskValidationCampaignEvent,
)
from flavourbench.reviewer_identity import (
    bind_reviewer_identity,
    derive_family_admission,
    issue_reviewer_credential,
    record_qualification_evidence,
)
from flavourbench.task_validation_replay_binding import (
    TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
    TASK_VALIDATION_V1_REPLAY_SHA256,
    rights_audit_plan,
)
from flavourbench.task_validation_runtime import (
    _verify_automated_replay_cached,
    verify_task_validation_runtime_evidence,
)

SERVICE_HEADERS = {"X-FlavourBench-Service-Token": "test-service-token"}
ADMIN_HEADERS = {
    **SERVICE_HEADERS,
    "X-FlavourBench-Admin-Token": "test-admin-token",
}
CAMPAIGN_SHA256 = "76b248477b3adc81b6eb198666a93538534db8e945567e2a99fc69085f709709"
FAMILIES = ("substitution", "composition", "cookability", "evidence")
CONCURRENT_SUBMIT_HELPER = Path(__file__).with_name("task_validation_concurrent_submit_helper.py")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _copy_replay_inputs(tmp_path, monkeypatch) -> None:
    settings = get_settings()
    path_fields = (
        "task_validation_candidate_bundle_path",
        "task_validation_assignment_path",
        "task_validation_acquisition_receipt_path",
        "task_validation_campaign_path",
        "task_validation_quality_report_path",
        "task_validation_readiness_path",
        "task_validation_automated_replay_path",
    )
    for field in path_fields:
        source = getattr(settings, field)
        destination = tmp_path / source.rsplit("/", 1)[-1]
        shutil.copy2(source, destination)
        monkeypatch.setattr(settings, field, str(destination))
    _verify_automated_replay_cached.cache_clear()


@pytest.fixture(autouse=True)
def clear_runtime_evidence() -> None:
    init_database()
    with SessionLocal.begin() as session:
        session.execute(delete(TaskValidationCampaignEvent))
        session.execute(delete(TaskValidationAuditAuthorization))


def _season(session) -> Season:
    season = session.scalar(select(Season).where(Season.slug == "season-1"))
    if season is not None:
        return season
    season = Season(
        id=str(uuid.uuid4()),
        slug="season-1",
        name="Task-validation test season",
        status="running",
        official=False,
        manifest_sha256=_sha("runtime-manifest"),
        prompt_registry_sha256=_sha("runtime-prompts"),
        tool_registry_sha256=_sha("runtime-tools"),
        epicure_release_id="epicure-test",
        epicure_bundle_sha256=_sha("runtime-epicure-bundle"),
        epicure_application_sha256=_sha("runtime-epicure-app"),
        analysis_plan_sha256=_sha("runtime-analysis"),
        protocol_bundle_json={"schema_version": "runtime-test"},
        protocol_bundle_sha256=_sha("runtime-protocol"),
    )
    session.add(season)
    session.flush()
    return season


def _admitted_reviewer(*, role: str, families: tuple[str, ...] = FAMILIES) -> tuple[str, str]:
    suffix = uuid.uuid4().hex
    now = datetime.now(UTC)
    with SessionLocal() as session:
        season = _season(session)
        reviewer = ExpertReviewer(
            id=str(uuid.uuid4()),
            reviewer_code=f"runtime-{suffix}",
            invitation_sha256=_sha(f"legacy-{suffix}"),
            qualification_json=list(families),
            qualification_verified=True,
            cohort="expert_independent",
            profile_json={"affiliation_class": "independent_external"},
            active=True,
        )
        session.add(reviewer)
        session.flush()
        binding = bind_reviewer_identity(
            session,
            season=season,
            reviewer=reviewer,
            identity_issuer="https://identity.test",
            issuer_subject=f"person-{suffix}",
            identity_evidence_sha256=_sha(f"identity-{suffix}"),
            roles=[role],
        )
        for family in families:
            qualification = record_qualification_evidence(
                session,
                binding=binding,
                family=family,
                affiliation_class="independent_external",
                independence_verified=True,
                conflict_cleared=True,
                qualification_evidence_sha256=_sha(f"qualification-{suffix}-{family}"),
                independence_evidence_sha256=_sha(f"independence-{suffix}-{family}"),
                conflict_disclosure_sha256=_sha(f"conflict-{suffix}-{family}"),
                consent_document_sha256=_sha("runtime-consent"),
                training_material_sha256=_sha("runtime-training"),
                verifier_principal_sha256=_sha("runtime-verifier"),
                verified_at=now - timedelta(days=2),
                valid_until=now + timedelta(days=365),
            )
            derive_family_admission(
                session,
                binding=binding,
                qualification=qualification,
                family=family,
                review_role=role,
                cohort="expert_independent",
                admission_policy={
                    "schema_version": "flavourbench-reviewer-admission-policy-v1",
                    "requires_calibration": False,
                    "minimum_accuracy_milli": 0,
                },
                decision_reference_sha256=_sha(f"decision-{suffix}-{family}"),
                valid_from=now - timedelta(days=1),
                valid_until=now + timedelta(days=180),
            )
        token, _ = issue_reviewer_credential(
            session,
            binding=binding,
            credential_kind="review_session",
            scopes=["expert_review"],
            maximum_uses=64,
            ttl_seconds=86_400,
        )
        session.commit()
        return token, reviewer.id


def _auth(token: str) -> dict[str, str]:
    return {**SERVICE_HEADERS, "Authorization": f"Bearer {token}"}


def test_runtime_replay_returns_both_exact_human_handoff_sets() -> None:
    evidence = verify_task_validation_runtime_evidence()
    replay = json.loads(
        Path(get_settings().task_validation_automated_replay_path).read_text(encoding="utf-8")
    )
    handoff = replay["human_audit_handoff"]
    assert evidence.rights_sample_ids == tuple(handoff["rights"]["sample_candidate_ids"])
    assert evidence.rights_anomaly_ids == tuple(handoff["rights"]["anomaly_or_hit_candidate_ids"])
    assert evidence.rights_required_ids == tuple(handoff["rights"]["required_candidate_ids"])
    assert evidence.contamination_sample_ids == tuple(
        handoff["contamination"]["sample_candidate_ids"]
    )
    assert evidence.local_prompt_risk_hit_ids == tuple(
        handoff["contamination"]["anomaly_or_hit_candidate_ids"]
    )
    assert evidence.contamination_required_ids == tuple(
        handoff["contamination"]["required_candidate_ids"]
    )
    assert evidence.rights_automated_evidence_verified is True
    assert evidence.local_prompt_risk_replay_verified is True
    assert evidence.contamination_automated_evidence_verified is False
    assert evidence.contamination_campaign_coverage_verified is False


def test_enabled_campaign_requires_every_replay_input_and_digest() -> None:
    with pytest.raises(
        ValidationError,
        match="enabled task-validation campaign requires every pinned artifact and hash",
    ):
        Settings(
            task_validation_campaign_enabled=True,
            task_validation_candidate_bundle_path="",
        )


@pytest.mark.parametrize(
    "path_field",
    ("task_validation_campaign_path", "task_validation_automated_replay_path"),
)
def test_runtime_replay_mutation_returns_only_generic_503(
    tmp_path, monkeypatch, path_field: str
) -> None:
    _copy_replay_inputs(tmp_path, monkeypatch)
    with TestClient(app) as client:
        replay_path = Path(getattr(get_settings(), path_field))
        replay_path.write_bytes(replay_path.read_bytes() + b"\n")
        response = client.get("/v1/task-validation/status", headers=SERVICE_HEADERS)
        assert response.status_code == 503
        assert response.json() == {
            "detail": "task-validation automated evidence failed verification"
        }
    _verify_automated_replay_cached.cache_clear()


def test_runtime_replay_missing_input_returns_only_generic_503(tmp_path, monkeypatch) -> None:
    _copy_replay_inputs(tmp_path, monkeypatch)
    with TestClient(app) as client:
        monkeypatch.setattr(
            get_settings(),
            "task_validation_candidate_bundle_path",
            str(tmp_path / "not-mounted.json"),
        )
        response = client.get("/v1/task-validation/status", headers=SERVICE_HEADERS)
        assert response.status_code == 503
        assert response.json() == {
            "detail": "task-validation automated evidence failed verification"
        }
    _verify_automated_replay_cached.cache_clear()


def _valid_ballot(*, family: str = "substitution", note: str = "") -> dict:
    cells = {
        "substitution": "functional_role_preservation",
        "composition": "bridge_ingredient_reasoning",
        "cookability": "sequencing_and_timing",
        "evidence": "score_interpretation",
    }
    return {
        "decision": "valid",
        "checks": {
            "constructFit": True,
            "contextComplete": True,
            "coherentQuestion": True,
            "generalTrackScope": True,
            "answerLeakageAbsent": True,
            "discriminationValue": True,
        },
        "family": family,
        "constructCellId": cells[family],
        "difficultyTier": "integrative",
        "independentSolutionOutline": (
            "Identify the ingredient's functional role, then compare the proposed method."
        ),
        "successCriteria": ["Explains the functional change with a practical consequence."],
        "permittedVariations": ["Accepts another workable method with a clear rationale."],
        "disqualifyingErrors": ["Ignores the stated culinary constraint."],
        "objectiveChecks": [],
        "issueTags": [],
        "note": note,
        "sourceMetadataSeen": False,
        "otherBallotSeen": False,
        "modelOutputsSeen": False,
        "schedulingFamilySeen": False,
    }


def _post_ballot(client: TestClient, token: str, candidate: str, key: str, body: dict):
    return client.post(
        f"/v1/expert/task-validation/candidates/{candidate}/ballots",
        headers={**_auth(token), "Idempotency-Key": key},
        json=body,
    )


def test_blind_consensus_idempotency_and_public_privacy() -> None:
    first_token, _ = _admitted_reviewer(role="task_validator")
    second_token, _ = _admitted_reviewer(role="task_validator")
    with TestClient(app) as client:
        first_queue = client.get(
            "/v1/expert/task-validation/ballots/next", headers=_auth(first_token)
        )
        assert first_queue.status_code == 200
        work = first_queue.json()["work"]
        candidate = work["candidateId"]
        assert work["workType"] == "blind_ballot"
        assert work["sourceMetadataVisible"] is False
        assert work["schedulingFamilyVisible"] is False
        assert work["otherBallotVisible"] is False
        assert work["modelOutputsVisible"] is False
        assert "sourceUrl" not in work
        assert "schedulingFamily" not in work

        first = _post_ballot(
            client,
            first_token,
            candidate,
            "blind-consensus-first",
            _valid_ballot(),
        )
        assert first.status_code == 200
        repeated = _post_ballot(
            client,
            first_token,
            candidate,
            "blind-consensus-first",
            _valid_ballot(),
        )
        assert repeated.status_code == 200
        assert repeated.json()["idempotent"] is True
        changed = _valid_ballot(note="This changes the already sealed event payload.")
        conflict = _post_ballot(
            client,
            first_token,
            candidate,
            "blind-consensus-first",
            changed,
        )
        assert conflict.status_code == 409

        second_queue = client.get(
            "/v1/expert/task-validation/ballots/next", headers=_auth(second_token)
        )
        assert second_queue.json()["work"]["candidateId"] == candidate
        second = _post_ballot(
            client,
            second_token,
            candidate,
            "blind-consensus-second",
            _valid_ballot(),
        )
        assert second.status_code == 200

        first_confirmation = client.get(
            "/v1/expert/task-validation/ballots/next", headers=_auth(first_token)
        ).json()["work"]
        assert first_confirmation["workType"] == "criterion_pack_confirmation"
        pack_hash = first_confirmation["criterionPackSha256"]
        sealed_first = client.post(
            f"/v1/expert/task-validation/candidates/{candidate}/criterion-pack-confirmations",
            headers={**_auth(first_token), "Idempotency-Key": "pack-consensus-first"},
            json={
                "criterionPackSha256": pack_hash,
                "accepted": True,
                "note": "The merged pack preserves both independent criteria.",
                "modelOutputsSeen": False,
            },
        )
        assert sealed_first.status_code == 200
        interim = client.get("/v1/task-validation/status", headers=SERVICE_HEADERS).json()
        assert interim["statusCounts"]["awaiting_criterion_pack_confirmations"] == 1

        second_confirmation = client.get(
            "/v1/expert/task-validation/ballots/next", headers=_auth(second_token)
        ).json()["work"]
        assert second_confirmation["criterionPackSha256"] == pack_hash
        sealed_second = client.post(
            f"/v1/expert/task-validation/candidates/{candidate}/criterion-pack-confirmations",
            headers={**_auth(second_token), "Idempotency-Key": "pack-consensus-second"},
            json={
                "criterionPackSha256": pack_hash,
                "accepted": True,
                "note": "The deterministic union is accurate and complete.",
                "modelOutputsSeen": False,
            },
        )
        assert sealed_second.status_code == 200

        status = client.get("/v1/task-validation/status", headers=SERVICE_HEADERS).json()
        assert status["validatedTasks"] == 1
        assert status["validatedByFamily"]["substitution"] == 1
        assert status["claimBoundary"]["rankEligible"] is False
        export = client.get("/v1/admin/task-validation/export", headers=ADMIN_HEADERS).json()
        rendered = json.dumps(export, sort_keys=True)
        assert "person_commitment_sha256" not in rendered
        assert "reviewer_admission_receipt_sha256" not in rendered
        assert export["privatePersonCommitmentsIncluded"] is False
        assert export["rankEligible"] is False
        assert export["official"] is False
        assert export["taskBankImportAuthorized"] is False
        assert export["contaminationFree"] is False
        assert export["automatedEvidence"]["rightsAutomatedEvidenceVerified"] is True
        assert export["automatedEvidence"]["localPromptRiskReplayVerified"] is True
        assert export["automatedEvidence"]["contaminationAutomatedEvidenceVerified"] is False


def test_family_scoped_eligibility_cannot_act_as_a_hidden_allocation_oracle() -> None:
    tokens = [
        _admitted_reviewer(role="task_validator", families=(family,))[0] for family in FAMILIES
    ]
    with TestClient(app) as client:
        responses = [
            client.get("/v1/expert/task-validation/ballots/next", headers=_auth(token))
            for token in tokens
        ]
    assert {response.status_code for response in responses} == {403}
    details = {response.json()["detail"] for response in responses}
    assert details == {
        "active task_validator admissions across all four blinded task families are required"
    }
    assert all(family not in next(iter(details)) for family in FAMILIES)


def test_disagreement_routes_only_to_a_distinct_adjudicator() -> None:
    first_token, _ = _admitted_reviewer(role="task_validator")
    second_token, _ = _admitted_reviewer(role="task_validator")
    adjudicator_token, _ = _admitted_reviewer(role="task_adjudicator")
    with TestClient(app) as client:
        candidate = client.get(
            "/v1/expert/task-validation/ballots/next", headers=_auth(first_token)
        ).json()["work"]["candidateId"]
        assert (
            _post_ballot(
                client, first_token, candidate, "disagreement-first", _valid_ballot()
            ).status_code
            == 200
        )
        exclusion = {
            **_valid_ballot(),
            "decision": "exclude",
            "family": None,
            "constructCellId": None,
            "difficultyTier": None,
            "independentSolutionOutline": "",
            "successCriteria": [],
            "permittedVariations": [],
            "disqualifyingErrors": [],
            "objectiveChecks": [],
            "issueTags": ["specialist_scope"],
            "note": "The prompt requires a specialist safety ruling outside the general track.",
        }
        assert (
            _post_ballot(
                client, second_token, candidate, "disagreement-second", exclusion
            ).status_code
            == 200
        )
        work = client.get(
            "/v1/expert/task-validation/adjudications/next",
            headers=_auth(adjudicator_token),
        ).json()["work"]
        assert work["candidateId"] == candidate
        assert work["validatorIdentitiesVisible"] is False
        assert work["sourceMetadataVisible"] is False
        assert work["schedulingFamilyVisible"] is False
        assert work["modelOutputsVisible"] is False

        source_reviewer_attempt = client.post(
            f"/v1/expert/task-validation/candidates/{candidate}/adjudications",
            headers={**_auth(first_token), "Idempotency-Key": "source-cannot-adjudicate"},
            json={
                "decision": "reject",
                "note": "Reject because the specialist ruling is out of scope.",
                "modelOutputsSeen": False,
                "independentAttestation": True,
            },
        )
        assert source_reviewer_attempt.status_code in {403, 409}

        adjudicated = client.post(
            f"/v1/expert/task-validation/candidates/{candidate}/adjudications",
            headers={**_auth(adjudicator_token), "Idempotency-Key": "distinct-adjudication"},
            json={
                "decision": "reject",
                "note": "Reject because the task requires a specialist safety determination.",
                "modelOutputsSeen": False,
                "independentAttestation": True,
            },
        )
        assert adjudicated.status_code == 200
        replayed = client.post(
            f"/v1/expert/task-validation/candidates/{candidate}/adjudications",
            headers={**_auth(adjudicator_token), "Idempotency-Key": "distinct-adjudication"},
            json={
                "decision": "reject",
                "note": "Reject because the task requires a specialist safety determination.",
                "modelOutputsSeen": False,
                "independentAttestation": True,
            },
        )
        assert replayed.status_code == 200
        assert replayed.json()["idempotent"] is True
        changed_replay = client.post(
            f"/v1/expert/task-validation/candidates/{candidate}/adjudications",
            headers={**_auth(adjudicator_token), "Idempotency-Key": "distinct-adjudication"},
            json={
                "decision": "reject",
                "note": "A changed rationale cannot reuse a sealed idempotency key.",
                "modelOutputsSeen": False,
                "independentAttestation": True,
            },
        )
        assert changed_replay.status_code == 409
        status = client.get("/v1/task-validation/status", headers=SERVICE_HEADERS).json()
        assert status["statusCounts"]["adjudicated_reject"] == 1
        assert status["validatedTasks"] == 0


def test_batch_auditor_is_distinct_and_public_source_claim_stays_bounded() -> None:
    auditor_token, auditor_id = _admitted_reviewer(role="task_validator")
    with TestClient(app) as client:
        forbidden_client_evidence = client.post(
            "/v1/admin/task-validation/auditors",
            headers=ADMIN_HEADERS,
            json={
                "reviewerId": auditor_id,
                "auditKind": "rights",
                "qualificationEvidenceSha256": _sha("rights-qualification-evidence"),
                "conflictEvidenceSha256": _sha("rights-conflict-evidence"),
                "automatedEvidenceSha256": _sha("rights-complete-replay-evidence"),
                "automatedCoverageComplete": True,
                "automatedHitCandidateIds": [],
                "decisionReferenceSha256": _sha("rights-authorization-decision"),
            },
        )
        assert forbidden_client_evidence.status_code == 422
        authorized = client.post(
            "/v1/admin/task-validation/auditors",
            headers=ADMIN_HEADERS,
            json={
                "reviewerId": auditor_id,
                "auditKind": "rights",
                "qualificationEvidenceSha256": _sha("rights-qualification-evidence"),
                "conflictEvidenceSha256": _sha("rights-conflict-evidence"),
                "decisionReferenceSha256": _sha("rights-authorization-decision"),
            },
        )
        assert authorized.status_code == 200
        assert authorized.json()["authorizationTrustBoundary"] == (
            "server_verified_frozen_replay_plus_admin_human_evidence"
        )
        assert authorized.json()["automatedEvidenceSha256"] == TASK_VALIDATION_V1_REPLAY_SHA256
        assert authorized.json()["automatedEvidenceVerified"] is True
        assert authorized.json()["rightsSnapshotIntegrityVerified"] is True
        assert authorized.json()["localPromptRiskReplayVerified"] is True
        assert authorized.json()["contaminationAutomatedEvidenceVerified"] is False
        assert authorized.json()["contaminationCampaignCoverageVerified"] is False
        second_role = client.post(
            "/v1/admin/task-validation/auditors",
            headers=ADMIN_HEADERS,
            json={
                "reviewerId": auditor_id,
                "auditKind": "contamination",
                "qualificationEvidenceSha256": _sha("contamination-qualification"),
                "conflictEvidenceSha256": _sha("contamination-conflict"),
                "decisionReferenceSha256": _sha("contamination-decision"),
            },
        )
        assert second_role.status_code == 409
        validation_attempt = client.get(
            "/v1/expert/task-validation/ballots/next", headers=_auth(auditor_token)
        )
        assert validation_attempt.status_code == 409

        queue = client.get("/v1/expert/task-validation/audits/next", headers=_auth(auditor_token))
        assert queue.status_code == 200
        work = queue.json()["work"]
        assert work["auditKind"] == "rights"
        assert work["sampleCandidateIds"] == rights_audit_plan()["sample_candidate_ids"]
        assert (
            work["anomalyOrHitCandidateIds"] == rights_audit_plan()["anomaly_or_hit_candidate_ids"]
        )
        assert work["automatedEvidenceVerified"] is True
        assert work["localPromptRiskReplayVerified"] is True
        assert work["contaminationAutomatedEvidenceVerified"] is False
        assert work["contaminationCampaignCoverageVerified"] is False
        assert len(work["records"]) >= 24
        assert work["modelOutputsVisible"] is False
        reviewed = [record["candidateId"] for record in work["records"]]
        audit_body = {
            "auditKind": "rights",
            "decision": "pass",
            "reviewedCandidateIds": reviewed,
            "findings": [],
            "unresolvedMaterialFindings": 0,
            "completeCoverageEvidenceChecked": True,
            "noModelOutputsSeen": True,
            "publicSourceContaminationLimitedAcknowledged": True,
            "noContaminationFreeClaim": True,
            "note": (
                "The frozen sample and every license anomaly were checked "
                "against the source records."
            ),
        }
        submitted = client.post(
            "/v1/expert/task-validation/audits",
            headers={**_auth(auditor_token), "Idempotency-Key": "rights-audit-submit"},
            json=audit_body,
        )
        assert submitted.status_code == 200
        exact_retry = client.post(
            "/v1/expert/task-validation/audits",
            headers={**_auth(auditor_token), "Idempotency-Key": "rights-audit-submit"},
            json=audit_body,
        )
        assert exact_retry.status_code == 200
        assert exact_retry.json()["idempotent"] is True
        conflicting_retry = client.post(
            "/v1/expert/task-validation/audits",
            headers={**_auth(auditor_token), "Idempotency-Key": "rights-audit-replacement"},
            json=audit_body,
        )
        assert conflicting_retry.status_code == 409
        assert conflicting_retry.json()["detail"] == (
            "task-validation batch audit is already sealed"
        )
        changed_exact_key = client.post(
            "/v1/expert/task-validation/audits",
            headers={**_auth(auditor_token), "Idempotency-Key": "rights-audit-submit"},
            json={**audit_body, "note": "A changed sealed audit cannot reuse its key."},
        )
        assert changed_exact_key.status_code == 409
        assert changed_exact_key.json()["detail"] == (
            "idempotency key was already used for different content"
        )
        status = client.get("/v1/task-validation/status", headers=SERVICE_HEADERS).json()
        assert status["batchAudits"] == 1
        assert status["audits"]["rights"]["humanDecision"] == "pass"
        assert status["audits"]["rights"]["automatedEvidenceVerified"] is True
        assert status["releaseGate"]["rightsHumanAuditPassed"] is True
        assert status["releaseGate"]["rightsAutomatedEvidenceVerified"] is True
        assert status["releaseGate"]["localPromptRiskReplayVerified"] is True
        assert status["releaseGate"]["contaminationAutomatedEvidenceVerified"] is False
        assert status["releaseGate"]["contaminationCampaignCoverageVerified"] is False
        assert status["releaseGate"]["rightsAuditPassed"] is True
        assert status["claimBoundary"]["contaminationFree"] is False
        assert status["releaseGate"]["taskBankImportAuthorized"] is False
        with SessionLocal() as session:
            authorization = session.scalar(select(TaskValidationAuditAuthorization))
            assert authorization is not None
            assert authorization.automated_evidence_sha256 == TASK_VALIDATION_V1_REPLAY_SHA256
            assert authorization.audit_plan_json == rights_audit_plan()
            assert authorization.audit_plan_sha256 == TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256


def test_concurrent_third_ballot_cannot_overfill_two_person_slots(tmp_path: Path) -> None:
    first_token, _ = _admitted_reviewer(role="task_validator")
    second_token, _ = _admitted_reviewer(role="task_validator")
    third_token, _ = _admitted_reviewer(role="task_validator")
    with TestClient(app) as client:
        candidate = client.get(
            "/v1/expert/task-validation/ballots/next", headers=_auth(first_token)
        ).json()["work"]["candidateId"]
        assert (
            _post_ballot(
                client, first_token, candidate, "concurrency-first", _valid_ballot()
            ).status_code
            == 200
        )

    barrier_directory = tmp_path / "cross-process-barrier"
    body_json = json.dumps(_valid_ballot(), sort_keys=True)
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(CONCURRENT_SUBMIT_HELPER),
                "--token",
                token,
                "--candidate",
                candidate,
                "--key",
                key,
                "--body-json",
                body_json,
                "--barrier-directory",
                str(barrier_directory),
                "--order",
                order,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for token, key, order in (
            (second_token, "concurrency-second", "first"),
            (third_token, "concurrency-third", "second"),
        )
    ]
    outputs: list[str] = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
            outputs.append(stdout)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
    statuses = [
        int(json.loads(output.strip().splitlines()[-1])["status_code"]) for output in outputs
    ]
    assert sorted(statuses) == [200, 409]
    with SessionLocal() as session:
        rows = list(
            session.scalars(
                select(TaskValidationCampaignEvent)
                .where(
                    TaskValidationCampaignEvent.campaign_sha256 == CAMPAIGN_SHA256,
                    TaskValidationCampaignEvent.candidate_id == candidate,
                    TaskValidationCampaignEvent.event_type == "blind_ballot",
                )
                .order_by(TaskValidationCampaignEvent.sequence)
            ).all()
        )
        assert len(rows) == 2
        all_rows = list(
            session.scalars(
                select(TaskValidationCampaignEvent)
                .where(TaskValidationCampaignEvent.campaign_sha256 == CAMPAIGN_SHA256)
                .order_by(TaskValidationCampaignEvent.sequence)
            ).all()
        )
        assert [row.sequence for row in all_rows] == list(range(1, len(all_rows) + 1))


def test_pre_freeze_withdrawal_filter_preserves_raw_task_event_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token, reviewer_id = _admitted_reviewer(role="task_validator")
    with TestClient(app) as client:
        candidate = client.get(
            "/v1/expert/task-validation/ballots/next", headers=_auth(token)
        ).json()["work"]["candidateId"]
        assert (
            _post_ballot(
                client,
                token,
                candidate,
                "pre-freeze-exclusion-ballot",
                _valid_ballot(),
            ).status_code
            == 200
        )

    monkeypatch.setattr(
        task_validation_runtime,
        "participant_record_analysis_eligible",
        lambda _session, **values: values["reviewer_id"] != reviewer_id,
    )
    with SessionLocal() as session:
        raw_rows, raw_documents = task_validation_runtime._campaign_events(
            session,
            CAMPAIGN_SHA256,
            analysis_eligible_only=False,
        )
        eligible_rows, eligible_documents = task_validation_runtime._campaign_events(
            session,
            CAMPAIGN_SHA256,
        )
    assert len(raw_rows) == len(raw_documents) == 1
    assert eligible_rows == []
    assert eligible_documents == []
