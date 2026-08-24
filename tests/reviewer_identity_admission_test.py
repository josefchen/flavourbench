from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from flavourbench.config import Settings, get_settings
from flavourbench.database import SessionLocal
from flavourbench.main import app
from flavourbench.models import (
    Base,
    Battle,
    CatalogModel,
    ExpertReviewer,
    ResponseArm,
    ReviewerAccessCredential,
    ReviewerConsentAcceptance,
    ReviewerEnrollmentOffer,
    ReviewerParticipationLifecycle,
    ReviewerRetentionSchedule,
    Season,
    Vote,
)
from flavourbench.reviewer_identity import (
    ReviewerIdentityError,
    apply_verified_vote_provenance,
    bind_reviewer_identity,
    consume_reviewer_credential,
    derive_family_admission,
    filter_ranking_vote_rows,
    freeze_calibration_set,
    issue_reviewer_credential,
    privacy_safe_vote_release,
    record_calibration_ballot,
    record_qualification_evidence,
    reviewer_rater_pseudonym,
    season_person_commitment,
    verified_vote_person_commitment,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.fixture
def engine():
    database = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(database)
    return database


def _settings() -> Settings:
    return Settings(
        reviewer_identity_hmac_secret="reviewer-identity-secret-with-more-than-32-characters",
        reviewer_credential_hmac_secret=("reviewer-credential-secret-with-more-than-32-characters"),
        pseudonym_secret="reviewer-pseudonym-secret-with-more-than-32-characters",
    )


def _season(session: Session, suffix: str = "one") -> Season:
    season = Season(
        id=f"season-{suffix}",
        slug=f"reviewer-season-{suffix}",
        name=f"Reviewer season {suffix}",
        status="running",
        official=True,
        manifest_sha256=_sha(f"manifest-{suffix}"),
        prompt_registry_sha256=_sha(f"prompts-{suffix}"),
        tool_registry_sha256=_sha(f"tools-{suffix}"),
        epicure_release_id="epicure-test",
        epicure_bundle_sha256=_sha(f"epicure-bundle-{suffix}"),
        epicure_application_sha256=_sha(f"epicure-application-{suffix}"),
        analysis_plan_sha256=_sha(f"analysis-{suffix}"),
        protocol_bundle_json={"version": suffix},
        protocol_bundle_sha256=_sha(f"protocol-{suffix}"),
    )
    session.add(season)
    session.flush()
    return season


def _reviewer(session: Session, suffix: str) -> ExpertReviewer:
    reviewer = ExpertReviewer(
        id=f"reviewer-{suffix}",
        reviewer_code=f"reviewer-{suffix}",
        invitation_sha256=_sha(f"legacy-invite-{suffix}"),
        qualification_json=["cookability"],
        qualification_verified=True,
        cohort="expert_independent",
        profile_json={"affiliation_class": "independent_external"},
        active=True,
    )
    session.add(reviewer)
    session.flush()
    return reviewer


def _calibration_set(session: Session, season: Season, now: datetime):
    return freeze_calibration_set(
        session,
        season=season,
        family="cookability",
        calibration_set_sha256=_sha(f"calibration-set-{season.id}"),
        source_artifact_sha256=_sha(f"calibration-source-{season.id}"),
        scoring_key_sha256=_sha(f"calibration-key-{season.id}"),
        item_count=10,
        real_source_arms=20,
        frozen_at=now - timedelta(days=4),
    )


def _admit_output_rater(
    session: Session,
    *,
    season: Season,
    calibration_set,
    suffix: str,
    now: datetime,
):
    reviewer = _reviewer(session, suffix)
    binding = bind_reviewer_identity(
        session,
        season=season,
        reviewer=reviewer,
        identity_issuer="https://identity.example.test",
        issuer_subject=f"opaque-subject-{suffix}",
        identity_evidence_sha256=_sha(f"identity-evidence-{suffix}"),
        roles=["output_rater"],
        settings=_settings(),
    )
    qualification = record_qualification_evidence(
        session,
        binding=binding,
        family="cookability",
        affiliation_class="independent_external",
        independence_verified=True,
        conflict_cleared=True,
        qualification_evidence_sha256=_sha(f"qualification-{suffix}"),
        independence_evidence_sha256=_sha(f"independence-{suffix}"),
        conflict_disclosure_sha256=_sha(f"conflict-{suffix}"),
        consent_document_sha256=_sha("consent"),
        training_material_sha256=_sha("training"),
        verifier_principal_sha256=_sha("admin-principal"),
        verified_at=now - timedelta(days=3),
        valid_until=now + timedelta(days=365),
    )
    ballot = record_calibration_ballot(
        session,
        binding=binding,
        calibration_set=calibration_set,
        ballot_sha256=_sha(f"ballot-{suffix}"),
        scoring_result_sha256=_sha(f"ballot-score-{suffix}"),
        correct_count=9,
        minimum_accuracy_milli=800,
        completed_at=now - timedelta(days=2),
    )
    policy = {
        "schema_version": "flavourbench-reviewer-admission-policy-v1",
        "requires_calibration": True,
        "minimum_accuracy_milli": 800,
    }
    admission = derive_family_admission(
        session,
        binding=binding,
        qualification=qualification,
        calibration_ballot=ballot,
        family="cookability",
        review_role="output_rater",
        cohort="expert_independent",
        admission_policy=policy,
        decision_reference_sha256=_sha(f"admission-decision-{suffix}"),
        valid_from=now - timedelta(days=1),
        valid_until=now + timedelta(days=180),
    )
    return reviewer, binding, admission


def _completed_battle(session: Session, season: Season, now: datetime) -> Battle:
    for suffix in ("left", "right"):
        session.execute(
            CatalogModel.__table__.insert().values(
                model_id=f"model-{suffix}",
                canonical_slug=f"model-{suffix}",
                name=f"Model {suffix}",
            )
        )
    session.execute(
        Battle.__table__.insert().values(
            id="verified-review-battle",
            season_id=season.id,
            run_class="official",
            rank_eligible=True,
            data_stratum="public_freeform",
            manifest_sha256=season.manifest_sha256,
            protocol_bundle_sha256=season.protocol_bundle_sha256,
            scheduler_version="reviewer-identity-test-v1",
            assignment_seed=_sha("assignment"),
            track_assignment_probability="1/1",
            model_assignment_probability="1/1",
            side_assignment_probability="1/2",
            track="model_arena",
            category="cookability",
            prompt="Design a practical supper.",
            prompt_sha256=_sha("prompt"),
            client_nonce_sha256=_sha("nonce"),
            requester_pseudonym=_sha("requester"),
            status="complete",
            left_arm_id="verified-left-arm",
            right_arm_id="verified-right-arm",
            completed_at=now - timedelta(minutes=1),
            retention_until=now + timedelta(days=30),
        )
    )
    for side in ("left", "right"):
        session.execute(
            ResponseArm.__table__.insert().values(
                id=f"verified-{side}-arm",
                battle_id="verified-review-battle",
                side=side,
                condition="epicure_on",
                model_id=f"model-{side}",
                actual_provider_slug="test-provider",
                actual_model_id=f"model-{side}",
                generation_id=f"generation-{side}",
                provider_generation_ids_json=[f"generation-{side}"],
                status="complete",
                answer_markdown=f"Answer {side}",
                answer_markdown_sha256=_sha(f"answer-{side}"),
                output_json={},
                output_json_sha256=_sha("{}"),
                prompt_sha256=_sha("prompt"),
                system_prompt_sha256=_sha("system"),
                schema_sha256=_sha("schema"),
                tool_schema_sha256=_sha("tool-schema"),
                protocol_bundle_sha256=season.protocol_bundle_sha256,
                epicure_release_id="epicure-test",
                epicure_bundle_sha256=season.epicure_bundle_sha256,
                epicure_application_sha256=season.epicure_application_sha256,
                cost_reconciled=True,
                cost_accounting_basis="test_exact",
                billing_reconciliation_status="test_exact",
                finish_reason="stop",
                completed_at=now - timedelta(minutes=2),
            )
        )
    session.flush()
    battle = session.get(Battle, "verified-review-battle")
    assert battle is not None
    return battle


def test_person_commitment_is_stable_within_season_and_unlinkable_across_seasons() -> None:
    first, issuer = season_person_commitment(
        season_id="season-one",
        identity_issuer="HTTPS://ID.EXAMPLE ",
        issuer_subject="opaque-123",
        settings=_settings(),
    )
    repeated, repeated_issuer = season_person_commitment(
        season_id="season-one",
        identity_issuer="https://id.example",
        issuer_subject="opaque-123",
        settings=_settings(),
    )
    next_season, _ = season_person_commitment(
        season_id="season-two",
        identity_issuer="https://id.example",
        issuer_subject="opaque-123",
        settings=_settings(),
    )

    assert first == repeated
    assert issuer == repeated_issuer
    assert first != next_season
    assert "opaque-123" not in first


def test_database_rejects_same_person_under_a_second_reviewer_code(engine) -> None:
    with Session(engine) as session:
        season = _season(session)
        first = _reviewer(session, "identity-a")
        bind_reviewer_identity(
            session,
            season=season,
            reviewer=first,
            identity_issuer="https://id.example",
            issuer_subject="one-natural-person",
            identity_evidence_sha256=_sha("first-identity-evidence"),
            roles=["task_validator"],
            settings=_settings(),
        )
        session.commit()

    with Session(engine) as session:
        season = session.get(Season, "season-one")
        assert season is not None
        second = _reviewer(session, "identity-b")
        with pytest.raises(IntegrityError):
            bind_reviewer_identity(
                session,
                season=season,
                reviewer=second,
                identity_issuer="https://id.example",
                issuer_subject="one-natural-person",
                identity_evidence_sha256=_sha("second-identity-evidence"),
                roles=["task_adjudicator"],
                settings=_settings(),
            )


def test_one_time_enrollment_credential_is_hash_only_and_exhausts(engine) -> None:
    with Session(engine) as session:
        season = _season(session)
        reviewer = _reviewer(session, "credential")
        binding = bind_reviewer_identity(
            session,
            season=season,
            reviewer=reviewer,
            identity_issuer="https://id.example",
            issuer_subject="credential-person",
            identity_evidence_sha256=_sha("credential-identity-evidence"),
            roles=["output_rater"],
            settings=_settings(),
        )
        token, credential = issue_reviewer_credential(
            session,
            binding=binding,
            credential_kind="enrollment_once",
            scopes=["exchange_review_session"],
            settings=_settings(),
        )
        credential_id = credential.id
        assert token not in credential.secret_hmac_sha256
        assert credential.maximum_uses == 1
        session.commit()

    with Session(engine) as session:
        consumed = consume_reviewer_credential(
            session,
            token=token,
            required_scope="exchange_review_session",
            settings=_settings(),
        )
        assert consumed.id == credential_id
        session.commit()

    with Session(engine) as session:
        with pytest.raises(ReviewerIdentityError, match="exhausted"):
            consume_reviewer_credential(
                session,
                token=token,
                required_scope="exchange_review_session",
                settings=_settings(),
            )


def test_verified_votes_require_two_distinct_people_and_release_no_locators(
    engine,
) -> None:
    now = datetime.now(UTC)
    with Session(engine) as session:
        season = _season(session)
        calibration_set = _calibration_set(session, season, now)
        first = _admit_output_rater(
            session,
            season=season,
            calibration_set=calibration_set,
            suffix="vote-a",
            now=now,
        )
        second = _admit_output_rater(
            session,
            season=season,
            calibration_set=calibration_set,
            suffix="vote-b",
            now=now,
        )
        battle = _completed_battle(session, season, now)
        verified_votes = []
        for ordinal, (reviewer, binding, admission) in enumerate((first, second), start=1):
            vote = Vote(
                id=f"verified-vote-{ordinal}",
                battle_id=battle.id,
                rater_pseudonym=reviewer_rater_pseudonym(binding, settings=_settings()),
                cohort="expert_independent",
                choice="left" if ordinal == 1 else "tie",
                reason_tags_json=["impractical"],
                rubric_json={},
                idempotency_key=f"verified-vote-key-{ordinal}",
            )
            apply_verified_vote_provenance(
                vote,
                reviewer=reviewer,
                binding=binding,
                admission=admission,
            )
            session.add(vote)
            verified_votes.append(vote)
        legacy = Vote(
            id="legacy-expert-vote",
            battle_id=battle.id,
            rater_pseudonym=_sha("legacy-rater"),
            cohort="expert_independent",
            choice="right",
            reason_tags_json=[],
            rubric_json={},
            idempotency_key="legacy-expert-vote-key",
            provenance_status="legacy_unverified",
        )
        session.add(legacy)
        session.commit()

        one_row = filter_ranking_vote_rows(
            session,
            [(verified_votes[0], battle)],
            expert_quorum=2,
        )
        all_rows = filter_ranking_vote_rows(
            session,
            [(verified_votes[0], battle), (verified_votes[1], battle), (legacy, battle)],
            expert_quorum=2,
        )
        assert one_row == []
        assert [vote.id for vote, _ in all_rows] == [
            "verified-vote-1",
            "verified-vote-2",
        ]

        released = privacy_safe_vote_release(verified_votes[0])
        serialized = str(released)
        assert set(released) == {"cohort", "choice", "reasonTags", "provenanceClass"}
        assert "reviewer-" not in serialized
        assert first[1].person_commitment_sha256 not in serialized
        assert first[2].id not in serialized

        session.execute(
            first[2]
            .__table__.update()
            .where(first[2].__table__.c.id == first[2].id)
            .values(evidence_bundle_sha256=_sha("forged-admission-evidence"))
        )
        session.expire_all()
        reloaded_battle = session.get(Battle, battle.id)
        assert reloaded_battle is not None
        reloaded_votes = [session.get(Vote, vote.id) for vote in verified_votes]
        assert all(vote is not None for vote in reloaded_votes)
        assert (
            filter_ranking_vote_rows(
                session,
                [(vote, reloaded_battle) for vote in reloaded_votes if vote is not None],
                expert_quorum=2,
            )
            == []
        )


def test_verified_vote_withdrawal_eligibility_uses_the_frozen_analysis_date(
    engine,
) -> None:
    now = datetime.now(UTC)
    with Session(engine) as session:
        season = _season(session, "withdrawal-freeze")
        calibration_set = _calibration_set(session, season, now)
        admitted = [
            _admit_output_rater(
                session,
                season=season,
                calibration_set=calibration_set,
                suffix=f"withdrawal-freeze-{suffix}",
                now=now,
            )
            for suffix in ("pre", "post")
        ]
        battle = _completed_battle(session, season, now)
        votes: list[Vote] = []
        for suffix, (reviewer, binding, admission) in zip(("pre", "post"), admitted, strict=True):
            vote = Vote(
                id=f"withdrawal-freeze-vote-{suffix}",
                battle_id=battle.id,
                rater_pseudonym=reviewer_rater_pseudonym(binding, settings=_settings()),
                cohort="expert_independent",
                choice="left",
                reason_tags_json=[],
                rubric_json={},
                idempotency_key=f"withdrawal-freeze-vote-key-{suffix}",
                created_at=now,
            )
            apply_verified_vote_provenance(
                vote,
                reviewer=reviewer,
                binding=binding,
                admission=admission,
            )
            session.add(vote)
            votes.append(vote)
        session.flush()

        for suffix, (reviewer, binding, _admission), freeze_at in zip(
            ("pre", "post"),
            admitted,
            (now + timedelta(hours=2), now + timedelta(minutes=30)),
            strict=True,
        ):
            offer_id = f"withdrawal-freeze-offer-{suffix}"
            acceptance_id = f"withdrawal-freeze-acceptance-{suffix}"
            lifecycle_id = f"withdrawal-freeze-lifecycle-{suffix}"
            session.execute(
                ReviewerEnrollmentOffer.__table__.insert().values(
                    id=offer_id,
                    season_id=season.id,
                    credential_prefix=f"freeze{suffix}",
                    secret_hmac_sha256=_sha(f"offer-secret-{suffix}"),
                    hmac_key_id="test",
                    consent_document_sha256=_sha("consent"),
                    activation_manifest_sha256=_sha("activation"),
                    status="accepted",
                    not_before=now - timedelta(days=2),
                    expires_at=now + timedelta(days=2),
                    accepted_at=now - timedelta(days=1),
                    accepted_request_sha256=_sha(f"accepted-request-{suffix}"),
                    created_at=now - timedelta(days=2),
                )
            )
            session.execute(
                ReviewerConsentAcceptance.__table__.insert().values(
                    id=acceptance_id,
                    enrollment_offer_id=offer_id,
                    season_id=season.id,
                    consent_document_sha256=_sha("consent"),
                    activation_manifest_sha256=_sha("activation"),
                    retention_policy_sha256=_sha("retention"),
                    acceptance_statement_sha256=_sha("acceptance-statement"),
                    confirmation_set_sha256=_sha("confirmation-set"),
                    request_sha256=_sha(f"accepted-request-{suffix}"),
                    receipt_prefix=f"receipt{suffix}",
                    receipt_secret_hmac_sha256=_sha(f"receipt-secret-{suffix}"),
                    hmac_key_id="test",
                    receipt_sha256=_sha(f"receipt-{suffix}"),
                    accepted_at=now - timedelta(days=1),
                    created_at=now - timedelta(days=1),
                )
            )
            withdrawn_at = now + timedelta(hours=1)
            session.execute(
                ReviewerParticipationLifecycle.__table__.insert().values(
                    id=lifecycle_id,
                    consent_acceptance_id=acceptance_id,
                    season_id=season.id,
                    reviewer_id=reviewer.id,
                    identity_binding_id=binding.id,
                    audit_marker_sha256=_sha(f"audit-marker-{suffix}"),
                    status="withdrawn",
                    withdrawn_at=withdrawn_at,
                    assignments_stopped_at=withdrawn_at,
                    withdrawal_receipt_sha256=_sha(f"withdrawal-receipt-{suffix}"),
                    created_at=now - timedelta(hours=1),
                )
            )
            session.execute(
                ReviewerRetentionSchedule.__table__.insert().values(
                    id=f"withdrawal-freeze-schedule-{suffix}",
                    lifecycle_id=lifecycle_id,
                    season_id=season.id,
                    reviewer_id=reviewer.id,
                    analysis_freeze_at=freeze_at,
                    first_public_release_at=freeze_at + timedelta(days=1),
                    direct_payload_delete_due_at=freeze_at + timedelta(days=365),
                    pseudonymous_audit_retain_until=freeze_at + timedelta(days=365 * 5),
                    retention_policy_sha256=_sha("retention"),
                    schedule_sha256=_sha(f"schedule-{suffix}"),
                    created_at=now - timedelta(hours=1),
                )
            )
            session.execute(
                ExpertReviewer.__table__.update()
                .where(ExpertReviewer.id == reviewer.id)
                .values(active=False, revoked_at=withdrawn_at)
            )
        session.commit()
        session.expire_all()

        pre_vote = session.get(Vote, votes[0].id)
        post_vote = session.get(Vote, votes[1].id)
        assert pre_vote is not None and post_vote is not None
        assert verified_vote_person_commitment(session, pre_vote) is None
        assert (
            verified_vote_person_commitment(session, post_vote)
            == admitted[1][1].person_commitment_sha256
        )


def test_concurrent_enrollment_consumption_has_one_winner(tmp_path) -> None:
    path = tmp_path / "reviewer-concurrency.sqlite3"
    database = create_engine(
        f"sqlite+pysqlite:///{path}",
        connect_args={"timeout": 10},
    )
    Base.metadata.create_all(database)
    sessions = sessionmaker(database, expire_on_commit=False)
    with sessions() as session:
        season = _season(session)
        reviewer = _reviewer(session, "concurrent")
        binding = bind_reviewer_identity(
            session,
            season=season,
            reviewer=reviewer,
            identity_issuer="https://id.example",
            issuer_subject="concurrent-person",
            identity_evidence_sha256=_sha("concurrent-identity-evidence"),
            roles=["output_rater"],
            settings=_settings(),
        )
        token, _ = issue_reviewer_credential(
            session,
            binding=binding,
            credential_kind="enrollment_once",
            scopes=["exchange_review_session"],
            settings=_settings(),
        )
        session.commit()

    def consume() -> str:
        with sessions() as session:
            try:
                consume_reviewer_credential(
                    session,
                    token=token,
                    required_scope="exchange_review_session",
                    settings=_settings(),
                )
                session.commit()
                return "accepted"
            except ReviewerIdentityError:
                session.rollback()
                return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _ordinal: consume(), range(2)))

    assert sorted(outcomes) == ["accepted", "rejected"]


def test_admin_can_build_verified_admission_and_exchange_enrollment_once() -> None:
    now = datetime.now(UTC)
    service_headers = {
        "X-FlavourBench-Service-Token": "test-service-token",
        "X-FlavourBench-Admin-Token": "test-admin-token",
    }
    with TestClient(app) as client:
        invited = client.post(
            "/v1/admin/experts",
            headers=service_headers,
            json={
                "reviewer_code": "season-identity-api-reviewer",
                "qualified_families": ["cookability"],
                "qualification_reference": "content-addressed external evidence",
                "qualification_verified": False,
                "affiliation_class": "independent_external",
                "conflict_disclosure_reference": "content-addressed disclosure",
                "consent_document_sha256": get_settings().active_expert_consent_sha256s[0],
                "training_material_sha256": _sha("api-training"),
                "calibration_set_sha256": _sha("api-calibration-placeholder"),
                "calibration_accuracy": 0,
                "compensation_reference": "unpaid external review",
            },
        )
        assert invited.status_code == 200, invited.text
        reviewer_id = invited.json()["reviewerId"]

        bound = client.post(
            f"/v1/admin/seasons/season-0/reviewers/{reviewer_id}/identity-binding",
            headers=service_headers,
            json={
                "identityIssuer": "https://identity.api.test",
                "issuerSubject": "raw-subject-never-retained",
                "identityEvidenceSha256": _sha("api-identity-evidence"),
                "roles": ["output_rater"],
            },
        )
        assert bound.status_code == 200, bound.text
        enrollment_token = bound.json()["enrollmentCredential"]

        qualification = client.post(
            f"/v1/admin/seasons/season-0/reviewers/{reviewer_id}/qualification-evidence",
            headers=service_headers,
            json={
                "family": "cookability",
                "affiliationClass": "independent_external",
                "independenceVerified": True,
                "conflictCleared": True,
                "qualificationEvidenceSha256": _sha("api-qualification"),
                "independenceEvidenceSha256": _sha("api-independence"),
                "conflictDisclosureSha256": _sha("api-conflict"),
                "consentDocumentSha256": get_settings().active_expert_consent_sha256s[0],
                "trainingMaterialSha256": _sha("api-training"),
                "verifierPrincipalSha256": _sha("api-admin-principal"),
                "verifiedAt": (now - timedelta(days=3)).isoformat(),
                "validUntil": (now + timedelta(days=365)).isoformat(),
            },
        )
        assert qualification.status_code == 200, qualification.text

        calibration_set = client.post(
            "/v1/admin/seasons/season-0/reviewer-calibration-sets",
            headers=service_headers,
            json={
                "family": "cookability",
                "calibrationSetSha256": _sha("api-frozen-calibration"),
                "sourceArtifactSha256": _sha("api-real-source-artifact"),
                "scoringKeySha256": _sha("api-scoring-key"),
                "itemCount": 10,
                "realSourceArms": 20,
                "frozenAt": (now - timedelta(days=4)).isoformat(),
            },
        )
        assert calibration_set.status_code == 200, calibration_set.text

        ballot = client.post(
            f"/v1/admin/seasons/season-0/reviewers/{reviewer_id}/calibration-ballots",
            headers=service_headers,
            json={
                "calibrationSetId": calibration_set.json()["calibrationSetId"],
                "ballotSha256": _sha("api-ballot"),
                "scoringResultSha256": _sha("api-ballot-score"),
                "correctCount": 9,
                "minimumAccuracyMilli": 800,
                "completedAt": (now - timedelta(days=2)).isoformat(),
            },
        )
        assert ballot.status_code == 200, ballot.text
        assert ballot.json()["passed"] is True

        admission = client.post(
            f"/v1/admin/seasons/season-0/reviewers/{reviewer_id}/family-admissions",
            headers=service_headers,
            json={
                "family": "cookability",
                "reviewRole": "output_rater",
                "qualificationEvidenceId": qualification.json()["qualificationEvidenceId"],
                "calibrationBallotId": ballot.json()["calibrationBallotId"],
                "requiresCalibration": True,
                "minimumAccuracyMilli": 800,
                "decisionReferenceSha256": _sha("api-admission-decision"),
                "validFrom": (now - timedelta(days=1)).isoformat(),
                "validUntil": (now + timedelta(days=180)).isoformat(),
            },
        )
        assert admission.status_code == 200, admission.text

        exchanged = client.post(
            "/v1/expert/credentials/exchange",
            headers={
                "X-FlavourBench-Service-Token": "test-service-token",
                "Authorization": f"Bearer {enrollment_token}",
            },
        )
        assert exchanged.status_code == 200, exchanged.text
        session_token = exchanged.json()["reviewerCredential"]

        replayed = client.post(
            "/v1/expert/credentials/exchange",
            headers={
                "X-FlavourBench-Service-Token": "test-service-token",
                "Authorization": f"Bearer {enrollment_token}",
            },
        )
        assert replayed.status_code == 401

        protocol = client.get(
            "/v1/expert/protocol",
            headers={
                "X-FlavourBench-Service-Token": "test-service-token",
                "Authorization": f"Bearer {session_token}",
            },
        )
        assert protocol.status_code == 200, protocol.text
        session_prefix = session_token.removeprefix("fbrv1_").split(".", 1)[0]
        with SessionLocal() as session:
            stored_session_credential = session.scalar(
                select(ReviewerAccessCredential).where(
                    ReviewerAccessCredential.credential_prefix == session_prefix
                )
            )
            assert stored_session_credential is not None
            assert stored_session_credential.use_count == 1
