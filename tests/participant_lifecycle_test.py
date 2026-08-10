from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import flavourbench.participant_lifecycle as participant_lifecycle
from flavourbench.config import Settings
from flavourbench.models import (
    Base,
    ControlledRun,
    ControlledRunReviewer,
    ExpertReviewer,
    ReviewerAccessCredential,
    ReviewerConsentAcceptance,
    ReviewerDeletionReceipt,
    ReviewerIdentityBinding,
    ReviewerParticipationLifecycle,
    ReviewerRetentionSchedule,
    ReviewerWithdrawalReceipt,
    Season,
    Vote,
)
from flavourbench.participant_lifecycle import (
    ActiveHumanStudyBinding,
    ParticipantLifecycleError,
    accept_participant_consent,
    create_retention_schedule,
    enroll_participant_identity,
    execute_participant_private_payload_deletion,
    execute_private_payload_deletion,
    issue_enrollment_offer,
    participant_record_analysis_eligible,
    privacy_safe_participant_status,
    withdraw_participant,
)
from flavourbench.reviewer_identity import ReviewerIdentityError, issue_reviewer_credential

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
CONSENT_SHA256 = "a" * 64
ACTIVATION_SHA256 = "b" * 64
RETENTION_SHA256 = "c" * 64


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _settings() -> Settings:
    return Settings(
        reviewer_identity_hmac_secret="participant-identity-hmac-secret-0000000001",
        reviewer_credential_hmac_secret="participant-credential-hmac-secret-0000001",
        human_study_activation_manifest_sha256=ACTIVATION_SHA256,
    )


@pytest.fixture
def engine():
    database = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(database)
    return database


@pytest.fixture
def governance(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    state = {"active": True, "original_valid": True}

    def active_gate(**_kwargs: object) -> ActiveHumanStudyBinding:
        if not state["active"]:
            raise ParticipantLifecycleError("human-study activation is suspended")
        return ActiveHumanStudyBinding(
            consent_document_sha256=CONSENT_SHA256,
            activation_manifest_sha256=ACTIVATION_SHA256,
            retention_policy_sha256=RETENTION_SHA256,
            consent_text="# Test consent\n\nParticipant-owned terms.",
        )

    def original_gate(_acceptance: ReviewerConsentAcceptance, **_kwargs: object) -> None:
        if not state["original_valid"]:
            raise ParticipantLifecycleError("original accepted manifest is invalid")

    monkeypatch.setattr(
        participant_lifecycle,
        "require_active_human_study",
        active_gate,
    )
    monkeypatch.setattr(
        participant_lifecycle,
        "require_original_accepted_manifest",
        original_gate,
    )
    return state


def _season(session: Session, suffix: str = "one") -> Season:
    season = Season(
        id=f"participant-season-{suffix}",
        slug=f"participant-season-{suffix}",
        name=f"Participant season {suffix}",
        epicure_release_id="epicure-test",
    )
    session.add(season)
    session.flush()
    return season


def _accept(
    session: Session,
    *,
    season: Season,
    suffix: str = "one",
    settings: Settings | None = None,
) -> tuple[str, str, ReviewerConsentAcceptance]:
    configured = settings or _settings()
    enrollment_token, _ = issue_enrollment_offer(
        session,
        season=season,
        consent_document_sha256=CONSENT_SHA256,
        now=NOW,
        settings=configured,
    )
    result = accept_participant_consent(
        session,
        enrollment_token=enrollment_token,
        consent_document_sha256=CONSENT_SHA256,
        activation_manifest_sha256=ACTIVATION_SHA256,
        confirmations=list(participant_lifecycle.CONSENT_CONFIRMATIONS),
        idempotency_key=f"acceptance-{suffix}",
        now=NOW + timedelta(seconds=1),
        settings=configured,
    )
    return enrollment_token, result.receipt_credential, result.acceptance


def _enroll(
    session: Session,
    *,
    season: Season,
    suffix: str = "one",
    settings: Settings | None = None,
):
    configured = settings or _settings()
    enrollment_token, receipt_credential, acceptance = _accept(
        session,
        season=season,
        suffix=suffix,
        settings=configured,
    )
    identity = enroll_participant_identity(
        session,
        receipt_credential=receipt_credential,
        identity_issuer="https://identity.example.test",
        issuer_subject=f"raw-issuer-subject-{suffix}",
        identity_evidence_sha256=_sha(f"identity-evidence-{suffix}"),
        roles=["output_rater"],
        qualified_families=["cookability"],
        affiliation_class="independent_external",
        now=NOW + timedelta(seconds=2),
        settings=configured,
    )
    return enrollment_token, receipt_credential, acceptance, identity


def test_consent_is_participant_owned_one_time_and_precedes_identity(
    engine, governance: dict[str, bool]
) -> None:
    del governance
    settings = _settings()
    with Session(engine) as session:
        season = _season(session)
        enrollment_token, receipt_credential, acceptance = _accept(
            session,
            season=season,
            settings=settings,
        )
        assert session.scalars(select(ExpertReviewer)).all() == []
        assert session.scalars(select(ReviewerIdentityBinding)).all() == []

        retry = accept_participant_consent(
            session,
            enrollment_token=enrollment_token,
            consent_document_sha256=CONSENT_SHA256,
            activation_manifest_sha256=ACTIVATION_SHA256,
            confirmations=list(participant_lifecycle.CONSENT_CONFIRMATIONS),
            idempotency_key="acceptance-one",
            now=NOW + timedelta(seconds=2),
            settings=settings,
        )
        assert retry.idempotent
        assert retry.receipt_credential == receipt_credential
        assert retry.acceptance.id == acceptance.id
        assert len(session.scalars(select(ReviewerConsentAcceptance)).all()) == 1

        identity = enroll_participant_identity(
            session,
            receipt_credential=receipt_credential,
            identity_issuer="https://identity.example.test",
            issuer_subject="raw-issuer-subject-one",
            identity_evidence_sha256=_sha("identity-evidence-one"),
            roles=["output_rater"],
            qualified_families=["cookability"],
            affiliation_class="independent_external",
            now=NOW + timedelta(seconds=3),
            settings=settings,
        )
        session.commit()

        assert identity.lifecycle.status == "active"
        assert identity.reviewer.profile_json["contact_data_persisted"] is False
        assert identity.reviewer.profile_json["consent_acceptance_sha256"] == (
            acceptance.receipt_sha256
        )
        assert identity.binding.person_commitment_sha256 != _sha("raw-issuer-subject-one")
        assert "raw-issuer-subject-one" not in str(identity.reviewer.profile_json)

    raw_connection = engine.raw_connection()
    try:
        database_dump = "\n".join(raw_connection.driver_connection.iterdump())
    finally:
        raw_connection.close()
    assert "raw-issuer-subject-one" not in database_dump
    assert "https://identity.example.test" not in database_dump


def test_expired_offer_cross_participant_access_and_suspended_forward_work_fail_closed(
    engine, governance: dict[str, bool]
) -> None:
    settings = _settings()
    with Session(engine) as session:
        first_season = _season(session, "first")
        second_season = _season(session, "second")
        expired_token, _ = issue_enrollment_offer(
            session,
            season=first_season,
            consent_document_sha256=CONSENT_SHA256,
            ttl_seconds=300,
            now=NOW,
            settings=settings,
        )
        with pytest.raises(ParticipantLifecycleError, match="expired"):
            accept_participant_consent(
                session,
                enrollment_token=expired_token,
                consent_document_sha256=CONSENT_SHA256,
                activation_manifest_sha256=ACTIVATION_SHA256,
                confirmations=list(participant_lifecycle.CONSENT_CONFIRMATIONS),
                idempotency_key="expired-acceptance",
                now=NOW + timedelta(seconds=301),
                settings=settings,
            )

        _, first_receipt, _ = _accept(
            session,
            season=first_season,
            suffix="first",
            settings=settings,
        )
        _, second_receipt, _ = _accept(
            session,
            season=second_season,
            suffix="second",
            settings=settings,
        )
        governance["active"] = False
        with pytest.raises(ParticipantLifecycleError, match="suspended"):
            enroll_participant_identity(
                session,
                receipt_credential=first_receipt,
                identity_issuer="https://identity.example.test",
                issuer_subject="first-subject",
                identity_evidence_sha256=_sha("first-evidence"),
                roles=["output_rater"],
                qualified_families=["cookability"],
                affiliation_class="independent_external",
                now=NOW + timedelta(seconds=3),
                settings=settings,
            )
        with pytest.raises(ParticipantLifecycleError, match="suspended"):
            enroll_participant_identity(
                session,
                receipt_credential=second_receipt,
                identity_issuer="https://identity.example.test",
                issuer_subject="second-subject",
                identity_evidence_sha256=_sha("second-evidence"),
                roles=["output_rater"],
                qualified_families=["cookability"],
                affiliation_class="independent_external",
                now=NOW + timedelta(seconds=3),
                settings=settings,
            )


def test_withdrawal_and_redaction_remain_available_after_suspension_and_preserve_history(
    engine, governance: dict[str, bool]
) -> None:
    settings = _settings()
    with Session(engine) as session:
        season = _season(session)
        _, receipt_credential, _acceptance, identity = _enroll(
            session,
            season=season,
            settings=settings,
        )
        run_id = "participant-controlled-run"
        session.execute(
            ControlledRun.__table__.insert().values(
                id=run_id,
                season_id=season.id,
                organization_reference_sha256=_sha("organization"),
                access_token_sha256=_sha("access-token"),
                protocol_version="test-v1",
                rater_plan_sha256=_sha("rater-plan"),
                analysis_plan_sha256=_sha("analysis-plan"),
                run_card_json={"test": True},
                run_card_sha256=_sha("run-card"),
                run_card_signature=_sha("run-card-signature"),
            )
        )
        assignment = ControlledRunReviewer(
            controlled_run_id=run_id,
            reviewer_id=identity.reviewer.id,
            authorization_reference_sha256=_sha("assignment-authorization"),
            active=True,
        )
        session.add(assignment)
        vote_id = "participant-preserved-vote"
        session.execute(
            Vote.__table__.insert().values(
                id=vote_id,
                battle_id="historical-battle",
                rater_pseudonym=_sha("rater"),
                cohort="expert_independent",
                choice="left",
                reason_tags_json=["generic"],
                rubric_json={"historical": True},
                idempotency_key="participant-preserved-vote-once",
                reviewer_id=identity.reviewer.id,
                reviewer_identity_binding_id=identity.binding.id,
                reviewer_family_admission_id="historical-admission",
                provenance_status="expert_verified_v1",
                provenance_sha256=_sha("historical-provenance"),
            )
        )
        schedule = create_retention_schedule(
            session,
            reviewer_id=identity.reviewer.id,
            analysis_freeze_at=NOW,
            first_public_release_at=NOW,
            now=NOW,
            settings=settings,
        )
        vote_before = dict(
            session.execute(select(Vote.__table__).where(Vote.id == vote_id)).mappings().one()
        )
        session.commit()

        governance["active"] = False
        withdrawal = withdraw_participant(
            session,
            receipt_credential=receipt_credential,
            idempotency_key="withdrawal-once",
            reason_code="voluntary_withdrawal",
            now=NOW + timedelta(hours=1),
            settings=settings,
        )
        session.commit()
        retry = withdraw_participant(
            session,
            receipt_credential=receipt_credential,
            idempotency_key="withdrawal-retry-different-key",
            reason_code="privacy_request",
            now=NOW + timedelta(hours=2),
            settings=settings,
        )
        session.commit()

        reviewer = session.get(ExpertReviewer, identity.reviewer.id)
        credentials = session.scalars(
            select(ReviewerAccessCredential).where(
                ReviewerAccessCredential.reviewer_id == identity.reviewer.id
            )
        ).all()
        assert reviewer is not None and not reviewer.active
        assert all(credential.status != "active" for credential in credentials)
        assert not session.get(ControlledRunReviewer, assignment.id).active
        assert withdrawal.receipt_sha256 == retry.receipt_sha256
        assert len(session.scalars(select(ReviewerWithdrawalReceipt)).all()) == 1
        vote_after_withdrawal = dict(
            session.execute(select(Vote.__table__).where(Vote.id == vote_id)).mappings().one()
        )
        assert vote_after_withdrawal == vote_before

        deletion = execute_participant_private_payload_deletion(
            session,
            receipt_credential=receipt_credential,
            idempotency_key="deletion-once",
            now=NOW + timedelta(hours=3),
            settings=settings,
        )
        session.commit()
        deletion_retry = execute_participant_private_payload_deletion(
            session,
            receipt_credential=receipt_credential,
            idempotency_key="deletion-retry-different-key",
            now=NOW + timedelta(hours=4),
            settings=settings,
        )
        session.commit()

        reviewer = session.get(ExpertReviewer, identity.reviewer.id)
        lifecycle = session.scalar(select(ReviewerParticipationLifecycle))
        assert reviewer is not None and lifecycle is not None
        assert reviewer.privacy_status == "redacted"
        assert reviewer.qualification_json == []
        assert not reviewer.qualification_verified
        assert set(reviewer.profile_json) == {
            "schema_version",
            "privacy_status",
            "audit_marker_sha256",
            "private_payload_before_sha256",
        }
        assert lifecycle.status == "redacted"
        assert deletion.receipt_sha256 == deletion_retry.receipt_sha256
        assert len(session.scalars(select(ReviewerDeletionReceipt)).all()) == 1
        assert session.get(ReviewerIdentityBinding, identity.binding.id) is not None
        assert (
            dict(session.execute(select(Vote.__table__).where(Vote.id == vote_id)).mappings().one())
            == vote_before
        )

        status = privacy_safe_participant_status(
            session,
            receipt_credential=receipt_credential,
            settings=settings,
        )
        assert status["participationStatus"] == "redacted"
        assert status["auditMarkerSha256"] == lifecycle.audit_marker_sha256
        assert "reviewerId" not in status
        assert "identityBindingId" not in status
        assert "receiptPrefix" not in status
        assert status["directPayloadDeleteDueAt"] == (
            schedule.direct_payload_delete_due_at.replace(tzinfo=UTC).isoformat()
        )


def test_scheduled_deletion_obeys_deadline_and_receipts_are_append_only(
    engine, governance: dict[str, bool]
) -> None:
    del governance
    settings = _settings()
    with Session(engine) as session:
        season = _season(session)
        _, receipt_credential, acceptance, identity = _enroll(
            session,
            season=season,
            settings=settings,
        )
        create_retention_schedule(
            session,
            reviewer_id=identity.reviewer.id,
            analysis_freeze_at=NOW,
            first_public_release_at=NOW,
            now=NOW,
            settings=settings,
        )
        withdraw_participant(
            session,
            receipt_credential=receipt_credential,
            idempotency_key="scheduled-withdrawal",
            reason_code="privacy_request",
            now=NOW + timedelta(hours=1),
            settings=settings,
        )
        session.commit()

        with pytest.raises(ParticipantLifecycleError, match="deadline is not due"):
            execute_private_payload_deletion(
                session,
                reviewer_id=identity.reviewer.id,
                idempotency_key="scheduled-too-early",
                execution_basis="scheduled_retention",
                now=NOW + timedelta(days=364),
                settings=settings,
            )
        session.rollback()
        deletion = execute_private_payload_deletion(
            session,
            reviewer_id=identity.reviewer.id,
            idempotency_key="scheduled-due",
            execution_basis="scheduled_retention",
            now=datetime(2027, 8, 9, 12, 0, tzinfo=UTC),
            settings=settings,
        )
        session.commit()

        acceptance.receipt_sha256 = _sha("mutated-acceptance")
        with pytest.raises(ValueError, match="immutable"):
            session.flush()
        session.rollback()
        stored_deletion = session.get(ReviewerDeletionReceipt, deletion.id)
        assert stored_deletion is not None
        session.delete(stored_deletion)
        with pytest.raises(ValueError, match="append-only"):
            session.flush()
        session.rollback()
        assert session.get(ReviewerRetentionSchedule, deletion.retention_schedule_id) is not None


def test_original_manifest_must_still_validate_for_terminal_rights(
    engine, governance: dict[str, bool]
) -> None:
    settings = _settings()
    with Session(engine) as session:
        season = _season(session)
        _, receipt_credential, _acceptance, _identity = _enroll(
            session,
            season=season,
            settings=settings,
        )
        governance["active"] = False
        governance["original_valid"] = False
        with pytest.raises(ParticipantLifecycleError, match="original accepted manifest"):
            withdraw_participant(
                session,
                receipt_credential=receipt_credential,
                idempotency_key="invalid-original-manifest",
                reason_code="voluntary_withdrawal",
                now=NOW + timedelta(hours=1),
                settings=settings,
            )
        binding = session.scalar(select(ReviewerIdentityBinding))
        assert binding is not None
        with pytest.raises(ReviewerIdentityError, match="current participant authority"):
            issue_reviewer_credential(
                session,
                binding=binding,
                credential_kind="review_session",
                scopes=["expert_review"],
                now=NOW + timedelta(hours=1),
                settings=settings,
            )


def test_participant_deletion_authentication_cannot_cross_reviewer_scope(
    engine, governance: dict[str, bool]
) -> None:
    del governance
    settings = _settings()
    with Session(engine) as session:
        season = _season(session)
        _, first_receipt, _first_acceptance, first = _enroll(
            session,
            season=season,
            suffix="first",
            settings=settings,
        )
        _, second_receipt, _second_acceptance, second = _enroll(
            session,
            season=season,
            suffix="second",
            settings=settings,
        )
        for suffix, receipt_credential, identity in (
            ("first", first_receipt, first),
            ("second", second_receipt, second),
        ):
            create_retention_schedule(
                session,
                reviewer_id=identity.reviewer.id,
                analysis_freeze_at=NOW,
                first_public_release_at=NOW,
                now=NOW,
                settings=settings,
            )
            withdraw_participant(
                session,
                receipt_credential=receipt_credential,
                idempotency_key=f"cross-scope-withdrawal-{suffix}",
                reason_code="privacy_request",
                now=NOW + timedelta(hours=1),
                settings=settings,
            )
        session.commit()

        with pytest.raises(ParticipantLifecycleError, match="crossed reviewer scope"):
            execute_private_payload_deletion(
                session,
                reviewer_id=second.reviewer.id,
                idempotency_key="cross-scope-deletion",
                execution_basis="participant_request",
                receipt_credential=first_receipt,
                now=NOW + timedelta(hours=2),
                settings=settings,
            )
        session.rollback()
        assert session.get(ExpertReviewer, first.reviewer.id).privacy_status == "retained"
        assert session.get(ExpertReviewer, second.reviewer.id).privacy_status == "retained"


def test_withdrawal_analysis_eligibility_is_freeze_aware_and_storage_preserving(
    engine, governance: dict[str, bool]
) -> None:
    del governance
    settings = _settings()
    with Session(engine) as session:
        season = _season(session, "freeze-aware")
        _, receipt_credential, _, identity = _enroll(
            session,
            season=season,
            suffix="freeze-aware",
            settings=settings,
        )
        lifecycle_created_at = identity.lifecycle.created_at.replace(tzinfo=UTC)
        record_time = lifecycle_created_at + timedelta(minutes=10)
        analysis_freeze_at = lifecycle_created_at + timedelta(hours=2)
        create_retention_schedule(
            session,
            reviewer_id=identity.reviewer.id,
            analysis_freeze_at=analysis_freeze_at,
            first_public_release_at=analysis_freeze_at + timedelta(days=1),
            now=lifecycle_created_at,
            settings=settings,
        )
        assert participant_record_analysis_eligible(
            session,
            reviewer_id=identity.reviewer.id,
            season_id=season.id,
            identity_binding_id=identity.binding.id,
            recorded_at=record_time,
        )
        withdraw_participant(
            session,
            receipt_credential=receipt_credential,
            idempotency_key="freeze-aware-pre-freeze-withdrawal",
            reason_code="voluntary_withdrawal",
            now=lifecycle_created_at + timedelta(hours=1),
            settings=settings,
        )
        session.commit()
        assert (
            participant_record_analysis_eligible(
                session,
                reviewer_id=identity.reviewer.id,
                season_id=season.id,
                identity_binding_id=identity.binding.id,
                recorded_at=record_time,
            )
            is False
        )
        assert session.scalar(select(ReviewerParticipationLifecycle)) is not None

    with Session(engine) as session:
        season = _season(session, "post-freeze")
        _, receipt_credential, _, identity = _enroll(
            session,
            season=season,
            suffix="post-freeze",
            settings=settings,
        )
        lifecycle_created_at = identity.lifecycle.created_at.replace(tzinfo=UTC)
        record_time = lifecycle_created_at + timedelta(minutes=10)
        analysis_freeze_at = lifecycle_created_at + timedelta(minutes=30)
        create_retention_schedule(
            session,
            reviewer_id=identity.reviewer.id,
            analysis_freeze_at=analysis_freeze_at,
            first_public_release_at=analysis_freeze_at + timedelta(days=1),
            now=lifecycle_created_at,
            settings=settings,
        )
        withdraw_participant(
            session,
            receipt_credential=receipt_credential,
            idempotency_key="freeze-aware-post-freeze-withdrawal",
            reason_code="voluntary_withdrawal",
            now=lifecycle_created_at + timedelta(hours=1),
            settings=settings,
        )
        session.commit()
        assert (
            participant_record_analysis_eligible(
                session,
                reviewer_id=identity.reviewer.id,
                season_id=season.id,
                identity_binding_id=identity.binding.id,
                recorded_at=record_time,
            )
            is True
        )


def test_sqlite_concurrent_double_consume_withdraw_and_redact_are_singleton(
    tmp_path, governance: dict[str, bool]
) -> None:
    del governance
    settings = _settings()
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'participant-races.sqlite3'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        season = _season(session)
        enrollment_token, _ = issue_enrollment_offer(
            session,
            season=season,
            consent_document_sha256=CONSENT_SHA256,
            now=NOW,
            settings=settings,
        )
        session.commit()

    acceptance_barrier = Barrier(2)

    def accept_once() -> tuple[str, str]:
        try:
            with Session(engine) as session:
                acceptance_barrier.wait()
                result = accept_participant_consent(
                    session,
                    enrollment_token=enrollment_token,
                    consent_document_sha256=CONSENT_SHA256,
                    activation_manifest_sha256=ACTIVATION_SHA256,
                    confirmations=list(participant_lifecycle.CONSENT_CONFIRMATIONS),
                    idempotency_key="concurrent-acceptance",
                    now=NOW + timedelta(seconds=1),
                    settings=settings,
                )
                session.commit()
                return "ok", result.receipt_credential
        except Exception as exc:  # The losing SQLite CAS may report lock or conflict.
            return "closed", type(exc).__name__

    with ThreadPoolExecutor(max_workers=2) as executor:
        acceptance_outcomes = list(executor.map(lambda _index: accept_once(), range(2)))
    assert any(outcome[0] == "ok" for outcome in acceptance_outcomes)
    with Session(engine) as session:
        acceptances = session.scalars(select(ReviewerConsentAcceptance)).all()
        assert len(acceptances) == 1
        successful_credentials = {value for status, value in acceptance_outcomes if status == "ok"}
        assert len(successful_credentials) == 1
        receipt_credential = successful_credentials.pop()
        season = session.scalar(select(Season))
        assert season is not None
        identity = enroll_participant_identity(
            session,
            receipt_credential=receipt_credential,
            identity_issuer="https://identity.example.test",
            issuer_subject="concurrent-raw-subject",
            identity_evidence_sha256=_sha("concurrent-identity"),
            roles=["output_rater"],
            qualified_families=["cookability"],
            affiliation_class="independent_external",
            now=NOW + timedelta(seconds=2),
            settings=settings,
        )
        create_retention_schedule(
            session,
            reviewer_id=identity.reviewer.id,
            analysis_freeze_at=NOW,
            first_public_release_at=NOW,
            now=NOW,
            settings=settings,
        )
        reviewer_id = identity.reviewer.id
        session.commit()

    withdrawal_barrier = Barrier(2)

    def withdraw_once() -> tuple[str, str]:
        try:
            with Session(engine) as session:
                withdrawal_barrier.wait()
                receipt = withdraw_participant(
                    session,
                    receipt_credential=receipt_credential,
                    idempotency_key="concurrent-withdrawal",
                    reason_code="voluntary_withdrawal",
                    now=NOW + timedelta(hours=1),
                    settings=settings,
                )
                session.commit()
                return "ok", receipt.receipt_sha256
        except Exception as exc:
            return "closed", type(exc).__name__

    with ThreadPoolExecutor(max_workers=2) as executor:
        withdrawal_outcomes = list(executor.map(lambda _index: withdraw_once(), range(2)))
    assert any(outcome[0] == "ok" for outcome in withdrawal_outcomes)
    with Session(engine) as session:
        assert len(session.scalars(select(ReviewerWithdrawalReceipt)).all()) == 1
        lifecycle = session.scalar(select(ReviewerParticipationLifecycle))
        assert lifecycle is not None and lifecycle.status == "withdrawn"
        assert not session.get(ExpertReviewer, reviewer_id).active
        assert (
            session.scalars(
                select(ReviewerAccessCredential).where(ReviewerAccessCredential.status == "active")
            ).all()
            == []
        )

    deletion_barrier = Barrier(2)

    def delete_once() -> tuple[str, str]:
        try:
            with Session(engine) as session:
                deletion_barrier.wait()
                receipt = execute_participant_private_payload_deletion(
                    session,
                    receipt_credential=receipt_credential,
                    idempotency_key="concurrent-deletion",
                    now=NOW + timedelta(hours=2),
                    settings=settings,
                )
                session.commit()
                return "ok", receipt.receipt_sha256
        except Exception as exc:
            return "closed", type(exc).__name__

    with ThreadPoolExecutor(max_workers=2) as executor:
        deletion_outcomes = list(executor.map(lambda _index: delete_once(), range(2)))
    assert any(outcome[0] == "ok" for outcome in deletion_outcomes)
    with Session(engine) as session:
        assert len(session.scalars(select(ReviewerDeletionReceipt)).all()) == 1
        lifecycle = session.scalar(select(ReviewerParticipationLifecycle))
        reviewer = session.get(ExpertReviewer, reviewer_id)
        assert lifecycle is not None and lifecycle.status == "redacted"
        assert reviewer is not None and reviewer.privacy_status == "redacted"
