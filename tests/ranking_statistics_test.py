from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import flavourbench.service_ranking as service_ranking
from flavourbench.models import Base, Battle, ExpertReviewer, RunEvent, Season, ValidatorResult
from flavourbench.ranking import _fit_bradley_terry, _paired_tie_aware_profile
from flavourbench.season0_analysis import _arena_rows, _uplift_rows
from flavourbench.validators import VALIDATOR_VERSION


def test_pinned_bradley_terry_recovers_known_order_with_ties() -> None:
    comparisons: list[tuple[str, str, float]] = []
    comparisons.extend(("strong", "middle", 1.0) for _ in range(40))
    comparisons.extend(("strong", "middle", 0.5) for _ in range(10))
    comparisons.extend(("strong", "weak", 1.0) for _ in range(45))
    comparisons.extend(("strong", "weak", 0.5) for _ in range(5))
    comparisons.extend(("middle", "weak", 1.0) for _ in range(35))
    comparisons.extend(("middle", "weak", 0.5) for _ in range(15))

    ratings = _fit_bradley_terry(comparisons, require_arena_rank=True)

    assert ratings["strong"][0] > ratings["middle"][0] > ratings["weak"][0]
    assert ratings["strong"][1] > ratings["weak"][2]


def test_tie_aware_profile_recovers_null_and_directional_effects() -> None:
    null_estimate, null_low, null_high = _paired_tie_aware_profile(40, 20, 40)
    positive_estimate, positive_low, positive_high = _paired_tie_aware_profile(80, 10, 10)

    assert null_estimate == pytest.approx(0.5, abs=0.02)
    assert null_low < 0.5 < null_high
    assert positive_estimate > 0.75
    assert positive_low > 0.5
    assert positive_high <= 1.0


@pytest.mark.parametrize(
    ("wins", "ties", "losses"),
    ((4, 11, 0), (0, 15, 5), (0, 17, 0), (5, 0, 0), (0, 0, 5)),
)
def test_tie_aware_profile_intervals_do_not_collapse_at_sparse_boundaries(
    wins: int, ties: int, losses: int
) -> None:
    estimate, low, high = _paired_tie_aware_profile(wins, ties, losses)

    assert 0.0 <= low <= estimate <= high <= 1.0
    assert high - low > 0.05


def test_both_bad_is_reported_but_excluded_from_preference_fits() -> None:
    arena = _arena_rows(
        [
            {
                "track": "model_arena",
                "task_family": "evidence",
                "primary_consensus_choice": choice,
                "left": {"season_model_id": "left"},
                "right": {"season_model_id": "right"},
            }
            for choice in ("left", "tie", "both_bad")
        ],
        {"left": "Left", "right": "Right"},
        None,
    )
    by_id = {row["season_model_id"]: row for row in arena}
    assert by_id["left"]["judgments"] == 3
    assert by_id["left"]["comparisons"] == 2
    assert by_id["left"]["both_bad"] == 1

    uplift = _uplift_rows(
        [
            {
                "track": "epicure_uplift",
                "task_family": "evidence",
                "season_model_id": "model",
                "primary_consensus_choice": choice,
                "left": {"condition": "epicure_on"},
                "right": {"condition": "epicure_off"},
            }
            for choice in ("left", "tie", "both_bad")
        ],
        {"model": "Model"},
        None,
    )[0]
    assert uplift["judgments"] == 3
    assert uplift["comparisons"] == 2
    assert uplift["both_bad"] == 1
    assert uplift["epicure_wins"] == 1
    assert uplift["ties"] == 1


def test_preference_gate_requires_identity_and_semantic_completion() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                ValidatorResult(
                    arm_id="complete-arm",
                    validator_name=name,
                    validator_version=VALIDATOR_VERSION,
                    status="pass",
                    score_milli=1000,
                    detail_json={},
                )
                for name in ("identity_blinding", "semantic_completion")
            ]
            + [
                ValidatorResult(
                    arm_id="incomplete-arm",
                    validator_name="identity_blinding",
                    validator_version=VALIDATOR_VERSION,
                    status="pass",
                    score_milli=1000,
                    detail_json={},
                ),
                ValidatorResult(
                    arm_id="incomplete-arm",
                    validator_name="semantic_completion",
                    validator_version=VALIDATOR_VERSION,
                    status="fail",
                    score_milli=0,
                    detail_json={"failure_reasons": ["ends_with_markdown_heading"]},
                ),
            ]
        )
        session.flush()
        assert service_ranking._arm_has_required_preference_validators(
            session,
            "complete-arm",
        )
        assert not service_ranking._arm_has_required_preference_validators(
            session,
            "incomplete-arm",
        )


def test_service_leaderboards_have_stable_model_id_tie_breakers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def metrics() -> dict[str, defaultdict[str, float]]:
        first: defaultdict[str, float] = defaultdict(float)
        second: defaultdict[str, float] = defaultdict(float)
        first["arms"] = second["arms"] = 1
        first["cost_reconciled_arms"] = second["cost_reconciled_arms"] = 1
        return {"z/model": first, "a/model": second}

    monkeypatch.setattr(service_ranking, "_votes", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service_ranking,
        "_operational_metrics",
        lambda *_args, **_kwargs: metrics(),
    )
    season = SimpleNamespace(manifest_sha256="manifest")

    model_rows = service_ranking.model_leaderboard(
        None,
        season,
        "public",
        "all",  # type: ignore[arg-type]
    )["rows"]
    uplift_rows = service_ranking.uplift_leaderboard(
        None,
        season,
        "public",
        "all",  # type: ignore[arg-type]
    )["rows"]

    assert [row["competitor_id"] for row in model_rows] == ["a/model", "z/model"]
    assert [row["competitor_id"] for row in uplift_rows] == ["a/model", "z/model"]


def test_ranking_controls_fail_closed_on_scope_and_containment_events() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    RunEvent.__table__.create(engine)
    public = SimpleNamespace(
        id="public-battle",
        left_arm_id="public-left",
        right_arm_id="public-right",
        task_id=None,
    )
    safety = SimpleNamespace(
        id="safety-battle",
        left_arm_id="safety-left",
        right_arm_id="safety-right",
        task_id="safety-task",
    )
    specialist = SimpleNamespace(
        id="specialist-battle",
        left_arm_id="specialist-left",
        right_arm_id="specialist-right",
        task_id="specialist-task",
    )
    retired = SimpleNamespace(
        id="retired-battle",
        left_arm_id="retired-left",
        right_arm_id="retired-right",
        task_id="retired-task",
    )
    with Session(engine) as session:
        no_attestation = service_ranking._ranking_control_sets(
            session,
            [public],
            data_stratum="public_freeform",
            evidence_cutoff_at=None,
        )
        assert no_attestation["preference_admissible"] == set()
        assert no_attestation["operational_admissible"] == set()

        session.add_all(
            [
                RunEvent(
                    entity_type="battle",
                    entity_id=public.id,
                    event_type="battle_general_track_scope_admitted",
                    payload_json={
                        "general_track_eligible": True,
                        "scope_protocol_sha256": "a" * 64,
                    },
                ),
                RunEvent(
                    entity_type="response_arm",
                    entity_id=safety.right_arm_id,
                    event_type="reviewer_reported_potential_safety_hazard",
                    payload_json={
                        "status": "pending_qualified_food_safety_adjudication",
                        "preference_exclusion_requested": False,
                    },
                ),
                RunEvent(
                    entity_type="task",
                    entity_id=specialist.task_id,
                    event_type="task_general_track_scope_quarantined",
                    payload_json={"general_track_eligible": False},
                ),
                RunEvent(
                    entity_type="task",
                    entity_id=retired.task_id,
                    event_type="confirmatory_task_retired",
                    payload_json={"ranking_use": False, "recomputation_required": True},
                ),
            ]
        )
        session.commit()

        invalid_admission = service_ranking._ranking_control_sets(
            session,
            [public],
            data_stratum="public_freeform",
            evidence_cutoff_at=None,
        )
        assert invalid_admission["preference_admissible"] == set()
        assert invalid_admission["operational_admissible"] == set()

        safety_controls = service_ranking._ranking_control_sets(
            session,
            [safety],
            data_stratum="controlled",
            evidence_cutoff_at=None,
        )
        assert safety_controls["preference_admissible"] == {safety.id}
        assert safety_controls["operational_admissible"] == {safety.id}
        assert safety_controls["restricted_arm_ids"] == set()
        assert safety_controls["reported_safety_arm_ids"] == {safety.right_arm_id}

        specialist_controls = service_ranking._ranking_control_sets(
            session,
            [specialist],
            data_stratum="controlled",
            evidence_cutoff_at=None,
        )
        assert specialist_controls["preference_admissible"] == set()
        assert specialist_controls["operational_admissible"] == set()

        retired_controls = service_ranking._ranking_control_sets(
            session,
            [retired],
            data_stratum="controlled",
            evidence_cutoff_at=None,
        )
        assert retired_controls["preference_admissible"] == set()
        assert retired_controls["operational_admissible"] == set()


def test_public_scope_event_is_the_append_only_rank_promotion_authority() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    with Session(engine) as session:
        season = Season(
            id="public-scope-season",
            slug="public-scope-season",
            name="Public scope season",
            status="running",
            official=True,
            manifest_sha256="1" * 64,
            prompt_registry_sha256="2" * 64,
            tool_registry_sha256="3" * 64,
            epicure_release_id="epicure-test-release",
            epicure_bundle_sha256="4" * 64,
            epicure_application_sha256="5" * 64,
            analysis_plan_sha256="6" * 64,
            protocol_bundle_json={"test": True},
            protocol_bundle_sha256="7" * 64,
        )
        battle = Battle(
            id="public-scope-promoted-battle",
            season_id=season.id,
            run_class="official",
            rank_eligible=False,
            data_stratum="public_freeform",
            manifest_sha256=season.manifest_sha256,
            protocol_bundle_sha256=season.protocol_bundle_sha256,
            scheduler_version="public-test",
            assignment_seed="8" * 64,
            track_assignment_probability="1/2",
            model_assignment_probability="1/2",
            side_assignment_probability="1/2",
            track="model_arena",
            category="cookability",
            prompt="Design a practical lentil supper.",
            prompt_sha256="9" * 64,
            client_nonce_sha256="a" * 64,
            requester_pseudonym="b" * 64,
            status="queued",
            retention_until=now + timedelta(days=30),
        )
        candidate = {
            "candidate_pack_sha256": "e" * 64,
            "candidate_pairs": 32,
            "source_arms": 64,
            "real_provider_calls": 113,
            "successful_real_epicure_calls": 73,
            "synthetic_arms": 0,
            "rank_eligible": False,
        }
        reviewer_profile = {
            "affiliation_class": "independent_external",
            "calibration_candidate": candidate,
            "qualification_reference": "verified-practice",
            "conflict_disclosure_reference": "no-conflict",
            "consent_document_sha256": "1" * 64,
            "training_material_sha256": "2" * 64,
            "calibration_set_sha256": "3" * 64,
            "calibration_item_count": 20,
            "calibration_gold_adjudicator_count": 2,
            "calibration_accuracy": 0.9,
            "admission_decision_reference": "public-scope-test-admission",
            "admission_decision_sha256": "4" * 64,
        }
        reviewer = ExpertReviewer(
            id="public-scope-independent-reviewer",
            reviewer_code="public-scope-independent-reviewer",
            invitation_sha256="5" * 64,
            qualification_json=["cookability"],
            qualification_verified=True,
            cohort="expert_independent",
            profile_json=reviewer_profile,
        )
        evidence_fields = (
            "qualification_reference",
            "conflict_disclosure_reference",
            "consent_document_sha256",
            "training_material_sha256",
            "calibration_set_sha256",
            "calibration_item_count",
            "calibration_gold_adjudicator_count",
            "calibration_accuracy",
            "admission_decision_reference",
            "admission_decision_sha256",
        )
        candidate_registration = RunEvent(
            id="public-scope-candidate-registration",
            entity_type="expert_reviewer",
            entity_id=reviewer.id,
            event_type="expert_calibration_candidate_registered",
            payload_json={
                "reviewer_code": reviewer.reviewer_code,
                "cohort": reviewer.cohort,
                "candidate": candidate,
                "candidate_record_sha256": service_ranking._canonical_sha256(candidate),
            },
            created_at=now - timedelta(minutes=4),
        )
        reviewer_admission = RunEvent(
            id="public-scope-reviewer-admission",
            entity_type="expert_reviewer",
            entity_id=reviewer.id,
            event_type="expert_reviewer_admitted",
            payload_json={
                "cohort": reviewer.cohort,
                "qualified_families": reviewer.qualification_json,
                "affiliation_class": "independent_external",
                "admission_protocol_version": "expert-admission-v2",
                "consent_active_at_admission": True,
                "calibration_candidate": candidate,
                "calibration_candidate_record_sha256": (
                    service_ranking._canonical_sha256(candidate)
                ),
                "evidence": {field: reviewer_profile[field] for field in evidence_fields},
            },
            created_at=now - timedelta(minutes=3),
        )
        assessment = {
            "task_validity": "valid",
            "task_issue_tags": [],
            "task_note": "",
            "answerability": "answerable",
            "family_fit": "in_family",
            "scope_eligibility": "general_track",
            "specialist_domains": [],
            "general_track_eligible": True,
        }
        assessment_sha256 = service_ranking._canonical_sha256(assessment)
        presentation_sha256 = "6" * 64
        review_session_id = "public-scope-review-session"
        review_assignment_id = "public-scope-review-assignment"
        task_assessed = RunEvent(
            entity_type="expert_review_assignment",
            entity_id=review_assignment_id,
            event_type="expert_review_task_assessed",
            payload_json={
                "review_session_id": review_session_id,
                "reviewer_id": reviewer.id,
                "battle_id": battle.id,
                "assessment": assessment,
                "assessment_sha256": assessment_sha256,
                "protocol_sha256": service_ranking.EXPERT_REVIEW_PROTOCOL_SHA256,
                "presentation_sha256": presentation_sha256,
            },
            created_at=now - timedelta(minutes=2),
        )
        session.add_all(
            [
                season,
                battle,
                reviewer,
                candidate_registration,
                reviewer_admission,
                task_assessed,
            ]
        )
        session.commit()
        session.execute(
            text(
                "UPDATE battles SET status = 'complete', completed_at = :completed_at "
                "WHERE id = :battle_id"
            ),
            {"completed_at": now, "battle_id": battle.id},
        )
        session.execute(
            text(
                "INSERT INTO votes (id, battle_id, rater_pseudonym, cohort, choice, "
                "reason_tags_json, rubric_json, idempotency_key, created_at) VALUES "
                "(:id, :battle_id, :rater, 'public', 'tie', '[]', '{}', :key, :created_at)"
            ),
            {
                "id": "public-scope-vote",
                "battle_id": battle.id,
                "rater": "c" * 64,
                "key": "public-scope-vote-key",
                "created_at": now,
            },
        )
        session.commit()
        session.refresh(battle)

        rows = service_ranking._votes(
            session,
            season,
            "model_arena",
            "public",
            "all",
            "public_freeform",
            None,
        )
        assert [row_battle.id for _row_vote, row_battle in rows] == [battle.id]
        controls = service_ranking._ranking_control_sets(
            session,
            [battle],
            data_stratum="public_freeform",
            evidence_cutoff_at=None,
        )
        assert controls["preference_admissible"] == set()

        scope_admission = RunEvent(
            entity_type="battle",
            entity_id=battle.id,
            event_type="battle_general_track_scope_admitted",
            payload_json={
                "general_track_eligible": True,
                "ranking_use": True,
                "scope_protocol_sha256": (service_ranking.EXPERT_REVIEW_PROTOCOL_SHA256),
                "scope_admission_quorum": 1,
                "reviewer_id": reviewer.id,
                "reviewer_cohort": reviewer.cohort,
                "reviewer_qualification_verified": True,
                "affiliation_class": "independent_external",
                "review_session_id": review_session_id,
                "review_assignment_id": review_assignment_id,
                "assessment_sha256": assessment_sha256,
                "presentation_sha256": presentation_sha256,
                "reviewer_admission_event_id": reviewer_admission.id,
                "reviewer_admission_evidence_sha256": (
                    service_ranking._canonical_sha256(reviewer_admission.payload_json)
                ),
            },
            created_at=now - timedelta(minutes=1),
        )
        session.add(scope_admission)
        session.commit()
        controls = service_ranking._ranking_control_sets(
            session,
            [battle],
            data_stratum="public_freeform",
            evidence_cutoff_at=None,
        )
        assert controls["preference_admissible"] == {battle.id}

        reviewer.qualification_verified = False
        reviewer.active = False
        reviewer.profile_json = {"consent_document_sha256": "retired"}
        session.add(
            RunEvent(
                entity_type="expert_reviewer",
                entity_id=reviewer.id,
                event_type="expert_reviewer_revoked",
                payload_json={"admission_event_id": reviewer_admission.id},
                created_at=now + timedelta(minutes=1),
            )
        )
        session.commit()

        historical = service_ranking._ranking_control_sets(
            session,
            [battle],
            data_stratum="public_freeform",
            evidence_cutoff_at=now,
        )
        assert historical["preference_admissible"] == {battle.id}
        after_revocation = service_ranking._ranking_control_sets(
            session,
            [battle],
            data_stratum="public_freeform",
            evidence_cutoff_at=now + timedelta(minutes=2),
        )
        assert after_revocation["preference_admissible"] == set()
