from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker

from flavourbench.config import Settings
from flavourbench.consent_documents import (
    ConsentDocumentResolution,
    resolve_expert_consent_document,
)
from flavourbench.human_study_activation import resolve_human_study_activation
from flavourbench.main import app

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "protocol/human-study/human-study-activation-current-v1.json"
SCHEMA = ROOT / "protocol/human-study/human-study-activation-manifest-v1.schema.json"
PACKAGE_V3 = ROOT / "protocol/human-study/HUMAN-STUDY-GO-PACKAGE-v3.sha256"
AT = datetime(2026, 8, 9, 3, 33, 21, tzinfo=UTC)
ADMIN_HEADERS = {
    "X-FlavourBench-Service-Token": "test-service-token",
    "X-FlavourBench-Pseudonym": "7" * 64,
    "X-FlavourBench-Admin-Token": "test-admin-token",
}


def _digest(label: str) -> str:
    return hashlib.sha256(f"human-study-gate-test:{label}".encode()).hexdigest()


def _approved_manifest(consent_sha256: str) -> dict:
    return {
        "schema_version": "flavourbench-human-study-activation-v1",
        "status": "approved_for_activation",
        "study_id": "flavourbench-season1-human-evaluation",
        "season_slug": "season-1",
        "research_lead": "Josef Chen",
        "research_lead_affiliation_disclosure": (
            "A complete fixture affiliation and product-conflict disclosure."
        ),
        "consent_document_sha256": consent_sha256,
        "study_scope": ["task_validation", "output_comparison"],
        "approved_at": "2026-08-09T00:00:00Z",
        "valid_until": "2027-08-09T00:00:00Z",
        "participants_enrolled_before_activation": 0,
        "human_judgments_collected_before_activation": 0,
        "research_contact": {
            "owner": "Josef Chen",
            "channel": "https://contact.flavourbench-fixture.org/research",
            "controlled_by_research_lead": True,
            "monitored": True,
            "supports_questions": True,
            "supports_withdrawal": True,
            "supports_corrections_and_data_rights": True,
            "monitoring_tested_at": "2026-08-09T00:00:00Z",
            "control_verification_sha256": _digest("contact-control"),
            "monitoring_test_sha256": _digest("contact-test"),
        },
        "funding_and_sponsorship": {
            "complete": True,
            "public_statement": (
                "A complete fixture funding statement for validation of the gate."
            ),
            "statement_sha256": _digest("funding-statement"),
            "conflict_mitigation_sha256": _digest("funding-conflicts"),
            "sources": [
                {
                    "name": "Fixture payer",
                    "kind": "cash",
                    "terms_or_no_support_statement_sha256": _digest("funding-terms"),
                    "covered_scope": ["participant_compensation"],
                    "conflict_disclosed": True,
                }
            ],
        },
        "compensation": {
            "status": "funded",
            "payer": "Fixture payer",
            "training_calibration_and_required_admin_time_paid": True,
            "abstentions_and_valid_withdrawal_time_paid": True,
            "payment_not_contingent_on_completion_or_outcome": True,
            "program_budget_authority_sha256": _digest("program-budget"),
            "payment_operation_sha256": _digest("payment-operation"),
            "payment_operation_test_sha256": _digest("payment-test"),
            "regions": [
                {
                    "work_location": "Test region",
                    "currency": "EUR",
                    "reviewer_count": 8,
                    "planned_hours_milli": 336_960,
                    "hourly_rate_micros": 25_000_000,
                    "applicable_minimum_hourly_rate_micros": 15_000_000,
                    "authorized_budget_micros": 8_424_000_000,
                    "minimum_wage_basis_sha256": _digest("wage-basis"),
                    "budget_authority_sha256": _digest("regional-budget"),
                }
            ],
        },
        "ethics_determination": {
            "determination": "equivalent_review_approved",
            "authority_name": "Fixture ethics authority",
            "determination_reference": "fixture-determination",
            "determination_document_sha256": _digest("ethics-document"),
            "reviewed_protocol_sha256": _digest("reviewed-protocol"),
            "conditions_acceptance_sha256": _digest("ethics-conditions"),
            "decision_at": "2026-08-08T00:00:00Z",
            "valid_until": "2027-12-31T00:00:00Z",
            "scope": ["task_validation", "output_comparison"],
            "conditions_implemented": True,
        },
        "participant_acceptance_operation": {
            "status": "implemented_and_tested",
            "acceptance_actor": "participant",
            "administrator_acceptance_prohibited": True,
            "exact_document_display_required": True,
            "unchecked_separate_confirmations_required": True,
            "document_digest_bound_to_acceptance": True,
            "acceptance_precedes_identity_verification_and_assignment": True,
            "participant_receipt_required": True,
            "idempotent_append_only_record": True,
            "operation_contract_sha256": _digest("acceptance-operation"),
            "end_to_end_test_report_sha256": _digest("acceptance-test"),
        },
        "withdrawal_operation": {
            "status": "implemented_and_tested",
            "request_actor": "participant",
            "receipt_code_authentication_supported": True,
            "future_credentials_and_assignments_revoked": True,
            "pre_freeze_linkable_judgments_excluded": True,
            "append_only_withdrawal_marker_retained": True,
            "participant_receipt_issued": True,
            "post_release_limit_disclosed": True,
            "response_sla_hours": 72,
            "operation_contract_sha256": _digest("withdrawal-operation"),
            "end_to_end_test_report_sha256": _digest("withdrawal-test"),
        },
        "retention_operation": {
            "status": "implemented_and_tested",
            "direct_identity_delete_within_months_after_analysis_freeze": 12,
            "pseudonymous_research_records_retained_years_after_first_release": 5,
            "contact_and_private_qualification_records_separated": True,
            "identity_documents_not_retained": True,
            "scheduled_deletion_has_named_owner": True,
            "deletion_receipt_and_exception_log_required": True,
            "sanitized_public_archive_may_persist_indefinitely_disclosed": True,
            "retention_schedule_sha256": _digest("retention-schedule"),
            "deletion_operation_sha256": _digest("deletion-operation"),
            "end_to_end_test_report_sha256": _digest("retention-test"),
        },
        "identity_and_conflicts": {
            "server_verified_distinct_real_person": True,
            "season_scoped_person_uniqueness_commitment": True,
            "raw_identity_handle_not_persisted": True,
            "raw_identity_documents_prohibited": True,
            "qualification_matched_to_task_family": True,
            "conflicts_disclosed_before_assignment": True,
            "task_author_reviewer_self_approval_prohibited": True,
            "prior_unblinded_access_checked": True,
            "affiliated_cohorts_excluded_from_independent_primary_claims": True,
            "required_conflict_classes": [
                "benchmark_authorship",
                "epicure_affiliation",
                "model_provider_affiliation",
                "funder_or_sponsor_affiliation",
                "task_authorship",
                "financial_interest",
                "prior_unblinded_access",
            ],
            "independent_admission_approver_count": 2,
            "identity_operation_test_sha256": _digest("identity-test"),
            "conflict_assignment_test_sha256": _digest("conflict-test"),
            "claim_separation_test_sha256": _digest("claim-separation-test"),
        },
        "approvals": {
            "human_pi": "Josef Chen",
            "independent_activation_reviewer": "Fixture independent reviewer",
            "approver_identities_and_conflicts_recorded": True,
            "human_pi_approval_sha256": _digest("pi-approval"),
            "independent_activation_review_sha256": _digest("independent-review"),
            "data_steward_readiness_sha256": _digest("data-steward"),
        },
    }


def _write_manifest(tmp_path: Path, manifest: dict) -> tuple[Path, str]:
    content = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(content).hexdigest()
    path = tmp_path / f"{digest}.json"
    path.write_bytes(content)
    return path, digest


def _production_settings(**values: object) -> Settings:
    base: dict[str, object] = {
        "environment": "production",
        "execution_mode": "live",
        "live_authorized": True,
        "database_url": "postgresql://test-only.invalid/flavourbench",
        "auto_create_schema": False,
        "service_token": "test-only-service-token-0000000001",
        "admin_token": "test-only-admin-token-000000000001",
        "expert_token": "test-only-expert-token-00000000001",
        "pseudonym_secret": "test-only-pseudonym-secret-000000001",
        "task_validator_identity_hmac_secret": ("test-only-task-validator-hmac-0000001"),
        "reviewer_identity_hmac_secret": "test-only-reviewer-identity-hmac-00001",
        "reviewer_credential_hmac_secret": "test-only-reviewer-credential-hmac-001",
        "organization_api_key_hmac_secret": "test-only-organization-key-hmac-000001",
        "run_card_signing_secret": "test-only-run-card-signing-secret-00001",
        "budget_authorization_signing_secret": "test-only-budget-signing-secret-000001",
    }
    base.update(values)
    return Settings(**base)


def test_current_record_is_schema_valid_and_explicitly_blocked() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    record = json.loads(CURRENT.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)

    assert record["status"] == "blocked"
    assert record["participants_enrolled_before_activation"] == 0
    assert record["human_judgments_collected_before_activation"] == 0
    assert record["human_evaluation_budget"] == {
        "EUR_micros": 0,
        "USD_micros": 0,
        "funded": False,
        "controlling_record_sha256": (
            "ca8863159378f9386b20512a9a6c8096d7754d70ea092a0845037d9f328048e2"
        ),
    }
    assert set(record["blockers"]) == {
        "monitored_research_contact_missing",
        "research_lead_affiliation_disclosure_inconsistent",
        "funding_and_sponsorship_disclosure_incomplete",
        "funded_fair_compensation_authority_missing",
        "ethics_or_equivalent_determination_missing",
        "participant_owned_acceptance_operation_missing",
        "participant_withdrawal_operation_missing",
        "retention_and_deletion_operation_missing",
        "identity_and_conflict_controls_incomplete",
        "activation_approvals_incomplete",
    }
    for source in record["source_records"]:
        source_path = ROOT / source["path"]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]


def test_v3_checksum_package_is_additive_exact_and_still_blocked() -> None:
    assert (
        hashlib.sha256(
            (ROOT / "protocol/human-study/HUMAN-STUDY-GO-PACKAGE-v1.sha256").read_bytes()
        ).hexdigest()
        == "2d383474732dd4d4e4ad9a67e71292b2a0aaf6ae3f20dc5213b44dd9301841ad"
    )
    assert (
        hashlib.sha256(
            (ROOT / "protocol/human-study/HUMAN-STUDY-GO-PACKAGE-v2.sha256").read_bytes()
        ).hexdigest()
        == "49d8ccc88cf1d20d12d250b189534ba85aa7bb14410edb92fb3c902e06607195"
    )

    rows = [line.split("  ", 1) for line in PACKAGE_V3.read_text().splitlines()]
    assert len(rows) == 61
    assert len({relative_path for _, relative_path in rows}) == len(rows)
    required = {
        "flavourbench/alembic/versions/0035_participant_lifecycle_privacy.py",
        "flavourbench/src/flavourbench/participant_lifecycle.py",
        "flavourbench/src/flavourbench/task_validation_runtime.py",
        "flavourbench/tests/postgresql_candidate_capacity_helper.py",
        "flavourbench/tests/task_validation_concurrent_submit_helper.py",
        "flavourbench/scripts/upgrade_legacy_0001_postgresql_v4.py",
        (
            "flavourbench/artifacts/migration-proofs/"
            "legacy-0001-upgrade-bridge-proof-v4-"
            "e2c41dddfe5af1907cf22dce41b3a8ea6c50b773882c9e09b01f0c7c82b4ec2f.json"
        ),
        "../../epicure/compose.yaml",
        "../../epicure/.env.example",
    }
    assert required <= {relative_path for _, relative_path in rows}
    for expected_sha256, relative_path in rows:
        assert len(expected_sha256) == 64
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == (expected_sha256)

    current = json.loads(CURRENT.read_text())
    assert current["status"] == "blocked"
    assert current["participants_enrolled_before_activation"] == 0
    assert current["human_judgments_collected_before_activation"] == 0


def test_blocked_current_record_can_never_be_activation_evidence(tmp_path: Path) -> None:
    content = CURRENT.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    path = tmp_path / f"{digest}.json"
    path.write_bytes(content)

    resolution = resolve_human_study_activation(
        str(path),
        digest,
        consent_sha256=("b593ba871f90d0dce505126f853e0c0ba1ae0d13fde971e0421e7533d048b2bb"),
        at=AT,
    )

    assert resolution.status == "blocked"
    assert resolution.blockers == ("activation_status_not_approved",)


@pytest.mark.parametrize(
    ("path", "bad_value", "blocker"),
    [
        (("research_contact", "monitored"), False, "monitored_research_contact_missing"),
        (
            ("funding_and_sponsorship", "complete"),
            False,
            "funding_and_sponsorship_disclosure_incomplete",
        ),
        (
            ("compensation", "regions", 0, "authorized_budget_micros"),
            0,
            "funded_fair_compensation_authority_missing",
        ),
        (
            ("ethics_determination", "determination"),
            "pending",
            "ethics_or_equivalent_determination_missing",
        ),
        (
            ("participant_acceptance_operation", "acceptance_actor"),
            "administrator",
            "participant_owned_acceptance_operation_missing",
        ),
        (
            ("withdrawal_operation", "status"),
            "planned",
            "participant_withdrawal_operation_missing",
        ),
        (
            ("retention_operation", "status"),
            "planned",
            "retention_and_deletion_operation_missing",
        ),
        (
            ("identity_and_conflicts", "raw_identity_documents_prohibited"),
            False,
            "identity_and_conflict_controls_incomplete",
        ),
        (
            ("approvals", "independent_activation_reviewer"),
            "Josef Chen",
            "activation_approvals_incomplete",
        ),
        (("participants_enrolled_before_activation",), 1, "preactivation_enrollment_detected"),
        (
            ("human_judgments_collected_before_activation",),
            1,
            "preactivation_judgments_detected",
        ),
    ],
)
def test_each_material_prerequisite_fails_closed(
    tmp_path: Path,
    path: tuple[str | int, ...],
    bad_value: object,
    blocker: str,
) -> None:
    consent_sha256 = _digest("consent")
    manifest = copy.deepcopy(_approved_manifest(consent_sha256))
    target: object = manifest
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = bad_value  # type: ignore[index]
    manifest_path, manifest_sha256 = _write_manifest(tmp_path, manifest)

    resolution = resolve_human_study_activation(
        str(manifest_path),
        manifest_sha256,
        consent_sha256=consent_sha256,
        at=AT,
    )

    assert not resolution.ready
    assert blocker in resolution.blockers


def test_fair_pay_gate_checks_rate_floor_roster_and_whole_budget(tmp_path: Path) -> None:
    consent_sha256 = _digest("consent")
    for mutation in (
        {"reviewer_count": 4},
        {"hourly_rate_micros": 14_999_999},
        {"authorized_budget_micros": 8_423_999_999},
    ):
        manifest = _approved_manifest(consent_sha256)
        manifest["compensation"]["regions"][0].update(mutation)
        path, digest = _write_manifest(tmp_path, manifest)
        resolution = resolve_human_study_activation(
            str(path), digest, consent_sha256=consent_sha256, at=AT
        )
        assert "funded_fair_compensation_authority_missing" in resolution.blockers


def test_only_a_complete_content_addressed_manifest_can_unlock_production_consent(
    tmp_path: Path,
) -> None:
    consent_text = "# Test consent\n\nStatus: active\n\nTest-only participant terms.\n"
    consent_sha256 = hashlib.sha256(consent_text.encode()).hexdigest()
    consent_dir = tmp_path / "consent"
    consent_dir.mkdir()
    (consent_dir / f"{consent_sha256}.md").write_text(consent_text, encoding="utf-8")

    blocked = resolve_expert_consent_document(
        consent_sha256,
        settings=_production_settings(
            active_expert_consent_sha256s=[consent_sha256],
            expert_consent_documents_dir=str(consent_dir),
        ),
    )
    assert blocked.status == "human_study_governance_manifest_unconfigured"
    assert blocked.text is None

    manifest = _approved_manifest(consent_sha256)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(manifest)
    manifest_path, manifest_sha256 = _write_manifest(tmp_path, manifest)
    active = resolve_expert_consent_document(
        consent_sha256,
        settings=_production_settings(
            active_expert_consent_sha256s=[consent_sha256],
            expert_consent_documents_dir=str(consent_dir),
            human_study_activation_manifest_path=str(manifest_path),
            human_study_activation_manifest_sha256=manifest_sha256,
        ),
    )

    assert active.status == "active"
    assert active.text == consent_text


def test_manifest_hash_mismatch_and_symlink_fail_closed(tmp_path: Path) -> None:
    consent_sha256 = _digest("consent")
    manifest_path, manifest_sha256 = _write_manifest(tmp_path, _approved_manifest(consent_sha256))
    mismatch = resolve_human_study_activation(
        str(manifest_path), "f" * 64, consent_sha256=consent_sha256, at=AT
    )
    assert mismatch.status == "manifest_hash_mismatch"

    link = tmp_path / "activation-link.json"
    link.symlink_to(manifest_path)
    symlink = resolve_human_study_activation(
        str(link), manifest_sha256, consent_sha256=consent_sha256, at=AT
    )
    assert symlink.status == "manifest_path_invalid"


def test_unresolved_governance_blocks_expert_invitation_before_enrollment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flavourbench.main.resolve_expert_consent_document",
        lambda _digest: ConsentDocumentResolution(_digest, "human_study_governance_blocked", None),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/admin/experts",
            headers=ADMIN_HEADERS,
            json={
                "reviewer_code": "governance-blocked-invitation-fixture",
                "qualified_families": ["cookability"],
                "qualification_reference": "fixture-qualification",
                "qualification_verified": False,
                "affiliation_class": "independent_external",
                "conflict_disclosure_reference": "fixture-conflict-disclosure",
                "consent_document_sha256": _digest("blocked-consent"),
                "training_material_sha256": _digest("training"),
                "calibration_set_sha256": _digest("calibration"),
                "calibration_accuracy": 0,
                "compensation_reference": "fixture-compensation",
            },
        )

    assert response.status_code == 409
    assert "operationally governed active consent" in response.text
    assert set(response.json()) == {"detail"}


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    content = b'{"schema_version":"x","schema_version":"y"}'
    digest = hashlib.sha256(content).hexdigest()
    path = tmp_path / f"{digest}.json"
    path.write_bytes(content)

    resolution = resolve_human_study_activation(
        str(path), digest, consent_sha256=_digest("consent"), at=AT
    )

    assert resolution.status == "manifest_json_invalid"
