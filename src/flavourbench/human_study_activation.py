"""Fail-closed validation for a prospective human-study activation manifest.

The consent document says what a participant sees.  This separate manifest binds
the operational and governance evidence that must exist before production may
treat that document as active.  The manifest contains references and digests,
never identity documents, contact-channel credentials, or payment credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_ACTIVATION_MANIFEST_BYTES = 256 * 1024
REQUIRED_STUDY_SCOPE = {"task_validation", "output_comparison"}
REQUIRED_CONFLICT_CLASSES = {
    "benchmark_authorship",
    "epicure_affiliation",
    "model_provider_affiliation",
    "funder_or_sponsor_affiliation",
    "task_authorship",
    "financial_interest",
    "prior_unblinded_access",
}
PLACEHOLDER_MARKERS = (
    "[required",
    "required before activation",
    "replace me",
    "placeholder",
    "test-only",
    ".invalid",
    "example.com",
    "tbd",
    "todo",
    "unknown",
    "unresolved",
)


@dataclass(frozen=True)
class HumanStudyActivationResolution:
    status: str
    manifest_sha256: str | None
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == "ready" and not self.blockers


class _DuplicateJsonKey(ValueError):
    pass


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _is_true(record: object, key: str) -> bool:
    return isinstance(record, dict) and record.get(key) is True


def _is_nonplaceholder_text(value: object, *, minimum: int = 3) -> bool:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        return False
    normalized = value.strip().casefold()
    return not any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed.astimezone(UTC)


def _all_sha256(record: object, fields: tuple[str, ...]) -> bool:
    return isinstance(record, dict) and all(_is_sha256(record.get(field)) for field in fields)


def _valid_contact(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    channel = record.get("channel")
    return bool(
        record.get("owner") == "Josef Chen"
        and isinstance(channel, str)
        and (channel.startswith("mailto:") or channel.startswith("https://"))
        and _is_nonplaceholder_text(channel, minimum=10)
        and _is_true(record, "controlled_by_research_lead")
        and _is_true(record, "monitored")
        and _is_true(record, "supports_questions")
        and _is_true(record, "supports_withdrawal")
        and _is_true(record, "supports_corrections_and_data_rights")
        and _parse_utc(record.get("monitoring_tested_at")) is not None
        and _all_sha256(record, ("control_verification_sha256", "monitoring_test_sha256"))
    )


def _valid_funding_disclosure(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    sources = record.get("sources")
    if not isinstance(sources, list) or not sources:
        return False
    valid_sources = all(
        isinstance(source, dict)
        and _is_nonplaceholder_text(source.get("name"))
        and source.get("kind") in {"cash", "compute_credit", "in_kind", "self_funded", "none"}
        and _is_sha256(source.get("terms_or_no_support_statement_sha256"))
        and isinstance(source.get("covered_scope"), list)
        and _is_true(source, "conflict_disclosed")
        for source in sources
    )
    return bool(
        _is_true(record, "complete")
        and _is_nonplaceholder_text(record.get("public_statement"), minimum=20)
        and _all_sha256(record, ("statement_sha256", "conflict_mitigation_sha256"))
        and valid_sources
    )


def _valid_compensation(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    regions = record.get("regions")
    if not isinstance(regions, list) or not regions:
        return False
    reviewer_total = 0
    for region in regions:
        if not isinstance(region, dict):
            return False
        reviewer_count = region.get("reviewer_count")
        planned_hours_milli = region.get("planned_hours_milli")
        hourly_rate_micros = region.get("hourly_rate_micros")
        minimum_wage_micros = region.get("applicable_minimum_hourly_rate_micros")
        authorized_budget_micros = region.get("authorized_budget_micros")
        if not (
            _is_nonplaceholder_text(region.get("work_location"))
            and isinstance(region.get("currency"), str)
            and re.fullmatch(r"[A-Z]{3}", region["currency"])
            and isinstance(reviewer_count, int)
            and not isinstance(reviewer_count, bool)
            and reviewer_count >= 1
            and isinstance(planned_hours_milli, int)
            and not isinstance(planned_hours_milli, bool)
            and planned_hours_milli > 0
            and isinstance(hourly_rate_micros, int)
            and not isinstance(hourly_rate_micros, bool)
            and hourly_rate_micros > 0
            and isinstance(minimum_wage_micros, int)
            and not isinstance(minimum_wage_micros, bool)
            and minimum_wage_micros > 0
            and hourly_rate_micros >= minimum_wage_micros
            and isinstance(authorized_budget_micros, int)
            and not isinstance(authorized_budget_micros, bool)
            and authorized_budget_micros
            >= math.ceil(planned_hours_milli * hourly_rate_micros / 1000)
            and _all_sha256(
                region,
                ("minimum_wage_basis_sha256", "budget_authority_sha256"),
            )
        ):
            return False
        reviewer_total += reviewer_count
    return bool(
        record.get("status") == "funded"
        and reviewer_total >= 5
        and _is_nonplaceholder_text(record.get("payer"))
        and _is_true(record, "training_calibration_and_required_admin_time_paid")
        and _is_true(record, "abstentions_and_valid_withdrawal_time_paid")
        and _is_true(record, "payment_not_contingent_on_completion_or_outcome")
        and _all_sha256(
            record,
            (
                "program_budget_authority_sha256",
                "payment_operation_sha256",
                "payment_operation_test_sha256",
            ),
        )
    )


def _valid_ethics(record: object, *, approved_at: datetime, valid_until: datetime) -> bool:
    if not isinstance(record, dict):
        return False
    decision_at = _parse_utc(record.get("decision_at"))
    determination_valid_until = _parse_utc(record.get("valid_until"))
    return bool(
        record.get("determination") in {"approved", "exempt", "equivalent_review_approved"}
        and _is_nonplaceholder_text(record.get("authority_name"))
        and _is_nonplaceholder_text(record.get("determination_reference"))
        and _all_sha256(
            record,
            (
                "determination_document_sha256",
                "reviewed_protocol_sha256",
                "conditions_acceptance_sha256",
            ),
        )
        and decision_at is not None
        and determination_valid_until is not None
        and decision_at <= approved_at < determination_valid_until
        and valid_until <= determination_valid_until
        and set(record.get("scope", [])) == REQUIRED_STUDY_SCOPE
        and _is_true(record, "conditions_implemented")
    )


def _valid_acceptance_operation(record: object) -> bool:
    return bool(
        isinstance(record, dict)
        and record.get("status") == "implemented_and_tested"
        and record.get("acceptance_actor") == "participant"
        and _is_true(record, "administrator_acceptance_prohibited")
        and _is_true(record, "exact_document_display_required")
        and _is_true(record, "unchecked_separate_confirmations_required")
        and _is_true(record, "document_digest_bound_to_acceptance")
        and _is_true(record, "acceptance_precedes_identity_verification_and_assignment")
        and _is_true(record, "participant_receipt_required")
        and _is_true(record, "idempotent_append_only_record")
        and _all_sha256(
            record,
            ("operation_contract_sha256", "end_to_end_test_report_sha256"),
        )
    )


def _valid_withdrawal_operation(record: object) -> bool:
    return bool(
        isinstance(record, dict)
        and record.get("status") == "implemented_and_tested"
        and record.get("request_actor") == "participant"
        and _is_true(record, "receipt_code_authentication_supported")
        and _is_true(record, "future_credentials_and_assignments_revoked")
        and _is_true(record, "pre_freeze_linkable_judgments_excluded")
        and _is_true(record, "append_only_withdrawal_marker_retained")
        and _is_true(record, "participant_receipt_issued")
        and _is_true(record, "post_release_limit_disclosed")
        and isinstance(record.get("response_sla_hours"), int)
        and not isinstance(record.get("response_sla_hours"), bool)
        and 1 <= record["response_sla_hours"] <= 168
        and _all_sha256(
            record,
            ("operation_contract_sha256", "end_to_end_test_report_sha256"),
        )
    )


def _valid_retention_operation(record: object) -> bool:
    return bool(
        isinstance(record, dict)
        and record.get("status") == "implemented_and_tested"
        and record.get("direct_identity_delete_within_months_after_analysis_freeze") == 12
        and record.get("pseudonymous_research_records_retained_years_after_first_release") == 5
        and _is_true(record, "contact_and_private_qualification_records_separated")
        and _is_true(record, "identity_documents_not_retained")
        and _is_true(record, "scheduled_deletion_has_named_owner")
        and _is_true(record, "deletion_receipt_and_exception_log_required")
        and _is_true(record, "sanitized_public_archive_may_persist_indefinitely_disclosed")
        and _all_sha256(
            record,
            (
                "retention_schedule_sha256",
                "deletion_operation_sha256",
                "end_to_end_test_report_sha256",
            ),
        )
    )


def _valid_identity_and_conflicts(record: object) -> bool:
    conflict_classes = record.get("required_conflict_classes") if isinstance(record, dict) else None
    return bool(
        isinstance(record, dict)
        and _is_true(record, "server_verified_distinct_real_person")
        and _is_true(record, "season_scoped_person_uniqueness_commitment")
        and _is_true(record, "raw_identity_handle_not_persisted")
        and _is_true(record, "raw_identity_documents_prohibited")
        and _is_true(record, "qualification_matched_to_task_family")
        and _is_true(record, "conflicts_disclosed_before_assignment")
        and _is_true(record, "task_author_reviewer_self_approval_prohibited")
        and _is_true(record, "prior_unblinded_access_checked")
        and _is_true(record, "affiliated_cohorts_excluded_from_independent_primary_claims")
        and isinstance(conflict_classes, list)
        and set(conflict_classes) == REQUIRED_CONFLICT_CLASSES
        and record.get("independent_admission_approver_count") == 2
        and _all_sha256(
            record,
            (
                "identity_operation_test_sha256",
                "conflict_assignment_test_sha256",
                "claim_separation_test_sha256",
            ),
        )
    )


def _valid_approvals(record: object) -> bool:
    return bool(
        isinstance(record, dict)
        and record.get("human_pi") == "Josef Chen"
        and _is_nonplaceholder_text(record.get("independent_activation_reviewer"))
        and record.get("independent_activation_reviewer") != "Josef Chen"
        and _is_true(record, "approver_identities_and_conflicts_recorded")
        and _all_sha256(
            record,
            (
                "human_pi_approval_sha256",
                "independent_activation_review_sha256",
                "data_steward_readiness_sha256",
            ),
        )
    )


def _manifest_blockers(
    manifest: object,
    *,
    consent_sha256: str,
    at: datetime,
) -> list[str]:
    if not isinstance(manifest, dict):
        return ["manifest_not_object"]
    if manifest.get("schema_version") != "flavourbench-human-study-activation-v1":
        return ["schema_version_invalid"]
    if manifest.get("status") != "approved_for_activation":
        return ["activation_status_not_approved"]
    blockers: list[str] = []
    if manifest.get("study_id") != "flavourbench-season1-human-evaluation":
        blockers.append("study_identity_invalid")
    if manifest.get("season_slug") != "season-1":
        blockers.append("season_scope_invalid")
    if manifest.get("research_lead") != "Josef Chen":
        blockers.append("research_lead_invalid")
    if not _is_nonplaceholder_text(
        manifest.get("research_lead_affiliation_disclosure"), minimum=10
    ):
        blockers.append("research_lead_affiliation_disclosure_missing")
    if manifest.get("consent_document_sha256") != consent_sha256:
        blockers.append("consent_document_mismatch")
    if set(manifest.get("study_scope", [])) != REQUIRED_STUDY_SCOPE:
        blockers.append("study_scope_invalid")
    approved_at = _parse_utc(manifest.get("approved_at"))
    valid_until = _parse_utc(manifest.get("valid_until"))
    if approved_at is None or valid_until is None or not approved_at <= at < valid_until:
        blockers.append("activation_window_invalid")
    if not _valid_contact(manifest.get("research_contact")):
        blockers.append("monitored_research_contact_missing")
    if not _valid_funding_disclosure(manifest.get("funding_and_sponsorship")):
        blockers.append("funding_and_sponsorship_disclosure_incomplete")
    if not _valid_compensation(manifest.get("compensation")):
        blockers.append("funded_fair_compensation_authority_missing")
    if (
        approved_at is None
        or valid_until is None
        or not _valid_ethics(
            manifest.get("ethics_determination"),
            approved_at=approved_at,
            valid_until=valid_until,
        )
    ):
        blockers.append("ethics_or_equivalent_determination_missing")
    if not _valid_acceptance_operation(manifest.get("participant_acceptance_operation")):
        blockers.append("participant_owned_acceptance_operation_missing")
    if not _valid_withdrawal_operation(manifest.get("withdrawal_operation")):
        blockers.append("participant_withdrawal_operation_missing")
    if not _valid_retention_operation(manifest.get("retention_operation")):
        blockers.append("retention_and_deletion_operation_missing")
    if not _valid_identity_and_conflicts(manifest.get("identity_and_conflicts")):
        blockers.append("identity_and_conflict_controls_incomplete")
    if not _valid_approvals(manifest.get("approvals")):
        blockers.append("activation_approvals_incomplete")
    if manifest.get("participants_enrolled_before_activation") != 0:
        blockers.append("preactivation_enrollment_detected")
    if manifest.get("human_judgments_collected_before_activation") != 0:
        blockers.append("preactivation_judgments_detected")
    return blockers


def resolve_human_study_activation(
    manifest_path: object,
    expected_sha256: object,
    *,
    consent_sha256: str,
    at: datetime | None = None,
) -> HumanStudyActivationResolution:
    """Resolve one exact manifest and return every semantic blocker.

    The manifest file must be a regular, non-symlinked, content-addressed JSON
    document.  A blocked or incomplete record never becomes activation evidence.
    """

    if not isinstance(manifest_path, str) or not manifest_path.strip():
        return HumanStudyActivationResolution(
            "manifest_unconfigured", None, ("activation_manifest_unconfigured",)
        )
    if not _is_sha256(expected_sha256):
        return HumanStudyActivationResolution(
            "manifest_digest_invalid", None, ("activation_manifest_digest_invalid",)
        )
    candidate = Path(manifest_path)
    if candidate.is_symlink():
        return HumanStudyActivationResolution(
            "manifest_path_invalid", expected_sha256, ("activation_manifest_path_invalid",)
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return HumanStudyActivationResolution(
            "manifest_missing", expected_sha256, ("activation_manifest_missing",)
        )
    if not resolved.is_file():
        return HumanStudyActivationResolution(
            "manifest_path_invalid", expected_sha256, ("activation_manifest_path_invalid",)
        )
    try:
        size = resolved.stat().st_size
        content = resolved.read_bytes()
    except OSError:
        return HumanStudyActivationResolution(
            "manifest_unreadable", expected_sha256, ("activation_manifest_unreadable",)
        )
    if size < 2 or size > MAX_ACTIVATION_MANIFEST_BYTES or len(content) != size:
        return HumanStudyActivationResolution(
            "manifest_size_invalid", expected_sha256, ("activation_manifest_size_invalid",)
        )
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if observed_sha256 != expected_sha256:
        return HumanStudyActivationResolution(
            "manifest_hash_mismatch", observed_sha256, ("activation_manifest_hash_mismatch",)
        )
    try:
        text = content.decode("utf-8")
        manifest = json.loads(text, object_pairs_hook=_without_duplicate_keys)
    except UnicodeDecodeError:
        return HumanStudyActivationResolution(
            "manifest_encoding_invalid", observed_sha256, ("activation_manifest_encoding_invalid",)
        )
    except (_DuplicateJsonKey, json.JSONDecodeError):
        return HumanStudyActivationResolution(
            "manifest_json_invalid", observed_sha256, ("activation_manifest_json_invalid",)
        )
    now = (at or datetime.now(UTC)).astimezone(UTC)
    blockers = tuple(_manifest_blockers(manifest, consent_sha256=consent_sha256, at=now))
    return HumanStudyActivationResolution(
        "ready" if not blockers else "blocked",
        observed_sha256,
        blockers,
    )
