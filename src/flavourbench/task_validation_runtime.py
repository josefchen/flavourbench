"""Live human runtime for the frozen v6 public-source task campaign.

This router never invokes a model, provider, Epicure, or source network. It
serves only the pinned human-written prompts and records evidence from real,
privately verified reviewers through the season-scoped 0030 identity system.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from .config import get_settings
from .construct_blueprint import CONSTRUCT_CELLS, DIFFICULTY_TIERS
from .database import get_db
from .models import (
    ExpertReviewer,
    ReviewerFamilyAdmission,
    ReviewerIdentityBinding,
    Season,
    TaskValidationAuditAuthorization,
    TaskValidationCampaignEvent,
    _task_validation_audit_authorization_sha256,
)
from .participant_lifecycle import (
    ParticipantLifecycleError,
    participant_record_analysis_eligible,
    require_active_participant_authority,
)
from .prospective_task_acquisition import ASSIGNMENT_SCHEMA, canonical_sha256, verify_artifact
from .reviewer_identity import (
    ReviewerIdentityError,
    consume_reviewer_credential,
    reviewer_rater_pseudonym,
)
from .security import require_admin_token, require_service_token
from .task_validation_automated_replay import (
    PINNED_INPUTS,
    ReplayInputPaths,
    TaskValidationReplayError,
    verify_pinned_replay,
)
from .task_validation_campaign import (
    QUALITY_REPORT_SCHEMA,
    READINESS_SCHEMA,
    ZERO_SHA256,
    TaskValidationCampaignError,
    derive_candidate_state,
    make_ledger_event,
    merged_criterion_pack,
    public_event_view,
    verify_campaign_packet,
    verify_event_chain,
)
from .task_validation_replay_binding import (
    TASK_VALIDATION_FORMAL_CONTAMINATION_METHODS,
    TASK_VALIDATION_RIGHTS_ANOMALY_IDS,
    TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
    TASK_VALIDATION_RIGHTS_REQUIRED_IDS,
    TASK_VALIDATION_RIGHTS_SAMPLE_IDS,
    TASK_VALIDATION_RIGHTS_SAMPLE_SEED_SHA256,
    TASK_VALIDATION_V1_REPLAY_PHYSICAL_SHA256,
    TASK_VALIDATION_V1_REPLAY_SHA256,
    TASK_VALIDATION_V6_CAMPAIGN_SHA256,
    rights_audit_plan,
)

CAMPAIGN_EVENT_TYPES = frozenset(
    {
        "blind_ballot",
        "criterion_pack_confirmation",
        "adjudication",
        "rights_batch_audit",
        "contamination_batch_audit",
    }
)
CANDIDATE_EVENT_CAPACITIES = {
    "blind_ballot": 2,
    "criterion_pack_confirmation": 2,
    "adjudication": 1,
}
FAMILIES = tuple(CONSTRUCT_CELLS)
CHECK_NAMES = (
    "construct_fit",
    "context_complete",
    "coherent_question",
    "general_track_scope",
    "answer_leakage_absent",
    "discrimination_value",
)
ISSUE_TAGS = frozenset(
    {
        "construct_mismatch",
        "missing_context",
        "incoherent_question",
        "specialist_scope",
        "answer_leakage",
        "low_discrimination",
        "visual_context_missing",
        "self_resolution",
        "duplicate_or_contaminated",
        "rights_uncertain",
        "other",
    }
)
TERMINAL_VALIDATED = frozenset({"validated_consensus", "validated_adjudicated"})
TERMINAL_FAILED = frozenset({"adjudicated_revise", "adjudicated_reject"})
# The database enforces the auditor's verified identity, uniqueness, and event
# authority. Qualification, conflict, and automated-evidence digests are
# separately attested by the authenticated admin workflow; their presence does
# not verify those artifacts or open a release/ranking gate.
AUDIT_AUTHORIZATION_TRUST_BOUNDARY = "server_verified_frozen_replay_plus_admin_human_evidence"

router = APIRouter(dependencies=[Depends(require_service_token)])
Db = Annotated[Session, Depends(get_db)]


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalize_lines(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        item = " ".join(value.split())
        if item and item not in normalized:
            normalized.append(item)
    return normalized


class BallotChecks(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    construct_fit: bool = Field(alias="constructFit")
    context_complete: bool = Field(alias="contextComplete")
    coherent_question: bool = Field(alias="coherentQuestion")
    general_track_scope: bool = Field(alias="generalTrackScope")
    answer_leakage_absent: bool = Field(alias="answerLeakageAbsent")
    discrimination_value: bool = Field(alias="discriminationValue")


class BlindBallotCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    decision: Literal["valid", "revise", "exclude"]
    checks: BallotChecks
    family: Literal["substitution", "composition", "cookability", "evidence"] | None = None
    construct_cell_id: str | None = Field(
        default=None, min_length=3, max_length=120, alias="constructCellId"
    )
    difficulty_tier: Literal["foundation", "integrative", "stress"] | None = Field(
        default=None, alias="difficultyTier"
    )
    independent_solution_outline: str = Field(
        default="", max_length=2000, alias="independentSolutionOutline"
    )
    success_criteria: list[str] = Field(
        default_factory=list, max_length=12, alias="successCriteria"
    )
    permitted_variations: list[str] = Field(
        default_factory=list, max_length=12, alias="permittedVariations"
    )
    disqualifying_errors: list[str] = Field(
        default_factory=list, max_length=12, alias="disqualifyingErrors"
    )
    objective_checks: list[str] = Field(
        default_factory=list, max_length=12, alias="objectiveChecks"
    )
    issue_tags: list[str] = Field(default_factory=list, max_length=8, alias="issueTags")
    note: str = Field(default="", max_length=1600)
    source_metadata_seen: Literal[False] = Field(alias="sourceMetadataSeen")
    other_ballot_seen: Literal[False] = Field(alias="otherBallotSeen")
    model_outputs_seen: Literal[False] = Field(alias="modelOutputsSeen")
    scheduling_family_seen: Literal[False] = Field(alias="schedulingFamilySeen")

    @field_validator(
        "success_criteria",
        "permitted_variations",
        "disqualifying_errors",
        "objective_checks",
    )
    @classmethod
    def normalize_criterion_lines(cls, values: list[str]) -> list[str]:
        return _normalize_lines(values)

    @field_validator("issue_tags")
    @classmethod
    def validate_issue_tags(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(values))
        if any(value not in ISSUE_TAGS for value in normalized):
            raise ValueError("unsupported task-validity issue tag")
        return normalized

    @model_validator(mode="after")
    def validate_decision_contract(self) -> BlindBallotCreate:
        checks = self.checks.model_dump()
        if self.decision == "valid":
            if not all(checks.values()):
                raise ValueError("valid ballots require all six checks")
            if self.issue_tags:
                raise ValueError("valid ballots cannot carry issue tags")
            if self.family is None or self.construct_cell_id is None:
                raise ValueError("valid ballots require a family and construct cell")
            if self.construct_cell_id not in CONSTRUCT_CELLS[self.family]:
                raise ValueError("construct cell does not belong to the selected family")
            if self.difficulty_tier not in DIFFICULTY_TIERS:
                raise ValueError("valid ballots require a frozen difficulty tier")
            if len(self.independent_solution_outline.strip()) < 20:
                raise ValueError("valid ballots require an independent solution outline")
            if not self.success_criteria:
                raise ValueError("valid ballots require at least one success criterion")
            if not self.permitted_variations:
                raise ValueError("valid ballots require at least one permitted variation")
            if not self.disqualifying_errors:
                raise ValueError("valid ballots require at least one disqualifying error")
        else:
            if not self.issue_tags or len(self.note.strip()) < 10:
                raise ValueError("revise and exclude ballots require an issue tag and note")
            if any(
                (
                    self.family is not None,
                    self.construct_cell_id is not None,
                    self.difficulty_tier is not None,
                    bool(self.independent_solution_outline.strip()),
                    bool(self.success_criteria),
                    bool(self.permitted_variations),
                    bool(self.disqualifying_errors),
                    bool(self.objective_checks),
                )
            ):
                raise ValueError("non-valid ballots cannot publish a criterion pack")
        return self


class CriterionPackConfirmationCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    criterion_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="criterionPackSha256")
    accepted: bool
    note: str = Field(default="", max_length=1200)
    model_outputs_seen: Literal[False] = Field(alias="modelOutputsSeen")

    @model_validator(mode="after")
    def require_refusal_note(self) -> CriterionPackConfirmationCreate:
        if not self.accepted and len(self.note.strip()) < 10:
            raise ValueError("a refused criterion pack requires a note")
        return self


class AdjudicationCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    decision: Literal["approve", "revise", "reject"]
    family: Literal["substitution", "composition", "cookability", "evidence"] | None = None
    construct_cell_id: str | None = Field(
        default=None, min_length=3, max_length=120, alias="constructCellId"
    )
    difficulty_tier: Literal["foundation", "integrative", "stress"] | None = Field(
        default=None, alias="difficultyTier"
    )
    success_criteria: list[str] = Field(
        default_factory=list, max_length=18, alias="successCriteria"
    )
    permitted_variations: list[str] = Field(
        default_factory=list, max_length=18, alias="permittedVariations"
    )
    disqualifying_errors: list[str] = Field(
        default_factory=list, max_length=18, alias="disqualifyingErrors"
    )
    objective_checks: list[str] = Field(
        default_factory=list, max_length=18, alias="objectiveChecks"
    )
    note: str = Field(min_length=10, max_length=2000)
    model_outputs_seen: Literal[False] = Field(alias="modelOutputsSeen")
    independent_attestation: Literal[True] = Field(alias="independentAttestation")

    @field_validator(
        "success_criteria",
        "permitted_variations",
        "disqualifying_errors",
        "objective_checks",
    )
    @classmethod
    def normalize_criterion_lines(cls, values: list[str]) -> list[str]:
        return _normalize_lines(values)

    @model_validator(mode="after")
    def validate_adjudication_contract(self) -> AdjudicationCreate:
        if self.decision == "approve":
            if self.family is None or self.construct_cell_id is None:
                raise ValueError("approved adjudications require final labels")
            if self.construct_cell_id not in CONSTRUCT_CELLS[self.family]:
                raise ValueError("construct cell does not belong to the selected family")
            if self.difficulty_tier not in DIFFICULTY_TIERS:
                raise ValueError("approved adjudications require a frozen difficulty tier")
            if not (
                self.success_criteria and self.permitted_variations and self.disqualifying_errors
            ):
                raise ValueError("approved adjudications require a complete criterion pack")
        elif any(
            (
                self.family is not None,
                self.construct_cell_id is not None,
                self.difficulty_tier is not None,
                bool(self.success_criteria),
                bool(self.permitted_variations),
                bool(self.disqualifying_errors),
                bool(self.objective_checks),
            )
        ):
            raise ValueError("revise and reject adjudications cannot freeze a final pack")
        return self


class AuditAuthorizationCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    reviewer_id: str = Field(min_length=36, max_length=36, alias="reviewerId")
    audit_kind: Literal["rights", "contamination"] = Field(alias="auditKind")
    qualification_evidence_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="qualificationEvidenceSha256"
    )
    conflict_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="conflictEvidenceSha256")
    decision_reference_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="decisionReferenceSha256"
    )


class AuditFinding(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    candidate_id: str = Field(min_length=36, max_length=36, alias="candidateId")
    severity: Literal["information", "minor", "material"]
    category: str = Field(min_length=3, max_length=80)
    note: str = Field(min_length=10, max_length=1200)
    resolved: bool


class BatchAuditCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    audit_kind: Literal["rights", "contamination"] = Field(alias="auditKind")
    decision: Literal["pass", "fail"]
    reviewed_candidate_ids: list[str] = Field(
        min_length=1, max_length=180, alias="reviewedCandidateIds"
    )
    findings: list[AuditFinding] = Field(default_factory=list, max_length=180)
    unresolved_material_findings: int = Field(
        ge=0, le=180, strict=True, alias="unresolvedMaterialFindings"
    )
    complete_coverage_evidence_checked: Literal[True] = Field(
        alias="completeCoverageEvidenceChecked"
    )
    no_model_outputs_seen: Literal[True] = Field(alias="noModelOutputsSeen")
    public_source_contamination_limited_acknowledged: Literal[True] = Field(
        alias="publicSourceContaminationLimitedAcknowledged"
    )
    no_contamination_free_claim: Literal[True] = Field(alias="noContaminationFreeClaim")
    note: str = Field(min_length=20, max_length=3000)

    @field_validator("reviewed_candidate_ids")
    @classmethod
    def unique_reviewed_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("reviewed candidate IDs must be unique")
        return values

    @model_validator(mode="after")
    def validate_audit_decision(self) -> BatchAuditCreate:
        material_unresolved = sum(
            finding.severity == "material" and not finding.resolved for finding in self.findings
        )
        if material_unresolved != self.unresolved_material_findings:
            raise ValueError("unresolved material finding count does not reconcile")
        if self.decision == "pass" and self.unresolved_material_findings:
            raise ValueError("an audit with unresolved material findings cannot pass")
        return self


@dataclass(frozen=True)
class CampaignContext:
    campaign: dict[str, Any]
    assignment: dict[str, Any]
    quality_report: dict[str, Any]
    readiness: dict[str, Any]
    assignment_by_candidate: dict[str, dict[str, Any]]
    automated_replay: AutomatedReplayEvidence


@dataclass(frozen=True)
class ReviewerContext:
    reviewer: ExpertReviewer
    binding: ReviewerIdentityBinding
    pseudonym: str


@dataclass(frozen=True)
class AutomatedReplayEvidence:
    replay_sha256: str
    replay_physical_sha256: str
    rights_sample_seed_sha256: str
    rights_sample_ids: tuple[str, ...]
    rights_anomaly_ids: tuple[str, ...]
    rights_required_ids: tuple[str, ...]
    contamination_sample_seed_sha256: str
    contamination_sample_ids: tuple[str, ...]
    local_prompt_risk_hit_ids: tuple[str, ...]
    contamination_required_ids: tuple[str, ...]
    rights_automated_evidence_verified: bool = True
    local_prompt_risk_replay_verified: bool = True
    contamination_automated_evidence_verified: bool = False
    contamination_campaign_coverage_verified: bool = False


class TaskValidationRuntimeEvidenceError(RuntimeError):
    """A configured campaign input or replay binding failed closed."""


def _physical_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise TaskValidationRuntimeEvidenceError("pinned task-validation input is unavailable")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise TaskValidationRuntimeEvidenceError(
            "pinned task-validation input is unreadable"
        ) from exc
    return digest.hexdigest()


def _handoff_ids(
    document: Mapping[str, Any], kind: str
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    try:
        handoff = document["human_audit_handoff"]
        record = handoff[kind]
        seed = str(record["sample_seed_commitment_sha256"])
        sample = tuple(str(value) for value in record["sample_candidate_ids"])
        hits = tuple(str(value) for value in record["anomaly_or_hit_candidate_ids"])
        required = tuple(str(value) for value in record["required_candidate_ids"])
    except (KeyError, TypeError) as exc:
        raise TaskValidationRuntimeEvidenceError(
            "verified replay lacks a complete human-audit handoff"
        ) from exc
    if (
        record.get("sample_algorithm") != "sha256-order-within-scheduling-family-v1"
        or len(sample) != 24
        or len(sample) != len(set(sample))
        or len(hits) != len(set(hits))
        or len(required) != len(set(required))
        or tuple(sorted(set(sample) | set(hits))) != required
    ):
        raise TaskValidationRuntimeEvidenceError(
            "verified replay human-audit handoff is internally inconsistent"
        )
    return seed, sample, hits, required


@lru_cache(maxsize=4)
def _verify_automated_replay_cached(
    configured: tuple[str, ...],
    observed_physical_sha256s: tuple[str, ...],
) -> AutomatedReplayEvidence:
    (
        candidate_bundle_path,
        candidate_bundle_sha256,
        assignment_path,
        assignment_sha256,
        acquisition_receipt_path,
        acquisition_receipt_sha256,
        campaign_path,
        campaign_sha256,
        quality_path,
        quality_sha256,
        readiness_path,
        readiness_sha256,
        replay_path,
        replay_sha256,
        replay_physical_sha256,
    ) = configured
    expected_semantic = {
        "candidate_bundle": candidate_bundle_sha256,
        "review_assignment": assignment_sha256,
        "acquisition_receipt": acquisition_receipt_sha256,
        "campaign": campaign_sha256,
        "quality_report": quality_sha256,
        "readiness": readiness_sha256,
    }
    if any(
        expected_semantic[role] != pin["semantic_sha256"] for role, pin in PINNED_INPUTS.items()
    ) or (
        replay_sha256 != TASK_VALIDATION_V1_REPLAY_SHA256
        or replay_physical_sha256 != TASK_VALIDATION_V1_REPLAY_PHYSICAL_SHA256
        or campaign_sha256 != TASK_VALIDATION_V6_CAMPAIGN_SHA256
    ):
        raise TaskValidationRuntimeEvidenceError(
            "task-validation replay configuration does not match the frozen v1 contract"
        )
    paths = ReplayInputPaths(
        candidate_bundle=Path(candidate_bundle_path),
        review_assignment=Path(assignment_path),
        acquisition_receipt=Path(acquisition_receipt_path),
        campaign=Path(campaign_path),
        quality_report=Path(quality_path),
        readiness=Path(readiness_path),
    )
    try:
        receipt = verify_pinned_replay(Path(replay_path), paths)
        document = json.loads(Path(replay_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TaskValidationReplayError) as exc:
        raise TaskValidationRuntimeEvidenceError(
            "task-validation automated replay failed verification"
        ) from exc
    if not isinstance(document, dict) or receipt != {
        "artifact_sha256": TASK_VALIDATION_V1_REPLAY_SHA256,
        "physical_sha256": TASK_VALIDATION_V1_REPLAY_PHYSICAL_SHA256,
        "rights_automatedEvidenceVerified": True,
        "contamination_automatedEvidenceVerified": True,
        "contamination_automated_hit_candidate_ids": document.get("runtime_projection", {}).get(
            "contamination_automated_hit_candidate_ids"
        ),
        "human_decision_fields_must_remain_unchanged": True,
        "campaign_audit_passed": False,
        "task_bank_import_authorized": False,
        "rank_eligible": False,
        "contamination_free": False,
    }:
        raise TaskValidationRuntimeEvidenceError(
            "task-validation automated replay receipt exceeds or differs from v1"
        )
    rights_seed, rights_sample, rights_anomalies, rights_required = _handoff_ids(document, "rights")
    contamination_seed, contamination_sample, prompt_risk_hits, contamination_required = (
        _handoff_ids(document, "contamination")
    )
    if (
        rights_seed != TASK_VALIDATION_RIGHTS_SAMPLE_SEED_SHA256
        or rights_sample != TASK_VALIDATION_RIGHTS_SAMPLE_IDS
        or rights_anomalies != TASK_VALIDATION_RIGHTS_ANOMALY_IDS
        or rights_required != TASK_VALIDATION_RIGHTS_REQUIRED_IDS
        or tuple(receipt["contamination_automated_hit_candidate_ids"]) != prompt_risk_hits
        or document.get("limitations", {}).get("external_benchmark_corpus_tested") is not False
        or document.get("limitations", {}).get("external_web_search_performed") is not False
        or document.get("limitations", {}).get("model_training_membership_tested") is not False
    ):
        raise TaskValidationRuntimeEvidenceError(
            "task-validation replay is not the bounded v1 evidence contract"
        )
    current_physical = tuple(
        _physical_sha256(path) for path in (*paths.as_mapping().values(), Path(replay_path))
    )
    if current_physical != observed_physical_sha256s:
        raise TaskValidationRuntimeEvidenceError(
            "task-validation replay inputs changed during verification"
        )
    return AutomatedReplayEvidence(
        replay_sha256=TASK_VALIDATION_V1_REPLAY_SHA256,
        replay_physical_sha256=TASK_VALIDATION_V1_REPLAY_PHYSICAL_SHA256,
        rights_sample_seed_sha256=rights_seed,
        rights_sample_ids=rights_sample,
        rights_anomaly_ids=rights_anomalies,
        rights_required_ids=rights_required,
        contamination_sample_seed_sha256=contamination_seed,
        contamination_sample_ids=contamination_sample,
        local_prompt_risk_hit_ids=prompt_risk_hits,
        contamination_required_ids=contamination_required,
    )


def verify_task_validation_runtime_evidence() -> AutomatedReplayEvidence:
    """Verify exact mounted bytes and rebuild the v1 replay from all six inputs."""

    settings = get_settings()
    if not settings.task_validation_campaign_enabled:
        raise TaskValidationRuntimeEvidenceError("task-validation campaign is disabled")
    configured = (
        settings.task_validation_candidate_bundle_path,
        settings.task_validation_candidate_bundle_sha256,
        settings.task_validation_assignment_path,
        settings.task_validation_assignment_sha256,
        settings.task_validation_acquisition_receipt_path,
        settings.task_validation_acquisition_receipt_sha256,
        settings.task_validation_campaign_path,
        settings.task_validation_campaign_sha256,
        settings.task_validation_quality_report_path,
        settings.task_validation_quality_report_sha256,
        settings.task_validation_readiness_path,
        settings.task_validation_readiness_sha256,
        settings.task_validation_automated_replay_path,
        settings.task_validation_automated_replay_sha256,
        settings.task_validation_automated_replay_physical_sha256,
    )
    file_paths = tuple(Path(configured[index]) for index in range(0, 14, 2))
    if len(set(file_paths)) != 7:
        raise TaskValidationRuntimeEvidenceError(
            "task-validation replay inputs are not seven distinct files"
        )
    observed = tuple(_physical_sha256(path) for path in file_paths)
    evidence = _verify_automated_replay_cached(configured, observed)
    if tuple(_physical_sha256(path) for path in file_paths) != observed:
        raise TaskValidationRuntimeEvidenceError(
            "task-validation replay inputs changed while consulting the verification cache"
        )
    return evidence


def _read_json_artifact(path_value: str, label: str) -> dict[str, Any]:
    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise HTTPException(status_code=503, detail=f"pinned {label} is unavailable")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail=f"pinned {label} is invalid") from exc
    if not isinstance(document, dict):
        raise HTTPException(status_code=503, detail=f"pinned {label} is not an object")
    return document


def _campaign_context() -> CampaignContext:
    settings = get_settings()
    if not settings.task_validation_campaign_enabled:
        raise HTTPException(status_code=503, detail="task-validation campaign is not enabled")
    try:
        automated_replay = verify_task_validation_runtime_evidence()
    except TaskValidationRuntimeEvidenceError as exc:
        raise HTTPException(
            status_code=503,
            detail="task-validation automated evidence failed verification",
        ) from exc
    campaign = _read_json_artifact(settings.task_validation_campaign_path, "campaign")
    assignment = _read_json_artifact(settings.task_validation_assignment_path, "assignment")
    quality = _read_json_artifact(
        settings.task_validation_quality_report_path, "campaign quality report"
    )
    readiness = _read_json_artifact(
        settings.task_validation_readiness_path, "campaign readiness decision"
    )
    try:
        verify_campaign_packet(campaign)
        verify_artifact(assignment, schema_version=ASSIGNMENT_SCHEMA)
        verify_artifact(quality, schema_version=QUALITY_REPORT_SCHEMA)
        verify_artifact(readiness, schema_version=READINESS_SCHEMA)
    except (TaskValidationCampaignError, ValueError) as exc:
        raise HTTPException(
            status_code=503, detail="task-validation artifact verification failed"
        ) from exc
    expected = {
        "campaign": settings.task_validation_campaign_sha256,
        "assignment": settings.task_validation_assignment_sha256,
        "quality": settings.task_validation_quality_report_sha256,
        "readiness": settings.task_validation_readiness_sha256,
    }
    observed = {
        "campaign": campaign.get("artifact_sha256"),
        "assignment": assignment.get("artifact_sha256"),
        "quality": quality.get("artifact_sha256"),
        "readiness": readiness.get("artifact_sha256"),
    }
    if expected != observed:
        raise HTTPException(status_code=503, detail="task-validation artifact pin differs")
    source = campaign.get("source_artifacts", {})
    bound = readiness.get("bound_artifacts", {})
    if (
        source.get("review_assignment_sha256") != observed["assignment"]
        or bound.get("campaign_sha256") != observed["campaign"]
        or bound.get("quality_report_sha256") != observed["quality"]
    ):
        raise HTTPException(status_code=503, detail="task-validation artifact chain differs")
    assignment_rows = assignment.get("assignment_rows")
    if not isinstance(assignment_rows, list) or len(assignment_rows) != 180:
        raise HTTPException(status_code=503, detail="task-validation assignment is incomplete")
    assignment_by_candidate = {
        str(row["candidate_id"]): row for row in assignment_rows if isinstance(row, dict)
    }
    schedule_ids = {str(row["candidate_id"]) for row in campaign["candidate_schedule"]}
    if set(assignment_by_candidate) != schedule_ids:
        raise HTTPException(status_code=503, detail="campaign schedule and assignment differ")
    replay_candidate_ids = set(automated_replay.rights_required_ids) | set(
        automated_replay.contamination_required_ids
    )
    if (
        automated_replay.replay_sha256 != settings.task_validation_automated_replay_sha256
        or campaign.get("artifact_sha256") != TASK_VALIDATION_V6_CAMPAIGN_SHA256
        or not replay_candidate_ids.issubset(schedule_ids)
    ):
        raise HTTPException(
            status_code=503, detail="task-validation automated evidence binding differs"
        )
    return CampaignContext(
        campaign=campaign,
        assignment=assignment,
        quality_report=quality,
        readiness=readiness,
        assignment_by_candidate=assignment_by_candidate,
        automated_replay=automated_replay,
    )


def _season(session: Session) -> Season:
    season = session.scalar(
        select(Season).where(Season.slug == get_settings().task_validation_season_slug)
    )
    if season is None:
        raise HTTPException(status_code=503, detail="task-validation season is unavailable")
    return season


def _authenticate_reviewer(session: Session, authorization: str) -> ReviewerContext:
    token = authorization.removeprefix("Bearer ").strip()
    if not token.startswith("fbrv1_"):
        raise HTTPException(
            status_code=401,
            detail="a bounded verified reviewer credential is required",
        )
    try:
        credential = consume_reviewer_credential(
            session,
            token=token,
            required_scope="expert_review",
        )
        session.commit()
    except (ReviewerIdentityError, DBAPIError) as exc:
        session.rollback()
        raise HTTPException(
            status_code=401,
            detail="reviewer credential is invalid, expired, or exhausted",
        ) from exc
    reviewer = session.get(ExpertReviewer, credential.reviewer_id)
    binding = session.get(ReviewerIdentityBinding, credential.identity_binding_id)
    if (
        reviewer is None
        or binding is None
        or not reviewer.active
        or reviewer.cohort != "expert_independent"
        or binding.reviewer_id != reviewer.id
        or binding.season_id != credential.season_id
        or binding.assurance_level != "server_verified"
    ):
        raise HTTPException(
            status_code=403, detail="reviewer identity is not independently admitted"
        )
    return ReviewerContext(
        reviewer=reviewer,
        binding=binding,
        pseudonym=reviewer_rater_pseudonym(binding),
    )


def _event_document(event: TaskValidationCampaignEvent) -> dict[str, Any]:
    return {
        "schema_version": "flavourbench-task-validation-ledger-event-v1",
        "campaign_sha256": event.campaign_sha256,
        "sequence": event.sequence,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "reviewer_pseudonym": event.reviewer_pseudonym,
        "person_commitment_sha256": event.person_commitment_sha256,
        "reviewer_admission_receipt_sha256": event.reviewer_admission_receipt_sha256,
        "payload": event.payload_json,
        "previous_event_sha256": event.previous_event_sha256,
        "event_sha256": event.event_sha256,
    }


def _campaign_events(
    session: Session,
    campaign_sha256: str,
    *,
    analysis_eligible_only: bool = True,
) -> tuple[list[TaskValidationCampaignEvent], list[dict[str, Any]]]:
    rows = list(
        session.scalars(
            select(TaskValidationCampaignEvent)
            .where(TaskValidationCampaignEvent.campaign_sha256 == campaign_sha256)
            .order_by(TaskValidationCampaignEvent.sequence)
        ).all()
    )
    try:
        documents = verify_event_chain(
            [_event_document(row) for row in rows],
            campaign_sha256=campaign_sha256,
        )
    except TaskValidationCampaignError as exc:
        raise HTTPException(
            status_code=503, detail="task-validation event chain failed verification"
        ) from exc
    if analysis_eligible_only:
        eligible = [
            participant_record_analysis_eligible(
                session,
                reviewer_id=row.reviewer_id,
                season_id=row.season_id,
                identity_binding_id=row.identity_binding_id,
                recorded_at=row.created_at,
            )
            is not False
            for row in rows
        ]
        rows = [row for row, keep in zip(rows, eligible, strict=True) if keep]
        documents = [document for document, keep in zip(documents, eligible, strict=True) if keep]
    return rows, documents


def _active_admission(
    session: Session,
    *,
    reviewer: ReviewerContext,
    season_id: str,
    family: str,
    role: Literal["task_validator", "task_adjudicator"],
    required: bool = True,
) -> ReviewerFamilyAdmission | None:
    now = datetime.now(UTC)
    rows = list(
        session.scalars(
            select(ReviewerFamilyAdmission).where(
                ReviewerFamilyAdmission.season_id == season_id,
                ReviewerFamilyAdmission.reviewer_id == reviewer.reviewer.id,
                ReviewerFamilyAdmission.identity_binding_id == reviewer.binding.id,
                ReviewerFamilyAdmission.family == family,
                ReviewerFamilyAdmission.review_role == role,
                ReviewerFamilyAdmission.cohort == "expert_independent",
            )
        ).all()
    )
    active = [row for row in rows if _utc(row.valid_from) <= now <= _utc(row.valid_until)]
    if len(active) > 1:
        raise HTTPException(status_code=503, detail="reviewer admission is ambiguous")
    if not active and required:
        raise HTTPException(status_code=403, detail=f"active {role} admission is required")
    return active[0] if active else None


def _all_family_role_admissions(
    session: Session,
    *,
    reviewer: ReviewerContext,
    season_id: str,
    role: Literal["task_validator", "task_adjudicator"],
) -> dict[str, ReviewerFamilyAdmission]:
    """Return a family-complete admission set without revealing schedule strata.

    The campaign asks reviewers to assign the family while the scheduling family
    remains sealed. A family-scoped eligibility check would make queue access an
    oracle for that hidden label, so both validating roles must be qualified for
    every family before the server allocates any campaign work.
    """

    admissions = {
        family: admission
        for family in FAMILIES
        if (
            admission := _active_admission(
                session,
                reviewer=reviewer,
                season_id=season_id,
                family=family,
                role=role,
                required=False,
            )
        )
        is not None
    }
    if set(admissions) != set(FAMILIES):
        raise HTTPException(
            status_code=403,
            detail=(f"active {role} admissions across all four blinded task families are required"),
        )
    return admissions


def _has_audit_authorization(session: Session, binding_id: str, campaign: str) -> bool:
    return (
        session.scalar(
            select(TaskValidationAuditAuthorization.id).where(
                TaskValidationAuditAuthorization.campaign_sha256 == campaign,
                TaskValidationAuditAuthorization.identity_binding_id == binding_id,
            )
        )
        is not None
    )


def _candidate_state(candidate_id: str, documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        return derive_candidate_state(candidate_id=candidate_id, events=documents)
    except TaskValidationCampaignError as exc:
        raise HTTPException(
            status_code=503, detail="task-validation candidate state failed verification"
        ) from exc


def _all_candidate_states(
    context: CampaignContext, documents: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        str(row["candidate_id"]): _candidate_state(str(row["candidate_id"]), documents)
        for row in context.campaign["candidate_schedule"]
    }


def _validated_family_counts(states: Mapping[str, Mapping[str, Any]]) -> Counter[str]:
    return Counter(
        str(state.get("family"))
        for state in states.values()
        if state.get("status") in TERMINAL_VALIDATED and state.get("family") in FAMILIES
    )


def _active_candidate_ids(
    context: CampaignContext, states: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    family_counts = _validated_family_counts(states)
    if all(family_counts[family] >= 30 for family in FAMILIES):
        return set()
    active: set[str] = set()
    for family in FAMILIES:
        rows = [
            row
            for row in context.campaign["candidate_schedule"]
            if row["scheduling_family"] == family
        ]
        failed = sum(
            states[str(row["candidate_id"])].get("status") in TERMINAL_FAILED for row in rows
        )
        migrated = sum(
            states[str(row["candidate_id"])].get("status") in TERMINAL_VALIDATED
            and states[str(row["candidate_id"])].get("family") != family
            for row in rows
        )
        boundary = min(45, 30 + failed + migrated)
        active.update(str(row["candidate_id"]) for row in rows[:boundary])
    return active


def _submission_event_id(
    *, campaign: str, event_type: str, candidate_id: str | None, binding_id: str, key: str
) -> str:
    if len(key) < 8 or len(key) > 120:
        raise HTTPException(status_code=400, detail="a valid Idempotency-Key is required")
    return canonical_sha256(
        {
            "schema_version": "flavourbench-task-validation-idempotency-v1",
            "campaign_sha256": campaign,
            "event_type": event_type,
            "candidate_id": candidate_id,
            "identity_binding_id": binding_id,
            "idempotency_key_sha256": _sha256_text(key),
        }
    )


def _matching_existing_event(
    session: Session,
    *,
    campaign: str,
    event_id: str,
    reviewer: ReviewerContext,
    event_type: str,
    candidate_id: str | None,
    payload: Mapping[str, Any],
    receipt_sha256: str,
) -> TaskValidationCampaignEvent | None:
    existing = session.scalar(
        select(TaskValidationCampaignEvent).where(
            TaskValidationCampaignEvent.campaign_sha256 == campaign,
            TaskValidationCampaignEvent.event_id == event_id,
        )
    )
    if existing is None:
        return None
    if not (
        existing.reviewer_id == reviewer.reviewer.id
        and existing.identity_binding_id == reviewer.binding.id
        and existing.event_type == event_type
        and existing.candidate_id == candidate_id
        and existing.payload_json == dict(payload)
        and existing.reviewer_admission_receipt_sha256 == receipt_sha256
    ):
        raise HTTPException(
            status_code=409, detail="idempotency key was already used for different content"
        )
    return existing


def _lock_campaign_season(session: Session, season_id: str) -> Season:
    season = session.scalar(select(Season).where(Season.id == season_id).with_for_update())
    if season is None:
        raise HTTPException(status_code=503, detail="task-validation season is unavailable")
    return season


def _append_event_locked(
    session: Session,
    *,
    season: Season,
    context: CampaignContext,
    reviewer: ReviewerContext,
    event_id: str,
    event_type: str,
    candidate_id: str | None,
    payload: Mapping[str, Any],
    family_admission: ReviewerFamilyAdmission | None = None,
    audit_authorization: TaskValidationAuditAuthorization | None = None,
) -> tuple[TaskValidationCampaignEvent, bool]:
    campaign_sha256 = str(context.campaign["artifact_sha256"])
    receipt_sha256 = (
        family_admission.evidence_bundle_sha256
        if family_admission is not None
        else audit_authorization.authorization_sha256
        if audit_authorization is not None
        else ""
    )
    existing = _matching_existing_event(
        session,
        campaign=campaign_sha256,
        event_id=event_id,
        reviewer=reviewer,
        event_type=event_type,
        candidate_id=candidate_id,
        payload=payload,
        receipt_sha256=receipt_sha256,
    )
    if existing is not None:
        return existing, True
    if audit_authorization is not None:
        sealed_audit = session.scalar(
            select(TaskValidationCampaignEvent.id).where(
                TaskValidationCampaignEvent.campaign_sha256 == campaign_sha256,
                TaskValidationCampaignEvent.audit_authorization_id == audit_authorization.id,
                TaskValidationCampaignEvent.event_type == event_type,
            )
        )
        if sealed_audit is not None:
            raise HTTPException(
                status_code=409,
                detail="task-validation batch audit is already sealed",
            )
    rows, documents = _campaign_events(
        session,
        campaign_sha256,
        analysis_eligible_only=False,
    )
    candidate_capacity = CANDIDATE_EVENT_CAPACITIES.get(event_type)
    if (
        candidate_id is not None
        and candidate_capacity is not None
        and sum(row.candidate_id == candidate_id and row.event_type == event_type for row in rows)
        >= candidate_capacity
    ):
        raise HTTPException(
            status_code=409,
            detail="task-validation candidate event capacity is already sealed",
        )
    sequence = len(rows) + 1
    previous = documents[-1]["event_sha256"] if documents else ZERO_SHA256
    document = make_ledger_event(
        campaign_sha256=campaign_sha256,
        sequence=sequence,
        event_id=event_id,
        event_type=event_type,  # type: ignore[arg-type]
        reviewer_pseudonym=reviewer.pseudonym,
        person_commitment_sha256=reviewer.binding.person_commitment_sha256,
        reviewer_admission_receipt_sha256=receipt_sha256,
        payload=payload,
        previous_event_sha256=str(previous),
    )
    record = TaskValidationCampaignEvent(
        season_id=season.id,
        campaign_sha256=campaign_sha256,
        sequence=sequence,
        event_id=event_id,
        event_type=event_type,
        candidate_id=candidate_id,
        reviewer_id=reviewer.reviewer.id,
        identity_binding_id=reviewer.binding.id,
        family_admission_id=family_admission.id if family_admission else None,
        audit_authorization_id=(audit_authorization.id if audit_authorization else None),
        reviewer_pseudonym=reviewer.pseudonym,
        person_commitment_sha256=reviewer.binding.person_commitment_sha256,
        reviewer_admission_receipt_sha256=receipt_sha256,
        payload_json=dict(payload),
        previous_event_sha256=str(previous),
        event_sha256=str(document["event_sha256"]),
        created_at=datetime.now(UTC),
    )
    session.add(record)
    try:
        session.commit()
    except (IntegrityError, DBAPIError) as exc:
        session.rollback()
        replay = _matching_existing_event(
            session,
            campaign=campaign_sha256,
            event_id=event_id,
            reviewer=reviewer,
            event_type=event_type,
            candidate_id=candidate_id,
            payload=payload,
            receipt_sha256=receipt_sha256,
        )
        if replay is not None:
            return replay, True
        raise HTTPException(
            status_code=409, detail="task-validation event conflicted with another submission"
        ) from exc
    return record, False


def _validator_progress(
    rows: Sequence[TaskValidationCampaignEvent], reviewer: ReviewerContext
) -> dict[str, int]:
    own = [row for row in rows if row.identity_binding_id == reviewer.binding.id]
    return {
        "blindBallots": sum(row.event_type == "blind_ballot" for row in own),
        "criterionPackConfirmations": sum(
            row.event_type == "criterion_pack_confirmation" for row in own
        ),
    }


def _blind_candidate_view(context: CampaignContext, candidate_id: str) -> dict[str, Any]:
    row = context.assignment_by_candidate[candidate_id]
    return {
        "workType": "blind_ballot",
        "candidateId": candidate_id,
        "prompt": row["prompt"],
        "promptSha256": row["prompt_sha256"],
        "sourceMetadataVisible": False,
        "schedulingFamilyVisible": False,
        "otherBallotVisible": False,
        "modelOutputsVisible": False,
        "constructCellsByFamily": {key: list(value) for key, value in CONSTRUCT_CELLS.items()},
        "difficultyTiers": sorted(DIFFICULTY_TIERS),
    }


def _confirmation_candidate_view(
    context: CampaignContext,
    candidate_id: str,
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ballots = [
        event
        for event in documents
        if event["event_type"] == "blind_ballot"
        and event["payload"].get("candidate_id") == candidate_id
    ]
    pack = merged_criterion_pack(candidate_id=candidate_id, events=documents)
    first = ballots[0]["payload"]
    row = context.assignment_by_candidate[candidate_id]
    return {
        "workType": "criterion_pack_confirmation",
        "candidateId": candidate_id,
        "prompt": row["prompt"],
        "promptSha256": row["prompt_sha256"],
        "labels": {
            "family": first["family"],
            "constructCellId": first["construct_cell_id"],
            "difficultyTier": first["difficulty_tier"],
        },
        "criterionPack": pack,
        "criterionPackSha256": canonical_sha256(pack),
        "sourceMetadataVisible": False,
        "schedulingFamilyVisible": False,
        "otherReviewerIdentityVisible": False,
        "modelOutputsVisible": False,
    }


@router.get("/expert/task-validation/ballots/next")
def next_task_validation_ballot(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    context = _campaign_context()
    reviewer = _authenticate_reviewer(session, authorization)
    season = _season(session)
    if reviewer.binding.season_id != season.id:
        raise HTTPException(status_code=403, detail="reviewer credential belongs to another season")
    campaign = str(context.campaign["artifact_sha256"])
    if _has_audit_authorization(session, reviewer.binding.id, campaign):
        raise HTTPException(status_code=409, detail="campaign auditors cannot validate tasks")
    _all_family_role_admissions(
        session,
        reviewer=reviewer,
        season_id=season.id,
        role="task_validator",
    )
    rows, documents = _campaign_events(session, campaign)
    states = _all_candidate_states(context, documents)
    own_ballot_candidates = {
        str(row.candidate_id)
        for row in rows
        if row.identity_binding_id == reviewer.binding.id and row.event_type == "blind_ballot"
    }
    own_confirmation_candidates = {
        str(row.candidate_id)
        for row in rows
        if row.identity_binding_id == reviewer.binding.id
        and row.event_type == "criterion_pack_confirmation"
    }
    for schedule in context.campaign["candidate_schedule"]:
        candidate_id = str(schedule["candidate_id"])
        if (
            candidate_id in own_ballot_candidates
            and candidate_id not in own_confirmation_candidates
            and states[candidate_id]["status"] == "awaiting_criterion_pack_confirmations"
        ):
            return {
                "campaignSha256": campaign,
                "work": _confirmation_candidate_view(context, candidate_id, documents),
                "progress": _validator_progress(rows, reviewer),
                "claimBoundary": {"rankEligible": False, "modelCalls": 0},
            }
    active_ids = _active_candidate_ids(context, states)
    for schedule in context.campaign["candidate_schedule"]:
        candidate_id = str(schedule["candidate_id"])
        state = states[candidate_id]
        candidate_ballots = [
            row
            for row in rows
            if row.candidate_id == candidate_id and row.event_type == "blind_ballot"
        ]
        if (
            candidate_id not in active_ids
            or candidate_id in own_ballot_candidates
            or state["status"] != "awaiting_blind_ballots"
            or len(candidate_ballots) >= 2
        ):
            continue
        return {
            "campaignSha256": campaign,
            "work": _blind_candidate_view(context, candidate_id),
            "progress": _validator_progress(rows, reviewer),
            "claimBoundary": {"rankEligible": False, "modelCalls": 0},
        }
    return {
        "campaignSha256": campaign,
        "work": None,
        "progress": _validator_progress(rows, reviewer),
        "claimBoundary": {"rankEligible": False, "modelCalls": 0},
    }


def _schedule_row(context: CampaignContext, candidate_id: str) -> dict[str, Any]:
    row = next(
        (
            item
            for item in context.campaign["candidate_schedule"]
            if item["candidate_id"] == candidate_id
        ),
        None,
    )
    if not isinstance(row, dict):
        raise HTTPException(status_code=404, detail="task-validation candidate not found")
    return row


@router.post("/expert/task-validation/candidates/{candidate_id}/ballots")
def record_task_validation_ballot(
    candidate_id: str,
    request: BlindBallotCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    context = _campaign_context()
    reviewer = _authenticate_reviewer(session, authorization)
    season = _season(session)
    _schedule_row(context, candidate_id)
    campaign = str(context.campaign["artifact_sha256"])
    if reviewer.binding.season_id != season.id:
        raise HTTPException(status_code=403, detail="reviewer credential belongs to another season")
    if _has_audit_authorization(session, reviewer.binding.id, campaign):
        raise HTTPException(status_code=409, detail="campaign auditors cannot validate tasks")
    admissions = _all_family_role_admissions(
        session,
        reviewer=reviewer,
        season_id=season.id,
        role="task_validator",
    )
    admission = admissions[request.family] if request.family else admissions[FAMILIES[0]]
    payload = request.model_dump(mode="json")
    payload["candidate_id"] = candidate_id
    payload["checks"] = {
        key: value for key, value in payload["checks"].items() if key in CHECK_NAMES
    }
    payload["ballot_sha256"] = canonical_sha256(payload)
    event_id = _submission_event_id(
        campaign=campaign,
        event_type="blind_ballot",
        candidate_id=candidate_id,
        binding_id=reviewer.binding.id,
        key=idempotency_key,
    )
    _lock_campaign_season(session, season.id)
    existing = _matching_existing_event(
        session,
        campaign=campaign,
        event_id=event_id,
        reviewer=reviewer,
        event_type="blind_ballot",
        candidate_id=candidate_id,
        payload=payload,
        receipt_sha256=admission.evidence_bundle_sha256,
    )
    if existing is not None:
        return {
            "candidateId": candidate_id,
            "eventSha256": existing.event_sha256,
            "sealed": True,
            "idempotent": True,
            "rankEligible": False,
        }
    rows, documents = _campaign_events(session, campaign)
    states = _all_candidate_states(context, documents)
    if candidate_id not in _active_candidate_ids(context, states):
        raise HTTPException(status_code=409, detail="candidate is not active in the fixed schedule")
    candidate_ballots = [
        row for row in rows if row.candidate_id == candidate_id and row.event_type == "blind_ballot"
    ]
    if any(row.identity_binding_id == reviewer.binding.id for row in candidate_ballots):
        raise HTTPException(status_code=409, detail="reviewer already sealed this ballot")
    if len(candidate_ballots) >= 2:
        raise HTTPException(status_code=409, detail="both blind ballot slots are already sealed")
    record, idempotent = _append_event_locked(
        session,
        season=season,
        context=context,
        reviewer=reviewer,
        event_id=event_id,
        event_type="blind_ballot",
        candidate_id=candidate_id,
        payload=payload,
        family_admission=admission,
    )
    return {
        "candidateId": candidate_id,
        "eventSha256": record.event_sha256,
        "sealed": True,
        "idempotent": idempotent,
        "rankEligible": False,
    }


@router.post("/expert/task-validation/candidates/{candidate_id}/criterion-pack-confirmations")
def record_criterion_pack_confirmation(
    candidate_id: str,
    request: CriterionPackConfirmationCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    context = _campaign_context()
    reviewer = _authenticate_reviewer(session, authorization)
    season = _season(session)
    _schedule_row(context, candidate_id)
    campaign = str(context.campaign["artifact_sha256"])
    admissions = _all_family_role_admissions(
        session,
        reviewer=reviewer,
        season_id=season.id,
        role="task_validator",
    )
    payload = {
        "candidate_id": candidate_id,
        **request.model_dump(mode="json"),
    }
    payload["confirmation_sha256"] = canonical_sha256(payload)
    event_id = _submission_event_id(
        campaign=campaign,
        event_type="criterion_pack_confirmation",
        candidate_id=candidate_id,
        binding_id=reviewer.binding.id,
        key=idempotency_key,
    )
    _lock_campaign_season(session, season.id)
    rows, documents = _campaign_events(session, campaign)
    ballot = next(
        (
            row
            for row in rows
            if row.candidate_id == candidate_id
            and row.event_type == "blind_ballot"
            and row.identity_binding_id == reviewer.binding.id
        ),
        None,
    )
    if ballot is None:
        raise HTTPException(status_code=403, detail="only a source validator may confirm this pack")
    ballot_family = str(ballot.payload_json.get("family"))
    if ballot_family not in admissions:
        raise HTTPException(status_code=503, detail="sealed ballot family is invalid")
    admission = admissions[ballot_family]
    existing = _matching_existing_event(
        session,
        campaign=campaign,
        event_id=event_id,
        reviewer=reviewer,
        event_type="criterion_pack_confirmation",
        candidate_id=candidate_id,
        payload=payload,
        receipt_sha256=admission.evidence_bundle_sha256,
    )
    if existing is not None:
        return {
            "candidateId": candidate_id,
            "eventSha256": existing.event_sha256,
            "sealed": True,
            "idempotent": True,
            "rankEligible": False,
        }
    if any(
        row.candidate_id == candidate_id
        and row.event_type == "criterion_pack_confirmation"
        and row.identity_binding_id == reviewer.binding.id
        for row in rows
    ):
        raise HTTPException(status_code=409, detail="criterion pack response is already sealed")
    state = _candidate_state(candidate_id, documents)
    if state["status"] != "awaiting_criterion_pack_confirmations":
        raise HTTPException(status_code=409, detail="candidate is not awaiting pack confirmation")
    if request.criterion_pack_sha256 != state["criterion_pack_sha256"]:
        raise HTTPException(
            status_code=409, detail="criterion pack hash differs from the server pack"
        )
    record, idempotent = _append_event_locked(
        session,
        season=season,
        context=context,
        reviewer=reviewer,
        event_id=event_id,
        event_type="criterion_pack_confirmation",
        candidate_id=candidate_id,
        payload=payload,
        family_admission=admission,
    )
    return {
        "candidateId": candidate_id,
        "eventSha256": record.event_sha256,
        "sealed": True,
        "idempotent": idempotent,
        "rankEligible": False,
    }


def _adjudication_view(
    context: CampaignContext,
    candidate_id: str,
    documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = context.assignment_by_candidate[candidate_id]
    ballots = [
        event
        for event in documents
        if event["event_type"] == "blind_ballot"
        and event["payload"].get("candidate_id") == candidate_id
    ]
    return {
        "workType": "adjudication",
        "candidateId": candidate_id,
        "prompt": row["prompt"],
        "promptSha256": row["prompt_sha256"],
        "validatorBallots": [
            {
                "ballot": index,
                "eventSha256": ballot["event_sha256"],
                "payload": ballot["payload"],
            }
            for index, ballot in enumerate(ballots, start=1)
        ],
        "sourceMetadataVisible": False,
        "schedulingFamilyVisible": False,
        "validatorIdentitiesVisible": False,
        "modelOutputsVisible": False,
        "constructCellsByFamily": {key: list(value) for key, value in CONSTRUCT_CELLS.items()},
        "difficultyTiers": sorted(DIFFICULTY_TIERS),
    }


@router.get("/expert/task-validation/adjudications/next")
def next_task_validation_adjudication(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    context = _campaign_context()
    reviewer = _authenticate_reviewer(session, authorization)
    season = _season(session)
    campaign = str(context.campaign["artifact_sha256"])
    if _has_audit_authorization(session, reviewer.binding.id, campaign):
        raise HTTPException(status_code=409, detail="campaign auditors cannot adjudicate tasks")
    _all_family_role_admissions(
        session,
        reviewer=reviewer,
        season_id=season.id,
        role="task_adjudicator",
    )
    rows, documents = _campaign_events(session, campaign)
    for schedule in context.campaign["candidate_schedule"]:
        candidate_id = str(schedule["candidate_id"])
        if _candidate_state(candidate_id, documents)["status"] != "awaiting_adjudication":
            continue
        if any(
            row.candidate_id == candidate_id and row.identity_binding_id == reviewer.binding.id
            for row in rows
        ):
            continue
        return {
            "campaignSha256": campaign,
            "work": _adjudication_view(context, candidate_id, documents),
            "claimBoundary": {"rankEligible": False, "modelCalls": 0},
        }
    return {
        "campaignSha256": campaign,
        "work": None,
        "claimBoundary": {"rankEligible": False, "modelCalls": 0},
    }


@router.post("/expert/task-validation/candidates/{candidate_id}/adjudications")
def record_task_validation_adjudication(
    candidate_id: str,
    request: AdjudicationCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    context = _campaign_context()
    reviewer = _authenticate_reviewer(session, authorization)
    season = _season(session)
    _schedule_row(context, candidate_id)
    campaign = str(context.campaign["artifact_sha256"])
    if _has_audit_authorization(session, reviewer.binding.id, campaign):
        raise HTTPException(status_code=409, detail="campaign auditors cannot adjudicate tasks")
    admissions_by_family = _all_family_role_admissions(
        session,
        reviewer=reviewer,
        season_id=season.id,
        role="task_adjudicator",
    )
    _lock_campaign_season(session, season.id)
    event_id = _submission_event_id(
        campaign=campaign,
        event_type="adjudication",
        candidate_id=candidate_id,
        binding_id=reviewer.binding.id,
        key=idempotency_key,
    )
    rows, documents = _campaign_events(session, campaign)
    if any(
        row.candidate_id == candidate_id
        and row.event_type in {"blind_ballot", "criterion_pack_confirmation"}
        and row.identity_binding_id == reviewer.binding.id
        for row in rows
    ):
        raise HTTPException(
            status_code=409, detail="a source validator cannot adjudicate this task"
        )
    state = _candidate_state(candidate_id, documents)
    existing_event_id = session.scalar(
        select(TaskValidationCampaignEvent.id).where(
            TaskValidationCampaignEvent.campaign_sha256 == campaign,
            TaskValidationCampaignEvent.event_id == event_id,
        )
    )
    if state["status"] != "awaiting_adjudication" and existing_event_id is None:
        raise HTTPException(status_code=409, detail="candidate is not awaiting adjudication")
    if request.decision == "approve":
        assert request.family is not None
        admission = admissions_by_family[request.family]
        criterion_pack = {
            "success_criteria": request.success_criteria,
            "permitted_variations": request.permitted_variations,
            "disqualifying_errors": request.disqualifying_errors,
            "objective_checks": request.objective_checks,
        }
        criterion_pack_sha256: str | None = canonical_sha256(criterion_pack)
    else:
        admission = admissions_by_family[FAMILIES[0]]
        criterion_pack = None
        criterion_pack_sha256 = None
    payload = {
        "candidate_id": candidate_id,
        "decision": request.decision,
        "family": request.family,
        "construct_cell_id": request.construct_cell_id,
        "difficulty_tier": request.difficulty_tier,
        "criterion_pack": criterion_pack,
        "criterion_pack_sha256": criterion_pack_sha256,
        "note": request.note,
        "model_outputs_seen": False,
        "independent_attestation": True,
        "source_ballot_sha256s": sorted(
            event["event_sha256"]
            for event in documents
            if event["event_type"] == "blind_ballot"
            and event["payload"].get("candidate_id") == candidate_id
        ),
    }
    payload["adjudication_sha256"] = canonical_sha256(payload)
    record, idempotent = _append_event_locked(
        session,
        season=season,
        context=context,
        reviewer=reviewer,
        event_id=event_id,
        event_type="adjudication",
        candidate_id=candidate_id,
        payload=payload,
        family_admission=admission,
    )
    return {
        "candidateId": candidate_id,
        "decision": request.decision,
        "eventSha256": record.event_sha256,
        "sealed": True,
        "idempotent": idempotent,
        "rankEligible": False,
    }


def _assert_authorization_replay_binding(
    authorization: TaskValidationAuditAuthorization,
    context: CampaignContext,
) -> None:
    expected_plan = rights_audit_plan()
    if not (
        authorization.campaign_sha256
        == context.campaign["artifact_sha256"]
        == TASK_VALIDATION_V6_CAMPAIGN_SHA256
        and authorization.audit_kind == "rights"
        and authorization.automated_evidence_sha256
        == context.automated_replay.replay_sha256
        == TASK_VALIDATION_V1_REPLAY_SHA256
        and authorization.audit_plan_json == expected_plan
        and authorization.audit_plan_sha256 == TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256
    ):
        raise HTTPException(
            status_code=503,
            detail="task-validation audit authorization replay binding is invalid",
        )


def _assert_campaign_replay_bindings(
    session: Session,
    context: CampaignContext,
    event_rows: Sequence[TaskValidationCampaignEvent] | None = None,
) -> dict[str, TaskValidationAuditAuthorization]:
    campaign = str(context.campaign["artifact_sha256"])
    authorizations = list(
        session.scalars(
            select(TaskValidationAuditAuthorization).where(
                TaskValidationAuditAuthorization.campaign_sha256 == campaign
            )
        ).all()
    )
    by_id: dict[str, TaskValidationAuditAuthorization] = {}
    for authorization in authorizations:
        _assert_authorization_replay_binding(authorization, context)
        by_id[authorization.id] = authorization
    rows = (
        event_rows
        if event_rows is not None
        else list(
            session.scalars(
                select(TaskValidationCampaignEvent).where(
                    TaskValidationCampaignEvent.campaign_sha256 == campaign,
                    TaskValidationCampaignEvent.event_type.in_(
                        {"rights_batch_audit", "contamination_batch_audit"}
                    ),
                )
            ).all()
        )
    )
    for row in rows:
        if row.event_type not in {"rights_batch_audit", "contamination_batch_audit"}:
            continue
        authorization = by_id.get(str(row.audit_authorization_id))
        payload = row.payload_json
        if (
            authorization is None
            or row.event_type != "rights_batch_audit"
            or payload.get("audit_kind") != "rights"
            or payload.get("audit_plan_sha256") != TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256
            or payload.get("automated_evidence_sha256") != TASK_VALIDATION_V1_REPLAY_SHA256
            or payload.get("automated_evidence_verified") is not True
            or payload.get("rights_snapshot_integrity_verified") is not True
            or payload.get("local_prompt_risk_replay_verified") is not True
            or payload.get("contamination_campaign_coverage_verified") is not False
            or payload.get("reviewed_candidate_ids") != list(TASK_VALIDATION_RIGHTS_REQUIRED_IDS)
        ):
            raise HTTPException(
                status_code=503,
                detail="task-validation audit event replay binding is invalid",
            )
    return by_id


@router.post(
    "/admin/task-validation/auditors",
    dependencies=[Depends(require_admin_token)],
)
def admin_authorize_task_validation_auditor(
    request: AuditAuthorizationCreate,
    session: Db,
) -> dict[str, Any]:
    context = _campaign_context()
    season = _season(session)
    campaign = str(context.campaign["artifact_sha256"])
    _assert_campaign_replay_bindings(session, context)
    if request.audit_kind == "contamination":
        raise HTTPException(
            status_code=409,
            detail=(
                "formal campaign contamination coverage is unavailable until a frozen "
                "exact/fuzzy/ngram/semantic/web successor verifies"
            ),
        )
    reviewer = session.get(ExpertReviewer, request.reviewer_id)
    binding = session.scalar(
        select(ReviewerIdentityBinding).where(
            ReviewerIdentityBinding.season_id == season.id,
            ReviewerIdentityBinding.reviewer_id == request.reviewer_id,
            ReviewerIdentityBinding.assurance_level == "server_verified",
        )
    )
    if (
        reviewer is None
        or binding is None
        or not reviewer.active
        or reviewer.cohort != "expert_independent"
    ):
        raise HTTPException(
            status_code=409, detail="independent verified auditor identity required"
        )
    try:
        lifecycle = require_active_participant_authority(
            session,
            reviewer_id=reviewer.id,
            season_id=season.id,
            identity_binding_id=binding.id,
        )
    except ParticipantLifecycleError as exc:
        raise HTTPException(
            status_code=409,
            detail="task-validation auditor lacks current participant authority",
        ) from exc
    if lifecycle is None and get_settings().environment == "production":
        raise HTTPException(
            status_code=409,
            detail="production task-validation audit requires participant-owned consent",
        )
    _lock_campaign_season(session, season.id)
    if (
        session.scalar(
            select(TaskValidationCampaignEvent.id).where(
                TaskValidationCampaignEvent.campaign_sha256 == campaign,
                TaskValidationCampaignEvent.identity_binding_id == binding.id,
            )
        )
        is not None
    ):
        raise HTTPException(
            status_code=409, detail="task validators and adjudicators cannot audit the campaign"
        )
    plan = rights_audit_plan()
    required_ids = plan["required_candidate_ids"]
    authorization = TaskValidationAuditAuthorization(
        season_id=season.id,
        campaign_sha256=campaign,
        reviewer_id=reviewer.id,
        identity_binding_id=binding.id,
        audit_kind=request.audit_kind,
        cohort="expert_independent",
        qualification_evidence_sha256=request.qualification_evidence_sha256,
        conflict_evidence_sha256=request.conflict_evidence_sha256,
        automated_evidence_sha256=context.automated_replay.replay_sha256,
        audit_plan_json=plan,
        audit_plan_sha256=TASK_VALIDATION_RIGHTS_AUDIT_PLAN_SHA256,
        decision_reference_sha256=request.decision_reference_sha256,
        authorization_sha256="0" * 64,
        created_at=datetime.now(UTC),
    )
    authorization.authorization_sha256 = _task_validation_audit_authorization_sha256(authorization)
    session.add(authorization)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="audit kind or auditor person is already authorized for this campaign",
        ) from exc
    return {
        "auditAuthorizationId": authorization.id,
        "auditKind": authorization.audit_kind,
        "authorizationSha256": authorization.authorization_sha256,
        "auditPlanSha256": authorization.audit_plan_sha256,
        "requiredCandidateCount": len(required_ids),
        "automatedEvidenceSha256": context.automated_replay.replay_sha256,
        "automatedEvidenceVerified": True,
        "rightsSnapshotIntegrityVerified": True,
        "localPromptRiskReplayVerified": True,
        "contaminationAutomatedEvidenceVerified": False,
        "contaminationCampaignCoverageVerified": False,
        "authorizationTrustBoundary": AUDIT_AUTHORIZATION_TRUST_BOUNDARY,
        "publicIdentity": "stable_pseudonym_only",
        "rankEligible": False,
    }


def _audit_source_view(context: CampaignContext, candidate_id: str) -> dict[str, Any]:
    row = context.assignment_by_candidate[candidate_id]
    source = row["source_metadata_visible_after_blind_decision"]
    return {
        "candidateId": candidate_id,
        "prompt": row["prompt"],
        "promptSha256": row["prompt_sha256"],
        "sourceUrl": source.get("url"),
        "sourceQuestionId": source.get("question_id"),
        "sourceRevisionGuid": source.get("revision_guid"),
        "sourceLicense": source.get("content_license"),
        "revisionLicense": source.get("revision_content_license"),
        "attribution": source.get("attribution"),
        "transformationLog": source.get("transformation_log"),
        "sourceAnswerRequested": source.get("source_answer_payload_requested"),
        "modelOutputsVisible": False,
    }


@router.get("/expert/task-validation/audits/next")
def next_task_validation_audit(
    session: Db,
    authorization: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    context = _campaign_context()
    reviewer = _authenticate_reviewer(session, authorization)
    campaign = str(context.campaign["artifact_sha256"])
    _assert_campaign_replay_bindings(session, context)
    authorizations = list(
        session.scalars(
            select(TaskValidationAuditAuthorization).where(
                TaskValidationAuditAuthorization.campaign_sha256 == campaign,
                TaskValidationAuditAuthorization.identity_binding_id == reviewer.binding.id,
            )
        ).all()
    )
    if len(authorizations) != 1:
        raise HTTPException(status_code=403, detail="one campaign audit authorization is required")
    authorization_row = authorizations[0]
    _assert_authorization_replay_binding(authorization_row, context)
    if (
        session.scalar(
            select(TaskValidationCampaignEvent.id).where(
                TaskValidationCampaignEvent.campaign_sha256 == campaign,
                TaskValidationCampaignEvent.identity_binding_id == reviewer.binding.id,
                TaskValidationCampaignEvent.event_type.in_(
                    {
                        "blind_ballot",
                        "criterion_pack_confirmation",
                        "adjudication",
                    }
                ),
            )
        )
        is not None
    ):
        raise HTTPException(status_code=409, detail="task reviewers cannot audit the campaign")
    completed = session.scalar(
        select(TaskValidationCampaignEvent).where(
            TaskValidationCampaignEvent.campaign_sha256 == campaign,
            TaskValidationCampaignEvent.audit_authorization_id == authorization_row.id,
        )
    )
    if completed is not None:
        return {
            "campaignSha256": campaign,
            "work": None,
            "complete": True,
            "rankEligible": False,
        }
    required = authorization_row.audit_plan_json["required_candidate_ids"]
    return {
        "campaignSha256": campaign,
        "work": {
            "workType": "batch_audit",
            "auditKind": authorization_row.audit_kind,
            "auditPlanSha256": authorization_row.audit_plan_sha256,
            "automatedEvidenceSha256": authorization_row.automated_evidence_sha256,
            "automatedEvidenceVerified": True,
            "rightsSnapshotIntegrityVerified": True,
            "localPromptRiskReplayVerified": True,
            "contaminationAutomatedEvidenceVerified": False,
            "contaminationCampaignCoverageVerified": False,
            "sampleSeedCommitmentSha256": authorization_row.audit_plan_json[
                "sample_seed_commitment_sha256"
            ],
            "sampleCandidateIds": authorization_row.audit_plan_json["sample_candidate_ids"],
            "anomalyOrHitCandidateIds": authorization_row.audit_plan_json[
                "anomaly_or_hit_candidate_ids"
            ],
            "records": [_audit_source_view(context, item) for item in required],
            "modelOutputsVisible": False,
        },
        "complete": False,
        "rankEligible": False,
    }


@router.post("/expert/task-validation/audits")
def record_task_validation_audit(
    request: BatchAuditCreate,
    session: Db,
    authorization: Annotated[str, Header()] = "",
    idempotency_key: Annotated[str, Header()] = "",
) -> dict[str, Any]:
    context = _campaign_context()
    reviewer = _authenticate_reviewer(session, authorization)
    season = _season(session)
    campaign = str(context.campaign["artifact_sha256"])
    _assert_campaign_replay_bindings(session, context)
    authorization_row = session.scalar(
        select(TaskValidationAuditAuthorization).where(
            TaskValidationAuditAuthorization.campaign_sha256 == campaign,
            TaskValidationAuditAuthorization.identity_binding_id == reviewer.binding.id,
            TaskValidationAuditAuthorization.audit_kind == request.audit_kind,
        )
    )
    if authorization_row is None:
        raise HTTPException(status_code=403, detail="campaign audit authorization is required")
    _assert_authorization_replay_binding(authorization_row, context)
    if set(request.reviewed_candidate_ids) != set(
        authorization_row.audit_plan_json["required_candidate_ids"]
    ):
        raise HTTPException(status_code=409, detail="audit did not cover the frozen required set")
    if any(
        finding.candidate_id not in request.reviewed_candidate_ids for finding in request.findings
    ):
        raise HTTPException(status_code=409, detail="audit finding points outside the reviewed set")
    payload = {
        "audit_kind": request.audit_kind,
        "decision": request.decision,
        "audit_plan_sha256": authorization_row.audit_plan_sha256,
        "automated_evidence_sha256": authorization_row.automated_evidence_sha256,
        "automated_evidence_verified": True,
        "rights_snapshot_integrity_verified": True,
        "local_prompt_risk_replay_verified": True,
        "contamination_campaign_coverage_verified": False,
        "reviewed_candidate_ids": sorted(request.reviewed_candidate_ids),
        "findings": [finding.model_dump(mode="json") for finding in request.findings],
        "unresolved_material_findings": request.unresolved_material_findings,
        "complete_coverage_evidence_checked": True,
        "model_outputs_seen": False,
        "public_source_contamination_limited_acknowledged": True,
        "no_contamination_free_claim": True,
        "note": request.note,
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    event_type = f"{request.audit_kind}_batch_audit"
    event_id = _submission_event_id(
        campaign=campaign,
        event_type=event_type,
        candidate_id=None,
        binding_id=reviewer.binding.id,
        key=idempotency_key,
    )
    _lock_campaign_season(session, season.id)
    if (
        session.scalar(
            select(TaskValidationCampaignEvent.id).where(
                TaskValidationCampaignEvent.campaign_sha256 == campaign,
                TaskValidationCampaignEvent.identity_binding_id == reviewer.binding.id,
                TaskValidationCampaignEvent.event_type.in_(
                    {
                        "blind_ballot",
                        "criterion_pack_confirmation",
                        "adjudication",
                    }
                ),
            )
        )
        is not None
    ):
        raise HTTPException(status_code=409, detail="task reviewers cannot audit the campaign")
    record, idempotent = _append_event_locked(
        session,
        season=season,
        context=context,
        reviewer=reviewer,
        event_id=event_id,
        event_type=event_type,
        candidate_id=None,
        payload=payload,
        audit_authorization=authorization_row,
    )
    return {
        "auditKind": request.audit_kind,
        "decision": request.decision,
        "eventSha256": record.event_sha256,
        "sealed": True,
        "idempotent": idempotent,
        "rankEligible": False,
    }


def _status_payload(session: Session, context: CampaignContext) -> dict[str, Any]:
    campaign = str(context.campaign["artifact_sha256"])
    rows, documents = _campaign_events(session, campaign)
    states = _all_candidate_states(context, documents)
    status_counts = Counter(str(state["status"]) for state in states.values())
    family_counts = _validated_family_counts(states)
    audit_rows = [
        row for row in rows if row.event_type in {"rights_batch_audit", "contamination_batch_audit"}
    ]
    _assert_campaign_replay_bindings(session, context, audit_rows)
    audits = {
        row.event_type.removesuffix("_batch_audit"): {
            "humanDecision": row.payload_json.get("decision"),
            "eventSha256": row.event_sha256,
            "unresolvedMaterialFindings": row.payload_json.get("unresolved_material_findings"),
            "automatedEvidenceVerified": True,
            "rightsSnapshotIntegrityVerified": True,
            "localPromptRiskReplayVerified": True,
            "contaminationAutomatedEvidenceVerified": False,
            "contaminationCampaignCoverageVerified": False,
        }
        for row in audit_rows
    }
    payload = {
        "schemaVersion": "flavourbench-task-validation-campaign-status-v1",
        "campaignSha256": campaign,
        "qualityReportSha256": context.quality_report["artifact_sha256"],
        "readinessDecisionSha256": context.readiness["artifact_sha256"],
        "candidateCount": 180,
        "targetTaskCount": 120,
        "blindBallots": sum(row.event_type == "blind_ballot" for row in rows),
        "criterionPackConfirmations": sum(
            row.event_type == "criterion_pack_confirmation" for row in rows
        ),
        "adjudications": sum(row.event_type == "adjudication" for row in rows),
        "batchAudits": len(audit_rows),
        "distinctPublicReviewerPseudonyms": len({row.reviewer_pseudonym for row in rows}),
        "validatedTasks": sum(status_counts[item] for item in TERMINAL_VALIDATED),
        "validatedByFamily": {family: family_counts[family] for family in FAMILIES},
        "statusCounts": dict(sorted(status_counts.items())),
        "audits": audits,
        "eventCount": len(rows),
        "eventChainHeadSha256": rows[-1].event_sha256 if rows else ZERO_SHA256,
        "releaseGate": {
            "familyQuotasMet": all(family_counts[family] >= 30 for family in FAMILIES),
            "rightsHumanAuditPassed": (audits.get("rights", {}).get("humanDecision") == "pass"),
            "contaminationHumanAuditPassed": (
                audits.get("contamination", {}).get("humanDecision") == "pass"
            ),
            "rightsAutomatedEvidenceVerified": True,
            "localPromptRiskReplayVerified": True,
            "contaminationAutomatedEvidenceVerified": False,
            "contaminationCampaignCoverageVerified": False,
            "rightsAuditPassed": (audits.get("rights", {}).get("humanDecision") == "pass"),
            "contaminationAuditPassed": False,
            "taskBankImportAuthorized": False,
        },
        "claimBoundary": {
            "humanWrittenPublicSourceTasks": True,
            "syntheticTasks": 0,
            "modelCalls": 0,
            "epicureCalls": 0,
            "publicSourceContaminationLimited": True,
            "contaminationFree": False,
            "official": False,
            "rankEligible": False,
        },
    }
    return {**payload, "statusSha256": canonical_sha256(payload)}


@router.get("/task-validation/status")
def task_validation_status(session: Db) -> dict[str, Any]:
    return _status_payload(session, _campaign_context())


@router.get(
    "/admin/task-validation/export",
    dependencies=[Depends(require_admin_token)],
)
def admin_task_validation_export(session: Db) -> dict[str, Any]:
    context = _campaign_context()
    campaign = str(context.campaign["artifact_sha256"])
    rows, documents = _campaign_events(session, campaign)
    _assert_campaign_replay_bindings(session, context, rows)
    states = _all_candidate_states(context, documents)
    records: list[dict[str, Any]] = []
    for schedule in context.campaign["candidate_schedule"]:
        candidate_id = str(schedule["candidate_id"])
        state = states[candidate_id]
        if state["status"] not in TERMINAL_VALIDATED:
            continue
        source = context.assignment_by_candidate[candidate_id][
            "source_metadata_visible_after_blind_decision"
        ]
        if state["status"] == "validated_consensus":
            pack = merged_criterion_pack(candidate_id=candidate_id, events=documents)
        else:
            adjudication = next(
                event
                for event in documents
                if event["event_type"] == "adjudication"
                and event["payload"].get("candidate_id") == candidate_id
            )
            pack = adjudication["payload"]["criterion_pack"]
        records.append(
            {
                "candidateId": candidate_id,
                "prompt": context.assignment_by_candidate[candidate_id]["prompt"],
                "promptSha256": context.assignment_by_candidate[candidate_id]["prompt_sha256"],
                "source": source,
                "validationState": state,
                "criterionPack": pack,
            }
        )
    payload = {
        "schemaVersion": "flavourbench-task-validation-private-export-preview-v1",
        "campaignSha256": campaign,
        "events": [public_event_view(event) for event in documents],
        "validatedRecords": records,
        "privatePersonCommitmentsIncluded": False,
        "privateAdmissionReceiptsIncluded": False,
        "stablePublicPseudonymsIncluded": True,
        "automatedEvidence": {
            "replaySha256": context.automated_replay.replay_sha256,
            "replayPhysicalSha256": context.automated_replay.replay_physical_sha256,
            "rightsAutomatedEvidenceVerified": True,
            "localPromptRiskReplayVerified": True,
            "contaminationAutomatedEvidenceVerified": False,
            "contaminationCampaignCoverageVerified": False,
            "formalContaminationMethodsRequired": list(
                TASK_VALIDATION_FORMAL_CONTAMINATION_METHODS
            ),
        },
        "taskBankImportAuthorized": False,
        "publicReleaseAuthorized": False,
        "official": False,
        "rankEligible": False,
        "contaminationFree": False,
    }
    return {**payload, "exportSha256": canonical_sha256(payload)}
