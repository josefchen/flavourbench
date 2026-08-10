from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictInt,
    computed_field,
    field_validator,
    model_validator,
)

from .construct_blueprint import (
    BLUEPRINT_SHA256,
    CONSTRUCT_CELLS,
    DIFFICULTY_TIERS,
    ConstructBlueprintError,
    validate_confirmatory_bank,
    validate_task_binding,
)
from .endpoint_contract import DECODING_PARAMETERS, REQUIRED_ENDPOINT_PARAMETERS
from .expert_review import (
    PROTOCOL_SHA256,
    PROTOCOL_VERSION,
    RESPONSE_FAILURE_TAGS,
    SPECIALIST_DOMAINS,
    TASK_FAMILIES,
    TASK_ISSUE_TAGS,
    WORKLOAD_TARGET,
    validate_acknowledgements,
)
from .season_design import (
    CONFIRMATORY_TASK_COUNT,
    CONFIRMATORY_TASKS_PER_FAMILY,
    SEASON_MODEL_COUNT,
    SEASON_TASK_SPLIT_COUNTS,
    SEASON_TASK_SPLIT_COUNTS_PER_FAMILY,
)
from .task_contributor_protocol import (
    PROTOCOL_SHA256 as TASK_CONTRIBUTOR_PROTOCOL_SHA256,
)
from .task_contributor_protocol import (
    PROTOCOL_VERSION as TASK_CONTRIBUTOR_PROTOCOL_VERSION,
)
from .task_evidence import (
    TaskContaminationAuditArtifact,
    TaskValidatorContractArtifact,
)


class TaskFamily(StrEnum):
    substitution = "substitution"
    composition = "composition"
    cookability = "cookability"
    evidence = "evidence"


class BattleTrack(StrEnum):
    model_arena = "model_arena"
    epicure_uplift = "epicure_uplift"


class ExecutionBackend(StrEnum):
    openrouter = "openrouter"
    bedrock = "bedrock"
    kimi_direct = "kimi_direct"
    qwencloud_direct = "qwencloud_direct"
    mock = "mock"


class SeasonTaskSplit(StrEnum):
    scored = "scored"
    development = "development"
    private_reserve = "private_reserve"


class VoteChoice(StrEnum):
    left = "left"
    right = "right"
    tie = "tie"
    both_bad = "both_bad"


class SeasonProvisionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    slug: str = Field(min_length=3, max_length=80, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=3, max_length=160)


class IndependentTaskReviewCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reviewer_id: str = Field(min_length=3, max_length=160, alias="reviewerId")
    blind_review_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="blindReviewEventSha256",
    )
    reconciliation_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="reconciliationEventSha256",
    )
    decision: str = Field(pattern=r"^approve$")
    independent_of_author: bool = Field(alias="independentOfAuthor")


class TaskCandidateAdjudicationBinding(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    adjudicator_reviewer_id: str = Field(
        min_length=3,
        max_length=160,
        alias="adjudicatorReviewerId",
    )
    adjudication_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="adjudicationEventSha256",
    )
    criterion_pack_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="criterionPackSha256",
    )
    decision: str = Field(pattern=r"^approve$")
    independent_of_author_and_reviewers: bool = Field(alias="independentOfAuthorAndReviewers")


class TaskEvidenceReviewBinding(BaseModel):
    """Human-review receipt over one reproducible task-evidence artifact."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    reviewer_id: str = Field(min_length=3, max_length=160, alias="reviewerId")
    evidence_type: Literal["validator_contract", "contamination_audit"] = Field(
        alias="evidenceType"
    )
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="artifactSha256")
    verification_receipt_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="verificationReceiptSha256",
    )
    review_event_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="reviewEventSha256",
    )
    decision: Literal["approve"]
    independent_of_task_roles: Literal[True] = Field(alias="independentOfTaskRoles")


class ConfirmatoryTaskCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    public_id: str = Field(min_length=3, max_length=80, alias="publicId")
    family: TaskFamily
    split: SeasonTaskSplit
    prompt: str = Field(min_length=10, max_length=4000)
    revision: int = Field(default=1, ge=1, le=100)
    construct_blueprint_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="constructBlueprintSha256",
    )
    construct_cell_id: str = Field(
        min_length=3,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_]*$",
        alias="constructCellId",
    )
    difficulty_tier: str = Field(
        pattern=r"^(foundation|integrative|stress)$",
        alias="difficultyTier",
    )
    human_author_id: str = Field(min_length=3, max_length=160, alias="humanAuthorId")
    source_candidate_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        alias="sourceCandidateId",
    )
    candidate_record_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="candidateRecordSha256",
    )
    independent_reviews: list[IndependentTaskReviewCreate] = Field(
        min_length=2, max_length=2, alias="independentReviews"
    )
    adjudication: TaskCandidateAdjudicationBinding
    validator_contract: TaskValidatorContractArtifact = Field(alias="validatorContract")
    validator_contract_review: TaskEvidenceReviewBinding = Field(alias="validatorContractReview")
    review_history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="reviewHistorySha256")
    task_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="taskRecordSha256")
    contamination_audit: TaskContaminationAuditArtifact = Field(alias="contaminationAudit")
    contamination_audit_review: TaskEvidenceReviewBinding = Field(alias="contaminationAuditReview")
    task_evidence_root_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="taskEvidenceRootSha256"
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("prompt must not have leading or trailing whitespace")
        if any(ord(character) < 32 and character not in "\n\t" for character in value):
            raise ValueError("prompt contains unsupported control characters")
        return value

    @model_validator(mode="after")
    def validate_independent_reviews(self) -> ConfirmatoryTaskCreate:
        reviewers = [review.reviewer_id for review in self.independent_reviews]
        if len(set(reviewers)) < 2:
            raise ValueError("confirmatory tasks require two distinct reviewers")
        if self.human_author_id in reviewers:
            raise ValueError("the task author cannot approve their own task")
        if any(not review.independent_of_author for review in self.independent_reviews):
            raise ValueError("every approving reviewer must be independent of the author")
        if (
            not self.adjudication.independent_of_author_and_reviewers
            or self.adjudication.adjudicator_reviewer_id == self.human_author_id
            or self.adjudication.adjudicator_reviewer_id in reviewers
        ):
            raise ValueError(
                "confirmatory tasks require a distinct independent third-person adjudicator"
            )
        role_ids = {
            self.human_author_id,
            self.adjudication.adjudicator_reviewer_id,
            *reviewers,
        }
        evidence_reviews = (
            self.validator_contract_review,
            self.contamination_audit_review,
        )
        if len({review.reviewer_id for review in evidence_reviews}) != 2:
            raise ValueError("validator and contamination evidence require distinct reviewers")
        if any(review.reviewer_id in role_ids for review in evidence_reviews):
            raise ValueError(
                "task-evidence reviewers must be independent of authorship, source review, "
                "and adjudication"
            )
        if (
            self.validator_contract_review.evidence_type != "validator_contract"
            or self.validator_contract_review.reviewer_id
            != self.validator_contract.verifier_reviewer_id
            or self.validator_contract_review.artifact_sha256
            != self.validator_contract.artifact_sha256
        ):
            raise ValueError("validatorContractReview does not bind the validator contract")
        if (
            self.contamination_audit_review.evidence_type != "contamination_audit"
            or self.contamination_audit_review.reviewer_id
            != self.contamination_audit.auditor_reviewer_id
            or self.contamination_audit_review.artifact_sha256
            != self.contamination_audit.artifact_sha256
        ):
            raise ValueError("contaminationAuditReview does not bind the contamination audit")
        try:
            validate_task_binding(
                family=self.family.value,
                construct_blueprint_sha256=self.construct_blueprint_sha256,
                construct_cell_id=self.construct_cell_id,
                difficulty_tier=self.difficulty_tier,
            )
        except ConstructBlueprintError as exc:
            raise ValueError(str(exc)) from exc
        return self


class TaskValidatorContractReviewCreate(BaseModel):
    """Sealed human inspection of an executable or explicitly human-only contract."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    validator_contract: TaskValidatorContractArtifact = Field(alias="validatorContract")
    decision: Literal["approve"]
    task_binding_checked: Literal[True] = Field(alias="taskBindingChecked")
    rules_and_fixtures_inspected: Literal[True] = Field(alias="rulesAndFixturesInspected")
    verification_receipt_reproduced: Literal[True] = Field(alias="verificationReceiptReproduced")
    model_outputs_not_consulted: Literal[True] = Field(alias="modelOutputsNotConsulted")
    independent_of_task_roles: Literal[True] = Field(alias="independentOfTaskRoles")
    note: str = Field(min_length=40, max_length=2000)


class TaskContaminationAuditReviewCreate(BaseModel):
    """Sealed human inspection of a replayed task-contamination audit."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    contamination_audit: TaskContaminationAuditArtifact = Field(alias="contaminationAudit")
    decision: Literal["approve"]
    task_binding_checked: Literal[True] = Field(alias="taskBindingChecked")
    replay_receipt_reproduced: Literal[True] = Field(alias="replayReceiptReproduced")
    hit_dispositions_inspected: Literal[True] = Field(alias="hitDispositionsInspected")
    model_outputs_not_consulted: Literal[True] = Field(alias="modelOutputsNotConsulted")
    independent_of_task_roles: Literal[True] = Field(alias="independentOfTaskRoles")
    note: str = Field(min_length=40, max_length=2000)


class TaskChallengeCreate(BaseModel):
    """Public, content-addressed report of a possible benchmark-item defect."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    issue_type: Literal[
        "invalid_task",
        "ambiguous_prompt",
        "criterion_pack_error",
        "validator_error",
        "contamination_or_leakage",
        "rights_or_privacy",
        "other",
    ] = Field(alias="issueType")
    description: str = Field(min_length=40, max_length=4000)
    evidence_reference: str | None = Field(
        default=None,
        min_length=8,
        max_length=1000,
        alias="evidenceReference",
    )
    client_nonce: str = Field(min_length=8, max_length=160, alias="clientNonce")
    no_personal_data_attestation: Literal[True] = Field(alias="noPersonalDataAttestation")


class TaskChallengeAdjudicationCreate(BaseModel):
    """Independent two-person disposition of a sealed task challenge."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    decision: Literal["confirmed", "dismissed", "deferred"]
    adjudicator_reviewer_ids: list[str] = Field(
        min_length=2,
        max_length=3,
        alias="adjudicatorReviewerIds",
    )
    rationale: str = Field(min_length=80, max_length=5000)
    evidence_bundle_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="evidenceBundleSha256",
    )
    correction_reference: str | None = Field(
        default=None,
        min_length=8,
        max_length=1000,
        alias="correctionReference",
    )
    independent_of_original_task_roles: Literal[True] = Field(
        alias="independentOfOriginalTaskRoles"
    )
    model_outputs_consulted_only_if_material: Literal[True] = Field(
        alias="modelOutputsConsultedOnlyIfMaterial"
    )

    @field_validator("adjudicator_reviewer_ids")
    @classmethod
    def distinct_challenge_adjudicators(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("challenge adjudicators must be distinct")
        return value

    @model_validator(mode="after")
    def require_correction_reference_for_confirmed_error(
        self,
    ) -> TaskChallengeAdjudicationCreate:
        if self.decision == "confirmed" and self.correction_reference is None:
            raise ValueError("confirmed task errors require a correctionReference")
        return self


class ConfirmatoryTaskBankCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tasks: list[ConfirmatoryTaskCreate] = Field(
        min_length=CONFIRMATORY_TASK_COUNT,
        max_length=CONFIRMATORY_TASK_COUNT,
    )
    bank_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="bankManifestSha256")
    validator_calibration_artifact_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="validatorCalibrationArtifactSha256",
    )
    contamination_calibration_artifact_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="contaminationCalibrationArtifactSha256",
    )
    import_reference: str = Field(min_length=3, max_length=240, alias="importReference")

    @model_validator(mode="after")
    def validate_balanced_bank(self) -> ConfirmatoryTaskBankCreate:
        public_ids = [task.public_id for task in self.tasks]
        if len(set(public_ids)) != len(public_ids):
            raise ValueError("confirmatory task publicIds must be unique")
        candidate_ids = [task.source_candidate_id for task in self.tasks]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("each confirmatory task requires a distinct source candidate")
        validator_artifacts = [task.validator_contract.artifact_sha256 for task in self.tasks]
        contamination_artifacts = [task.contamination_audit.artifact_sha256 for task in self.tasks]
        evidence_review_events = [
            review.review_event_sha256
            for task in self.tasks
            for review in (
                task.validator_contract_review,
                task.contamination_audit_review,
            )
        ]
        evidence_roots = [task.task_evidence_root_sha256 for task in self.tasks]
        if len(set(validator_artifacts)) != len(validator_artifacts):
            raise ValueError("validator artifacts cannot be reused across confirmatory tasks")
        if len(set(contamination_artifacts)) != len(contamination_artifacts):
            raise ValueError("contamination audits cannot be reused across confirmatory tasks")
        if len(set(evidence_review_events)) != len(evidence_review_events):
            raise ValueError("human evidence-review events cannot be reused across tasks")
        if len(set(evidence_roots)) != len(evidence_roots):
            raise ValueError("task evidence roots must be unique")
        counts = {
            family: sum(task.family == family for task in self.tasks) for family in TaskFamily
        }
        if any(count != CONFIRMATORY_TASKS_PER_FAMILY for count in counts.values()):
            raise ValueError(
                "confirmatory task bank requires exactly "
                f"{CONFIRMATORY_TASKS_PER_FAMILY} tasks per family"
            )
        split_counts = {
            split: sum(task.split.value == split for task in self.tasks)
            for split in SEASON_TASK_SPLIT_COUNTS
        }
        if split_counts != SEASON_TASK_SPLIT_COUNTS:
            raise ValueError(
                "confirmatory task bank requires the frozen scored/development/private-reserve "
                "split"
            )
        cell_counts = {
            (family.value, split): sum(
                task.family == family and task.split.value == split for task in self.tasks
            )
            for family in TaskFamily
            for split in SEASON_TASK_SPLIT_COUNTS
        }
        if any(
            cell_counts[(family.value, split)] != per_family
            for family in TaskFamily
            for split, per_family in SEASON_TASK_SPLIT_COUNTS_PER_FAMILY.items()
        ):
            raise ValueError(
                "each family requires 40 scored, 10 development, and 10 private-reserve tasks"
            )
        try:
            validate_confirmatory_bank(self.tasks)
        except ConstructBlueprintError as exc:
            raise ValueError(str(exc)) from exc
        return self


class TaskContributorInviteCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    contributor_code: str = Field(
        min_length=3,
        max_length=80,
        pattern=r"^[a-zA-Z0-9_-]+$",
        alias="contributorCode",
    )
    qualified_families: list[TaskFamily] = Field(
        min_length=1,
        max_length=4,
        alias="qualifiedFamilies",
    )
    protocol_version: str = Field(
        min_length=3,
        max_length=80,
        alias="protocolVersion",
    )
    protocol_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="protocolSha256",
    )
    verified_identity_handle: str = Field(
        min_length=3,
        max_length=500,
        alias="verifiedIdentityHandle",
    )
    person_uniqueness_evidence_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="personUniquenessEvidenceSha256",
    )
    person_uniqueness_verified_by_admin: Literal[True] = Field(
        alias="personUniquenessVerifiedByAdmin"
    )

    @field_validator("qualified_families")
    @classmethod
    def unique_task_contributor_families(cls, value: list[TaskFamily]) -> list[TaskFamily]:
        if len(value) != len(set(value)):
            raise ValueError("qualifiedFamilies must not contain duplicates")
        return value

    @model_validator(mode="after")
    def require_current_task_contributor_protocol(self) -> TaskContributorInviteCreate:
        if (
            self.protocol_version != TASK_CONTRIBUTOR_PROTOCOL_VERSION
            or self.protocol_sha256 != TASK_CONTRIBUTOR_PROTOCOL_SHA256
        ):
            raise ValueError("task contributor invitation must bind the current frozen protocol")
        return self


class TaskContributorProtocolAcceptanceCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    protocol_version: str = Field(
        min_length=3,
        max_length=80,
        alias="protocolVersion",
    )
    protocol_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="protocolSha256",
    )
    voluntary_participation_accepted: bool = Field(
        strict=True,
        alias="voluntaryParticipationAccepted",
    )
    task_contribution_agreement_accepted: bool = Field(
        strict=True,
        alias="taskContributionAgreementAccepted",
    )
    human_only_methods_acknowledged: bool = Field(
        strict=True,
        alias="humanOnlyMethodsAcknowledged",
    )

    @model_validator(mode="after")
    def require_current_protocol_and_affirmative_acceptance(
        self,
    ) -> TaskContributorProtocolAcceptanceCreate:
        if (
            self.protocol_version != TASK_CONTRIBUTOR_PROTOCOL_VERSION
            or self.protocol_sha256 != TASK_CONTRIBUTOR_PROTOCOL_SHA256
        ):
            raise ValueError("task contributor acceptance does not match the current protocol")
        if not (
            self.voluntary_participation_accepted
            and self.task_contribution_agreement_accepted
            and self.human_only_methods_acknowledged
        ):
            raise ValueError("task contributor protocol requires all three acceptances")
        return self


class TaskValidatorInviteCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    validator_code: str = Field(
        min_length=8,
        max_length=120,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$",
        alias="validatorCode",
    )
    qualified_families: list[TaskFamily] = Field(
        min_length=1,
        max_length=4,
        alias="qualifiedFamilies",
    )
    qualification_reference: str = Field(
        min_length=8,
        max_length=500,
        alias="qualificationReference",
    )
    verified_identity_handle: str = Field(
        min_length=3,
        max_length=500,
        alias="verifiedIdentityHandle",
    )
    qualification_evidence_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="qualificationEvidenceSha256",
    )
    independence_attestation_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="independenceAttestationSha256",
    )
    verification_record_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="verificationRecordSha256",
    )
    affiliation_class: str = Field(
        pattern=r"^(independent_external|product_affiliated)$",
        alias="affiliationClass",
    )
    conflict_disclosure_reference: str = Field(
        min_length=3,
        max_length=500,
        alias="conflictDisclosureReference",
    )
    consent_document_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="consentDocumentSha256",
    )
    adjudication_authorized: bool = Field(default=False, alias="adjudicationAuthorized")
    evidence_verified_by_admin: bool = Field(alias="evidenceVerifiedByAdmin")

    @field_validator("qualified_families")
    @classmethod
    def unique_task_validator_families(cls, value: list[TaskFamily]) -> list[TaskFamily]:
        if len(value) != len(set(value)):
            raise ValueError("qualifiedFamilies must be unique")
        return value

    @model_validator(mode="after")
    def require_independent_adjudicator(self) -> TaskValidatorInviteCreate:
        if not self.evidence_verified_by_admin:
            raise ValueError("task-validator evidence must be verified before invitation")
        if self.adjudication_authorized and self.affiliation_class != "independent_external":
            raise ValueError("task adjudicators must be independent_external")
        return self


class TaskContributionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    family: TaskFamily
    prompt: str = Field(min_length=40, max_length=4000)
    construct_blueprint_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="constructBlueprintSha256",
    )
    construct_cell_id: str = Field(
        min_length=3,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_]*$",
        alias="constructCellId",
    )
    difficulty_tier: str = Field(
        pattern=r"^(foundation|integrative|stress)$",
        alias="difficultyTier",
    )
    subskills: list[str] = Field(min_length=1, max_length=8)
    explicit_constraints: list[str] = Field(
        min_length=1,
        max_length=12,
        alias="explicitConstraints",
    )
    unacceptable_outcomes: list[str] = Field(
        min_length=1,
        max_length=12,
        alias="unacceptableOutcomes",
    )
    acceptable_solution_outline: str = Field(
        min_length=20,
        max_length=2000,
        alias="acceptableSolutionOutline",
    )
    objective_validator_possible: bool = Field(alias="objectiveValidatorPossible")
    validator_notes: str = Field(default="", max_length=1000, alias="validatorNotes")
    rights_basis: str = Field(
        pattern=r"^(original_personal_authorship|employer_authorized_original)$",
        alias="rightsBasis",
    )
    human_authorship_attestation: bool = Field(alias="humanAuthorshipAttestation")
    no_personal_data_attestation: bool = Field(alias="noPersonalDataAttestation")
    research_use_consent: bool = Field(alias="researchUseConsent")
    client_nonce: str = Field(min_length=8, max_length=120, alias="clientNonce")

    @field_validator("prompt", "acceptable_solution_outline", "validator_notes")
    @classmethod
    def normalize_task_contribution_text(cls, value: str) -> str:
        normalized = value.strip()
        if any(ord(character) < 32 and character not in "\n\t" for character in normalized):
            raise ValueError("task contribution contains unsupported control characters")
        return normalized

    @field_validator("subskills", "explicit_constraints", "unacceptable_outcomes")
    @classmethod
    def normalize_task_contribution_lists(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if len(normalized) != len(value) or len(normalized) != len(set(normalized)):
            raise ValueError("task contribution lists require unique non-empty entries")
        if any(len(item) > 400 for item in normalized):
            raise ValueError("task contribution list entries may not exceed 400 characters")
        return normalized

    @model_validator(mode="after")
    def enforce_task_contributor_attestations(self) -> TaskContributionCreate:
        if not (
            self.human_authorship_attestation
            and self.no_personal_data_attestation
            and self.research_use_consent
        ):
            raise ValueError("all task-contribution attestations are required")
        if self.objective_validator_possible and len(self.validator_notes) < 10:
            raise ValueError("validator notes are required when an objective check is possible")
        if (
            self.construct_blueprint_sha256 != BLUEPRINT_SHA256
            or self.construct_cell_id not in CONSTRUCT_CELLS[self.family.value]
            or self.difficulty_tier not in DIFFICULTY_TIERS
            or self.construct_cell_id not in self.subskills
        ):
            raise ValueError("task contribution does not match the frozen construct blueprint")
        return self


class TaskContributionWithdrawalCreate(BaseModel):
    """Contributor-authorized withdrawal of an unimported task candidate."""

    model_config = ConfigDict(populate_by_name=True)

    candidate_record_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="candidateRecordSha256",
    )
    reason_category: str = Field(
        pattern=(
            r"^(voluntary_withdrawal|rights_concern|content_error|"
            r"privacy_concern|other)$"
        ),
        alias="reasonCategory",
    )
    note: str = Field(default="", max_length=1000)
    withdrawal_confirmed: Literal[True] = Field(alias="withdrawalConfirmed")
    client_nonce: str = Field(min_length=8, max_length=120, alias="clientNonce")

    @field_validator("note")
    @classmethod
    def normalize_task_withdrawal_note(cls, value: str) -> str:
        normalized = value.strip()
        if any(ord(character) < 32 and character not in "\n\t" for character in normalized):
            raise ValueError("withdrawal note contains unsupported control characters")
        return normalized


class TaskCandidateReviewCreate(BaseModel):
    """Legacy one-stage candidate review.

    Kept only so historical development events remain parseable.  The API rejects new
    submissions of this shape and official task-bank import never accepts it.
    """

    model_config = ConfigDict(populate_by_name=True)

    decision: str = Field(pattern=r"^(approve|revise|reject)$")
    construct_fit: bool = Field(alias="constructFit")
    context_complete: bool = Field(alias="contextComplete")
    specialist_scope_clear: bool = Field(alias="specialistScopeClear")
    answer_leakage_absent: bool = Field(alias="answerLeakageAbsent")
    rights_basis_credible: bool = Field(alias="rightsBasisCredible")
    validator_plan_adequate: bool = Field(alias="validatorPlanAdequate")
    issue_tags: list[str] = Field(default_factory=list, max_length=8, alias="issueTags")
    note: str = Field(default="", max_length=1200)

    @field_validator("issue_tags")
    @classmethod
    def unique_task_review_issue_tags(cls, value: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in value if tag.strip()]
        if len(normalized) != len(value) or len(normalized) != len(set(normalized)):
            raise ValueError("issueTags require unique non-empty entries")
        return normalized

    @model_validator(mode="after")
    def enforce_task_candidate_approval(self) -> TaskCandidateReviewCreate:
        checks = (
            self.construct_fit,
            self.context_complete,
            self.specialist_scope_clear,
            self.answer_leakage_absent,
            self.rights_basis_credible,
            self.validator_plan_adequate,
        )
        if self.decision == "approve" and (not all(checks) or self.issue_tags):
            raise ValueError("approval requires every admission check to pass without issue tags")
        if self.decision != "approve" and not self.issue_tags:
            raise ValueError("revise and reject decisions require at least one issue tag")
        return self


class TaskCandidateBlindValidityCreate(BaseModel):
    """Prompt-only, independently authored assessment of a human task candidate."""

    model_config = ConfigDict(populate_by_name=True)

    decision: str = Field(pattern=r"^(valid|revise|exclude)$")
    construct_fit: bool = Field(alias="constructFit")
    context_complete: bool = Field(alias="contextComplete")
    coherent_question: bool = Field(alias="coherentQuestion")
    general_track_scope: bool = Field(alias="generalTrackScope")
    answer_leakage_absent: bool = Field(alias="answerLeakageAbsent")
    discrimination_value: bool = Field(alias="discriminationValue")
    family_classification: TaskFamily | None = Field(default=None, alias="familyClassification")
    construct_cell_classification: str | None = Field(
        default=None,
        min_length=3,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_]*$",
        alias="constructCellClassification",
    )
    difficulty_tier_classification: str | None = Field(
        default=None,
        pattern=r"^(foundation|integrative|stress)$",
        alias="difficultyTierClassification",
    )
    independent_solution_outline: str = Field(
        default="",
        max_length=2000,
        alias="independentSolutionOutline",
    )
    success_criteria: list[str] = Field(
        default_factory=list,
        max_length=8,
        alias="successCriteria",
    )
    disqualifying_errors: list[str] = Field(
        default_factory=list,
        max_length=8,
        alias="disqualifyingErrors",
    )
    issue_tags: list[str] = Field(default_factory=list, max_length=8, alias="issueTags")
    criteria_authored_by_reviewer: bool = Field(alias="criteriaAuthoredByReviewer")
    author_pack_not_seen: bool = Field(alias="authorPackNotSeen")
    model_outputs_not_consulted: bool = Field(alias="modelOutputsNotConsulted")
    note: str = Field(default="", max_length=1600)

    @field_validator("success_criteria", "disqualifying_errors")
    @classmethod
    def normalize_blind_criterion_rows(cls, value: list[str]) -> list[str]:
        normalized = [" ".join(item.strip().split()) for item in value]
        if any(len(item) < 8 or len(item) > 500 for item in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError("criterion rows must be unique and contain 8 to 500 characters")
        return normalized

    @field_validator("issue_tags")
    @classmethod
    def validate_blind_issue_tags(cls, value: list[str]) -> list[str]:
        allowed = {
            "construct_mismatch",
            "missing_context",
            "multiple_unrelated_questions",
            "specialist_scope",
            "answer_leakage",
            "low_discrimination",
            "duplicate_or_contaminated",
            "other",
        }
        normalized = [tag.strip() for tag in value if tag.strip()]
        if (
            len(normalized) != len(value)
            or len(normalized) != len(set(normalized))
            or any(tag not in allowed for tag in normalized)
        ):
            raise ValueError("issueTags contain invalid or duplicate values")
        return normalized

    @model_validator(mode="after")
    def enforce_blind_candidate_validity(self) -> TaskCandidateBlindValidityCreate:
        if not (
            self.criteria_authored_by_reviewer
            and self.author_pack_not_seen
            and self.model_outputs_not_consulted
        ):
            raise ValueError(
                "blind review requires human criterion authorship, author-pack blindness, "
                "and model-output blindness attestations"
            )
        checks = (
            self.construct_fit,
            self.context_complete,
            self.coherent_question,
            self.general_track_scope,
            self.answer_leakage_absent,
            self.discrimination_value,
        )
        if self.decision == "valid":
            if (
                not all(checks)
                or self.issue_tags
                or self.family_classification is None
                or self.construct_cell_classification is None
                or self.difficulty_tier_classification is None
                or len(self.independent_solution_outline.strip()) < 40
                or len(self.success_criteria) < 2
                or not self.disqualifying_errors
            ):
                raise ValueError(
                    "valid requires every blind check, an independent classification, "
                    "solution outline, and complete criterion pack"
                )
        elif (
            not self.issue_tags
            or len(self.note.strip()) < 10
            or self.family_classification is not None
            or self.construct_cell_classification is not None
            or self.difficulty_tier_classification is not None
            or self.independent_solution_outline.strip()
            or self.success_criteria
            or self.disqualifying_errors
        ):
            raise ValueError(
                "revise and exclude require an issue note and must not publish a criterion pack"
            )
        return self


class TaskCandidateReconciliationCreate(BaseModel):
    """Second-stage review after the sealed author pack is revealed."""

    model_config = ConfigDict(populate_by_name=True)

    decision: str = Field(pattern=r"^(approve|revise|reject)$")
    author_pack_adequacy: str = Field(
        pattern=r"^(adequate|partial|misleading)$",
        alias="authorPackAdequacy",
    )
    construct_label_agreement: bool = Field(alias="constructLabelAgreement")
    difficulty_label_agreement: bool = Field(alias="difficultyLabelAgreement")
    constraint_set_adequate: bool = Field(alias="constraintSetAdequate")
    solution_outline_adequate: bool = Field(alias="solutionOutlineAdequate")
    validator_plan_adequate: bool = Field(alias="validatorPlanAdequate")
    rights_basis_credible: bool = Field(alias="rightsBasisCredible")
    success_criteria: list[str] = Field(min_length=2, max_length=8, alias="successCriteria")
    permitted_variations: list[str] = Field(
        min_length=1,
        max_length=6,
        alias="permittedVariations",
    )
    disqualifying_errors: list[str] = Field(
        min_length=1,
        max_length=8,
        alias="disqualifyingErrors",
    )
    objective_checks: list[str] = Field(default_factory=list, max_length=8, alias="objectiveChecks")
    issue_tags: list[str] = Field(default_factory=list, max_length=8, alias="issueTags")
    criteria_authored_by_reviewer: bool = Field(alias="criteriaAuthoredByReviewer")
    independent_of_author: bool = Field(alias="independentOfAuthor")
    model_outputs_not_consulted: bool = Field(alias="modelOutputsNotConsulted")
    note: str = Field(default="", max_length=2000)

    @field_validator(
        "success_criteria",
        "permitted_variations",
        "disqualifying_errors",
        "objective_checks",
    )
    @classmethod
    def normalize_reconciled_criterion_rows(cls, value: list[str]) -> list[str]:
        normalized = [" ".join(item.strip().split()) for item in value]
        if any(len(item) < 8 or len(item) > 500 for item in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError("criterion rows must be unique and contain 8 to 500 characters")
        return normalized

    @field_validator("issue_tags")
    @classmethod
    def validate_reconciliation_issue_tags(cls, value: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in value if tag.strip()]
        if len(normalized) != len(value) or len(normalized) != len(set(normalized)):
            raise ValueError("issueTags require unique non-empty entries")
        return normalized

    @model_validator(mode="after")
    def enforce_reconciliation_contract(self) -> TaskCandidateReconciliationCreate:
        if not (
            self.criteria_authored_by_reviewer
            and self.independent_of_author
            and self.model_outputs_not_consulted
        ):
            raise ValueError(
                "reconciliation requires human authorship, independence, and model-output "
                "blindness attestations"
            )
        checks = (
            self.construct_label_agreement,
            self.difficulty_label_agreement,
            self.constraint_set_adequate,
            self.solution_outline_adequate,
            self.validator_plan_adequate,
            self.rights_basis_credible,
        )
        if self.decision == "approve" and (
            not all(checks) or self.author_pack_adequacy != "adequate" or self.issue_tags
        ):
            raise ValueError(
                "approval requires an adequate author pack, every reconciliation check, "
                "and no issue tags"
            )
        if self.decision != "approve" and (not self.issue_tags or len(self.note.strip()) < 10):
            raise ValueError("revise and reject require an issue tag and explanatory note")
        return self


class TaskCandidateAdjudicationCreate(BaseModel):
    """Third-person answer-blind adjudication and canonical criterion pack."""

    model_config = ConfigDict(populate_by_name=True)

    decision: str = Field(pattern=r"^(approve|revise|reject)$")
    family: TaskFamily | None = None
    construct_cell_id: str | None = Field(
        default=None,
        min_length=3,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_]*$",
        alias="constructCellId",
    )
    difficulty_tier: str | None = Field(
        default=None,
        pattern=r"^(foundation|integrative|stress)$",
        alias="difficultyTier",
    )
    success_criteria: list[str] = Field(default_factory=list, max_length=8, alias="successCriteria")
    permitted_variations: list[str] = Field(
        default_factory=list,
        max_length=6,
        alias="permittedVariations",
    )
    disqualifying_errors: list[str] = Field(
        default_factory=list,
        max_length=8,
        alias="disqualifyingErrors",
    )
    objective_checks: list[str] = Field(default_factory=list, max_length=8, alias="objectiveChecks")
    criteria_authored_by_adjudicator: bool = Field(alias="criteriaAuthoredByAdjudicator")
    independent_of_author_and_reviewers: bool = Field(alias="independentOfAuthorAndReviewers")
    model_outputs_not_consulted: bool = Field(alias="modelOutputsNotConsulted")
    note: str = Field(min_length=10, max_length=2400)

    @field_validator(
        "success_criteria",
        "permitted_variations",
        "disqualifying_errors",
        "objective_checks",
    )
    @classmethod
    def normalize_adjudicated_candidate_rows(cls, value: list[str]) -> list[str]:
        normalized = [" ".join(item.strip().split()) for item in value]
        if any(len(item) < 8 or len(item) > 500 for item in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError("criterion rows must be unique and contain 8 to 500 characters")
        return normalized

    @model_validator(mode="after")
    def enforce_candidate_adjudication(self) -> TaskCandidateAdjudicationCreate:
        if not self.independent_of_author_and_reviewers or not self.model_outputs_not_consulted:
            raise ValueError("adjudication requires independence and model-output blindness")
        if self.decision == "approve":
            if (
                self.family is None
                or self.construct_cell_id is None
                or self.difficulty_tier is None
                or len(self.success_criteria) < 2
                or not self.permitted_variations
                or not self.disqualifying_errors
                or not self.criteria_authored_by_adjudicator
            ):
                raise ValueError(
                    "approval requires a final construct classification and complete "
                    "human-authored criterion pack"
                )
        elif (
            self.family is not None
            or self.construct_cell_id is not None
            or self.difficulty_tier is not None
            or self.success_criteria
            or self.permitted_variations
            or self.disqualifying_errors
            or self.objective_checks
            or self.criteria_authored_by_adjudicator
        ):
            raise ValueError("revise and reject must not freeze a criterion pack")
        return self


class DevelopmentTaskBlindValidityCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    decision: str = Field(pattern=r"^(valid|revise|exclude)$")
    construct_fit: bool = Field(alias="constructFit")
    context_complete: bool = Field(alias="contextComplete")
    coherent_question: bool = Field(alias="coherentQuestion")
    general_track_scope: bool = Field(alias="generalTrackScope")
    answer_leakage_absent: bool = Field(alias="answerLeakageAbsent")
    discrimination_value: bool = Field(alias="discriminationValue")
    issue_tags: list[str] = Field(default_factory=list, max_length=8, alias="issueTags")
    note: str = Field(default="", max_length=1200)

    @field_validator("issue_tags")
    @classmethod
    def unique_development_task_issue_tags(cls, value: list[str]) -> list[str]:
        allowed = {
            "construct_mismatch",
            "missing_context",
            "multiple_unrelated_questions",
            "specialist_scope",
            "answer_leakage",
            "low_discrimination",
            "duplicate_or_contaminated",
            "other",
        }
        normalized = [tag.strip() for tag in value if tag.strip()]
        if (
            len(normalized) != len(value)
            or len(normalized) != len(set(normalized))
            or any(tag not in allowed for tag in normalized)
        ):
            raise ValueError("issueTags contain invalid or duplicate values")
        return normalized

    @model_validator(mode="after")
    def enforce_blind_validity_decision(self) -> DevelopmentTaskBlindValidityCreate:
        checks = (
            self.construct_fit,
            self.context_complete,
            self.coherent_question,
            self.general_track_scope,
            self.answer_leakage_absent,
            self.discrimination_value,
        )
        if self.decision == "valid" and (not all(checks) or self.issue_tags):
            raise ValueError("valid requires every blind check to pass without issue tags")
        if self.decision != "valid" and (not self.issue_tags or len(self.note.strip()) < 10):
            raise ValueError("revise and exclude require an issue tag and explanatory note")
        return self


class DevelopmentTaskCriteriaCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reference_adequacy: str = Field(
        pattern=r"^(adequate|partial|misleading)$", alias="referenceAdequacy"
    )
    success_criteria: list[str] = Field(min_length=2, max_length=8, alias="successCriteria")
    permitted_variations: list[str] = Field(min_length=1, max_length=6, alias="permittedVariations")
    disqualifying_errors: list[str] = Field(min_length=1, max_length=8, alias="disqualifyingErrors")
    objective_checks: list[str] = Field(default_factory=list, max_length=8, alias="objectiveChecks")
    criteria_authored_by_reviewer: bool = Field(alias="criteriaAuthoredByReviewer")
    note: str = Field(default="", max_length=1600)

    @field_validator(
        "success_criteria",
        "permitted_variations",
        "disqualifying_errors",
        "objective_checks",
    )
    @classmethod
    def normalize_criterion_rows(cls, value: list[str]) -> list[str]:
        normalized = [" ".join(item.strip().split()) for item in value]
        if any(len(item) < 8 or len(item) > 500 for item in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError("criterion rows must be unique and contain 8 to 500 characters")
        return normalized

    @model_validator(mode="after")
    def enforce_human_criterion_authorship(self) -> DevelopmentTaskCriteriaCreate:
        if not self.criteria_authored_by_reviewer:
            raise ValueError("reviewer authorship attestation is required")
        if self.reference_adequacy != "adequate" and len(self.note.strip()) < 10:
            raise ValueError("partial or misleading references require an explanatory note")
        return self


class DevelopmentTaskAdjudicationCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    decision: str = Field(pattern=r"^(valid|revise|exclude)$")
    reference_adequacy: str | None = Field(
        default=None,
        pattern=r"^(adequate|partial|misleading)$",
        alias="referenceAdequacy",
    )
    success_criteria: list[str] = Field(default_factory=list, max_length=8, alias="successCriteria")
    permitted_variations: list[str] = Field(
        default_factory=list,
        max_length=6,
        alias="permittedVariations",
    )
    disqualifying_errors: list[str] = Field(
        default_factory=list,
        max_length=8,
        alias="disqualifyingErrors",
    )
    objective_checks: list[str] = Field(default_factory=list, max_length=8, alias="objectiveChecks")
    criteria_authored_by_adjudicator: bool = Field(alias="criteriaAuthoredByAdjudicator")
    independent_of_source_reviewers: bool = Field(alias="independentOfSourceReviewers")
    model_outputs_not_consulted: bool = Field(alias="modelOutputsNotConsulted")
    note: str = Field(min_length=10, max_length=2000)

    @field_validator(
        "success_criteria",
        "permitted_variations",
        "disqualifying_errors",
        "objective_checks",
    )
    @classmethod
    def normalize_adjudicated_criterion_rows(cls, value: list[str]) -> list[str]:
        normalized = [" ".join(item.strip().split()) for item in value]
        if any(len(item) < 8 or len(item) > 500 for item in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError("criterion rows must be unique and contain 8 to 500 characters")
        return normalized

    @model_validator(mode="after")
    def enforce_adjudication_contract(self) -> DevelopmentTaskAdjudicationCreate:
        if not self.independent_of_source_reviewers or not self.model_outputs_not_consulted:
            raise ValueError("adjudication requires independence and answer-blindness attestations")
        if self.decision == "valid":
            if (
                self.reference_adequacy is None
                or len(self.success_criteria) < 2
                or not self.permitted_variations
                or not self.disqualifying_errors
                or not self.criteria_authored_by_adjudicator
            ):
                raise ValueError(
                    "valid adjudication requires a complete human-authored criterion pack"
                )
        elif (
            self.reference_adequacy is not None
            or self.success_criteria
            or self.permitted_variations
            or self.disqualifying_errors
            or self.objective_checks
            or self.criteria_authored_by_adjudicator
        ):
            raise ValueError("revise and exclude adjudications must not publish a criterion pack")
        return self


class BattleCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(min_length=10, max_length=2000)
    category: TaskFamily
    research_consent: bool = Field(default=False, alias="researchConsent")
    client_nonce: str = Field(min_length=8, max_length=120, alias="clientNonce")

    @field_validator("prompt")
    @classmethod
    def clean_prompt(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if any(ord(character) < 32 and character not in "\n\t" for character in cleaned):
            raise ValueError("prompt contains unsupported control characters")
        return cleaned


class ControlledRunCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    season: str = Field(default="season-0", min_length=3, max_length=80)
    organization_reference: str = Field(min_length=3, max_length=240, alias="organizationReference")
    protocol_version: str = Field(
        default="flavourbench-controlled-run-v1",
        min_length=3,
        max_length=80,
        alias="protocolVersion",
    )
    rater_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="raterPlanSha256")
    analysis_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="analysisPlanSha256")
    submitted_endpoint_model_id: str = Field(
        min_length=3, max_length=200, alias="submittedEndpointModelId"
    )
    submitted_model_card_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="submittedModelCardSha256"
    )
    data_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="dataPolicySha256")
    model_ids: list[str] = Field(
        min_length=1,
        max_length=SEASON_MODEL_COUNT,
        alias="modelIds",
    )
    task_schedule: list[ControlledScheduleEntry] = Field(
        min_length=1, max_length=5000, alias="taskSchedule"
    )
    budget_cap_micros: int = Field(gt=0, le=10_000_000_000, alias="budgetCapMicros")

    @field_validator("model_ids")
    @classmethod
    def unique_model_ids(cls, value: list[str]) -> list[str]:
        if any(len(model_id) < 3 or len(model_id) > 200 for model_id in value):
            raise ValueError("modelIds contains an invalid model identifier")
        if len(value) != len(set(value)):
            raise ValueError("modelIds must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_frozen_schedule(self) -> ControlledRunCreate:
        roster = set(self.model_ids)
        if self.submitted_endpoint_model_id not in roster:
            raise ValueError("submittedEndpointModelId must be present in modelIds")
        seen: set[tuple[str, str, tuple[str, ...], int]] = set()
        repetitions: dict[tuple[str, str, tuple[str, ...]], list[int]] = {}
        for entry in self.task_schedule:
            entry_models = (
                tuple(sorted(entry.model_ids))
                if entry.track == BattleTrack.model_arena
                else tuple(entry.model_ids)
            )
            if not set(entry_models) <= roster:
                raise ValueError("taskSchedule contains a model outside modelIds")
            if self.submitted_endpoint_model_id not in entry_models:
                raise ValueError("every scheduled comparison must include submittedEndpointModelId")
            key = (
                entry.task_public_id,
                entry.track.value,
                entry_models,
                entry.repetition_index,
            )
            if key in seen:
                raise ValueError("taskSchedule contains a duplicate repetition")
            seen.add(key)
            group = key[:3]
            repetitions.setdefault(group, []).append(entry.repetition_index)
        for indexes in repetitions.values():
            if sorted(indexes) != list(range(1, len(indexes) + 1)):
                raise ValueError("repetitionIndex values must be contiguous from 1")
        return self


class ControlledScheduleEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_public_id: str = Field(min_length=3, max_length=80, alias="taskPublicId")
    track: BattleTrack
    model_ids: list[str] = Field(min_length=1, max_length=2, alias="modelIds")
    repetition_index: int = Field(ge=1, le=1000, alias="repetitionIndex")

    @model_validator(mode="after")
    def validate_track_arms(self) -> ControlledScheduleEntry:
        if self.track == BattleTrack.model_arena:
            if len(self.model_ids) != 2 or len(set(self.model_ids)) != 2:
                raise ValueError("model_arena requires two distinct modelIds")
        elif len(self.model_ids) != 1:
            raise ValueError("epicure_uplift requires exactly one modelId")
        return self


class ControlledBattleCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_public_id: str | None = Field(
        default=None, min_length=3, max_length=80, alias="taskPublicId"
    )
    expected_assignment_ordinal: int = Field(ge=0, alias="expectedAssignmentOrdinal")
    client_nonce: str = Field(min_length=8, max_length=120, alias="clientNonce")


class ControlledRunLifecycleAction(StrEnum):
    collection_complete = "collection_complete"
    close = "close"
    revoke = "revoke"


class ControlledRunLifecycleCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    action: ControlledRunLifecycleAction
    authorization_reference: str = Field(
        min_length=3, max_length=240, alias="authorizationReference"
    )


class ControlledTokenRotationCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    authorization_reference: str = Field(
        min_length=3, max_length=240, alias="authorizationReference"
    )


class CostSettlementCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    arm_costs_micros: dict[str, int] = Field(min_length=1, max_length=2, alias="armCostsMicros")
    authorization_reference: str = Field(
        min_length=3, max_length=240, alias="authorizationReference"
    )

    @field_validator("arm_costs_micros")
    @classmethod
    def validate_costs(cls, value: dict[str, int]) -> dict[str, int]:
        if any(
            not arm_id or not isinstance(cost, int) or cost < 0 or cost > 1_000_000_000
            for arm_id, cost in value.items()
        ):
            raise ValueError("armCostsMicros contains an invalid settlement")
        return value


class ControlledRunReleaseCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    authorized: bool
    publication_acceptance_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        alias="publicationAcceptanceId",
    )
    authorization_reference: str = Field(
        min_length=3, max_length=240, alias="authorizationReference"
    )

    @model_validator(mode="after")
    def require_publication_acceptance(self) -> ControlledRunReleaseCreate:
        if not self.authorized and self.publication_acceptance_id is not None:
            raise ValueError("revocation cannot name a publication acceptance")
        return self


class GovernanceAcceptanceRevocationCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="documentSha256")
    external_envelope_reference: str = Field(
        min_length=8,
        max_length=240,
        alias="externalEnvelopeReference",
    )
    signatory_principal_reference: str = Field(
        min_length=8,
        max_length=240,
        alias="signatoryPrincipalReference",
    )
    authority_basis: str = Field(min_length=3, max_length=160, alias="authorityBasis")
    accepted_at: datetime = Field(alias="acceptedAt")
    reason_code: str = Field(
        min_length=3,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_:-]*$",
        alias="reasonCode",
    )

    @field_validator("accepted_at")
    @classmethod
    def require_aware_accepted_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("acceptedAt must be timezone-aware")
        return value


class ControlledReviewerAuthorizationCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    authorization_reference: str = Field(
        min_length=3, max_length=240, alias="authorizationReference"
    )
    active: bool = True


class SnapshotPublishCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    publication_reference: str = Field(min_length=3, max_length=240, alias="publicationReference")


class Season1PostcollectionItemAuditCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    artifact: dict[str, Any]
    supersedes_event_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        alias="supersedesEventId",
    )


class Season1ArenaMethodValidationCreate(BaseModel):
    artifact: dict[str, Any]


class VoteCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    choice: VoteChoice
    reason_tags: list[str] = Field(default_factory=list, max_length=8, alias="reasonTags")
    rubric: dict[str, int | str | bool | None] = Field(default_factory=dict, max_length=0)

    @field_validator("reason_tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        allowed = {
            "ignored_constraint",
            "weak_flavour_logic",
            "unclear",
            "generic",
            "overconfident",
            "safety_hazard",
            "unsupported_safety_claim",
            "allergen_or_dietary_risk",
            "impractical",
            "evidence_trace_mismatch",
            "entity_resolution_mismatch",
            "similarity_as_functional_proof",
            "similarity_as_mechanism",
            "axis_as_measured_quantity",
            "score_as_normative_truth",
            "selective_evidence",
            "irrelevant_evidence",
            "false_precision",
        }
        if any(tag not in allowed for tag in value):
            raise ValueError("unsupported reason tag")
        return list(dict.fromkeys(value))


class ExpertArmRubric(BaseModel):
    task_completion: int = Field(ge=1, le=5, strict=True)
    constraint_compliance: int = Field(ge=1, le=5, strict=True)
    coherence: int = Field(ge=1, le=5, strict=True)
    sensory_promise: int = Field(ge=1, le=5, strict=True)
    cookability: int = Field(ge=1, le=5, strict=True)
    clarity: int = Field(ge=1, le=5, strict=True)
    originality: int = Field(ge=1, le=5, strict=True)
    evidence_use: int = Field(ge=1, le=5, strict=True)
    calibration: int = Field(ge=1, le=5, strict=True)


class ExpertRubric(BaseModel):
    left: ExpertArmRubric
    right: ExpertArmRubric


class ExpertVoteCreate(VoteCreate):
    rubric: ExpertRubric


class ExpertTaskAssessmentCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_validity: str = Field(pattern=r"^(valid|minor_issue|invalid)$")
    task_issue_tags: list[str] = Field(default_factory=list, max_length=7)
    task_note: str = Field(default="", max_length=600)
    answerability: str = Field(pattern=r"^(answerable|minor_ambiguity|unanswerable)$")
    family_fit: str = Field(pattern=r"^(in_family|borderline|out_of_family)$")
    scope_eligibility: str = Field(pattern=r"^(general_track|specialist_track|exclude)$")
    specialist_domains: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("task_issue_tags")
    @classmethod
    def validate_task_issue_tags(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value))
        if any(tag not in TASK_ISSUE_TAGS for tag in normalized):
            raise ValueError("unsupported task issue tag")
        return normalized

    @field_validator("specialist_domains")
    @classmethod
    def validate_specialist_domains(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value))
        if any(domain not in SPECIALIST_DOMAINS for domain in normalized):
            raise ValueError("unsupported specialist domain")
        return normalized

    @model_validator(mode="after")
    def require_issue_evidence(self) -> ExpertTaskAssessmentCreate:
        if self.task_validity == "valid" and self.task_issue_tags:
            raise ValueError("valid tasks cannot include task issue tags")
        if self.task_validity != "valid" and not self.task_issue_tags:
            raise ValueError("task limitations require at least one task issue tag")
        if self.scope_eligibility == "general_track" and self.specialist_domains:
            raise ValueError("general-track tasks cannot include specialist domains")
        if self.scope_eligibility == "specialist_track" and not self.specialist_domains:
            raise ValueError("specialist-track tasks require at least one specialist domain")
        return self

    @computed_field
    @property
    def general_track_eligible(self) -> bool:
        return bool(
            self.task_validity != "invalid"
            and self.answerability != "unanswerable"
            and self.family_fit != "out_of_family"
            and self.scope_eligibility == "general_track"
        )


class ExpertReviewMetadata(ExpertTaskAssessmentCreate):
    confidence: int = Field(ge=1, le=5, strict=True)
    left_failure_tags: list[str] = Field(default_factory=list, max_length=12)
    right_failure_tags: list[str] = Field(default_factory=list, max_length=12)
    practical_check: str = Field(pattern=r"^(not_performed|reasoned_only|partial_cook|full_cook)$")
    comparative_rationale: str = Field(min_length=20, max_length=1200)

    @field_validator("left_failure_tags", "right_failure_tags")
    @classmethod
    def validate_failure_tags(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value))
        if any(tag not in RESPONSE_FAILURE_TAGS for tag in normalized):
            raise ValueError("unsupported response failure tag")
        return normalized


class ExpertReviewRubricV2(ExpertRubric):
    rubric_version: str = Field(
        default=PROTOCOL_VERSION,
        pattern=rf"^{PROTOCOL_VERSION}$",
    )
    review_metadata: ExpertReviewMetadata


class ExpertReviewCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    choice: VoteChoice
    reason_tags: list[str] = Field(default_factory=list, max_length=8, alias="reasonTags")
    rubric: ExpertReviewRubricV2

    @field_validator("reason_tags")
    @classmethod
    def validate_reason_tags(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value))
        if any(tag not in RESPONSE_FAILURE_TAGS for tag in normalized):
            raise ValueError("unsupported reason tag")
        return normalized


class ExpertSessionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    protocol_sha256: str = Field(
        default=PROTOCOL_SHA256,
        pattern=r"^[0-9a-f]{64}$",
        alias="protocolSha256",
    )
    controlled_run_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        alias="controlledRunId",
    )
    target_judgments: int = Field(
        default=WORKLOAD_TARGET["total_presentations"],
        ge=12,
        le=2000,
        alias="targetJudgments",
    )
    acknowledgements: list[str] = Field(min_length=7, max_length=7)

    @field_validator("acknowledgements")
    @classmethod
    def validate_acknowledgement_set(cls, value: list[str]) -> list[str]:
        return validate_acknowledgements(value)


class ExpertInviteCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    reviewer_code: str = Field(min_length=3, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    qualified_families: list[TaskFamily] = Field(min_length=1, max_length=4)
    qualification_reference: str = Field(min_length=3, max_length=512)
    qualification_verified: bool
    affiliation_class: str = Field(
        pattern=r"^(independent_external|product_affiliated|provider_affiliated)$"
    )
    conflict_disclosure_reference: str = Field(min_length=3, max_length=512)
    consent_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_accuracy: float = Field(ge=0, le=1)
    compensation_reference: str = Field(min_length=3, max_length=512)

    @field_validator("qualified_families")
    @classmethod
    def unique_families(cls, value: list[TaskFamily]) -> list[TaskFamily]:
        if len(value) != len(set(value)):
            raise ValueError("qualified_families must not contain duplicates")
        return value

    @model_validator(mode="after")
    def enforce_qualification(self) -> ExpertInviteCreate:
        if self.qualification_verified and self.calibration_accuracy < 0.8:
            raise ValueError("verified experts require at least 0.8 calibration accuracy")
        return self


class ParticipantEnrollmentOfferCreate(BaseModel):
    """Admin-issued, identity-free invitation to review one exact consent."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    consent_document_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="consentDocumentSha256",
    )
    ttl_seconds: int = Field(default=3600, ge=300, le=86_400, alias="ttlSeconds")


class ParticipantConsentAcceptanceCreate(BaseModel):
    """Participant-owned confirmation over exact consent and activation hashes."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    consent_document_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="consentDocumentSha256",
    )
    activation_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="activationManifestSha256",
    )
    confirmations: list[
        Literal[
            "participation_is_voluntary",
            "exact_consent_document_read",
            "withdrawal_and_post_release_limits_understood",
            "retention_and_deletion_schedule_understood",
        ]
    ] = Field(min_length=4, max_length=4)
    idempotency_key: str = Field(
        min_length=8,
        max_length=160,
        alias="idempotencyKey",
    )


class ParticipantIdentityEnrollmentCreate(BaseModel):
    """Transient identity input accepted only after participant consent."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    identity_issuer: str = Field(min_length=3, max_length=512, alias="identityIssuer")
    issuer_subject: SecretStr = Field(alias="issuerSubject")
    identity_evidence_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        alias="identityEvidenceSha256",
    )
    roles: list[
        Literal["task_author", "task_validator", "task_adjudicator", "output_rater"]
    ] = Field(min_length=1, max_length=4)
    qualified_families: list[TaskFamily] = Field(
        min_length=1,
        max_length=4,
        alias="qualifiedFamilies",
    )
    affiliation_class: Literal[
        "independent_external", "product_affiliated", "provider_affiliated"
    ] = Field(alias="affiliationClass")

    @field_validator("roles", "qualified_families")
    @classmethod
    def unique_participant_scopes(cls, value: list[Any]) -> list[Any]:
        if len(value) != len(set(value)):
            raise ValueError("participant reviewer scopes must not contain duplicates")
        return value


class ParticipantWithdrawalCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    idempotency_key: str = Field(
        min_length=8,
        max_length=160,
        alias="idempotencyKey",
    )
    reason_code: Literal[
        "voluntary_withdrawal", "privacy_request", "safety_concern"
    ] = Field(default="voluntary_withdrawal", alias="reasonCode")


class ReviewerRetentionScheduleCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    analysis_freeze_at: datetime = Field(alias="analysisFreezeAt")
    first_public_release_at: datetime = Field(alias="firstPublicReleaseAt")

    @field_validator("analysis_freeze_at", "first_public_release_at")
    @classmethod
    def require_utc_deadline(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retention schedule timestamps must carry a UTC offset")
        return value


class ReviewerPrivatePayloadDeletionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    idempotency_key: str = Field(
        min_length=8,
        max_length=160,
        alias="idempotencyKey",
    )


class ReviewerIdentityBindingCreate(BaseModel):
    """Raw issuer subjects are accepted once and must never be persisted."""

    model_config = ConfigDict(populate_by_name=True)

    identity_issuer: str = Field(min_length=3, max_length=512, alias="identityIssuer")
    issuer_subject: SecretStr = Field(alias="issuerSubject")
    identity_evidence_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="identityEvidenceSha256"
    )
    roles: list[
        Literal["task_author", "task_validator", "task_adjudicator", "output_rater"]
    ] = Field(min_length=1, max_length=4)

    @field_validator("roles")
    @classmethod
    def unique_reviewer_roles(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("reviewer roles must not contain duplicates")
        return value


class ReviewerQualificationEvidenceCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    family: TaskFamily
    affiliation_class: Literal[
        "independent_external", "product_affiliated", "provider_affiliated"
    ] = Field(alias="affiliationClass")
    independence_verified: bool = Field(alias="independenceVerified")
    conflict_cleared: bool = Field(alias="conflictCleared")
    qualification_evidence_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="qualificationEvidenceSha256"
    )
    independence_evidence_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="independenceEvidenceSha256"
    )
    conflict_disclosure_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="conflictDisclosureSha256"
    )
    consent_document_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="consentDocumentSha256"
    )
    training_material_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="trainingMaterialSha256"
    )
    verifier_principal_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="verifierPrincipalSha256"
    )
    verified_at: datetime = Field(alias="verifiedAt")
    valid_until: datetime | None = Field(default=None, alias="validUntil")


class ReviewerCalibrationSetCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    family: TaskFamily
    calibration_set_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="calibrationSetSha256"
    )
    source_artifact_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="sourceArtifactSha256"
    )
    scoring_key_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="scoringKeySha256"
    )
    item_count: int = Field(ge=1, le=500, strict=True, alias="itemCount")
    real_source_arms: int = Field(ge=2, le=2000, strict=True, alias="realSourceArms")
    frozen_at: datetime = Field(alias="frozenAt")

    @model_validator(mode="after")
    def require_two_real_arms_per_calibration_item(self) -> ReviewerCalibrationSetCreate:
        if self.real_source_arms < self.item_count * 2:
            raise ValueError("calibration sets require two real source arms per item")
        return self


class ReviewerCalibrationBallotCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    calibration_set_id: str = Field(min_length=36, max_length=36, alias="calibrationSetId")
    ballot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="ballotSha256")
    scoring_result_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="scoringResultSha256"
    )
    correct_count: int = Field(ge=0, le=500, strict=True, alias="correctCount")
    minimum_accuracy_milli: int = Field(
        ge=0, le=1000, strict=True, alias="minimumAccuracyMilli"
    )
    completed_at: datetime = Field(alias="completedAt")


class ReviewerFamilyAdmissionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    family: TaskFamily
    review_role: Literal[
        "task_author", "task_validator", "task_adjudicator", "output_rater"
    ] = Field(alias="reviewRole")
    qualification_evidence_id: str = Field(
        min_length=36, max_length=36, alias="qualificationEvidenceId"
    )
    calibration_ballot_id: str | None = Field(
        default=None, min_length=36, max_length=36, alias="calibrationBallotId"
    )
    requires_calibration: bool = Field(alias="requiresCalibration")
    minimum_accuracy_milli: int = Field(
        ge=0, le=1000, strict=True, alias="minimumAccuracyMilli"
    )
    decision_reference_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="decisionReferenceSha256"
    )
    valid_from: datetime = Field(alias="validFrom")
    valid_until: datetime = Field(alias="validUntil")

    @model_validator(mode="after")
    def validate_family_admission_window(self) -> ReviewerFamilyAdmissionCreate:
        if self.valid_until <= self.valid_from:
            raise ValueError("reviewer admission validity window is empty")
        if self.requires_calibration and self.calibration_ballot_id is None:
            raise ValueError("calibration policy requires a ballot")
        return self


class ExpertAdmissionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    qualification_reference: str = Field(min_length=3, max_length=512)
    conflict_disclosure_reference: str = Field(min_length=3, max_length=512)
    consent_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_item_count: int = Field(ge=20, le=200, strict=True)
    calibration_gold_adjudicator_count: int = Field(ge=2, le=20, strict=True)
    calibration_accuracy: float = Field(ge=0.8, le=1)
    admission_decision_reference: str = Field(min_length=3, max_length=512)


def _validate_isolated_family_counts(
    primary_judgments: int,
    primary_by_family: dict[TaskFamily, StrictInt] | None,
) -> None:
    if primary_by_family is None:
        return
    normalized = {str(family): int(value) for family, value in primary_by_family.items()}
    if set(normalized) != set(TASK_FAMILIES):
        raise ValueError("primaryByFamily must contain each task family exactly once")
    if any(value < 2 for value in normalized.values()):
        raise ValueError("primaryByFamily requires at least two judgments per family")
    if sum(normalized.values()) != primary_judgments:
        raise ValueError("primaryByFamily must sum to primary_judgments")


class AuthorEvaluatorAdmissionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    qualification_reference: str = Field(min_length=3, max_length=512)
    conflict_disclosure_reference: str = Field(min_length=3, max_length=512)
    candidate_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_judgments: int = Field(ge=8, le=200, strict=True)
    primary_by_family: dict[TaskFamily, StrictInt] | None = Field(
        default=None,
        alias="primaryByFamily",
    )
    admission_decision_reference: str = Field(min_length=3, max_length=512)
    independent_validation_claim: bool = False

    @model_validator(mode="after")
    def prohibit_independence_claim(self) -> AuthorEvaluatorAdmissionCreate:
        if self.independent_validation_claim:
            raise ValueError("the author-evaluator pathway cannot claim independent validation")
        _validate_isolated_family_counts(self.primary_judgments, self.primary_by_family)
        return self


class AnonymousExternalAdmissionCreate(BaseModel):
    """Admit a pseudonymous external rater without manufacturing expert credentials."""

    model_config = ConfigDict(populate_by_name=True)

    candidate_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_judgments: int = Field(ge=8, le=200, strict=True)
    primary_by_family: dict[TaskFamily, StrictInt] | None = Field(
        default=None,
        alias="primaryByFamily",
    )
    admission_decision_reference: str = Field(min_length=3, max_length=512)
    identity_collection_prohibited: bool = True
    independence_self_attestation_required: bool = True
    qualification_self_attestation_required: bool = True

    @model_validator(mode="after")
    def enforce_anonymous_external_boundary(
        self,
    ) -> AnonymousExternalAdmissionCreate:
        if not (
            self.identity_collection_prohibited
            and self.independence_self_attestation_required
            and self.qualification_self_attestation_required
        ):
            raise ValueError(
                "anonymous external admission requires identity minimization and "
                "independence and qualification self-attestations"
            )
        _validate_isolated_family_counts(self.primary_judgments, self.primary_by_family)
        return self


class AnonymousExternalReconsentCreate(BaseModel):
    """Record reviewer acceptance for one exact response-pool activation."""

    model_config = ConfigDict(populate_by_name=True)

    candidate_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pool_activation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    consent_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    consent_statement_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    voluntary_participation_accepted: bool = Field(strict=True)
    pool_specific_consent_accepted: bool = Field(strict=True)

    @model_validator(mode="after")
    def require_affirmative_reconsent(self) -> AnonymousExternalReconsentCreate:
        if not (self.voluntary_participation_accepted and self.pool_specific_consent_accepted):
            raise ValueError("anonymous external re-consent requires both acceptances")
        return self


class ExpertCalibrationCandidateRegister(BaseModel):
    candidate_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_commitment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_class: Literal[
        "paid_real_legacy_pilot_quarantined_from_season1",
        "paid_real_frontier_pilot_development",
    ]
    candidate_pack_reference: str = Field(min_length=3, max_length=512)
    candidate_pairs: int = Field(ge=20, le=200, strict=True)
    candidate_pairs_by_family: dict[str, int] = Field(min_length=4, max_length=4)
    source_arms: int = Field(ge=40, le=400, strict=True)
    real_provider_calls: int = Field(ge=40, strict=True)
    real_epicure_calls: int = Field(ge=20, strict=True)
    successful_real_epicure_calls: int = Field(ge=20, strict=True)
    synthetic_arms: int = Field(ge=0, le=0, strict=True)
    rank_eligible: bool
    status: str = Field(pattern=r"^candidate_pending_independent_gold_adjudication$")

    @model_validator(mode="after")
    def enforce_candidate_evidence(self) -> ExpertCalibrationCandidateRegister:
        expected_families = {family.value for family in TaskFamily}
        if set(self.candidate_pairs_by_family) != expected_families:
            raise ValueError("candidate pairs must cover the four registered families")
        if sum(self.candidate_pairs_by_family.values()) != self.candidate_pairs:
            raise ValueError("candidate family counts must sum to candidate_pairs")
        if min(self.candidate_pairs_by_family.values()) < 5:
            raise ValueError("each family requires at least five candidate pairs")
        if self.source_arms != self.candidate_pairs * 2:
            raise ValueError("every candidate must bind exactly two source arms")
        if self.successful_real_epicure_calls > self.real_epicure_calls:
            raise ValueError("successful Epicure calls cannot exceed total Epicure calls")
        if self.rank_eligible:
            raise ValueError("calibration candidates cannot be rank eligible")
        return self


class ReleaseReviewCreate(BaseModel):
    status: str = Field(pattern=r"^(approved|rejected)$")
    review_reference: str = Field(min_length=3, max_length=240)


class TaskRegistryFreezeCreate(BaseModel):
    task_hashes: dict[str, str] = Field(
        min_length=CONFIRMATORY_TASK_COUNT,
        max_length=CONFIRMATORY_TASK_COUNT,
    )
    review_reference: str = Field(min_length=3, max_length=240)


class EndpointDecodingCreate(BaseModel):
    max_tokens: int = Field(gt=0, le=1_000_000, strict=True)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, strict=True)


class EndpointRateCardCreate(BaseModel):
    currency: str = Field(default="USD", pattern=r"^USD$")
    prompt_price_per_token: str = Field(min_length=1, max_length=40)
    completion_price_per_token: str = Field(min_length=1, max_length=40)
    request_price: str = Field(default="0", min_length=1, max_length=40)
    internal_reasoning_price_per_token: str = Field(default="0", min_length=1, max_length=40)
    input_cache_read_price_per_token: str = Field(default="0", min_length=1, max_length=40)
    input_cache_write_price_per_token: str = Field(default="0", min_length=1, max_length=40)
    input_cache_write_1h_price_per_token: str = Field(default="0", min_length=1, max_length=40)
    image_price_per_unit: str = Field(default="0", min_length=1, max_length=40)
    web_search_price_per_request: str = Field(default="0", min_length=1, max_length=40)
    context_length: int = Field(gt=0, le=10_000_000, strict=True)
    pricing_source_uri: str = Field(
        min_length=12,
        max_length=1000,
        pattern=r"^https://",
    )
    pricing_source_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pricing_observed_at: datetime

    @model_validator(mode="after")
    def validate_prices(self) -> EndpointRateCardCreate:
        for field in (
            "prompt_price_per_token",
            "completion_price_per_token",
            "request_price",
            "internal_reasoning_price_per_token",
            "input_cache_read_price_per_token",
            "input_cache_write_price_per_token",
            "input_cache_write_1h_price_per_token",
            "image_price_per_unit",
            "web_search_price_per_request",
        ):
            try:
                price = Decimal(getattr(self, field))
            except InvalidOperation as exc:
                raise ValueError(f"{field} must be a decimal USD amount") from exc
            if not price.is_finite() or price < 0:
                raise ValueError(f"{field} must be a finite non-negative amount")
        return self


class EndpointContractCreate(BaseModel):
    execution_backend: ExecutionBackend = ExecutionBackend.openrouter
    provider_slug: str = Field(min_length=2, max_length=120)
    expected_actual_model_id: str = Field(min_length=3, max_length=240)
    expected_actual_provider_slug: str = Field(min_length=2, max_length=160)
    supported_parameters: list[str] = Field(min_length=1, max_length=80)
    decoding: EndpointDecodingCreate
    # Zero is the database/API representation for provider metadata unknown.
    # A positive value is an asserted upstream ceiling and must be enforced.
    endpoint_max_completion_tokens: int = Field(ge=0, le=1_000_000, strict=True)
    endpoint_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_rate_card: EndpointRateCardCreate | None = None
    backend_contract: dict = Field(default_factory=dict)
    backend_contract_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_backend_contract_presence(self) -> EndpointContractCreate:
        contract_backends = {
            ExecutionBackend.bedrock,
            ExecutionBackend.kimi_direct,
            ExecutionBackend.qwencloud_direct,
        }
        if self.execution_backend in contract_backends and (
            not self.backend_contract or self.backend_contract_sha256 is None
        ):
            raise ValueError(
                "managed direct endpoints require a content-addressed backend contract"
            )
        if self.execution_backend not in contract_backends and (
            self.backend_contract or self.backend_contract_sha256 is not None
        ):
            raise ValueError(
                "backend_contract is reserved for managed direct endpoint identity"
            )
        return self

    @field_validator("provider_slug")
    @classmethod
    def validate_endpoint_tag(cls, value: str) -> str:
        if value != value.strip() or any(character.isspace() for character in value):
            raise ValueError("provider_slug must be the exact whitespace-free endpoint tag")
        return value

    @field_validator("supported_parameters")
    @classmethod
    def validate_supported_parameters(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("supported_parameters must not contain duplicates")
        if any(
            not item
            or any(
                not (character.islower() or character.isdigit() or character == "_")
                for character in item
            )
            for item in value
        ):
            raise ValueError("supported_parameters must contain lowercase API parameter names")
        return sorted(value)


class ModelSmokeArtifactCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="requestSha256")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="responseSha256")
    provider_request_id_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="providerRequestIdSha256"
    )
    generation_id: str = Field(min_length=1, max_length=200, alias="generationId")
    actual_model_id: str = Field(min_length=3, max_length=240, alias="actualModelId")
    actual_provider_slug: str = Field(min_length=2, max_length=160, alias="actualProviderSlug")
    tools_passed: bool = Field(alias="toolsPassed")
    structured_output_passed: bool = Field(alias="structuredOutputPassed")
    data_collection_denied: bool = Field(alias="dataCollectionDenied")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="schemaSha256")
    tool_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="toolSchemaSha256")
    tool_trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="toolTraceSha256")
    structured_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="structuredOutputSha256")
    cost_micros: int = Field(ge=0, le=1_000_000_000, alias="costMicros")
    cost_reconciled: bool = Field(alias="costReconciled")
    completed_at: str = Field(min_length=20, max_length=40, alias="completedAt")


class ModelSmokeCreate(EndpointContractCreate):
    tools_passed: bool
    structured_output_passed: bool
    data_collection_denied: bool
    zdr_compatible: bool = False
    evidence_reference: str = Field(min_length=3, max_length=240)
    evidence_artifact: ModelSmokeArtifactCreate
    evidence_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ManifestEntry(EndpointContractCreate):
    model_id: str = Field(min_length=3, max_length=200)
    slot_role: str = Field(pattern=r"^(closed_family|open_weight|efficiency|reasoning)$")
    worst_case_cost_micros: int = Field(ge=0, le=100_000_000)

    @model_validator(mode="after")
    def validate_executable_endpoint_contract(self) -> ManifestEntry:
        supported = set(self.supported_parameters)
        missing = REQUIRED_ENDPOINT_PARAMETERS - supported
        if missing:
            raise ValueError(
                "season endpoint is missing required parameters: " + ", ".join(sorted(missing))
            )
        decoding = self.decoding.model_dump(exclude_none=True)
        unsupported_decoding = set(decoding) - supported - DECODING_PARAMETERS
        if unsupported_decoding:
            raise ValueError(
                "decoding contains unknown parameters: " + ", ".join(sorted(unsupported_decoding))
            )
        not_supported = set(decoding) - supported
        if not_supported:
            raise ValueError(
                "decoding parameters are not supported by the endpoint: "
                + ", ".join(sorted(not_supported))
            )
        if (
            self.endpoint_max_completion_tokens > 0
            and self.decoding.max_tokens > self.endpoint_max_completion_tokens
        ):
            raise ValueError("max_tokens exceeds the frozen endpoint completion limit")
        return self


class ProviderBudgetAuthorizationCreate(BaseModel):
    execution_backend: ExecutionBackend
    currency: str = Field(default="USD", pattern=r"^USD$")
    budget_cap_micros: int = Field(gt=0, le=10_000_000_000, strict=True)
    account_scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    account_authorization_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    valid_until: datetime


class ProviderOpeningBalanceSourceCreate(BaseModel):
    source_kind: str = Field(
        pattern=(
            r"^(bedrock_smoke_ledger|bedrock_b2_ledger|season0_legacy_ledger|"
            r"openrouter_generation_export|kimi_usage_export|provider_billing_export|"
            r"initial_zero_balance_authorization)$"
        )
    )
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    governed_used_micros: int = Field(ge=0, le=10_000_000_000, strict=True)
    governed_reserved_micros: int = Field(ge=0, le=10_000_000_000, strict=True)


class ProviderCredentialBindingCreate(BaseModel):
    binding_kind: str = Field(
        pattern=(
            r"^(bedrock_control_plane_v1|openrouter_account_endpoint_v1|"
            r"kimi_catalog_endpoint_v1|qwencloud_catalog_endpoint_v1)$"
        )
    )
    evidence_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    credential_scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_arn_sha256s: list[str] = Field(default_factory=list, max_length=128)
    observed_at: datetime

    @field_validator("target_arn_sha256s")
    @classmethod
    def validate_target_arn_sha256s(cls, values: list[str]) -> list[str]:
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in values
        ):
            raise ValueError("target ARN identities must be lowercase SHA-256 values")
        if len(set(values)) != len(values):
            raise ValueError("target ARN identities must be unique")
        return values


class ProviderAccountAuthorizationCreate(BaseModel):
    execution_backend: ExecutionBackend
    currency: str = Field(default="USD", pattern=r"^USD$")
    budget_cap_micros: int = Field(gt=0, le=5_000_000_000, strict=True)
    opening_balance_sources: list[ProviderOpeningBalanceSourceCreate] = Field(
        min_length=1,
        max_length=32,
    )
    credential_binding: ProviderCredentialBindingCreate
    authorization_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supersedes_authorization_envelope_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    valid_until: datetime


class ProviderAccountAuthorizationRevokeCreate(BaseModel):
    authorization_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    revocation_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BedrockBillingCrosscheckCreate(BaseModel):
    arm_ids: list[str] = Field(min_length=1, max_length=20_000)
    source_kind: str = Field(pattern=r"^(aws_cur|aws_data_export)$")
    source_artifact_uri: str = Field(
        min_length=12,
        max_length=1000,
        pattern=r"^(s3|https)://",
    )
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    statement_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_request_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage_start: datetime
    coverage_end: datetime
    billed_usage_micros: int = Field(
        ge=0,
        le=9_000_000_000_000_000_000,
        strict=True,
    )
    credits_policy: str = Field(pattern=r"^gross_usage_before_credits_excluding_tax$")
    authorization_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supersedes_crosscheck_id: str | None = Field(default=None, min_length=36, max_length=36)

    @model_validator(mode="after")
    def validate_billing_crosscheck(self) -> BedrockBillingCrosscheckCreate:
        if len(set(self.arm_ids)) != len(self.arm_ids):
            raise ValueError("billing crosscheck arm IDs must be unique")
        if not self.source_artifact_uri.isascii() or any(
            character.isspace() for character in self.source_artifact_uri
        ):
            raise ValueError("billing artifact URI must use ASCII URI encoding without whitespace")
        if (
            self.coverage_start.tzinfo is None
            or self.coverage_end.tzinfo is None
            or self.coverage_end <= self.coverage_start
        ):
            raise ValueError("billing coverage must be an ordered timezone-aware interval")
        return self


class SeasonFreezeCreate(BaseModel):
    models: list[ManifestEntry] = Field(
        min_length=SEASON_MODEL_COUNT,
        max_length=SEASON_MODEL_COUNT,
    )
    epicure_release_id: str = Field(min_length=3, max_length=160)
    epicure_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    epicure_application_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_cap_micros: int = Field(gt=0, le=10_000_000_000)
    lineage_reference: str = Field(min_length=3, max_length=240)
    budget_authorization_reference: str = Field(min_length=3, max_length=240)
    provider_budget_authorizations: list[ProviderBudgetAuthorizationCreate] = Field(
        default_factory=list,
        max_length=3,
    )


class EpicureReleaseRegisterCreate(BaseModel):
    release_id: str = Field(
        min_length=3,
        max_length=160,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]*$",
    )
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_release_uri: str = Field(
        min_length=12,
        max_length=1000,
        pattern=r"^https://",
    )
    release_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rights_clearance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_release_match: bool
    redistribution_rights_cleared: bool
    reproducibility_verified: bool


class SeasonOfficializeCreate(BaseModel):
    gate_a_decision_reference: str = Field(min_length=3, max_length=240)
    privacy_review_reference: str = Field(min_length=3, max_length=240)
    security_review_reference: str = Field(min_length=3, max_length=240)
    expert_access_reference: str = Field(min_length=3, max_length=240)
    task_registry_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    statistical_approval_reference: str = Field(min_length=3, max_length=240)
    reproducibility_approval_reference: str = Field(min_length=3, max_length=240)
    data_steward_approval_reference: str = Field(min_length=3, max_length=240)


class PublicArm(BaseModel):
    side: str
    answer_markdown: str


class BattlePublic(BaseModel):
    battle_id: str
    status: str
    category: str
    prompt: str | None = None
    answers: list[PublicArm] = Field(default_factory=list)
    reveal: dict | None = None
    error: str | None = None


ORGANIZATION_API_SCOPES = frozenset(
    {
        "models:read",
        "models:submit",
        "orders:read",
        "orders:create",
        "orders:cancel",
        "bundles:read",
        "keys:manage",
    }
)
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "client_secret",
        "cookie",
        "password",
        "private_key",
        "secret",
        "token",
    }
)


def _reject_secret_material(value: object, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if (
                normalized in _SECRET_FIELD_NAMES
                or normalized.endswith("_api_key")
                or normalized.endswith("_password")
                or normalized.endswith("_private_key")
                or normalized.endswith("_secret")
            ):
                raise ValueError(f"secret-like field is not accepted at {path}.{key}")
            _reject_secret_material(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_material(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("sk-", "Bearer ", "-----BEGIN PRIVATE KEY-----")) or (
            len(stripped) >= 20 and stripped.startswith("AKIA") and stripped[4:].isalnum()
        ):
            raise ValueError(f"secret-like value is not accepted at {path}")


def _https_document_uri(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(character.isspace() for character in value)
        or not value.isascii()
    ):
        raise ValueError(
            "document URI must be an ASCII HTTPS URL without credentials, whitespace, or fragment"
        )
    return value


class ManagedRouteKind(StrEnum):
    managed_bedrock = "managed_bedrock"
    managed_openrouter = "managed_openrouter"


class OrganizationCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    slug: str = Field(min_length=3, max_length=80, pattern=r"^[a-z0-9][a-z0-9-]*$")
    legal_name: str = Field(min_length=2, max_length=240, alias="legalName")
    display_name: str = Field(min_length=2, max_length=160, alias="displayName")
    idp_tenant_reference: str = Field(min_length=8, max_length=240, alias="idpTenantReference")
    billing_reference: str | None = Field(
        default=None, min_length=8, max_length=240, alias="billingReference"
    )
    data_region: str = Field(
        default="eu",
        min_length=2,
        max_length=32,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
        alias="dataRegion",
    )
    retention_policy: dict = Field(alias="retentionPolicy")
    activate: bool = False

    @field_validator("retention_policy")
    @classmethod
    def validate_retention_policy(cls, value: dict) -> dict:
        _reject_secret_material(value)
        if not value:
            raise ValueError("retentionPolicy must be explicit")
        return value


class OrganizationApiKeyCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    label: str = Field(min_length=2, max_length=120)
    scopes: list[str] = Field(min_length=1, max_length=len(ORGANIZATION_API_SCOPES))
    created_by_principal_reference: str = Field(
        min_length=8,
        max_length=240,
        alias="createdByPrincipalReference",
    )
    expires_at: datetime = Field(alias="expiresAt")
    rate_limit_profile: str = Field(
        default="standard",
        min_length=3,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        alias="rateLimitProfile",
    )
    network_policy: dict = Field(default_factory=dict, alias="networkPolicy")

    @model_validator(mode="after")
    def validate_key_contract(self) -> OrganizationApiKeyCreate:
        if len(self.scopes) != len(set(self.scopes)):
            raise ValueError("scopes must be unique")
        unknown = set(self.scopes) - ORGANIZATION_API_SCOPES
        if unknown:
            raise ValueError("unsupported scopes: " + ", ".join(sorted(unknown)))
        if self.expires_at.tzinfo is None:
            raise ValueError("expiresAt must be timezone-aware")
        _reject_secret_material(self.network_policy)
        return self


class GovernanceAcceptanceCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    agreement_type: str = Field(
        min_length=3,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_:-]*$",
        alias="agreementType",
    )
    agreement_version: str = Field(min_length=1, max_length=80, alias="agreementVersion")
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="documentSha256")
    external_envelope_reference: str = Field(
        min_length=8,
        max_length=240,
        alias="externalEnvelopeReference",
    )
    signatory_principal_reference: str = Field(
        min_length=8,
        max_length=240,
        alias="signatoryPrincipalReference",
    )
    authority_basis: str = Field(min_length=3, max_length=160, alias="authorityBasis")
    accepted_at: datetime = Field(alias="acceptedAt")
    expires_at: datetime | None = Field(default=None, alias="expiresAt")
    model_submission_id: str | None = Field(
        default=None, min_length=36, max_length=36, alias="modelSubmissionId"
    )
    route_revision_id: str | None = Field(
        default=None, min_length=36, max_length=36, alias="routeRevisionId"
    )
    evaluation_order_id: str | None = Field(
        default=None, min_length=36, max_length=36, alias="evaluationOrderId"
    )
    binding: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_acceptance(self) -> GovernanceAcceptanceCreate:
        if self.accepted_at.tzinfo is None:
            raise ValueError("acceptedAt must be timezone-aware")
        if self.expires_at is not None and (
            self.expires_at.tzinfo is None or self.expires_at <= self.accepted_at
        ):
            raise ValueError("expiresAt must follow acceptedAt")
        subjects = (
            self.model_submission_id,
            self.route_revision_id,
            self.evaluation_order_id,
        )
        if sum(value is not None for value in subjects) > 1:
            raise ValueError("an acceptance may name at most one specific subject")
        _reject_secret_material(self.binding)
        return self


class ManagedRouteSubmissionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    route_kind: ManagedRouteKind = Field(alias="routeKind")
    managed_route_reference: str = Field(
        min_length=8, max_length=240, alias="managedRouteReference"
    )
    requested_model_id: str = Field(min_length=3, max_length=240, alias="requestedModelId")
    expected_actual_model_id: str = Field(
        min_length=3, max_length=240, alias="expectedActualModelId"
    )
    expected_actual_provider_slug: str = Field(
        min_length=2, max_length=160, alias="expectedActualProviderSlug"
    )
    supported_parameters: list[str] = Field(
        min_length=1, max_length=64, alias="supportedParameters"
    )
    decoding_bounds: dict = Field(alias="decodingBounds")
    endpoint_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="endpointDocumentSha256")
    data_policy: dict = Field(alias="dataPolicy")
    rate_card: dict = Field(alias="rateCard")

    @model_validator(mode="after")
    def validate_managed_route(self) -> ManagedRouteSubmissionCreate:
        if len(self.supported_parameters) != len(set(self.supported_parameters)):
            raise ValueError("supportedParameters must be unique")
        missing = REQUIRED_ENDPOINT_PARAMETERS - set(self.supported_parameters)
        if missing:
            raise ValueError(
                "managed route is missing required parameters: " + ", ".join(sorted(missing))
            )
        _reject_secret_material(self.decoding_bounds)
        _reject_secret_material(self.data_policy)
        _reject_secret_material(self.rate_card)
        return self


class ModelSubmissionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    display_name: str = Field(min_length=2, max_length=240, alias="displayName")
    publisher: str = Field(min_length=2, max_length=240)
    requested_canonical_model_id: str = Field(
        min_length=3,
        max_length=240,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
        alias="requestedCanonicalModelId",
    )
    exact_model_version: str = Field(min_length=1, max_length=240, alias="exactModelVersion")
    release_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", alias="releaseDate")
    model_card_uri: str = Field(min_length=12, max_length=1000, alias="modelCardUri")
    model_card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="modelCardSha256")
    license_uri: str = Field(min_length=12, max_length=1000, alias="licenseUri")
    license_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", alias="licenseDocumentSha256")
    capability_claims: dict = Field(alias="capabilityClaims")
    contamination_disclosure: dict = Field(alias="contaminationDisclosure")
    route: ManagedRouteSubmissionCreate
    supersedes_submission_id: str | None = Field(
        default=None,
        min_length=36,
        max_length=36,
        alias="supersedesSubmissionId",
    )

    @field_validator("model_card_uri", "license_uri")
    @classmethod
    def validate_document_uri(cls, value: str) -> str:
        return _https_document_uri(value)

    @model_validator(mode="after")
    def validate_submission(self) -> ModelSubmissionCreate:
        _reject_secret_material(self.capability_claims)
        _reject_secret_material(self.contamination_disclosure)
        return self


class ModelSubmissionDecisionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    decision: str = Field(pattern=r"^(approve|changes_requested|reject)$")
    decision_reference_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="decisionReferenceSha256"
    )
    season: str = Field(default="season-0", min_length=3, max_length=80)


class EvaluationOrderCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_submission_id: str = Field(min_length=36, max_length=36, alias="modelSubmissionId")
    route_revision_id: str = Field(min_length=36, max_length=36, alias="routeRevisionId")
    season: str = Field(default="season-0", min_length=3, max_length=80)
    evaluation_profile_id: str = Field(
        default="private-comparative-v1",
        pattern=r"^private-comparative-v1$",
        alias="evaluationProfileId",
    )
    requested_visibility: str = Field(
        default="private",
        pattern=r"^(private|public_candidate)$",
        alias="requestedVisibility",
    )
    budget_cap_micros: int = Field(gt=0, le=10_000_000_000, alias="budgetCapMicros")
    client_reference: str = Field(min_length=8, max_length=240, alias="clientReference")


class EvaluationOrderDecisionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    decision: str = Field(pattern=r"^(approve|reject)$")
    decision_reference_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="decisionReferenceSha256"
    )
    quote_reference_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        alias="quoteReferenceSha256",
    )
    forecast_cost_micros: int | None = Field(
        default=None, ge=0, le=10_000_000_000, alias="forecastCostMicros"
    )


class EvaluationOrderProvisionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provision_reference_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="provisionReferenceSha256"
    )


class OrganizationApiKeyRevokeCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    revocation_reference_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", alias="revocationReferenceSha256"
    )
