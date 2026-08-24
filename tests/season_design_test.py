import hashlib
from datetime import UTC, datetime

import pytest

from flavourbench.construct_blueprint import BLUEPRINT_SHA256, CONSTRUCT_CELLS
from flavourbench.schemas import (
    ConfirmatoryTaskBankCreate,
    ControlledRunCreate,
    SeasonFreezeCreate,
    TaskRegistryFreezeCreate,
)
from flavourbench.season_design import (
    CONFIRMATORY_TASK_COUNT,
    CONFIRMATORY_TASKS_PER_FAMILY,
    SEASON_MODEL_COUNT,
    SEASON_SLOT_ROLE_COUNTS,
    SEASON_TASK_SPLIT_COUNTS,
    SEASON_TASK_SPLIT_COUNTS_PER_FAMILY,
)
from flavourbench.task_evidence import (
    CONTAMINATION_AUDIT_SCHEMA_VERSION,
    TASK_EVIDENCE_IMPLEMENTATION_SHA256,
    VALIDATOR_CONTRACT_SCHEMA_VERSION,
    VALIDATOR_DSL_VERSION,
    canonical_sha256,
    normalized_prompt_sha256,
)


def _array_bounds(model: type, field: str) -> tuple[int, int]:
    schema = model.model_json_schema()
    definition = schema["properties"][field]
    return definition["minItems"], definition["maxItems"]


def _task_payloads() -> list[dict]:
    tasks: list[dict] = []
    ordinal = 0
    for family in ("substitution", "composition", "cookability", "evidence"):
        for cell in CONSTRUCT_CELLS[family]:
            splits = ["scored"] * 8 + ["development"] * 2 + ["private_reserve"] * 2
            difficulties = ["foundation"] * 4 + ["integrative"] * 4 + ["stress"] * 4
            for cell_ordinal, (split, difficulty) in enumerate(
                zip(splits, difficulties, strict=True)
            ):
                ordinal += 1
                prompt = (
                    f"{family} {cell} {difficulty} case {ordinal} asks about "
                    f"ingredient{ordinal} method{ordinal} constraint{ordinal} outcome{ordinal}."
                )
                prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
                observed_at = datetime(2026, 8, 1, tzinfo=UTC).isoformat()
                query_sha256 = normalized_prompt_sha256(prompt)
                executable = cell_ordinal < 5
                rules = (
                    [
                        {
                            "kind": "required_entity",
                            "ruleId": "required_entity",
                            "description": "The response must name the required ingredient.",
                            "aliases": [f"ingredient{ordinal}"],
                            "minimumMentions": 1,
                        }
                    ]
                    if executable
                    else []
                )
                fixtures = (
                    [
                        {
                            "fixtureId": "passes_rule",
                            "responseText": f"Use ingredient{ordinal}.",
                            "expectedRuleStatus": {"required_entity": "pass"},
                        },
                        {
                            "fixtureId": "fails_rule",
                            "responseText": "Use something else.",
                            "expectedRuleStatus": {"required_entity": "fail"},
                        },
                    ]
                    if executable
                    else []
                )
                tasks.append(
                    {
                        "publicId": f"fb-s1-{ordinal:03d}",
                        "family": family,
                        "split": split,
                        "prompt": prompt,
                        "constructBlueprintSha256": BLUEPRINT_SHA256,
                        "constructCellId": cell,
                        "difficultyTier": difficulty,
                        "humanAuthorId": f"author-{ordinal:03d}",
                        "sourceCandidateId": f"00000000-0000-4000-8000-{ordinal:012d}",
                        "candidateRecordSha256": f"{ordinal + 1000:064x}",
                        "independentReviews": [
                            {
                                "reviewerId": "reviewer-one",
                                "blindReviewEventSha256": "1" * 64,
                                "reconciliationEventSha256": "2" * 64,
                                "decision": "approve",
                                "independentOfAuthor": True,
                            },
                            {
                                "reviewerId": "reviewer-two",
                                "blindReviewEventSha256": "3" * 64,
                                "reconciliationEventSha256": "4" * 64,
                                "decision": "approve",
                                "independentOfAuthor": True,
                            },
                        ],
                        "adjudication": {
                            "adjudicatorReviewerId": "reviewer-adjudicator",
                            "adjudicationEventSha256": "5" * 64,
                            "criterionPackSha256": "6" * 64,
                            "decision": "approve",
                            "independentOfAuthorAndReviewers": True,
                        },
                        "validatorContract": {
                            "schemaVersion": VALIDATOR_CONTRACT_SCHEMA_VERSION,
                            "artifactSha256": f"{ordinal + 3000:064x}",
                            "taskPublicId": f"fb-s1-{ordinal:03d}",
                            "taskFamily": family,
                            "taskRevision": 1,
                            "promptSha256": prompt_sha256,
                            "objectiveScope": ("executable_subset" if executable else "human_only"),
                            "humanOnlyReason": (
                                None
                                if executable
                                else "This fixture reserves sensory quality for human review."
                            ),
                            "validatorDslVersion": VALIDATOR_DSL_VERSION,
                            "evaluatorImplementationSha256": (TASK_EVIDENCE_IMPLEMENTATION_SHA256),
                            "validatorContainerImageDigest": "sha256:" + "a" * 64,
                            "rules": rules,
                            "fixtures": fixtures,
                            "fixtureSetSha256": canonical_sha256(fixtures),
                            "status": "verified",
                            "verifiedAt": observed_at,
                            "verifierReviewerId": "reviewer-three",
                        },
                        "validatorContractReview": {
                            "reviewerId": "reviewer-three",
                            "evidenceType": "validator_contract",
                            "artifactSha256": f"{ordinal + 3000:064x}",
                            "verificationReceiptSha256": f"{ordinal + 6000:064x}",
                            "reviewEventSha256": f"{ordinal + 7000:064x}",
                            "decision": "approve",
                            "independentOfTaskRoles": True,
                        },
                        "reviewHistorySha256": "4" * 64,
                        "taskRecordSha256": f"{ordinal:064x}",
                        "contaminationAudit": {
                            "schemaVersion": CONTAMINATION_AUDIT_SCHEMA_VERSION,
                            "artifactSha256": f"{ordinal + 4000:064x}",
                            "taskPublicId": f"fb-s1-{ordinal:03d}",
                            "taskFamily": family,
                            "taskRevision": 1,
                            "promptSha256": prompt_sha256,
                            "normalizedPromptSha256": query_sha256,
                            "auditImplementationSha256": TASK_EVIDENCE_IMPLEMENTATION_SHA256,
                            "scanBundleSha256": "7" * 64,
                            "auditContainerImageDigest": "sha256:" + "a" * 64,
                            "methods": [
                                {
                                    "method": method,
                                    "implementationVersion": "schema-fixture-v1",
                                    "querySha256": query_sha256,
                                    "corpusSnapshotSha256": f"{index + 1:064x}",
                                    "resultSetSha256": f"{index + 101:064x}",
                                    "hitCount": 0,
                                    "completedAt": observed_at,
                                }
                                for index, method in enumerate(
                                    ("exact", "fuzzy", "ngram", "semantic", "web")
                                )
                            ],
                            "hits": [],
                            "conclusion": "pass",
                            "auditorReviewerId": "reviewer-four",
                            "observedAt": observed_at,
                        },
                        "contaminationAuditReview": {
                            "reviewerId": "reviewer-four",
                            "evidenceType": "contamination_audit",
                            "artifactSha256": f"{ordinal + 4000:064x}",
                            "verificationReceiptSha256": f"{ordinal + 8000:064x}",
                            "reviewEventSha256": f"{ordinal + 9000:064x}",
                            "decision": "approve",
                            "independentOfTaskRoles": True,
                        },
                        "taskEvidenceRootSha256": f"{ordinal + 5000:064x}",
                    }
                )
    return tasks


def test_prospective_season_design_is_16_models_and_240_tasks() -> None:
    assert SEASON_MODEL_COUNT == 16
    assert CONFIRMATORY_TASK_COUNT == 240
    assert CONFIRMATORY_TASKS_PER_FAMILY == 60
    assert SEASON_TASK_SPLIT_COUNTS == {
        "scored": 160,
        "development": 40,
        "private_reserve": 40,
    }
    assert SEASON_TASK_SPLIT_COUNTS_PER_FAMILY == {
        "scored": 40,
        "development": 10,
        "private_reserve": 10,
    }
    assert sum(SEASON_SLOT_ROLE_COUNTS.values()) == SEASON_MODEL_COUNT
    assert SEASON_SLOT_ROLE_COUNTS == {
        "closed_family": 4,
        "open_weight": 8,
        "efficiency": 2,
        "reasoning": 2,
    }


def test_production_schemas_bind_the_prospective_design() -> None:
    assert _array_bounds(ConfirmatoryTaskBankCreate, "tasks") == (240, 240)
    assert _array_bounds(SeasonFreezeCreate, "models") == (16, 16)
    assert _array_bounds(ControlledRunCreate, "modelIds") == (1, 16)
    task_hash_schema = TaskRegistryFreezeCreate.model_json_schema()["properties"]["task_hashes"]
    assert task_hash_schema["minProperties"] == 240
    assert task_hash_schema["maxProperties"] == 240


def test_confirmatory_bank_requires_balanced_frozen_splits() -> None:
    bank = ConfirmatoryTaskBankCreate.model_validate(
        {
            "tasks": _task_payloads(),
            "bankManifestSha256": "a" * 64,
            "validatorCalibrationArtifactSha256": "b" * 64,
            "contaminationCalibrationArtifactSha256": "c" * 64,
            "importReference": "sealed human task registry",
        }
    )

    assert len(bank.tasks) == 240
    assert sum(task.split.value == "scored" for task in bank.tasks) == 160
    assert sum(task.split.value == "development" for task in bank.tasks) == 40
    assert sum(task.split.value == "private_reserve" for task in bank.tasks) == 40


def _validate_task_bank(tasks: list[dict]) -> None:
    ConfirmatoryTaskBankCreate.model_validate(
        {
            "tasks": tasks,
            "bankManifestSha256": "a" * 64,
            "validatorCalibrationArtifactSha256": "b" * 64,
            "contaminationCalibrationArtifactSha256": "c" * 64,
            "importReference": "sealed human task registry",
        }
    )


def test_confirmatory_bank_rejects_construct_cell_collapse() -> None:
    tasks = _task_payloads()
    for task in tasks:
        task["constructCellId"] = CONSTRUCT_CELLS[task["family"]][0]

    with pytest.raises(ValueError, match="requires exactly 12 tasks"):
        _validate_task_bank(tasks)


def test_confirmatory_bank_rejects_author_concentration() -> None:
    tasks = _task_payloads()
    for task in tasks:
        task["humanAuthorId"] = "one-overrepresented-author"

    with pytest.raises(ValueError, match="insufficient author diversity"):
        _validate_task_bank(tasks)


def test_confirmatory_bank_rejects_insufficient_executable_coverage() -> None:
    tasks = _task_payloads()
    for task in tasks:
        contract = task["validatorContract"]
        contract["objectiveScope"] = "human_only"
        contract["humanOnlyReason"] = "No objective subset is defined for this test task."
        contract["rules"] = []
        contract["fixtures"] = []
        contract["fixtureSetSha256"] = canonical_sha256([])

    with pytest.raises(ValueError, match="executable-validator coverage floor"):
        _validate_task_bank(tasks)


def test_confirmatory_bank_rejects_near_duplicate_prompts() -> None:
    tasks = _task_payloads()
    shared = (
        "Design a practical weeknight plate using beans, greens, lemon, toasted seeds, "
        "and a creamy sauce while preserving contrast, timing, texture, and acidity"
    )
    tasks[0]["prompt"] = f"{shared} for guest alpha."
    tasks[1]["prompt"] = f"{shared} for guest beta."

    with pytest.raises(ValueError, match="near-duplicate prompts exceed"):
        _validate_task_bank(tasks)
