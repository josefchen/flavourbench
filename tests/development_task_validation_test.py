from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from flavourbench.development_task_validation import (
    DevelopmentTaskValidationError,
    build_validation_packet,
    verify_validation_packet,
)
from flavourbench.real_task_bank import sha256_json

ROOT = Path(__file__).resolve().parents[1]
TASK_BANK = (
    ROOT
    / "data/season0/frozen"
    / (
        "season0-real-task-bank-"
        "1ce969bdee4124fa44bab46a04feda2a0ebeddf4d37c49c0264b48b3833a4313.json"
    )
)
TASK_VALIDITY = (
    ROOT
    / "artifacts/season1/task-validity/development-v2"
    / (
        "development-task-validity-v2-"
        "5ffd81a44267291413bc8a638d15391ec2b51decdda270550f81ca17ec587846.json"
    )
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _packet() -> dict:
    return build_validation_packet(
        task_bank=_load(TASK_BANK),
        task_validity=_load(TASK_VALIDITY),
    )


def test_packet_is_balanced_and_contains_only_real_human_material() -> None:
    packet = _packet()

    assert packet["counts"]["tasks"] == 40
    assert packet["counts"]["per_family"] == {
        "substitution": 10,
        "composition": 10,
        "cookability": 10,
        "evidence": 10,
    }
    assert packet["counts"]["synthetic_tasks"] == 0
    assert packet["counts"]["human_questions"] == 40
    assert packet["counts"]["accepted_human_references"] == 40
    assert packet["counts"]["sealed_human_reviews"] == 0
    assert packet["review_policy"]["required_independent_reviewers_per_task"] == 3
    assert (
        packet["review_policy"]["assignment_policy"]
        == "unfinished-criteria-then-least-reviewed-hmac-tiebreak-v1"
    )
    assert packet["review_policy"]["unanimous_valid_completes_source_review"] is True
    assert packet["review_policy"]["adjudication_trigger"] == "any_nonunanimous_decision"
    assert (
        packet["statistics_policy"]["version"]
        == "three-label-null-safe-descriptive-statistics-v1"
    )
    assert packet["statistics_policy"]["missing_reviews_imputed"] is False
    assert packet["statistics_policy"]["undefined_metrics_remain_null"] is True
    assert packet["claim_boundary"]["packet_itself_is_human_validation_evidence"] is False


def test_packet_enforces_answer_blind_validity_before_reference_unlock() -> None:
    packet = _packet()

    for task in packet["tasks"]:
        assert task["blind_validity_stage"]["answer_visible"] is False
        assert task["blind_validity_stage"]["source_url_visible"] is False
        assert (
            task["sealed_human_reference_stage"]["unlock_condition"]
            == "blind_validity_decision_sealed"
        )
        assert task["independent_review_count"] == 0
        assert task["criteria_status"] == "not_authored"


def test_verifier_rejects_mutated_prompt_and_false_review_claim() -> None:
    packet = _packet()
    document = {**packet, "artifact_sha256": sha256_json(packet)}
    verify_validation_packet(document)

    mutated_prompt = copy.deepcopy(document)
    mutated_prompt["tasks"][0]["prompt"] += " altered"
    mutated_prompt["artifact_sha256"] = sha256_json(
        {key: value for key, value in mutated_prompt.items() if key != "artifact_sha256"}
    )
    with pytest.raises(DevelopmentTaskValidationError, match="prompt hash"):
        verify_validation_packet(mutated_prompt)

    false_claim = copy.deepcopy(document)
    false_claim["tasks"][0]["independent_review_count"] = 3
    false_claim["artifact_sha256"] = sha256_json(
        {key: value for key, value in false_claim.items() if key != "artifact_sha256"}
    )
    with pytest.raises(DevelopmentTaskValidationError, match="review claims"):
        verify_validation_packet(false_claim)


def test_builder_rejects_mutated_task_validity_selection() -> None:
    task_validity = _load(TASK_VALIDITY)
    task_validity["tasks"][0]["task_id"] = "fb-s0-substitution-999"

    with pytest.raises(DevelopmentTaskValidationError, match="content address"):
        build_validation_packet(task_bank=_load(TASK_BANK), task_validity=task_validity)
