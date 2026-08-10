from __future__ import annotations

import json
from pathlib import Path

import pytest

from flavourbench.expert_calibration import (
    ACCEPTED_FINAL_FINISH_REASONS,
    BALLOT_SCHEMA_VERSION,
    BLINDING_LEAK_PATTERN,
    CURRENT_FRONTIER_TASK_QUARANTINE,
    CURRENT_FRONTIER_TASK_QUARANTINE_BINDING,
    MANUAL_RESPONSE_QUARANTINE,
    TASK_FAMILIES,
    ExpertCalibrationError,
    _artifact_document,
    _assert_governance_review_contracts,
    _eligible_arm,
    ballot_template,
    build_candidate_payload,
    freeze_gold_set,
    score_reviewer_ballot,
    sha256_json,
    sha256_text,
)


def _write_arm(
    directory: Path,
    *,
    family: str,
    task_number: int,
    model_number: int,
    condition: str,
    task_id_override: str | None = None,
) -> None:
    task_id = task_id_override or f"task-{family}-{task_number:02d}"
    model_id = f"endpoint-{model_number:02d}"
    arm_id = sha256_text(f"{task_id}|{model_id}|{condition}")
    prompt = f"Design a careful {family} answer for task {task_number}."
    if condition == "epicure_on":
        answer = (
            f"Response supported by measured ingredient relationships for {family} "
            f"task {task_number} and endpoint {model_number}."
        )
        trace = [
            {
                "name": "find_pairings",
                "arguments": {"ingredient": f"ingredient-{task_number}"},
                "arguments_sha256": "1" * 64,
                "is_error": False,
                "latency_ms": 12,
                "model_visible_result_sha256": "2" * 64,
                "result_sha256": "3" * 64,
                "round_index": 0,
            }
        ]
    else:
        answer = (
            f"Baseline culinary response for {family} task {task_number} "
            f"and endpoint {model_number}."
        )
        trace = []
    record = {
        "schema_version": "test-arm-v1",
        "arm_id": arm_id,
        "phase": "scored",
        "status": "success",
        "synthetic": False,
        "rank_eligible": True,
        "condition": condition,
        "model": {
            "season_model_id": model_id,
            "canonical_model_id": f"canonical/{model_id}",
            "display_name": f"Endpoint {model_number}",
            "provider": "test-provider",
            "requested_endpoint_id": f"route-{model_number}",
        },
        "task": {
            "family": family,
            "task_id": task_id,
            "task_sha256": sha256_text(f"task|{prompt}"),
            "prompt_sha256": sha256_text(prompt),
            "prompt": prompt,
        },
        "result": {
            "answer_markdown": answer,
            "finish_reason": "stop",
            "provider_calls": 1,
            "real_epicure_calls": len(trace),
            "request_id_sha256s": ["4" * 64],
            "request_payload_sha256s": ["5" * 64],
            "returned_model_ids": [],
            "actual_provider_name": "Test provider",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
            "tool_trace": trace,
        },
    }
    digest = sha256_json(record)
    record["artifact_sha256"] = digest
    path = directory / f"arm-{arm_id}-{digest}.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _arms(directory: Path) -> None:
    directory.mkdir(parents=True)
    for family in TASK_FAMILIES:
        for index in range(8):
            for condition in ("epicure_on", "epicure_off"):
                _write_arm(
                    directory,
                    family=family,
                    task_number=index,
                    model_number=index,
                    condition=condition,
                )


def _gold_ballot(candidate: dict, reviewer_code: str) -> dict:
    ballot = ballot_template(candidate, candidate["artifact_sha256"])
    ballot["role"] = "independent_gold_adjudicator"
    ballot["reviewer"] = {
        "reviewer_code": reviewer_code,
        "affiliation": f"Independent kitchen {reviewer_code}",
    }
    ballot["attestations"] = {
        "worked_independently": True,
        "no_model_or_condition_identity_lookup": True,
        "reviewed_complete_unedited_answers": True,
        "independent_of_epicure_and_model_providers": True,
        "product_affiliation_disclosed": False,
    }
    for index, row in enumerate(ballot["items"]):
        row.update(
            {
                "task_validity": "valid",
                "choice": ("left", "right", "tie", "both_bad")[index % 4],
                "confidence": 4,
                "rationale": "The selected label follows the comparative culinary evidence.",
                "flags": [],
            }
        )
    return ballot


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def test_candidate_pack_uses_verified_real_tool_pairs_and_stays_blinded(
    tmp_path: Path,
) -> None:
    arms_dir = tmp_path / "arms"
    _arms(arms_dir)

    candidate, identity = build_candidate_payload(arms_dir)
    repeated, repeated_identity = build_candidate_payload(arms_dir)

    assert sha256_json(candidate) == sha256_json(repeated)
    assert identity == repeated_identity
    assert candidate["observed"] == {
        "candidate_pairs": 32,
        "candidate_pairs_by_family": {
            "composition": 8,
            "cookability": 8,
            "evidence": 8,
            "substitution": 8,
        },
        "unique_tasks": 32,
        "unique_tasks_by_family": {
            "substitution": 8,
            "composition": 8,
            "cookability": 8,
            "evidence": 8,
        },
        "unique_models": 8,
        "unique_models_by_family": {
            "substitution": 8,
            "composition": 8,
            "cookability": 8,
            "evidence": 8,
        },
        "source_arms": 64,
        "real_provider_calls": 64,
        "real_epicure_calls": 32,
        "successful_real_epicure_calls": 32,
        "synthetic_arms": 0,
        "quality_judgments": 0,
    }
    assert candidate["use_policy"]["rank_eligible"] is False
    assert candidate["selection_policy"][
        "current_frontier_task_quarantine_artifact_sha256"
    ] == CURRENT_FRONTIER_TASK_QUARANTINE_BINDING["artifact_sha256"]
    assert set(
        candidate["selection_policy"]["current_frontier_task_quarantine_task_ids"]
    ) == set(CURRENT_FRONTIER_TASK_QUARANTINE)
    assert all("model" not in item for item in candidate["items"])
    assert all("condition" not in item for item in candidate["items"])
    assert len(identity["items"]) == 32


def test_candidate_builder_excludes_current_frontier_quarantine(tmp_path: Path) -> None:
    arms_dir = tmp_path / "arms"
    _arms(arms_dir)
    held_task = sorted(CURRENT_FRONTIER_TASK_QUARANTINE)[0]
    for condition in ("epicure_on", "epicure_off"):
        _write_arm(
            arms_dir,
            family="composition",
            task_number=99,
            model_number=99,
            condition=condition,
            task_id_override=held_task,
        )

    candidate, _ = build_candidate_payload(arms_dir)
    assert candidate["selection_policy"]["excluded_current_frontier_task_pairs"] == 1
    assert held_task not in {item["task_id"] for item in candidate["items"]}


def test_two_independent_ballots_freeze_twenty_items_and_score_affiliated_reviewer(
    tmp_path: Path,
) -> None:
    arms_dir = tmp_path / "arms"
    _arms(arms_dir)
    candidate_payload, _ = build_candidate_payload(arms_dir)
    candidate, _ = _artifact_document(candidate_payload)
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate)
    ballot_a = _gold_ballot(candidate, "gold-a")
    ballot_b = _gold_ballot(candidate, "gold-b")
    ballot_a_path = tmp_path / "gold-a.json"
    ballot_b_path = tmp_path / "gold-b.json"
    _write_json(ballot_a_path, ballot_a)
    _write_json(ballot_b_path, ballot_b)

    frozen_path = tmp_path / "private" / "frozen.json"
    reviewer_pack_path = tmp_path / "reviewer-pack.json"
    reviewer_workspace_path = tmp_path / "reviewer.html"
    reviewer_template_path = tmp_path / "reviewer-template.json"
    result = freeze_gold_set(
        candidate_path=candidate_path,
        ballot_paths=[ballot_a_path, ballot_b_path],
        output_path=frozen_path,
        reviewer_pack_path=reviewer_pack_path,
        reviewer_workspace_path=reviewer_workspace_path,
        reviewer_ballot_template_path=reviewer_template_path,
    )

    assert result["items"] == 20
    assert result["items_by_family"] == {
        "composition": 5,
        "cookability": 5,
        "evidence": 5,
        "substitution": 5,
    }
    assert result["independent_gold_adjudicators"] == 2
    assert frozen_path.stat().st_mode & 0o777 == 0o600
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    reviewer_pack = json.loads(reviewer_pack_path.read_text(encoding="utf-8"))
    assert len(reviewer_pack["items"]) == 20
    assert "choice" not in json.dumps(reviewer_pack["items"])
    assert frozen["reviewer_pack_sha256"] == reviewer_pack["artifact_sha256"]
    assert frozen["current_frontier_task_quarantine"]["artifact_sha256"] == (
        CURRENT_FRONTIER_TASK_QUARANTINE_BINDING["artifact_sha256"]
    )
    assert reviewer_pack["current_frontier_task_quarantine"] == (
        frozen["current_frontier_task_quarantine"]
    )

    gold_by_id = {
        row["calibration_item_id"]: row["choice"] for row in frozen["items"]
    }
    reviewer_ballot = ballot_template(
        reviewer_pack,
        reviewer_pack["artifact_sha256"],
    )
    reviewer_ballot["role"] = "affiliated_reviewer_calibration"
    reviewer_ballot["reviewer"] = {
        "reviewer_code": "affiliated-reviewer",
        "affiliation": "Product team",
    }
    reviewer_ballot["attestations"] = {
        "worked_independently": True,
        "no_model_or_condition_identity_lookup": True,
        "reviewed_complete_unedited_answers": True,
        "independent_of_epicure_and_model_providers": False,
        "product_affiliation_disclosed": True,
    }
    for row in reviewer_ballot["items"]:
        row.update(
            {
                "task_validity": "valid",
                "choice": gold_by_id[row["calibration_item_id"]],
                "confidence": 4,
                "rationale": "This judgment matches the decisive culinary comparison.",
                "flags": [],
            }
        )
    reviewer_ballot_path = tmp_path / "reviewer-ballot.json"
    _write_json(reviewer_ballot_path, reviewer_ballot)
    score_path = tmp_path / "private" / "score.json"
    score = score_reviewer_ballot(
        reviewer_pack_path=reviewer_pack_path,
        frozen_path=frozen_path,
        ballot_path=reviewer_ballot_path,
        output_path=score_path,
    )
    assert score["items"] == 20
    assert score["accuracy"] == 1
    assert score["passed"] is True


def test_disagreement_cannot_be_silently_frozen(tmp_path: Path) -> None:
    arms_dir = tmp_path / "arms"
    _arms(arms_dir)
    candidate_payload, _ = build_candidate_payload(arms_dir)
    candidate, _ = _artifact_document(candidate_payload)
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate)
    ballot_a = _gold_ballot(candidate, "gold-a")
    ballot_b = _gold_ballot(candidate, "gold-b")
    ballot_b["items"][0]["choice"] = "tie"
    if ballot_a["items"][0]["choice"] == "tie":
        ballot_b["items"][0]["choice"] = "left"
    ballot_a_path = tmp_path / "gold-a.json"
    ballot_b_path = tmp_path / "gold-b.json"
    _write_json(ballot_a_path, ballot_a)
    _write_json(ballot_b_path, ballot_b)

    with pytest.raises(ExpertCalibrationError, match="require a third ballot"):
        freeze_gold_set(
            candidate_path=candidate_path,
            ballot_paths=[ballot_a_path, ballot_b_path],
            output_path=tmp_path / "frozen.json",
            reviewer_pack_path=tmp_path / "reviewer-pack.json",
            reviewer_workspace_path=tmp_path / "reviewer.html",
            reviewer_ballot_template_path=tmp_path / "reviewer-template.json",
        )


def test_ballot_schema_is_pinned() -> None:
    assert BALLOT_SCHEMA_VERSION == "flavourbench-expert-calibration-ballot-v1"


def test_non_normal_final_completion_is_never_calibration_eligible() -> None:
    base = {
        "phase": "scored",
        "status": "success",
        "synthetic": False,
        "rank_eligible": True,
        "condition": "epicure_on",
        "result": {"finish_reason": "stop"},
    }
    assert _eligible_arm(base) is True
    assert "length" not in ACCEPTED_FINAL_FINISH_REASONS
    assert _eligible_arm({**base, "result": {"finish_reason": "length"}}) is False
    assert _eligible_arm({**base, "result": {"finish_reason": "max_tokens"}}) is False
    assert _eligible_arm({**base, "result": {}}) is False


@pytest.mark.parametrize(
    "answer",
    (
        "<reasoning>We should inspect the request.</reasoning>",
        "Now craft the final answer for the user.",
        "Let's craft final answer with practical guidance.",
        "The pairing-score query returned a high result.",
        "The flavour tools do not cover texture.",
        "The model backs up the classic moves.",
    ),
)
def test_blinding_filter_rejects_scratchpad_tool_and_condition_disclosure(
    answer: str,
) -> None:
    assert BLINDING_LEAK_PATTERN.search(answer) is not None


def test_manual_response_quarantine_is_full_sha256_coordinate_set() -> None:
    assert len(MANUAL_RESPONSE_QUARANTINE) >= 12
    assert all(
        len(answer_sha256) == 64
        and set(answer_sha256).issubset("0123456789abcdef")
        and reason
        for answer_sha256, reason in MANUAL_RESPONSE_QUARANTINE.items()
    )


def test_final_candidate_coordinates_are_bound_to_governance_reviews() -> None:
    root = Path(__file__).resolve().parents[1]
    candidate = json.loads(
        (
            root
            / "artifacts"
            / "expert-calibration"
            / "candidate-v11"
            / (
                "candidate-pack-"
                "94e917b6c202eb49953f3a8c22f897301eaa7ffba47116b83c915d17a6850b69.json"
            )
        ).read_text(encoding="utf-8")
    )
    _assert_governance_review_contracts(candidate["items"])
    tampered = json.loads(json.dumps(candidate["items"]))
    tampered[0]["left"]["answer_sha256"] = "0" * 64
    with pytest.raises(
        ExpertCalibrationError,
        match="candidate coordinates differ from governance reviews",
    ):
        _assert_governance_review_contracts(tampered)
