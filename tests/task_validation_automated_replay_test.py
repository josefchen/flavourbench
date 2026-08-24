from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from flavourbench.prospective_task_acquisition import canonical_sha256
from flavourbench.task_validation_automated_replay import (
    PINNED_INPUTS,
    PINNED_REPLAY_PHYSICAL_SHA256,
    PINNED_REPLAY_SEMANTIC_SHA256,
    ReplayInputPaths,
    TaskValidationReplayError,
    build_replay_artifact,
    verify_pinned_replay,
    verify_replay_document,
    write_replay_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ReplayInputPaths.from_root(ROOT)
REPLAY_PATH = (
    ROOT
    / "artifacts/season1/task-validation-campaign-v6"
    / f"automated-replay-{PINNED_REPLAY_SEMANTIC_SHA256}.json"
)
SCHEMA_PATH = ROOT / "contracts/season1/task-validation-automated-replay-v1.schema.json"
RUNTIME_CONTRACT_PATH = ROOT / "contracts/season1/task-validation-automated-replay-runtime-v1.json"


def _document() -> dict:
    value = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_pinned_replay_rebuilds_byte_for_byte_and_is_schema_valid() -> None:
    document = _document()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    assert build_replay_artifact(INPUTS) == document
    verified = verify_pinned_replay(REPLAY_PATH, INPUTS)
    assert verified == {
        "artifact_sha256": PINNED_REPLAY_SEMANTIC_SHA256,
        "physical_sha256": PINNED_REPLAY_PHYSICAL_SHA256,
        "rights_automatedEvidenceVerified": True,
        "contamination_automatedEvidenceVerified": True,
        "contamination_automated_hit_candidate_ids": document["contamination_and_prompt_risk"][
            "automated_hit_candidate_ids"
        ],
        "human_decision_fields_must_remain_unchanged": True,
        "campaign_audit_passed": False,
        "task_bank_import_authorized": False,
        "rank_eligible": False,
        "contamination_free": False,
    }


def test_replay_covers_every_source_and_scheduled_record_without_overclaiming() -> None:
    document = _document()
    assert document["coverage"] == {
        "captured_source_records_verified": 1052,
        "scheduled_records_verified": 180,
        "scheduled_records_scanned": 180,
        "scheduled_records_by_family": {
            "composition": 45,
            "cookability": 45,
            "evidence": 45,
            "substitution": 45,
        },
        "rights_coverage_percent": 100,
        "scan_coverage_percent": 100,
        "scan_categories": [
            "public_source_exposure_baseline",
            "duplicate",
            "answer_leak",
            "self_resolution",
            "link",
            "visual",
            "specialist",
        ],
    }
    rights = document["rights"]
    scans = document["contamination_and_prompt_risk"]
    assert len(rights["records"]) == len({row["candidate_id"] for row in rights["records"]})
    assert len(rights["records"]) == 180
    assert all(all(row["checks"].values()) for row in rights["records"])
    assert rights["integrity_failure_count"] == 0
    assert rights["anomaly_candidate_ids"] == ["210d72af-ea8f-54ba-b231-5ced2e448195"]
    assert len(scans["records"]) == len({row["candidate_id"] for row in scans["records"]})
    assert len(scans["records"]) == 180
    assert all(row["public_source_exposure_baseline"] is True for row in scans["records"])
    assert scans["finding_count_by_category"] == {
        "link": 18,
        "self_resolution": 3,
        "visual": 3,
    }
    assert len(scans["automated_hit_candidate_ids"]) == 22
    assert scans["contamination_free"] is False
    assert document["limitations"] == {
        "public_source_contamination_limited_not_contamination_free": True,
        "model_training_membership_tested": False,
        "external_benchmark_corpus_tested": False,
        "external_web_search_performed": False,
        "raw_api_response_bodies_preserved": False,
        "raw_source_html_reconstructable": False,
        "recorded_html_hash_commitments_verified_for_shape_only": True,
        "source_revision_independently_refetched": False,
        "task_validity_established": False,
        "human_rights_decision_observed": False,
        "human_contamination_decision_observed": False,
        "release_go_issued": False,
    }


def test_seeded_human_handoff_matches_the_campaign_sampling_contract() -> None:
    handoff = _document()["human_audit_handoff"]
    assert handoff["rights"]["sample_seed_commitment_sha256"] == (
        "aae16a208727c7f64b4a89607929c783891afc580f7def0b6fdb4d6849ea49f7"
    )
    assert handoff["contamination"]["sample_seed_commitment_sha256"] == (
        "cbefa9086a14e25d388436ff118b06463f94cbf4cd0d46954f65e368414f1c9e"
    )
    assert len(handoff["rights"]["sample_candidate_ids"]) == 24
    assert len(handoff["rights"]["required_candidate_ids"]) == 24
    assert len(handoff["contamination"]["sample_candidate_ids"]) == 24
    assert len(handoff["contamination"]["anomaly_or_hit_candidate_ids"]) == 22
    assert len(handoff["contamination"]["required_candidate_ids"]) == 42
    assert handoff["replay_sets_human_decision"] is False


@pytest.mark.parametrize("role", sorted(PINNED_INPUTS))
def test_any_pinned_input_byte_mutation_fails_closed(tmp_path: Path, role: str) -> None:
    copied: dict[str, Path] = {}
    for input_role, source in INPUTS.as_mapping().items():
        destination = tmp_path / f"{input_role}.json"
        destination.write_bytes(source.read_bytes())
        copied[input_role] = destination
    mutated = copied[role]
    document = json.loads(mutated.read_text(encoding="utf-8"))
    document["mutation_probe"] = role
    mutated.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths = ReplayInputPaths(**copied)
    with pytest.raises(TaskValidationReplayError, match=f"{role} physical SHA-256 mismatch"):
        build_replay_artifact(paths)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("rights", "human_decision"), "pass"),
        (("contamination_and_prompt_risk", "contamination_free"), True),
        (("runtime_projection", "campaign_audit_passed"), True),
        (("claim_boundary", "rank_eligible"), True),
    ],
)
def test_rehashed_replay_policy_mutation_still_fails_deterministic_rebuild(
    field_path: tuple[str, str], replacement: object
) -> None:
    forged = copy.deepcopy(_document())
    forged[field_path[0]][field_path[1]] = replacement
    forged.pop("artifact_sha256")
    forged["artifact_sha256"] = canonical_sha256(forged)
    with pytest.raises(TaskValidationReplayError, match="deterministic rebuild"):
        verify_replay_document(forged, INPUTS)


def test_published_replay_cannot_be_substituted_by_a_symlink(tmp_path: Path) -> None:
    link = tmp_path / "replay.json"
    link.symlink_to(REPLAY_PATH)
    with pytest.raises(TaskValidationReplayError, match="symlinked"):
        verify_pinned_replay(link, INPUTS)


def test_writer_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    document = _document()
    first = write_replay_artifact(document, tmp_path)
    second = write_replay_artifact(document, tmp_path)
    assert first == second
    assert first.name == f"automated-replay-{PINNED_REPLAY_SEMANTIC_SHA256}.json"
    assert first.read_bytes() == REPLAY_PATH.read_bytes()


def test_runtime_handoff_binds_exact_verifier_schema_and_candidate_sets() -> None:
    contract = json.loads(RUNTIME_CONTRACT_PATH.read_text(encoding="utf-8"))
    document = _document()
    module_path = ROOT / contract["server_verifier"]["module_relative_path"]
    assert contract["campaign_sha256"] == PINNED_INPUTS["campaign"]["semantic_sha256"]
    assert contract["replay"]["semantic_sha256"] == PINNED_REPLAY_SEMANTIC_SHA256
    assert contract["replay"]["physical_sha256"] == PINNED_REPLAY_PHYSICAL_SHA256
    assert (
        contract["replay"]["json_schema_physical_sha256"]
        == hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()
    )
    assert (
        contract["server_verifier"]["module_physical_sha256"]
        == hashlib.sha256(module_path.read_bytes()).hexdigest()
    )
    assert contract["authorization_binding"]["rights"][
        "anomaly_candidate_ids_sha256"
    ] == canonical_sha256(document["rights"]["anomaly_candidate_ids"])
    assert contract["authorization_binding"]["contamination"][
        "automated_hit_candidate_ids_sha256"
    ] == canonical_sha256(document["contamination_and_prompt_risk"]["automated_hit_candidate_ids"])
    assert (
        contract["allowed_runtime_projection"]["human_decision_fields_must_remain_unchanged"]
        is True
    )
    assert contract["allowed_runtime_projection"]["campaign_audit_passed"] is False
    assert contract["claim_boundary"]["release_go_issued"] is False
