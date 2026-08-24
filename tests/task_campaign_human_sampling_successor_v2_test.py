from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from flavourbench.real_task_bank import sha256_json
from flavourbench.task_campaign_human_sampling_successor import (
    build_sampling_artifact as build_v1,
)
from flavourbench.task_campaign_human_sampling_successor import (
    materialize_sampling_frame as materialize_v1,
)
from flavourbench.task_campaign_human_sampling_successor_v2 import (
    DEFAULT_GO_PACKAGE_V1,
    DEFAULT_GO_PACKAGE_V2,
    DEFAULT_HUMAN_REVIEW,
    DEFAULT_READINESS_V1,
    DEFAULT_READINESS_V2,
    DEFAULT_STUDY_YAML,
    DEFAULT_SUPERSESSION,
    DEFAULT_V1_ARTIFACT,
    GO_PACKAGE_V1_PHYSICAL_SHA256,
    GO_PACKAGE_V2_PHYSICAL_SHA256,
    HUMAN_REVIEW_PHYSICAL_SHA256,
    READINESS_V1_PHYSICAL_SHA256,
    READINESS_V2_PHYSICAL_SHA256,
    STUDY_YAML_PHYSICAL_SHA256,
    SUPERSESSION_PHYSICAL_SHA256,
    V1_PHYSICAL_SHA256,
    V1_SEMANTIC_SHA256,
    HumanSamplingV2Error,
    build_sampling_artifact_v2,
    materialize_sampling_frame_v2,
    verify_sampling_artifact_v2,
    write_sampling_artifact_v2,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "artifacts/season1/human-judgment-sampling-v2-candidate"


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    return build_sampling_artifact_v2()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_bytes_and_historical_readiness_remain_immutable(
    artifact: dict[str, Any],
) -> None:
    assert _sha256(DEFAULT_V1_ARTIFACT) == V1_PHYSICAL_SHA256
    assert _sha256(DEFAULT_READINESS_V1) == READINESS_V1_PHYSICAL_SHA256
    v1_document = json.loads(DEFAULT_V1_ARTIFACT.read_text(encoding="utf-8"))
    assert v1_document["artifact_sha256"] == V1_SEMANTIC_SHA256
    assert build_v1() == v1_document

    supersession = artifact["supersession"]
    assert supersession["supersedes_semantic_sha256"] == V1_SEMANTIC_SHA256
    assert supersession["supersedes_physical_sha256"] == V1_PHYSICAL_SHA256
    assert supersession["v1_bytes_modified"] is False
    assert supersession["sampling_coordinate_changes"] == 0


def test_corrected_sources_and_both_checksum_packages_are_exact(
    artifact: dict[str, Any],
) -> None:
    assert _sha256(DEFAULT_HUMAN_REVIEW) == HUMAN_REVIEW_PHYSICAL_SHA256
    assert _sha256(DEFAULT_STUDY_YAML) == STUDY_YAML_PHYSICAL_SHA256
    assert _sha256(DEFAULT_READINESS_V2) == READINESS_V2_PHYSICAL_SHA256
    assert _sha256(DEFAULT_SUPERSESSION) == SUPERSESSION_PHYSICAL_SHA256
    assert _sha256(DEFAULT_GO_PACKAGE_V1) == GO_PACKAGE_V1_PHYSICAL_SHA256
    assert _sha256(DEFAULT_GO_PACKAGE_V2) == GO_PACKAGE_V2_PHYSICAL_SHA256
    assert artifact["checksum_verification"] == {
        "v1_package_preserved_and_verified": True,
        "v1_package_rows": 16,
        "v2_package_verified": True,
        "v2_package_rows": 27,
    }


def test_v2_inherits_every_v1_coordinate_and_identifier(
    artifact: dict[str, Any],
) -> None:
    v1_document = build_v1()
    v1_frame = materialize_v1(v1_document)
    v2_frame = materialize_sampling_frame_v2(artifact)
    assert v2_frame == v1_frame
    assert len(v2_frame["arena_comparisons"]) == 800
    assert len(v2_frame["uplift_comparisons"]) == 800
    assert len(v2_frame["primary_judgment_slots"]) == 3200
    assert len(v2_frame["concealed_repeat_presentations"]) == 400


def test_corrected_arithmetic_and_no_go_boundary_are_explicit(
    artifact: dict[str, Any],
) -> None:
    arithmetic = artifact["corrected_workload_arithmetic"]
    assert arithmetic["primary"] == {
        "arena_unique_comparisons": 800,
        "uplift_unique_comparisons": 800,
        "distinct_raters_per_comparison": 2,
        "judgments": 3200,
        "minutes_at_3_each": 9600,
        "hours_at_3_each": "160",
        "minutes_at_4_each": 12800,
        "hours_at_4_each": "640/3 (213.333...)",
    }
    assert arithmetic["reliability"]["repeat_presentations"] == 400
    assert arithmetic["complete_output_review"]["presentations"] == 3600
    assert arithmetic["validation_plus_complete_output_hours"] == {
        "minimum": "224",
        "maximum": "316",
    }
    assert arithmetic["with_20_percent_paid_time_hours"] == {
        "minimum": "268.8",
        "maximum": "379.2",
    }
    assert arithmetic[
        "illustrative_envelope_after_separate_10_percent_operational_reserve_eur"
    ] == {"minimum": "5913.60", "maximum": "10428.00", "authorized": False}

    boundary = artifact["claim_boundary"]
    for key in (
        "official",
        "rank_eligible",
        "calls_authorized",
        "model_calls_authorized",
        "epicure_calls_authorized",
        "human_contact_authorized",
        "human_judgment_collection_authorized",
        "compensation_or_spend_authorized",
        "quality_evidence_observed",
        "research_result",
        "paper_or_public_claim_authorized",
    ):
        assert boundary[key] is False
    assert boundary["quality_observations"] == 0
    assert boundary["human_judgments"] == 0
    assert artifact["validation_status"]["power_validated"] is False
    assert artifact["validation_status"]["ethics_approved"] is False
    assert artifact["validation_status"]["funding_approved"] is False


def test_source_and_rehashed_document_tampering_fail_closed(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    tampered_review = tmp_path / "human-review.md"
    shutil.copyfile(DEFAULT_HUMAN_REVIEW, tampered_review)
    tampered_review.write_text(
        tampered_review.read_text(encoding="utf-8").replace("3,200", "3,201", 1),
        encoding="utf-8",
    )
    with pytest.raises(HumanSamplingV2Error, match="physical digest mismatch"):
        build_sampling_artifact_v2(human_review_path=tampered_review)

    tampered = json.loads(json.dumps(artifact))
    tampered["claim_boundary"]["official"] = True
    body = {key: value for key, value in tampered.items() if key != "artifact_sha256"}
    tampered["artifact_sha256"] = sha256_json(body)
    with pytest.raises(HumanSamplingV2Error, match="differs from exact corrected successor"):
        verify_sampling_artifact_v2(tampered)


def test_writer_is_idempotent_but_existing_conflicts_fail_closed(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "idempotent"
    first = write_sampling_artifact_v2(artifact, output_dir)
    second = write_sampling_artifact_v2(artifact, output_dir)
    assert first == second
    assert json.loads(first.read_text(encoding="utf-8")) == artifact

    conflict_dir = tmp_path / "preexisting-conflict"
    conflict_dir.mkdir()
    conflict = conflict_dir / first.name
    conflict.write_text("{}\n", encoding="utf-8")
    with pytest.raises(HumanSamplingV2Error, match="existing final artifact conflicts"):
        write_sampling_artifact_v2(artifact, conflict_dir)


def test_racing_final_creation_uses_no_replace_and_fails_closed(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "racing-conflict"
    output_dir.mkdir()

    def racing_link(source: os.PathLike[str], destination: os.PathLike[str], **_: Any) -> None:
        del source
        Path(destination).write_text("racing conflict\n", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(HumanSamplingV2Error, match="appeared during no-replace publication"):
        write_sampling_artifact_v2(artifact, output_dir)
    assert not list(output_dir.glob(".human-sampling-v2-*"))

    module_source = (
        ROOT / "src/flavourbench/task_campaign_human_sampling_successor_v2.py"
    ).read_text(encoding="utf-8")
    assert "os.replace(" not in module_source
    assert "os.link(" in module_source


def test_checked_in_v2_candidate_exactly_matches_builder(artifact: dict[str, Any]) -> None:
    candidates = list(CANDIDATE_DIR.glob("human-judgment-sampling-v2-candidate-*.json"))
    assert len(candidates) == 1
    assert candidates[0].name == (
        f"human-judgment-sampling-v2-candidate-{artifact['artifact_sha256']}.json"
    )
    assert json.loads(candidates[0].read_text(encoding="utf-8")) == artifact
