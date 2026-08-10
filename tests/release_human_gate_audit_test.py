from __future__ import annotations

import json
from pathlib import Path

from flavourbench.release_human_gate_audit import (
    HUMAN_QA_SHA256,
    KIT_SHA256,
    RELEASE_BOUNDARY_SHA256,
    REVIEW_POOL_SHA256,
    TASK_PACKET_SHA256,
    TASK_STATUS_SHA256,
    build_audit,
    verify_audit,
    write_audit,
)


def test_current_release_and_human_evidence_fail_closed() -> None:
    audit = build_audit()

    assert audit["overall_decision"] == "NO_GO_OFFICIAL_RANKING"
    assert audit["epicure_release"]["decision"] == "NO_GO"
    assert audit["epicure_release"]["technical_public_input_reconstruction"] == "verified"
    assert audit["epicure_release"]["official_release"] is False
    assert audit["epicure_release"]["rank_eligible"] is False
    assert len(audit["epicure_release"]["rights_rows"]) == 11
    assert all(row["gate"] == "blocked" for row in audit["epicure_release"]["rights_rows"])
    assert audit["epicure_release"]["training_lineage"]["status"] == (
        "candidate_rejected_as_exact_lineage"
    )
    assert audit["epicure_release"]["external_reproduction"]["status"] == "missing"

    human = audit["human_rank_readiness"]
    assert human["decision"] == "NO_GO"
    assert human["current_development_task_packet"] == {
        "tasks": 40,
        "public_development_tasks": True,
        "complete_independent_reviews": 0,
        "distinct_independent_reviewers": 0,
        "human_criterion_packs": 0,
        "independently_validated_tasks": 0,
        "confirmatory_eligible": False,
        "rank_eligible": False,
    }
    assert human["current_real_output_review_pool"]["pairs"] == 43
    assert human["current_real_output_review_pool"]["quality_judgments"] == 0
    assert human["restricted_historical_qa"]["ranking_use"] is False
    assert human["prospective_official_task_bank"]["tasks"] == 240
    assert human["prospective_official_task_bank"]["distinct_people_per_task"] == 6
    assert human["prospective_expert_judgment_thresholds"][
        "distinct_independent_raters_per_comparison"
    ] == 2
    assert human["public_anonymity_boundary"][
        "self_attested_path_counts_as_verified_independent_expert"
    ] is False

    sources = audit["source_artifacts"]
    assert sources["public_reconstruction_kit"]["artifact_sha256"] == KIT_SHA256
    assert sources["release_boundary"]["artifact_sha256"] == RELEASE_BOUNDARY_SHA256
    assert sources["development_task_packet"]["artifact_sha256"] == TASK_PACKET_SHA256
    assert sources["development_task_status"]["artifact_sha256"] == TASK_STATUS_SHA256
    assert sources["restricted_human_qa"]["artifact_sha256"] == HUMAN_QA_SHA256
    assert sources["unjudged_real_output_review_pool"][
        "artifact_sha256"
    ] == REVIEW_POOL_SHA256
    assert audit["claim_boundary"]["provider_calls_made"] == 0
    assert audit["claim_boundary"]["epicure_calls_made"] == 0
    assert audit["claim_boundary"]["reviewers_invented"] == 0
    assert audit["claim_boundary"]["judgments_invented"] == 0


def test_audit_is_content_addressed_and_reproducible(tmp_path: Path) -> None:
    first = write_audit(tmp_path, build_audit())
    second = write_audit(tmp_path, build_audit())
    assert first == second
    document = json.loads(first.read_text(encoding="utf-8"))
    assert verify_audit(document)
    assert first.name == f"release-human-gate-audit-{document['artifact_sha256']}.json"


def test_audit_checklist_has_one_unambiguous_terminal_gate() -> None:
    checklist = build_audit()["minimum_execution_checklist"]
    assert [row["order"] for row in checklist] == list(range(1, 9))
    assert checklist[-1]["gate"] == "official_rank_release"
    assert "only after every prior gate" in checklist[-1]["action"]
