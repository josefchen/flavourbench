"""Audit the Epicure release and human-evidence gates without network access.

The output is deliberately fail closed.  It records what the frozen artifacts prove,
what they do not prove, and the smallest evidence package that can change either gate.
It never upgrades same-operator reconstruction, development review, or self-attested
anonymous review into official ranking evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .development_task_validation import verify_validation_packet

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "flavourbench-release-human-gate-audit-v1"

PUBLIC_RECONSTRUCTION = (
    ROOT / "flavourbench/artifacts/season1/epicure-lineage/public-reconstruction"
)
KIT_SHA256 = "bc5b109840c3c7d468fdc050eff7b907b66a0bc3a31899bd4c39ea8bcb1ae4b6"
RECEIPT_SHA256 = "e1140807dfd2feb0286318f6ab6a8a60246273b06b6f06b14ca7baccf2750cef"
RELEASE_BOUNDARY_SHA256 = (
    "1757620645fabc9f758152315b7c09a802143f5790db58b5844b647f8ca22e59"
)
LINEAGE_AUDIT_SHA256 = "332a3e20e1de351307d2ece3d37275ac7599dcc14f42305cd1e77bd7621cdfaf"
TASK_PACKET_SHA256 = "c45023aee6cf8ff91437c08c16ae20498b2d025e9a0155aedd44898de1d7fbb1"
TASK_STATUS_SHA256 = "eca1cee283c9ae916f15fd819134b9dedece044ef8768d689423086c3860f7fd"
HUMAN_QA_SHA256 = "f1c262e075ccc73a4db0bb3c328e6c90d66d7d01eceaf54dfc9912e8c96e9fea"
REVIEW_POOL_SHA256 = "f4daaef029dfc46d739be479d601938eb75ee73d957b73bd5607762dc6a8e9b2"
STUDY_DESIGN_SHA256 = "7a63cfd6117338a3af16a422d5ee3458298fdc0ff2fd0abfe45fe851a7e54506"

KIT = PUBLIC_RECONSTRUCTION / f"epicure-public-reconstruction-kit-{KIT_SHA256}.tar.gz"
RECEIPT = PUBLIC_RECONSTRUCTION / (
    f"epicure-public-input-reconstruction-receipt-{RECEIPT_SHA256}.json"
)
RELEASE_BOUNDARY = PUBLIC_RECONSTRUCTION / (
    f"epicure-public-reconstruction-release-candidate-{RELEASE_BOUNDARY_SHA256}.json"
)
LINEAGE_AUDIT = ROOT / (
    "flavourbench/artifacts/season1/epicure-lineage/candidate-source-trace/"
    f"epicure-candidate-training-lineage-audit-{LINEAGE_AUDIT_SHA256}.json"
)
TASK_PACKET = ROOT / (
    "flavourbench/artifacts/season1/task-validity/human-validation-current-v2/"
    f"development-task-human-validation-v2-{TASK_PACKET_SHA256}.json"
)
TASK_STATUS = ROOT / (
    "flavourbench/artifacts/season1/task-validity/live-status-v1/"
    f"development-task-validation-status-{TASK_STATUS_SHA256}.json"
)
HUMAN_QA = ROOT / (
    "flavourbench/artifacts/season1/human-review/operational-qa/"
    f"restricted-operational-qa-{HUMAN_QA_SHA256}.json"
)
REVIEW_POOL = ROOT / (
    "flavourbench/artifacts/season1/current-quality-run/"
    "pilot-v24-required-epicure/review-pool/"
    f"required-frontier-review-pool-{REVIEW_POOL_SHA256}.json"
)
STUDY_DESIGN = ROOT / "flavourbench/contracts/season1/season1-study-design-v5.json"
DEFAULT_OUTPUT_DIR = ROOT / (
    "flavourbench/artifacts/season1/readiness/release-human-gate-v1"
)


class ReleaseHumanGateAuditError(RuntimeError):
    """A frozen input is absent, altered, or crosses its claim boundary."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_content_addressed(
    path: Path,
    *,
    expected: str,
    digest_field: str = "artifact_sha256",
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseHumanGateAuditError(f"frozen input must be a regular file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseHumanGateAuditError(f"cannot read frozen input: {path}") from exc
    if not isinstance(document, dict):
        raise ReleaseHumanGateAuditError(f"frozen input is not an object: {path}")
    claimed = document.get(digest_field)
    payload = {key: value for key, value in document.items() if key != digest_field}
    if claimed != expected or canonical_sha256(payload) != expected:
        raise ReleaseHumanGateAuditError(f"content address does not verify: {path}")
    return document


def _kit_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    if KIT.is_symlink() or not KIT.is_file() or file_sha256(KIT) != KIT_SHA256:
        raise ReleaseHumanGateAuditError("public reconstruction kit digest does not verify")
    try:
        with tarfile.open(KIT, mode="r:gz") as archive:
            members = archive.getnames()
            if len(members) != len(set(members)) or any("/" in name for name in members):
                raise ReleaseHumanGateAuditError("public reconstruction kit layout is unsafe")
            manifest_handle = archive.extractfile("KIT-MANIFEST.json")
            rights_handle = archive.extractfile("rights-boundary.json")
            data_handle = archive.extractfile("data-sources.json")
            if manifest_handle is None or rights_handle is None or data_handle is None:
                raise ReleaseHumanGateAuditError("public reconstruction kit is incomplete")
            manifest = json.load(manifest_handle)
            rights = json.load(rights_handle)
            data_sources = json.load(data_handle)
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise ReleaseHumanGateAuditError("public reconstruction kit cannot be verified") from exc
    if not all(isinstance(value, dict) for value in (manifest, rights, data_sources)):
        raise ReleaseHumanGateAuditError("public reconstruction control files are malformed")
    data_files = data_sources.get("files")
    if not isinstance(data_files, list) or len(data_files) != 11:
        raise ReleaseHumanGateAuditError(
            "public data-source inventory is not the frozen 11-file set"
        )
    if (
        manifest.get("technical_reconstruction_candidate") is not True
        or manifest.get("payload_redistribution_cleared") is not False
        or manifest.get("independent_reproduction") is not False
        or manifest.get("rank_eligible") is not False
        or rights.get("payload_redistribution_cleared") is not False
        or rights.get("rank_eligible") is not False
        or data_sources.get("redistribution_cleared") is not False
    ):
        raise ReleaseHumanGateAuditError("public reconstruction kit crosses its evidence boundary")
    return manifest, data_sources


def _source_rights_rows(data_sources: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = data_sources.get("files")
    assert isinstance(files, list)
    return [
        {
            "path": str(item["path"]),
            "sha256": str(item["sha256"]),
            "bytes": int(item["bytes"]),
            "origin_and_upstream_sources": "missing",
            "copyright_or_database_rights_holder": "missing",
            "license_identifier_and_text": "missing",
            "authority_to_publish_exact_bytes": "missing",
            "authority_to_redistribute_and_serve_derivatives": "missing",
            "required_attribution_or_restrictions": "missing",
            "signed_rights_evidence": "missing",
            "gate": "blocked",
        }
        for item in files
    ]


def build_audit() -> dict[str, Any]:
    """Build a deterministic audit of the current frozen evidence."""

    manifest, data_sources = _kit_evidence()
    receipt = _load_content_addressed(RECEIPT, expected=RECEIPT_SHA256)
    release = _load_content_addressed(RELEASE_BOUNDARY, expected=RELEASE_BOUNDARY_SHA256)
    lineage = _load_content_addressed(LINEAGE_AUDIT, expected=LINEAGE_AUDIT_SHA256)
    task_packet = _load_content_addressed(TASK_PACKET, expected=TASK_PACKET_SHA256)
    task_status = _load_content_addressed(
        TASK_STATUS,
        expected=TASK_STATUS_SHA256,
        digest_field="artifactSha256",
    )
    human_qa = _load_content_addressed(HUMAN_QA, expected=HUMAN_QA_SHA256)
    review_pool = _load_content_addressed(REVIEW_POOL, expected=REVIEW_POOL_SHA256)
    study_design = _load_content_addressed(STUDY_DESIGN, expected=STUDY_DESIGN_SHA256)
    verify_validation_packet(task_packet)

    if not (
        release.get("technical_public_input_reconstruction_verified") is True
        and release.get("payload_redistribution_cleared") is False
        and release.get("training_lineage_recovered") is False
        and release.get("clean_signed_release") is False
        and release.get("immutable_oci_identity") is None
        and release.get("independent_reproduction") is False
        and release.get("official_release") is False
        and release.get("rank_eligible") is False
        and receipt.get("operator_independence") == "not_adjudicated"
        and receipt.get("rank_eligible") is False
        and lineage.get("exact_lineage_test", {}).get("decision")
        == "candidate_rejected_as_exact_lineage"
    ):
        raise ReleaseHumanGateAuditError("Epicure release evidence crosses its frozen boundary")

    task_counts = task_packet.get("counts", {})
    qa_boundary = human_qa.get("claim_boundary", {})
    pool_observed = review_pool.get("observed", {})
    pool_boundary = review_pool.get("claim_boundary", {})
    if not (
        task_status.get("packetSha256") == TASK_PACKET_SHA256
        and task_status.get("taskCount") == 40
        and task_status.get("completeIndependentReviews") == 0
        and task_status.get("distinctIndependentReviewers") == 0
        and task_status.get("humanCriterionPacks") == 0
        and task_status.get("independentlyValidatedTasks") == 0
        and task_status.get("claimBoundary", {}).get("rankEligible") is False
        and task_counts.get("sealed_human_reviews") == 0
        and task_counts.get("independently_validated_tasks") == 0
        and qa_boundary.get("research_use") is False
        and qa_boundary.get("paper_use") is False
        and qa_boundary.get("rank_eligible") is False
        and qa_boundary.get("leaderboard_use") is False
        and pool_observed.get("quality_judgments") == 0
        and pool_observed.get("synthetic_arms") == 0
        and pool_boundary.get("official") is False
        and pool_boundary.get("rank_eligible") is False
        and pool_boundary.get("research_result") is False
    ):
        raise ReleaseHumanGateAuditError("human-review evidence crosses its frozen boundary")

    task_bank = study_design["task_bank"]
    task_admission = task_bank["admission"]
    expert_evaluation = study_design["expert_evaluation"]
    release_requirements = study_design["release_requirements"]
    rights_rows = _source_rights_rows(data_sources)

    source_files = {
        "public_reconstruction_kit": {
            "path": str(KIT.relative_to(ROOT)),
            "artifact_sha256": KIT_SHA256,
            "file_sha256": KIT_SHA256,
        },
        "same_operator_reconstruction_receipt": {
            "path": str(RECEIPT.relative_to(ROOT)),
            "artifact_sha256": RECEIPT_SHA256,
            "file_sha256": file_sha256(RECEIPT),
        },
        "release_boundary": {
            "path": str(RELEASE_BOUNDARY.relative_to(ROOT)),
            "artifact_sha256": RELEASE_BOUNDARY_SHA256,
            "file_sha256": file_sha256(RELEASE_BOUNDARY),
        },
        "candidate_lineage_falsification": {
            "path": str(LINEAGE_AUDIT.relative_to(ROOT)),
            "artifact_sha256": LINEAGE_AUDIT_SHA256,
            "file_sha256": file_sha256(LINEAGE_AUDIT),
        },
        "development_task_packet": {
            "path": str(TASK_PACKET.relative_to(ROOT)),
            "artifact_sha256": TASK_PACKET_SHA256,
            "file_sha256": file_sha256(TASK_PACKET),
        },
        "development_task_status": {
            "path": str(TASK_STATUS.relative_to(ROOT)),
            "artifact_sha256": TASK_STATUS_SHA256,
            "file_sha256": file_sha256(TASK_STATUS),
        },
        "restricted_human_qa": {
            "path": str(HUMAN_QA.relative_to(ROOT)),
            "artifact_sha256": HUMAN_QA_SHA256,
            "file_sha256": file_sha256(HUMAN_QA),
        },
        "unjudged_real_output_review_pool": {
            "path": str(REVIEW_POOL.relative_to(ROOT)),
            "artifact_sha256": REVIEW_POOL_SHA256,
            "file_sha256": file_sha256(REVIEW_POOL),
        },
        "prospective_study_design": {
            "path": str(STUDY_DESIGN.relative_to(ROOT)),
            "artifact_sha256": STUDY_DESIGN_SHA256,
            "file_sha256": file_sha256(STUDY_DESIGN),
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "record_role": "offline_fail_closed_release_and_human_rank_readiness_audit",
        "as_of_evidence": "latest_repository_artifacts_bound_above",
        "overall_decision": "NO_GO_OFFICIAL_RANKING",
        "epicure_release": {
            "decision": "NO_GO",
            "technical_public_input_reconstruction": "verified",
            "release_id": release["release_id"],
            "runtime_id": release["runtime_id"],
            "kit_registered_data_files": len(rights_rows),
            "same_operator_or_unadjudicated_receipt": RECEIPT_SHA256,
            "rights_rows": rights_rows,
            "training_lineage": {
                "status": "candidate_rejected_as_exact_lineage",
                "recovered_candidate_run_id": lineage["candidate_private_source"][
                    "candidate_job"
                ]["run_id"],
                "deployed_embedding_sha256": lineage["deployed_runtime_payload"][
                    "embeddings_sha256"
                ],
                "candidate_input_embedding_sha256": lineage["candidate_private_source"][
                    "candidate_export_manifest"
                ]["input_embedding_sha256"],
                "candidate_input_embedding_available": False,
                "candidate_export_dirty": True,
                "candidate_statistics_match_deployed": False,
                "required_closure": [
                    "exact deployed-output training-run identity",
                    "exact input graph and input embedding bytes with hashes",
                    "clean training and export source revisions or archived exact dirty bytes",
                    (
                        "preprocessing, objective, hyperparameters, seed, environment and "
                        "export transform"
                    ),
                    "source-by-source training-data rights and signed lineage attestation",
                    (
                        "or a formally approved opaque-artifact boundary that preserves "
                        "these limitations"
                    ),
                ],
            },
            "signed_executable_identity": {
                "clean_signed_source_tag": "missing",
                "immutable_oci_reference": "missing",
                "required_oci_form": "registry/repository@sha256:<64 hex>",
                "required_attestations": [
                    "signature binding the clean source tag and commit",
                    "reproducible source archive and build-recipe digests",
                    "Linux x86-64 OCI manifest digest for the studied runtime",
                    "image signature, SBOM and build-provenance attestation",
                    (
                        "runtime /provenance output matching application, bundle and "
                        "tool-schema hashes"
                    ),
                ],
            },
            "external_reproduction": {
                "status": "missing",
                "same_operator_receipt_does_not_count": True,
                "required_evidence": [
                    (
                        "named or pseudonymously published operator with privately verified "
                        "independence"
                    ),
                    "conflict and no-study-author-operation attestation",
                    "fresh target environment and exact kit digest",
                    "complete command log and content-addressed reconstruction receipt",
                    "1,790 x 300 provenance and frozen fixture parity",
                    "operator signature over the receipt and discrepancy report",
                ],
            },
            "official_release": False,
            "rank_eligible": False,
        },
        "human_rank_readiness": {
            "decision": "NO_GO",
            "current_development_task_packet": {
                "tasks": task_status["taskCount"],
                "public_development_tasks": True,
                "complete_independent_reviews": task_status["completeIndependentReviews"],
                "distinct_independent_reviewers": task_status["distinctIndependentReviewers"],
                "human_criterion_packs": task_status["humanCriterionPacks"],
                "independently_validated_tasks": task_status["independentlyValidatedTasks"],
                "confirmatory_eligible": False,
                "rank_eligible": False,
            },
            "current_real_output_review_pool": {
                "pairs": pool_observed["candidate_pairs"],
                "real_source_arms": pool_observed["source_arms"],
                "synthetic_arms": pool_observed["synthetic_arms"],
                "quality_judgments": pool_observed["quality_judgments"],
                "official": False,
                "rank_eligible": False,
            },
            "restricted_historical_qa": {
                "unique_primary_judgments": human_qa["review_progress"][
                    "unique_primary_judgments"
                ],
                "paper_use": False,
                "research_use": False,
                "ranking_use": False,
            },
            "public_anonymity_boundary": {
                "self_attested_no_pii_path": (
                    "may be reported only as a pseudonymous self-attested external rater"
                ),
                "self_attested_path_counts_as_verified_independent_expert": False,
                "verified_path": (
                    "civil identity and qualification are verified in a restricted admin record; "
                    "only a pseudonym and season-specific HMAC commitment are released"
                ),
                "anti_duplication": (
                    "one admin-witnessed season-HMAC person commitment per reviewer; reject "
                    "duplicate commitments and enforce one vote per battle/rater/cohort"
                ),
                "author_or_product_affiliated_reviews": "separate cohort; never independent",
            },
            "prospective_official_task_bank": {
                "tasks": task_bank["total"],
                "splits": task_bank["splits"],
                "minimum_distinct_verified_authors": task_admission[
                    "minimum_distinct_verified_task_authors"
                ],
                "distinct_people_per_task": task_admission["distinct_people_per_task"],
                "minimum_human_validity_records": task_admission[
                    "minimum_human_validity_records"
                ],
                "minimum_human_evidence_reviews": task_admission[
                    "minimum_human_evidence_reviews"
                ],
                "observed_official_tasks": 0,
                "observed_admissible_task_reviews": 0,
            },
            "prospective_expert_judgment_thresholds": {
                "distinct_independent_raters_per_comparison": expert_evaluation[
                    "minimum_distinct_independent_raters_per_comparison"
                ],
                "minimum_unique_comparisons": expert_evaluation[
                    "minimum_unique_comparisons"
                ],
                "minimum_unique_comparison_appearances_per_model": expert_evaluation[
                    "minimum_unique_comparison_appearances_per_model"
                ],
                "reliability_repeat_rate": expert_evaluation["reliability_repeat_rate"],
                "current_admissible_judgments": 0,
            },
            "industry_standard_claim_thresholds": release_requirements[
                "industry_standard_claim"
            ],
        },
        "minimum_execution_checklist": [
            {
                "order": 1,
                "owner": "Epicure rights holder and release governor",
                "action": (
                    "Complete and sign the 11-row source-rights matrix, payload licence and "
                    "redistribution/serving attestation."
                ),
                "gate": "epicure_payload_rights",
            },
            {
                "order": 2,
                "owner": "Epicure training owner",
                "action": (
                    "Recover the exact deployed training/export lineage or approve and publish "
                    "an explicit opaque-artifact boundary."
                ),
                "gate": "epicure_training_lineage",
            },
            {
                "order": 3,
                "owner": "Epicure release engineer",
                "action": (
                    "Create a clean signed tag and publish the exact Linux x86-64 image by OCI "
                    "digest with signature, SBOM and build provenance."
                ),
                "gate": "epicure_signed_executable",
            },
            {
                "order": 4,
                "owner": "External reproducibility operator",
                "action": (
                    "Reconstruct from the frozen kit in a fresh environment and publish a signed "
                    "receipt and fixture-parity report without study-author operation."
                ),
                "gate": "epicure_independent_reproduction",
            },
            {
                "order": 5,
                "owner": "Human PI and reviewer administrator",
                "action": (
                    "Recruit unique people through privacy-preserving verified records. Public "
                    "pseudonyms are allowed; self-attested no-PII records remain a separate cohort."
                ),
                "gate": "reviewer_independence_and_anti_duplication",
            },
            {
                "order": 6,
                "owner": "Task-bank team",
                "action": (
                    "Author and admit the frozen 240-task hidden bank under the "
                    "six-distinct-person "
                    "task evidence contract; do not promote the 40 public development tasks."
                ),
                "gate": "confirmatory_task_validity",
            },
            {
                "order": 7,
                "owner": "Human rating team",
                "action": (
                    "Collect two qualification-matched independent judgments per comparison and "
                    "meet the frozen per-model, comparison and reliability thresholds."
                ),
                "gate": "admissible_quality_judgments",
            },
            {
                "order": 8,
                "owner": "Release governor",
                "action": (
                    "Re-run officialization and ranking from the append-only archive only after "
                    "every prior gate has machine-verifiable evidence."
                ),
                "gate": "official_rank_release",
            },
        ],
        "source_artifacts": source_files,
        "claim_boundary": {
            "provider_calls_made": 0,
            "epicure_calls_made": 0,
            "reviewers_invented": 0,
            "judgments_invented": 0,
            "same_operator_reconstruction_is_independent": False,
            "self_attested_anonymous_review_is_verified_expert_review": False,
            "development_tasks_are_confirmatory": False,
            "official_result_supported": False,
            "rank_eligible": False,
        },
    }


def verify_audit(document: object) -> bool:
    if not isinstance(document, Mapping) or document.get("schema_version") != SCHEMA_VERSION:
        return False
    digest = document.get("artifact_sha256")
    payload = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return bool(
        isinstance(digest, str)
        and len(digest) == 64
        and canonical_sha256(payload) == digest
        and document.get("overall_decision") == "NO_GO_OFFICIAL_RANKING"
        and document.get("claim_boundary", {}).get("rank_eligible") is False
    )


def write_audit(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    digest = canonical_sha256(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"release-human-gate-audit-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise ReleaseHumanGateAuditError("content-addressed audit conflict")
        return destination
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args(argv)
    path = write_audit(arguments.output_dir, build_audit())
    document = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(path.resolve()),
                "artifact_sha256": document["artifact_sha256"],
                "overall_decision": document["overall_decision"],
                "provider_calls_made": 0,
                "epicure_calls_made": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
