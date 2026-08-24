from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from flavourbench.epicure_public_rebuild_candidate import (
    CANDIDATE_SCHEMA,
    CandidateError,
    _canonical_sha256,
    _verify_execution_receipt,
    build_kit,
    verify_kit_archive,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "artifacts/season1/epicure-lineage/public-reconstruction"
KIT_SHA256 = "bc5b109840c3c7d468fdc050eff7b907b66a0bc3a31899bd4c39ea8bcb1ae4b6"
KIT_MANIFEST_SHA256 = "7ba21a10af4c7f66dcf968326be75ed9fe59ea8e3d81aa7862fbde3a82332f50"
RECEIPT_ARTIFACT = "e1140807dfd2feb0286318f6ab6a8a60246273b06b6f06b14ca7baccf2750cef"
CANDIDATE_ARTIFACT = "1757620645fabc9f758152315b7c09a802143f5790db58b5844b647f8ca22e59"
KIT = PUBLIC / f"epicure-public-reconstruction-kit-{KIT_SHA256}.tar.gz"
RECEIPT = PUBLIC / f"epicure-public-input-reconstruction-receipt-{RECEIPT_ARTIFACT}.json"
CANDIDATE = PUBLIC / (
    f"epicure-public-reconstruction-release-candidate-{CANDIDATE_ARTIFACT}.json"
)
PAPER = ROOT.parent / "paper/flavourbench"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rewrite_digest(document: dict) -> dict:
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return {**unhashed, "artifact_sha256": _canonical_sha256(unhashed)}


def test_frozen_reconstruction_kit_is_deterministic_and_self_verifying(
    tmp_path: Path,
) -> None:
    report = verify_kit_archive(KIT, expected_sha256=KIT_SHA256)
    assert report == {
        "kit_manifest_sha256": KIT_MANIFEST_SHA256,
        "runtime_id": "epicure-mcp-1790-r1+bundle.98d0403115bf.app.be4216ae799f",
        "source_files_verified": 28,
        "data_files_registered": 11,
        "dependency_wheels_registered": 41,
        "network_calls_made": 0,
        "payload_redistribution_cleared": False,
        "independent_reproduction": False,
        "rank_eligible": False,
        "status": "public_input_reconstruction_kit_verified_offline",
    }
    rebuilt = build_kit(ROOT, tmp_path)
    assert rebuilt.read_bytes() == KIT.read_bytes()
    assert hashlib.sha256(rebuilt.read_bytes()).hexdigest() == KIT_SHA256


def test_kit_has_exact_flat_source_only_control_plane() -> None:
    expected = {
        "KIT-MANIFEST.json",
        "README.md",
        "application-source.tar.gz",
        "data-sources.json",
        "functional-fixtures.json",
        "reconstruct.py",
        "rights-boundary.json",
        "runtime-dependency-lock.txt",
        "runtime-manifest.json",
        "runtime-sbom.cdx.json",
        "tool-catalog.json",
    }
    secret_patterns = (
        re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(rb"cohere_[A-Za-z0-9._-]{20,}"),
        re.compile(rb"AKIA[0-9A-Z]{16}"),
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    )
    with tarfile.open(KIT, mode="r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == sorted(expected)
        assert all(member.isfile() for member in members)
        assert all(not member.issym() and not member.islnk() for member in members)
        assert all(member.mode == 0o644 for member in members)
        assert all(member.mtime == 0 for member in members)
        assert all(member.uid == 0 and member.gid == 0 for member in members)
        assert all(member.uname == "" and member.gname == "" for member in members)
        manifest_handle = archive.extractfile("KIT-MANIFEST.json")
        assert manifest_handle is not None
        manifest = json.load(manifest_handle)
        for member in members:
            handle = archive.extractfile(member)
            assert handle is not None
            data = handle.read()
            assert not any(pattern.search(data) for pattern in secret_patterns)
    assert manifest["contains_runtime_data"] is False
    assert manifest["contains_dependency_wheels"] is False
    assert manifest["contains_training_material"] is False
    assert manifest["contains_credentials"] is False
    assert manifest["independent_reproduction"] is False
    assert manifest["payload_redistribution_cleared"] is False
    assert manifest["rank_eligible"] is False
    assert all(item["contains_runtime_data"] is False for item in manifest["members"])
    assert all(item["contains_dependency_wheel"] is False for item in manifest["members"])


def test_same_operator_execution_receipt_is_exact_and_fail_closed() -> None:
    receipt = _read(RECEIPT)
    assert receipt["artifact_sha256"] == RECEIPT_ARTIFACT
    assert receipt["data_input_mode"] == "immutable_public_git_urls"
    assert receipt["dependency_input_mode"] == "public_pypi_exact_hashes"
    assert receipt["dependency_wheels_verified"] == 41
    assert receipt["runtime_payload_environment_sha256"] == (
        "f715e57d4c8879916d56aeb6c0983c75ed9f6ad7a96187906cfbb375e8e49fd0"
    )
    assert receipt["installed_distribution_integrity"][
        "all_declared_record_hashes_match_physical_files"
    ] is True
    assert receipt["functional_fixture_passed"] is True
    assert receipt["functional_fixture"]["provenance"] == {
        "application_sha256": (
            "be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313"
        ),
        "bundle_sha256": (
            "98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1"
        ),
        "embedding_dimensions": 300,
        "ingredient_count": 1790,
        "release_id": "exploratory-unmatched-1790-runtime",
    }
    assert receipt["operator_independence"] == "not_adjudicated"
    assert receipt["independent_reproduction"] is False
    assert receipt["payload_license"] is None
    assert receipt["payload_rights_attestation"] is None
    assert receipt["payload_redistribution_cleared"] is False
    assert receipt["training_lineage_recovered"] is False
    assert receipt["immutable_oci_identity"] is None
    assert receipt["rank_eligible"] is False
    _verify_execution_receipt(RECEIPT, KIT_MANIFEST_SHA256)


def test_release_candidate_distinguishes_fixed_technical_gaps_from_open_gates() -> None:
    candidate = _read(CANDIDATE)
    unhashed = {key: value for key, value in candidate.items() if key != "artifact_sha256"}
    assert candidate["schema_version"] == CANDIDATE_SCHEMA
    assert candidate["artifact_sha256"] == CANDIDATE_ARTIFACT
    assert candidate["artifact_sha256"] == _canonical_sha256(unhashed)
    assert candidate["kit"]["sha256"] == KIT_SHA256
    assert candidate["kit"]["manifest_sha256"] == KIT_MANIFEST_SHA256
    assert candidate["technical_public_input_reconstruction_verified"] is True
    assert {
        "portable_exact_application_source_bytes",
        "immutable_exact_runtime_data_source_map",
        "public_pypi_acquisition_of_all_41_hash_locked_wheels",
        "offline_exact_wheel_installation",
        "installed_distribution_record_and_runtime_payload_verification",
        "cross_file_runtime_manifest_and_sbom_verification",
        "operator_generated_deterministic_tool_parity_fixture",
        "public_input_runtime_materialization_and_functional_probe",
    } == set(candidate["fixed_technical_gaps"])
    assert {
        "payload_license_and_rights_attestation",
        "training_lineage",
        "clean_signed_release",
        "immutable_oci_identity",
        "independent_reproduction",
        "public_runtime_provenance",
        "official_benchmark_governance",
    } == {item["gate"] for item in candidate["remaining_governance_blockers"]}
    assert all(
        item["status"] == "unresolved"
        for item in candidate["remaining_governance_blockers"]
    )
    assert candidate["operator_independence"] == "not_adjudicated"
    assert candidate["independent_reproduction"] is False
    assert candidate["payload_redistribution_cleared"] is False
    assert candidate["training_lineage_recovered"] is False
    assert candidate["clean_signed_release"] is False
    assert candidate["immutable_oci_identity"] is None
    assert candidate["public_runtime_provenance_verified"] is False
    assert candidate["official_release"] is False
    assert candidate["rank_eligible"] is False


def test_rehashed_receipt_cannot_claim_independence(tmp_path: Path) -> None:
    receipt = _read(RECEIPT)
    receipt["operator_independence"] = "independent"
    receipt["independent_reproduction"] = True
    mutated = tmp_path / "receipt.json"
    mutated.write_text(
        json.dumps(_rewrite_digest(receipt), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CandidateError, match="crosses the evidence boundary"):
        _verify_execution_receipt(mutated, KIT_MANIFEST_SHA256)


def test_materialization_requires_explicit_rights_boundary_acknowledgement(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "kit"
    extracted.mkdir()
    with tarfile.open(KIT, mode="r:gz") as archive:
        archive.extractall(extracted, filter="data")
    process = subprocess.run(
        [
            sys.executable,
            str(extracted / "reconstruct.py"),
            "materialize",
            "--kit-root",
            str(extracted),
            "--workspace",
            str(tmp_path / "workspace"),
            "--online",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "--acknowledge-unattested-payload-rights" in process.stderr


def test_paper_claims_follow_the_release_blocked_reconstruction_receipts() -> None:
    candidate = _read(CANDIDATE)
    receipt = _read(RECEIPT)
    assert candidate["technical_public_input_reconstruction_verified"] is True
    assert candidate["operator_independence"] == "not_adjudicated"
    assert candidate["independent_reproduction"] is False
    assert candidate["payload_redistribution_cleared"] is False
    assert candidate["rank_eligible"] is False
    assert receipt["independent_reproduction"] is False
    assert receipt["payload_redistribution_cleared"] is False
    assert receipt["rank_eligible"] is False

    manuscript = (PAPER / "main.tex").read_text(encoding="utf-8")
    source_notes = (PAPER / "SOURCE_NOTES.md").read_text(encoding="utf-8")
    reconstruction_note = (
        PAPER
        / "provenance/EPICURE-PUBLIC-RECONSTRUCTION-CANDIDATE-SOURCE-NOTE.md"
    ).read_text(encoding="utf-8")
    readme = (PAPER / "README.md").read_text(encoding="utf-8")

    for digest in (KIT_SHA256, RECEIPT_ARTIFACT, CANDIDATE_ARTIFACT):
        assert digest in manuscript
        assert digest in source_notes
        assert digest in reconstruction_note
    assert r"\texttt{independent\_reproduction: false}" in manuscript
    assert r"\texttt{payload\_redistribution\_cleared: false}" in manuscript
    assert r"\texttt{rank\_eligible: false}" in manuscript
    assert "candidate is not rank eligible" in manuscript
    assert "`rank_eligible: false`" in source_notes
    assert "`rank_eligible: false`" in reconstruction_note
    assert "not currently include that kit" in readme

    stale_claims = (
        "whose training lineage and redistributable payload\ncannot be independently reconstructed",
        "does not make the Epicure runtime publicly executable",
        "They still cannot execute the exact\nruntime from the research archive alone "
        "because the data and private wheelhouse",
    )
    combined = "\n".join((manuscript, source_notes, reconstruction_note, readme))
    assert all(claim not in combined for claim in stale_claims)
