from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from flavourbench import epicure_public_reconstruction as reconstruction
from flavourbench.epicure_public_reconstruction import (
    ABSENT_EXACT_APPLICATION_BLOBS,
    APPLICATION_SOURCE_ARCHIVE_FILENAME,
    APPLICATION_SOURCE_ARCHIVE_SHA256,
    PUBLIC_BUNDLE_COMMIT,
    PUBLIC_BUNDLE_FILES,
    PublicReconstructionError,
    build_application_source_archive,
    build_public_packet,
    verify_application_source_archive,
    verify_public_packet,
)

ROOT = Path(__file__).resolve().parents[1]
PACKET_DIR = ROOT / "artifacts/season1/epicure-lineage/public-reconstruction"
EXPECTED_ARTIFACT = "8defca7356994d32126ad8b328d0727cd9cc20af4c4c5497c45d279a096706ad"
PACKET = PACKET_DIR / f"epicure-public-reconstruction-packet-{EXPECTED_ARTIFACT}.json"
SOURCE_ARCHIVE = PACKET_DIR / APPLICATION_SOURCE_ARCHIVE_FILENAME
RUNTIME_MANIFEST = ROOT / (
    "artifacts/season1/epicure-lineage/reproducibility/"
    "epicure-exact-runtime-manifest-"
    "a37e5d25f9c5f7a1ec32708b17e0301bbd88248b4c0aeacecf89579106d8edf5.json"
)


def _canonical_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _rewrite_digest(document: dict) -> dict:
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return {**unhashed, "artifact_sha256": _canonical_sha256(unhashed)}


def _write(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_frozen_packet_matches_current_public_evidence() -> None:
    frozen = json.loads(PACKET.read_text(encoding="utf-8"))
    built = build_public_packet(ROOT)
    assert frozen["artifact_sha256"] == EXPECTED_ARTIFACT
    assert {key: value for key, value in frozen.items() if key != "artifact_sha256"} == built

    verified = verify_public_packet(
        packet_path=PACKET,
        root=ROOT,
        layout="repository",
    )
    assert verified["status"] == (
        "public_source_and_data_verifiable_release_governance_blocked"
    )
    assert verified["rank_eligible"] is False
    assert verified["redistributable"] is False


def test_archive_layout_is_self_verifying(tmp_path: Path) -> None:
    document = json.loads(PACKET.read_text(encoding="utf-8"))
    archive = tmp_path / "archive"
    for entry in document["public_files"]:
        source = ROOT / entry["repository_path"]
        destination = archive / entry["archive_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    packet = archive / "provenance/epicure-public-reconstruction-packet.json"
    shutil.copy2(PACKET, packet)

    verified = verify_public_packet(
        packet_path=packet,
        root=archive,
        layout="archive",
    )
    assert verified["artifact_sha256"] == EXPECTED_ARTIFACT


def test_packet_records_irreducible_public_omissions() -> None:
    document = json.loads(PACKET.read_text(encoding="utf-8"))
    assert document["rights_inventory"] == {
        "application_code_license_observed": "MIT",
        "application_copyright_notice": "Copyright (c) 2026 KAIKAKU-AI",
        "application_license_file_sha256": (
            "65baefcdc9727c4db42391a219389e97970496fc2d252dff32aae7c5c76319fb"
        ),
        "application_relicensed_for_archive": False,
        "exact_application_source_included": True,
        "payload_license": None,
        "payload_rights_attestation": None,
        "source_rights_matrix": None,
        "payload_redistribution_cleared": False,
        "status": "code_license_observed_payload_rights_unresolved",
    }
    omissions = {item["requirement"]: item for item in document["publicly_omitted_requirements"]}
    assert all(item["included"] is False for item in omissions.values())
    assert {
        "exact_runtime_data_payload_bytes_in_research_archive",
        "original_training_input_embedding",
        "payload_rights_attestation",
        "source_rights_matrix",
        "immutable_oci_image",
        "independent_reproduction_receipt",
    } <= omissions.keys()
    assert document["not_possible_from_public_packet"] == {
        "claim_independent_reproduction": True,
        "claim_official_rank_eligibility": True,
        "execute_exact_runtime_from_research_archive_alone": True,
        "redistribute_exact_payload": True,
        "retrain_exact_representation": True,
    }
    public_data = document["public_runtime_data"]
    assert public_data["availability_status"] == "public_exact_bytes_hash_verified"
    assert public_data["commit"] == PUBLIC_BUNDLE_COMMIT
    assert public_data["publicly_fetchable_without_credentials"] is True
    assert public_data["included_in_research_archive"] is False
    assert public_data["payload_redistribution_cleared"] is False
    assert [
        {key: item[key] for key in ("path", "bytes", "sha256")}
        for item in public_data["files"]
    ] == list(PUBLIC_BUNDLE_FILES)
    source = document["application_source_availability"]
    assert source["manifest_file_count"] == 28
    assert source["exact_match_count_at_reference_commit"] == 21
    assert source["exact_study_source_in_research_archive"] is True
    assert source["source_archive"]["sha256"] == APPLICATION_SOURCE_ARCHIVE_SHA256
    assert source["source_archive"]["model_or_data_payload_included"] is False
    assert source["source_archive"]["relicensed_for_archive"] is False
    assert source["absent_or_different_exact_blobs"] == list(
        ABSENT_EXACT_APPLICATION_BLOBS
    )
    assert document["release_gates"]["research_archive_exact_application_bytes"] is True


def test_source_archive_is_exact_source_only_and_deterministic(tmp_path: Path) -> None:
    manifest = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    report = verify_application_source_archive(
        archive_path=SOURCE_ARCHIVE,
        runtime_manifest=manifest,
        expected_sha256=APPLICATION_SOURCE_ARCHIVE_SHA256,
    )
    assert report == {
        "archive_sha256": APPLICATION_SOURCE_ARCHIVE_SHA256,
        "member_count": 30,
        "source_file_count": 28,
        "source_bytes": 116_668,
        "application_sha256": (
            "be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313"
        ),
        "license_sha256": (
            "65baefcdc9727c4db42391a219389e97970496fc2d252dff32aae7c5c76319fb"
        ),
        "credential_material_included": False,
        "model_or_data_payload_included": False,
    }

    reconstructed_root = tmp_path / "mcp"
    with tarfile.open(SOURCE_ARCHIVE, mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.name != "LICENSE" and not member.name.startswith("src/epicure_mcp/"):
                continue
            handle = archive.extractfile(member)
            assert handle is not None
            destination = reconstructed_root / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(handle.read())
    rebuilt = build_application_source_archive(
        root=ROOT,
        mcp_root=reconstructed_root,
        output_dir=tmp_path / "rebuilt",
    )
    assert rebuilt.read_bytes() == SOURCE_ARCHIVE.read_bytes()


def test_source_archive_contains_no_model_or_data_payload() -> None:
    with tarfile.open(SOURCE_ARCHIVE, mode="r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
        manifest_file = archive.extractfile("MANIFEST.json")
        assert manifest_file is not None
        manifest = json.load(manifest_file)
    assert len(names) == 30
    assert names == sorted(names)
    assert all(
        name in {"LICENSE", "MANIFEST.json"}
        or (name.startswith("src/epicure_mcp/") and name.endswith(".py"))
        for name in names
    )
    assert manifest["license"] == {
        "archive_path": "LICENSE",
        "bytes": 1_067,
        "copyright_notice": "Copyright (c) 2026 KAIKAKU-AI",
        "relicensed_for_archive": False,
        "sha256": "65baefcdc9727c4db42391a219389e97970496fc2d252dff32aae7c5c76319fb",
        "spdx_expression": "MIT",
    }
    assert manifest["archive_boundary"] == {
        "application_source_included": True,
        "credential_material_included": False,
        "dependency_wheels_included": False,
        "model_or_data_payload_included": False,
        "training_corpus_included": False,
    }


def test_network_data_check_is_explicit_and_hashes_all_frozen_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = json.loads(PACKET.read_text(encoding="utf-8"))
    expected = {
        item["raw_url"]: (item["bytes"], item["sha256"])
        for item in document["public_runtime_data"]["files"]
    }
    calls: list[str] = []

    def fake_download(url: str, *, timeout: float) -> tuple[int, str]:
        assert timeout == 7.0
        calls.append(url)
        return expected[url]

    monkeypatch.setattr(reconstruction, "_download_sha256", fake_download)
    verify_public_packet(
        packet_path=PACKET,
        root=ROOT,
        layout="repository",
        verify_runtime_data_online=True,
        network_timeout=7.0,
    )
    assert calls == list(expected)


def test_default_verification_never_opens_a_network_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(url: str, *, timeout: float) -> tuple[int, str]:
        raise AssertionError((url, timeout))

    monkeypatch.setattr(reconstruction, "_download_sha256", fail_if_called)
    verify_public_packet(
        packet_path=PACKET,
        root=ROOT,
        layout="repository",
    )


def test_offline_data_check_rejects_a_changed_payload(tmp_path: Path) -> None:
    first = PUBLIC_BUNDLE_FILES[0]
    (tmp_path / first["path"]).write_bytes(b"not the frozen payload")
    with pytest.raises(PublicReconstructionError, match="runtime data hash mismatch"):
        verify_public_packet(
            packet_path=PACKET,
            root=ROOT,
            layout="repository",
            runtime_data_dir=tmp_path,
        )


def test_rehashed_overclaim_still_fails_closed(tmp_path: Path) -> None:
    document = json.loads(PACKET.read_text(encoding="utf-8"))
    document["release_gates"]["payload_rights_attested"] = True
    mutated = tmp_path / "overclaim.json"
    _write(mutated, _rewrite_digest(document))

    with pytest.raises(PublicReconstructionError, match="release gates"):
        verify_public_packet(
            packet_path=mutated,
            root=ROOT,
            layout="repository",
        )


def test_rehashed_path_traversal_still_fails_closed(tmp_path: Path) -> None:
    document = json.loads(PACKET.read_text(encoding="utf-8"))
    document["public_files"][0]["repository_path"] = "../private-payload.bin"
    mutated = tmp_path / "traversal.json"
    _write(mutated, _rewrite_digest(document))

    with pytest.raises(PublicReconstructionError, match="frozen evidence|safe relative path"):
        verify_public_packet(
            packet_path=mutated,
            root=ROOT,
            layout="repository",
        )
