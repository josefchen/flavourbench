from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from flavourbench.models import Base, ResearchReleaseArchive, Season
from flavourbench.research_release import (
    ArchiveMember,
    ResearchReleaseError,
    _canonical_sha256,
    _robustness_members,
    load_signing_key,
    seal_members,
    verify_archive,
)


def _sealed(tmp_path: Path):
    key = Ed25519PrivateKey.generate()
    metadata = {
        "release_slug": "flavourbench-season-1-test",
        "archive_class": "internal_official",
        "snapshot_set_sha256": "1" * 64,
        "requirements_lock_sha256": "2" * 64,
        "build_image_digest": f"sha256:{'3' * 64}",
        "counts": {"synthetic_arms": 0, "votes": 2},
    }
    members = [
        ArchiveMember("README.md", b"# Test release\n"),
        ArchiveMember(
            "records/votes.jsonl",
            b'{"choice":"left","id":"v1"}\n{"choice":"tie","id":"v2"}\n',
            2,
        ),
    ]
    first = seal_members(
        members=members,
        manifest_metadata=metadata,
        output_dir=tmp_path / "first",
        private_key=key,
        signing_key_id="test-key-2026",
    )
    second = seal_members(
        members=members,
        manifest_metadata=metadata,
        output_dir=tmp_path / "second",
        private_key=key,
        signing_key_id="test-key-2026",
    )
    return key, first, second


def test_sealed_release_is_byte_deterministic_and_signature_verifies(tmp_path: Path) -> None:
    _key, first, second = _sealed(tmp_path)
    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.archive_sha256 == second.archive_sha256
    assert first.signature_base64 == second.signature_base64
    assert first.manifest_sha256 == _canonical_sha256(first.manifest)
    report = verify_archive(
        archive_path=first.archive_path,
        signature_base64=first.signature_base64,
        public_key_pem=first.public_key_pem,
        expected_archive_sha256=first.archive_sha256,
    )
    assert report == {
        "schema_version": "flavourbench-research-release-verification-v1",
        "archive_sha256": first.archive_sha256,
        "manifest_sha256": first.manifest_sha256,
        "member_count": 4,
        "signature_valid": True,
        "inventory_valid": True,
        "reproducible_metadata_valid": True,
    }


def test_archive_tampering_and_wrong_key_fail_closed(tmp_path: Path) -> None:
    _key, sealed, _second = _sealed(tmp_path)
    tampered = tmp_path / "tampered.tar.gz"
    data = bytearray(sealed.archive_path.read_bytes())
    data[-1] ^= 1
    tampered.write_bytes(data)
    with pytest.raises(ResearchReleaseError, match="signature is invalid"):
        verify_archive(
            archive_path=tampered,
            signature_base64=sealed.signature_base64,
            public_key_pem=sealed.public_key_pem,
        )
    wrong_public = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    with pytest.raises((ResearchReleaseError, ValueError, TypeError)):
        verify_archive(
            archive_path=sealed.archive_path,
            signature_base64=sealed.signature_base64,
            public_key_pem=wrong_public.decode("ascii"),
        )


def test_signing_key_loader_requires_ed25519_and_private_permissions(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    path = tmp_path / "release-key.pem"
    path.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    path.chmod(0o644)
    with pytest.raises(ResearchReleaseError, match="group/world readable"):
        load_signing_key(path)
    path.chmod(0o600)
    assert isinstance(load_signing_key(path), Ed25519PrivateKey)


def test_archive_metadata_is_append_only_and_signature_bound(tmp_path: Path) -> None:
    _key, sealed, _second = _sealed(tmp_path)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        season = Season(
            id="season-id",
            slug="season-1",
            name="Season 1",
            epicure_release_id="epicure-v1",
        )
        session.add(season)
        snapshot_ids = ["s1", "s2", "s3", "s4"]
        record = ResearchReleaseArchive(
            id="archive-id",
            season_id=season.id,
            archive_class="internal_official",
            schema_version="flavourbench-research-release-v1",
            snapshot_ids_json=snapshot_ids,
            snapshot_set_sha256=_canonical_sha256({"snapshot_ids": snapshot_ids}),
            manifest_json=sealed.manifest,
            manifest_sha256=sealed.manifest_sha256,
            archive_sha256=sealed.archive_sha256,
            storage_object_key=str(sealed.archive_path),
            size_bytes=sealed.archive_size_bytes,
            member_count=4,
            source_date_epoch=0,
            requirements_lock_sha256=hashlib.sha256(b"lock").hexdigest(),
            build_image_digest=f"sha256:{'3' * 64}",
            signature_algorithm="Ed25519",
            signing_key_id="test-key-2026",
            public_key_pem=sealed.public_key_pem,
            public_key_sha256=sealed.public_key_sha256,
            signature_base64=sealed.signature_base64,
        )
        session.add(record)
        session.commit()
        record.storage_object_key = "changed"
        with pytest.raises(Exception, match="append-only"):
            session.commit()


def test_secret_scan_refuses_credentials(tmp_path: Path) -> None:
    with pytest.raises(ResearchReleaseError, match="aws_access_key"):
        seal_members(
            members=[ArchiveMember("records/unsafe.json", b'{"key":"AKIA1234567890ABCDEF"}')],
            manifest_metadata={"release_slug": "unsafe"},
            output_dir=tmp_path,
            private_key=Ed25519PrivateKey.generate(),
            signing_key_id="test",
        )


def test_robustness_artifacts_are_bound_into_the_release_manifest(tmp_path: Path) -> None:
    design = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "contracts/season1/season1-study-design-v5.json"
        ).read_text(encoding="utf-8")
    )
    schemas = {
        "post_collection_item_audit": "flavourbench-season1-post-collection-item-audit-v1",
        "generation_reliability_panel": (
            "flavourbench-season1-generation-reliability-panel-v1"
        ),
        "prompt_sensitivity_audit": "flavourbench-season1-prompt-sensitivity-audit-v1",
        "practical_cookability_execution": (
            "flavourbench-season1-practical-cookability-execution-v1"
        ),
    }
    paths: dict[str, Path] = {}
    for name, schema_version in schemas.items():
        payload = {
            "schema_version": schema_version,
            "status": "complete",
            "study_design_artifact_sha256": design["artifact_sha256"],
            "synthetic_observations": 0,
        }
        document = {**payload, "artifact_sha256": _canonical_sha256(payload)}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        paths[name] = path

    members, digests = _robustness_members(paths)

    assert len(members) == 4
    assert set(digests) == set(schemas)
    assert all(member.path.startswith("validity-and-robustness/") for member in members)

    tampered = json.loads(paths["prompt_sensitivity_audit"].read_text(encoding="utf-8"))
    tampered["synthetic_observations"] = 1
    paths["prompt_sensitivity_audit"].write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ResearchReleaseError, match="common contract"):
        _robustness_members(paths)
