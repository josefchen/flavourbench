"""Build and verify Epicure's public reconstruction-boundary packet.

The packet makes the evidence already safe to publish independently checkable.
It deliberately does not turn hashes or a same-operator private rebuild into a
redistributable payload, recovered training lineage, or independent
reproduction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "epicure-public-reconstruction-packet-v3"
INVENTORY_SCHEMA = "epicure-recovered-runtime-inventory-v2"
RUNTIME_MANIFEST_SCHEMA = "epicure-exact-runtime-manifest-v1"
REBUILD_RECEIPT_SCHEMA = "epicure-private-offline-rebuild-receipt-v1"
CANDIDATE_LINEAGE_SCHEMA = "epicure-candidate-training-lineage-audit-v1"

INVENTORY_ARTIFACT = "70d00d933aa1340841a82a9637de8b75de380f8aeba2179beab419fb6542ab5f"
CORRECTION_ARTIFACT = "d739a1b08be79c8a116ea86687ef8f4c983fe8cc9c312257f85ef107e32e90e7"
ATTESTATION_ARTIFACT = "825d087713ef525a1643a81bb9b94c26f3be64794ac84b4699d1cc8380922220"
RUNTIME_MANIFEST_ARTIFACT = "a37e5d25f9c5f7a1ec32708b17e0301bbd88248b4c0aeacecf89579106d8edf5"
REBUILD_RECEIPT_ARTIFACT = "35854e9f50f8f3756ab480a6ded012e15bb2e2cc4673948923068cc9deb88255"
AUTHORITY_ARTIFACT = "83b5f3109242478f6def0acbc434112900760ae92890626da23838bac0ea5a6a"
CANDIDATE_LINEAGE_ARTIFACT = "332a3e20e1de351307d2ece3d37275ac7599dcc14f42305cd1e77bd7621cdfaf"

RUNTIME_ID = "epicure-mcp-1790-r1+bundle.98d0403115bf.app.be4216ae799f"
BUNDLE_SHA256 = "98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1"
APPLICATION_SHA256 = "be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313"
TOOL_SCHEMA_SHA256 = "666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd"
APPLICATION_LICENSE_SHA256 = "65baefcdc9727c4db42391a219389e97970496fc2d252dff32aae7c5c76319fb"
APPLICATION_LICENSE_BYTES = 1_067
APPLICATION_SOURCE_ARCHIVE_SCHEMA = "epicure-study-application-source-archive-v1"
APPLICATION_SOURCE_ARCHIVE_SHA256 = (
    "d08fb475e9c325a8c41daf5b789e6b4bca547228139eece4578f9b06c324703c"
)
APPLICATION_SOURCE_ARCHIVE_FILENAME = (
    f"epicure-study-application-source-{APPLICATION_SOURCE_ARCHIVE_SHA256}.tar.gz"
)

PUBLIC_BUNDLE_REPOSITORY = "https://github.com/KAIKAKU-AI/epicure-mcp"
PUBLIC_BUNDLE_COMMIT = "14ddf04aba81a76b75efa6554041f6bff48992c6"
PUBLIC_BUNDLE_RAW_ROOT = (
    "https://raw.githubusercontent.com/KAIKAKU-AI/epicure-mcp/"
    f"{PUBLIC_BUNDLE_COMMIT}/data"
)
PUBLIC_BUNDLE_OBSERVED_AT = "2026-08-03"

PUBLIC_BUNDLE_FILES: tuple[dict[str, Any], ...] = (
    {
        "path": "consolidated_nodes.csv",
        "bytes": 66_703,
        "sha256": "b935b9a31f7ea87e04a29f6e4e6c523723343d2c054149cfcc258b6b06259752",
    },
    {
        "path": "embeddings.csv",
        "bytes": 10_425_903,
        "sha256": "b27fe776a59b59d703ae24170ccfcf384b89a753fd358307d085514a4fde6f69",
    },
    {
        "path": "factor_dirs_ica_n20.npy",
        "bytes": 24_128,
        "sha256": "dd20d0c5c6fe1db763f0edbef601de4d01544c5e1cb7f44cf3879040e4664336",
    },
    {
        "path": "factor_labels_ica_cooc.json",
        "bytes": 75_043,
        "sha256": "f11f4fd01e39f039665bb9874e72fe4a13957051744420a62e4f45254d526f91",
    },
    {
        "path": "ingredient_list.csv",
        "bytes": 75_127,
        "sha256": "31f46abfce52a18647b07a5a60f971b97286849f1936d43b5dc9104f8942c849",
    },
    {
        "path": "ingredient_tags.csv",
        "bytes": 101_492,
        "sha256": "8f52e83a072069f436ab7d851ed0251e775da92afc46e0deaa61d49d91014772",
    },
    {
        "path": "mode_explorer_cooc.json",
        "bytes": 1_918_235,
        "sha256": "382deb62764fa1459656373e21fb81bf3ab610c0d1900e7dfd784d5aa61465b6",
    },
    {
        "path": "mode_poles_cooc.npy",
        "bytes": 180_128,
        "sha256": "47477fe844280ec840cec18959f5e62ce468bd42374e8057c828e8f35d415fcb",
    },
    {
        "path": "supervised_directions.npz",
        "bytes": 55_310,
        "sha256": "44f0a5964a96d521b503c577979afd1c48a9b7e62f5826e651d2dfa2088b4d74",
    },
    {
        "path": "umap_coords.csv",
        "bytes": 54_018,
        "sha256": "a97761fe7648885a94a96bcd5f0eb652ca71f767810874ac6e5c0bba4992cd2d",
    },
    {
        "path": "umap_coords_3d.csv",
        "bytes": 72_836,
        "sha256": "9d09666cd003fa4d86d9263dd89359cdf60f05616c613ab7e28b9e133defe028",
    },
)

ABSENT_EXACT_APPLICATION_BLOBS: tuple[dict[str, Any], ...] = (
    {
        "repository_path": "src/epicure_mcp/config.py",
        "manifest_path": "config.py",
        "bytes": 4_050,
        "sha256": "6221163a4d0afe793b62b876e34e8a6f6af056a4194527a826ed9ecf9bccfbfe",
        "reference_commit_state": "different_blob",
    },
    {
        "repository_path": "src/epicure_mcp/geometry.py",
        "manifest_path": "geometry.py",
        "bytes": 3_733,
        "sha256": "152d0b6f5da863b5f2ddbcefa7176a87eb9c90c58dbaea83a0ace4d024b6846f",
        "reference_commit_state": "different_blob",
    },
    {
        "repository_path": "src/epicure_mcp/security.py",
        "manifest_path": "security.py",
        "bytes": 1_333,
        "sha256": "437205948f4a3610353cc500a52fbe96ddea00c52a2da5119bad19774ab44fed",
        "reference_commit_state": "different_blob",
    },
    {
        "repository_path": "src/epicure_mcp/server.py",
        "manifest_path": "server.py",
        "bytes": 9_996,
        "sha256": "83ff0f10bb39c3a08d6fc96f944f010df21e0baf25020928ff58147598d4b693",
        "reference_commit_state": "different_blob",
    },
    {
        "repository_path": "src/epicure_mcp/tools/find_pairings.py",
        "manifest_path": "tools/find_pairings.py",
        "bytes": 1_339,
        "sha256": "11e634f793f98fab9ebabf549d10899b8c9b10881c84889fac7e8a33f84854b6",
        "reference_commit_state": "different_blob",
    },
    {
        "repository_path": "src/epicure_mcp/tools/morph.py",
        "manifest_path": "tools/morph.py",
        "bytes": 5_007,
        "sha256": "544d64f4325443637734b5f56d9a2536b5698f45a10b140e0aaad958e64c1e97",
        "reference_commit_state": "different_blob",
    },
    {
        "repository_path": "src/epicure_mcp/provenance.py",
        "manifest_path": "provenance.py",
        "bytes": 5_360,
        "sha256": "822b4e2a3af78b9cca76548d11654998e90e438f195cdf62c0ed363aa5d44223",
        "reference_commit_state": "absent",
    },
)

_SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"cohere_[A-Za-z0-9._-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"(?i:bearer)[ \t]+[A-Za-z0-9._~-]{24,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
)


class PublicReconstructionError(RuntimeError):
    """The packet is malformed, inconsistent, unsafe, or overclaims evidence."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_sha256(handle: Any) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


def _download_sha256(url: str, *, timeout: float = 30.0) -> tuple[int, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FlavourBench-Epicure-Reconstruction-Verifier/3"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _stream_sha256(response)
    except (OSError, urllib.error.URLError) as error:
        raise PublicReconstructionError(
            f"could not retrieve frozen public data URL: {url}"
        ) from error


def _expected_public_runtime_data() -> dict[str, Any]:
    return {
        "availability_status": "public_exact_bytes_hash_verified",
        "observed_at": PUBLIC_BUNDLE_OBSERVED_AT,
        "repository": PUBLIC_BUNDLE_REPOSITORY,
        "commit": PUBLIC_BUNDLE_COMMIT,
        "tree_url": f"{PUBLIC_BUNDLE_REPOSITORY}/tree/{PUBLIC_BUNDLE_COMMIT}/data",
        "bundle_sha256": BUNDLE_SHA256,
        "publicly_fetchable_without_credentials": True,
        "included_in_research_archive": False,
        "payload_license": None,
        "payload_rights_attestation": None,
        "source_rights_matrix": None,
        "payload_redistribution_cleared": False,
        "files": [
            {**entry, "raw_url": f"{PUBLIC_BUNDLE_RAW_ROOT}/{entry['path']}"}
            for entry in PUBLIC_BUNDLE_FILES
        ],
        "interpretation_rule": (
            "The immutable public Git commit supplies the exact bytes named by the frozen runtime "
            "manifest. Public byte availability is a technical fact, not a payload-license or "
            "upstream-source-rights attestation, and the research archive does not mirror the "
            "files."
        ),
    }


def _expected_application_source_availability() -> dict[str, Any]:
    return {
        "study_source_manifest_sha256": APPLICATION_SHA256,
        "reference_repository": PUBLIC_BUNDLE_REPOSITORY,
        "reference_commit": PUBLIC_BUNDLE_COMMIT,
        "manifest_file_count": 28,
        "manifest_bytes": 116_668,
        "exact_match_count_at_reference_commit": 21,
        "absent_or_different_count_at_reference_commit": 7,
        "exact_study_source_in_research_archive": True,
        "exact_study_source_in_reference_commit": False,
        "source_archive": {
            "archive_path": "provenance/epicure-study-application-source.tar.gz",
            "repository_filename": APPLICATION_SOURCE_ARCHIVE_FILENAME,
            "sha256": APPLICATION_SOURCE_ARCHIVE_SHA256,
            "member_count": 30,
            "source_file_count": 28,
            "source_bytes": 116_668,
            "license_sha256": APPLICATION_LICENSE_SHA256,
            "credential_material_included": False,
            "model_or_data_payload_included": False,
            "relicensed_for_archive": False,
        },
        "absent_or_different_exact_blobs": [
            dict(entry) for entry in ABSENT_EXACT_APPLICATION_BLOBS
        ],
        "interpretation_rule": (
            "The source-only research archive now preserves all 28 exact study application "
            "files under the existing KAIKAKU-AI MIT licence. The seven-entry comparison records "
            "what was absent or different at the public Git reference; it does not imply a clean "
            "signed release, exact OCI image, payload rights, or independent reproduction."
        ),
    }


def _verify_manifest_matches_frozen_public_data(manifest: Mapping[str, Any]) -> None:
    data = manifest.get("data")
    if not isinstance(data, Mapping):
        raise PublicReconstructionError("runtime manifest has no data register")
    entries = data.get("entries")
    if entries != list(PUBLIC_BUNDLE_FILES) or data.get("sha256") != BUNDLE_SHA256:
        raise PublicReconstructionError("frozen public data register differs from runtime manifest")


def _verify_manifest_matches_application_gap(manifest: Mapping[str, Any]) -> None:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise PublicReconstructionError("runtime manifest has no application source register")
    entries = source.get("entries")
    if not isinstance(entries, list) or len(entries) != 28:
        raise PublicReconstructionError("runtime application source register is incomplete")
    by_path = {entry.get("path"): entry for entry in entries if isinstance(entry, Mapping)}
    for absent in ABSENT_EXACT_APPLICATION_BLOBS:
        observed = by_path.get(absent["manifest_path"])
        if not isinstance(observed, Mapping) or {
            "bytes": observed.get("bytes"),
            "sha256": observed.get("sha256"),
        } != {"bytes": absent["bytes"], "sha256": absent["sha256"]}:
            raise PublicReconstructionError(
                "application gap register differs from runtime manifest"
            )
    if source.get("sha256") != APPLICATION_SHA256:
        raise PublicReconstructionError("application gap register has the wrong source identity")


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicReconstructionError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise PublicReconstructionError(f"{field} is not a safe relative path: {value}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicReconstructionError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise PublicReconstructionError(f"JSON evidence is not an object: {path}")
    return value


def _content_addressed_json(path: Path, expected: str) -> dict[str, Any]:
    value = _read_json(path)
    digest = value.get("artifact_sha256")
    unhashed = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if digest != expected or digest != _canonical_sha256(unhashed):
        raise PublicReconstructionError(f"invalid content-addressed evidence: {path}")
    return value


def _application_source_archive_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise PublicReconstructionError("runtime manifest has no application source register")
    entries = source.get("entries")
    if not isinstance(entries, list) or len(entries) != 28:
        raise PublicReconstructionError("runtime application source register is incomplete")
    return {
        "schema_version": APPLICATION_SOURCE_ARCHIVE_SCHEMA,
        "record_role": "exact_study_application_source_bytes",
        "runtime_id": RUNTIME_ID,
        "application_sha256": APPLICATION_SHA256,
        "source_manifest_version": source.get("manifest_version"),
        "source_entries": entries,
        "license": {
            "archive_path": "LICENSE",
            "spdx_expression": "MIT",
            "copyright_notice": "Copyright (c) 2026 KAIKAKU-AI",
            "bytes": APPLICATION_LICENSE_BYTES,
            "sha256": APPLICATION_LICENSE_SHA256,
            "relicensed_for_archive": False,
        },
        "archive_boundary": {
            "application_source_included": True,
            "credential_material_included": False,
            "dependency_wheels_included": False,
            "model_or_data_payload_included": False,
            "training_corpus_included": False,
        },
        "interpretation_rule": (
            "The archive preserves the exact studied Python application and its existing "
            "KAIKAKU-AI MIT licence. It contains no runtime data, model payload, dependency "
            "wheelhouse, credentials, training corpus, new licence, or signed release claim."
        ),
    }


def _render_source_archive_manifest(manifest: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _application_source_archive_manifest(manifest),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _assert_no_credential_material(data: bytes, *, member: str) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(data):
            raise PublicReconstructionError(
                f"credential-like material detected in application source: {member}"
            )


def _deterministic_source_tar_gz(members: Mapping[str, bytes]) -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=compressed,
        compresslevel=9,
        mtime=0,
    ) as handle:
        handle.write(tar_buffer.getvalue())
    return compressed.getvalue()


def verify_application_source_archive(
    *,
    archive_path: Path,
    runtime_manifest: Mapping[str, Any],
    expected_sha256: str,
) -> dict[str, Any]:
    """Verify exact members, metadata, hashes, licence, and the payload exclusion boundary."""

    if archive_path.is_symlink() or not archive_path.is_file():
        raise PublicReconstructionError(f"application source archive is missing: {archive_path}")
    raw = archive_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PublicReconstructionError("application source archive hash mismatch")
    if len(raw) < 10 or raw[:2] != b"\x1f\x8b" or int.from_bytes(raw[4:8], "little") != 0:
        raise PublicReconstructionError("application source archive has a non-deterministic header")

    source = runtime_manifest.get("source")
    if not isinstance(source, Mapping) or source.get("sha256") != APPLICATION_SHA256:
        raise PublicReconstructionError("application source archive has the wrong source identity")
    entries = source.get("entries")
    if not isinstance(entries, list) or len(entries) != 28:
        raise PublicReconstructionError("runtime application source register is incomplete")
    expected: dict[str, tuple[int, str]] = {
        "LICENSE": (APPLICATION_LICENSE_BYTES, APPLICATION_LICENSE_SHA256),
        "MANIFEST.json": (
            len(_render_source_archive_manifest(runtime_manifest)),
            hashlib.sha256(_render_source_archive_manifest(runtime_manifest)).hexdigest(),
        ),
    }
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PublicReconstructionError("runtime application source register is malformed")
        relative = _safe_relative_path(entry.get("path"), field="application_source_path")
        if not relative.endswith(".py"):
            raise PublicReconstructionError("application source archive may contain only Python")
        expected[f"src/epicure_mcp/{relative}"] = (entry.get("bytes"), entry.get("sha256"))

    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if names != sorted(expected) or len(names) != len(set(names)):
                raise PublicReconstructionError(
                    "application source archive member register is not exact"
                )
            for member in members:
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.mode != 0o644
                    or member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                ):
                    raise PublicReconstructionError(
                        f"application source archive metadata is not deterministic: {member.name}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise PublicReconstructionError(
                        f"application source archive member is unreadable: {member.name}"
                    )
                data = handle.read()
                expected_bytes, expected_digest = expected[member.name]
                if (
                    len(data) != expected_bytes
                    or hashlib.sha256(data).hexdigest() != expected_digest
                ):
                    raise PublicReconstructionError(
                        f"application source archive member hash mismatch: {member.name}"
                    )
                if member.name.endswith(".py"):
                    _assert_no_credential_material(data, member=member.name)
    except (tarfile.TarError, OSError) as error:
        raise PublicReconstructionError("invalid application source archive") from error

    return {
        "archive_sha256": expected_sha256,
        "member_count": len(expected),
        "source_file_count": len(entries),
        "source_bytes": sum(entry["bytes"] for entry in entries),
        "application_sha256": APPLICATION_SHA256,
        "license_sha256": APPLICATION_LICENSE_SHA256,
        "credential_material_included": False,
        "model_or_data_payload_included": False,
    }


def build_application_source_archive(
    *, root: Path, mcp_root: Path, output_dir: Path
) -> Path:
    """Build the deterministic, source-only archive from the exact recovered study checkout."""

    manifest_path = root.resolve() / (
        "artifacts/season1/epicure-lineage/reproducibility/"
        f"epicure-exact-runtime-manifest-{RUNTIME_MANIFEST_ARTIFACT}.json"
    )
    manifest = _content_addressed_json(manifest_path, RUNTIME_MANIFEST_ARTIFACT)
    _verify_manifest_matches_application_gap(manifest)
    source = manifest["source"]
    members: dict[str, bytes] = {
        "MANIFEST.json": _render_source_archive_manifest(manifest),
    }
    mcp_root = mcp_root.resolve()
    license_path = mcp_root / "LICENSE"
    if license_path.is_symlink() or not license_path.is_file():
        raise PublicReconstructionError("application licence is missing")
    license_bytes = license_path.read_bytes()
    if (
        len(license_bytes) != APPLICATION_LICENSE_BYTES
        or hashlib.sha256(license_bytes).hexdigest() != APPLICATION_LICENSE_SHA256
    ):
        raise PublicReconstructionError("application licence differs from the frozen MIT licence")
    members["LICENSE"] = license_bytes
    for entry in source["entries"]:
        relative = _safe_relative_path(entry.get("path"), field="application_source_path")
        if not relative.endswith(".py"):
            raise PublicReconstructionError("application source archive may contain only Python")
        path = mcp_root / "src/epicure_mcp" / relative
        resolved = path.resolve()
        source_root = (mcp_root / "src/epicure_mcp").resolve()
        if not resolved.is_relative_to(source_root) or path.is_symlink() or not path.is_file():
            raise PublicReconstructionError(f"application source file is missing: {path}")
        data = path.read_bytes()
        if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get(
            "sha256"
        ):
            raise PublicReconstructionError(f"application source hash mismatch: {relative}")
        _assert_no_credential_material(data, member=relative)
        members[f"src/epicure_mcp/{relative}"] = data

    rendered = _deterministic_source_tar_gz(members)
    digest = hashlib.sha256(rendered).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"epicure-study-application-source-{digest}.tar.gz"
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise PublicReconstructionError("content-addressed source archive conflict")
    else:
        with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o644)
    verify_application_source_archive(
        archive_path=destination,
        runtime_manifest=manifest,
        expected_sha256=digest,
    )
    return destination


def _verify_evidence_chain(
    *, inventory: Mapping[str, Any], manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise PublicReconstructionError("recovered inventory has the wrong schema")
    if manifest.get("schema_version") != RUNTIME_MANIFEST_SCHEMA:
        raise PublicReconstructionError("runtime manifest has the wrong schema")
    if receipt.get("schema_version") != REBUILD_RECEIPT_SCHEMA:
        raise PublicReconstructionError("private rebuild receipt has the wrong schema")
    manifest_unhashed = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if receipt.get("runtime_manifest") != manifest_unhashed:
        raise PublicReconstructionError("receipt does not embed the exact runtime manifest")
    if manifest.get("data", {}).get("sha256") != inventory.get("bundle", {}).get("sha256"):
        raise PublicReconstructionError("runtime data differs from the recovered inventory")
    if manifest.get("source", {}).get("sha256") != inventory.get("application", {}).get("sha256"):
        raise PublicReconstructionError("runtime source differs from the recovered inventory")
    recovered = receipt.get("recovered_inventory")
    if not isinstance(recovered, Mapping) or recovered.get("artifact_sha256") != inventory.get(
        "artifact_sha256"
    ):
        raise PublicReconstructionError("receipt does not bind the recovered inventory")
    gates = receipt.get("release_gates")
    required_true = {
        "exact_source_and_data_manifest",
        "hash_locked_platform_runtime",
        "machine_readable_sbom",
        "private_offline_runtime_rebuild",
    }
    required_false = {
        "independent_reproduction",
        "immutable_oci_identity",
        "training_lineage_recovered",
        "payload_rights_attested",
        "public_redistributable_payload",
    }
    if not isinstance(gates, Mapping):
        raise PublicReconstructionError("receipt has no release-gate record")
    if any(gates.get(gate) is not True for gate in required_true):
        raise PublicReconstructionError("receipt understates a verified private gate")
    if any(gates.get(gate) is not False for gate in required_false):
        raise PublicReconstructionError("receipt crosses an unresolved release gate")
    if receipt.get("rank_eligible") is not False or receipt.get("redistributable") is not False:
        raise PublicReconstructionError("receipt crosses the ranking or redistribution boundary")


def _verify_candidate_lineage(candidate: Mapping[str, Any]) -> None:
    test = candidate.get("exact_lineage_test")
    if (
        candidate.get("schema_version") != CANDIDATE_LINEAGE_SCHEMA
        or not isinstance(test, Mapping)
        or test.get("candidate_and_deployed_statistics_match") is not False
        or test.get("candidate_input_embedding_artifact_available_in_checkout") is not False
        or test.get("candidate_export_was_recorded_from_dirty_worktree") is not True
        or test.get("decision") != "candidate_rejected_as_exact_lineage"
        or candidate.get("rank_eligible") is not False
        or candidate.get("redistributable") is not False
        or candidate.get("provider_calls_made") != 0
        or candidate.get("epicure_network_calls_made") != 0
        or candidate.get("synthetic_observations") != 0
    ):
        raise PublicReconstructionError("candidate lineage is not rejected as exact")


def _evidence_specifications() -> tuple[dict[str, Any], ...]:
    return (
        {
            "role": "authoritative_recovered_runtime_inventory",
            "repository_path": (
                "artifacts/season1/epicure-lineage/"
                f"epicure-recovered-runtime-inventory-{INVENTORY_ARTIFACT}.json"
            ),
            "archive_path": "provenance/epicure-recovered-runtime-inventory.json",
            "artifact_sha256": INVENTORY_ARTIFACT,
        },
        {
            "role": "inventory_supersession_correction",
            "repository_path": (
                "artifacts/season1/epicure-lineage/"
                f"epicure-lineage-inventory-correction-{CORRECTION_ARTIFACT}.json"
            ),
            "archive_path": "provenance/epicure-lineage-inventory-correction.json",
            "artifact_sha256": CORRECTION_ARTIFACT,
        },
        {
            "role": "loopback_runtime_provenance_attestation",
            "repository_path": (
                "artifacts/season1/epicure-lineage/"
                "local-attested-20260803-18082-corrected/attestations/"
                f"runtime-provenance-attestation-{ATTESTATION_ARTIFACT}.json"
            ),
            "archive_path": "provenance/epicure-runtime-provenance-attestation.json",
            "artifact_sha256": None,
            "canonical_json_sha256": ATTESTATION_ARTIFACT,
        },
        {
            "role": "exact_source_data_dependency_manifest",
            "repository_path": (
                "artifacts/season1/epicure-lineage/reproducibility/"
                f"epicure-exact-runtime-manifest-{RUNTIME_MANIFEST_ARTIFACT}.json"
            ),
            "archive_path": "provenance/epicure-exact-runtime-manifest.json",
            "artifact_sha256": RUNTIME_MANIFEST_ARTIFACT,
        },
        {
            "role": "exact_study_application_source_archive",
            "repository_path": (
                "artifacts/season1/epicure-lineage/public-reconstruction/"
                f"{APPLICATION_SOURCE_ARCHIVE_FILENAME}"
            ),
            "archive_path": "provenance/epicure-study-application-source.tar.gz",
            "artifact_sha256": None,
            "frozen_file_sha256": APPLICATION_SOURCE_ARCHIVE_SHA256,
            "contains_application_source": True,
        },
        {
            "role": "same_operator_private_rebuild_receipt",
            "repository_path": (
                "artifacts/season1/epicure-lineage/reproducibility/"
                f"epicure-private-offline-rebuild-receipt-{REBUILD_RECEIPT_ARTIFACT}.json"
            ),
            "archive_path": "provenance/epicure-private-offline-rebuild-receipt.json",
            "artifact_sha256": REBUILD_RECEIPT_ARTIFACT,
        },
        {
            "role": "runtime_reconstruction_authority",
            "repository_path": (
                "artifacts/season1/epicure-lineage/reproducibility/"
                f"epicure-runtime-reconstruction-authority-{AUTHORITY_ARTIFACT}.json"
            ),
            "archive_path": "provenance/epicure-runtime-reconstruction-authority.json",
            "artifact_sha256": AUTHORITY_ARTIFACT,
        },
        {
            "role": "candidate_training_lineage_falsification",
            "repository_path": (
                "artifacts/season1/epicure-lineage/candidate-source-trace/"
                f"epicure-candidate-training-lineage-audit-{CANDIDATE_LINEAGE_ARTIFACT}.json"
            ),
            "archive_path": "provenance/epicure-candidate-training-lineage-audit.json",
            "artifact_sha256": CANDIDATE_LINEAGE_ARTIFACT,
        },
        {
            "role": "hash_locked_runtime_dependencies",
            "repository_path": (
                "contracts/epicure/reproducibility/runtime-linux-x86_64-cp312.lock"
            ),
            "archive_path": "provenance/epicure-runtime-dependency-lock.txt",
            "artifact_sha256": None,
        },
        {
            "role": "cyclonedx_runtime_sbom",
            "repository_path": (
                "contracts/epicure/reproducibility/runtime-linux-x86_64-cp312.cdx.json"
            ),
            "archive_path": "provenance/epicure-runtime-sbom.cdx.json",
            "artifact_sha256": None,
        },
        {
            "role": "private_rebuild_verifier_source",
            "repository_path": "contracts/epicure/reproducibility/rebuild_verifier.py",
            "archive_path": "provenance/epicure-private-rebuild-verifier.py",
            "artifact_sha256": None,
        },
        {
            "role": "private_rebuild_recipe",
            "repository_path": "contracts/epicure/reproducibility/PRIVATE-REBUILD.md",
            "archive_path": "provenance/epicure-private-rebuild-recipe.md",
            "artifact_sha256": None,
        },
        {
            "role": "runtime_container_recipe_not_immutable_image",
            "repository_path": "contracts/epicure/reproducibility/Dockerfile.runtime",
            "archive_path": "provenance/epicure-runtime.Dockerfile",
            "artifact_sha256": None,
        },
        {
            "role": "semantic_tool_catalog",
            "repository_path": (f"contracts/epicure/tool-catalog-{TOOL_SCHEMA_SHA256}.json"),
            "archive_path": "provenance/epicure-tool-catalog.json",
            "artifact_sha256": None,
        },
        {
            "role": "reconstruction_and_redistribution_boundary",
            "repository_path": ("contracts/epicure/EPICURE-MCP-1790-R1-REPRODUCIBILITY.md"),
            "archive_path": "provenance/epicure-reproducibility-boundary.md",
            "artifact_sha256": None,
        },
        {
            "role": "public_packet_verifier_source",
            "repository_path": "src/flavourbench/epicure_public_reconstruction.py",
            "archive_path": "provenance/epicure-public-reconstruction-verifier.py",
            "artifact_sha256": None,
        },
    )


def _public_files(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for specification in _evidence_specifications():
        relative = _safe_relative_path(specification["repository_path"], field="repository_path")
        _safe_relative_path(specification["archive_path"], field="archive_path")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise PublicReconstructionError(f"public evidence must be a regular file: {path}")
        file_digest = _file_sha256(path)
        frozen_file_digest = specification.get("frozen_file_sha256")
        if frozen_file_digest is not None and file_digest != frozen_file_digest:
            raise PublicReconstructionError(f"public evidence hash mismatch: {path}")
        entries.append(
            {
                **specification,
                "bytes": path.stat().st_size,
                "file_sha256": file_digest,
                "included_in_public_research_archive": True,
                "contains_application_source": specification.get(
                    "contains_application_source", False
                ),
                "contains_model_or_data_payload": False,
                "contains_credential_material": False,
            }
        )
    return entries


def build_public_packet(root: Path) -> dict[str, Any]:
    """Build a packet from only the evidence safe to include in the archive."""

    root = root.resolve()
    files = _public_files(root)
    by_role = {item["role"]: item for item in files}
    inventory = _content_addressed_json(
        root / by_role["authoritative_recovered_runtime_inventory"]["repository_path"],
        INVENTORY_ARTIFACT,
    )
    manifest = _content_addressed_json(
        root / by_role["exact_source_data_dependency_manifest"]["repository_path"],
        RUNTIME_MANIFEST_ARTIFACT,
    )
    receipt = _content_addressed_json(
        root / by_role["same_operator_private_rebuild_receipt"]["repository_path"],
        REBUILD_RECEIPT_ARTIFACT,
    )
    authority = _content_addressed_json(
        root / by_role["runtime_reconstruction_authority"]["repository_path"],
        AUTHORITY_ARTIFACT,
    )
    candidate = _content_addressed_json(
        root / by_role["candidate_training_lineage_falsification"]["repository_path"],
        CANDIDATE_LINEAGE_ARTIFACT,
    )
    _verify_evidence_chain(inventory=inventory, manifest=manifest, receipt=receipt)
    _verify_candidate_lineage(candidate)
    _verify_manifest_matches_frozen_public_data(manifest)
    _verify_manifest_matches_application_gap(manifest)
    verify_application_source_archive(
        archive_path=root
        / by_role["exact_study_application_source_archive"]["repository_path"],
        runtime_manifest=manifest,
        expected_sha256=APPLICATION_SOURCE_ARCHIVE_SHA256,
    )
    if authority.get("status") != "private_runtime_reconstruction_verified_release_blocked":
        raise PublicReconstructionError("runtime reconstruction authority is not release-blocked")

    if (
        inventory.get("runtime_id") != RUNTIME_ID
        or manifest.get("runtime_id") != RUNTIME_ID
        or manifest.get("data", {}).get("sha256") != BUNDLE_SHA256
        or manifest.get("source", {}).get("sha256") != APPLICATION_SHA256
        or receipt.get("rank_eligible") is not False
        or receipt.get("redistributable") is not False
    ):
        raise PublicReconstructionError("authoritative evidence crosses the runtime boundary")
    inventory_rights = inventory.get("rights")
    if not isinstance(inventory_rights, Mapping) or inventory_rights != {
        "artifact_rights_attestation": None,
        "code_license": "MIT",
        "license_file_sha256": APPLICATION_LICENSE_SHA256,
        "payload_license": None,
        "redistributable_payload": False,
        "source_rights_matrix": None,
        "status": "blocked_pending_data_steward_attestation",
    }:
        raise PublicReconstructionError("runtime rights evidence is missing or overstated")

    return {
        "schema_version": SCHEMA_VERSION,
        "record_role": "public_source_and_data_byte_verification_packet_not_payload_release",
        "release_id": "exploratory-unmatched-1790-runtime",
        "runtime_identity": {
            "runtime_id": RUNTIME_ID,
            "bundle_sha256": BUNDLE_SHA256,
            "application_sha256": APPLICATION_SHA256,
            "tool_schema_sha256": TOOL_SCHEMA_SHA256,
            "ingredient_count": 1790,
            "embedding_dimensions": 300,
        },
        "rights_inventory": {
            "application_code_license_observed": "MIT",
            "application_copyright_notice": "Copyright (c) 2026 KAIKAKU-AI",
            "application_license_file_sha256": APPLICATION_LICENSE_SHA256,
            "application_relicensed_for_archive": False,
            "exact_application_source_included": True,
            "payload_license": None,
            "payload_rights_attestation": None,
            "source_rights_matrix": None,
            "payload_redistribution_cleared": False,
            "status": "code_license_observed_payload_rights_unresolved",
        },
        "public_runtime_data": _expected_public_runtime_data(),
        "application_source_availability": _expected_application_source_availability(),
        "public_files": files,
        "publicly_omitted_requirements": [
            {
                "requirement": "exact_runtime_data_payload_bytes_in_research_archive",
                "included": False,
                "publicly_fetchable_elsewhere": True,
                "reason": (
                    "not_mirrored_in_archive; exact bytes are hash-verified at the frozen public "
                    "Git commit, while payload redistribution rights remain unattested"
                ),
            },
            {
                "requirement": "private_runtime_wheelhouse",
                "included": False,
                "reason": "private_rebuild_input_not_a_public_release",
            },
            {
                "requirement": "original_training_input_embedding",
                "included": False,
                "reason": "not_recovered",
            },
            {
                "requirement": "complete_training_corpus_source_payloads",
                "included": False,
                "reason": "lineage_and_source_rights_incomplete",
            },
            {
                "requirement": "payload_rights_attestation",
                "included": False,
                "reason": "not_attested",
            },
            {
                "requirement": "source_rights_matrix",
                "included": False,
                "reason": "not_attested",
            },
            {
                "requirement": "clean_signed_application_release",
                "included": False,
                "reason": "not_available",
            },
            {
                "requirement": "immutable_oci_image",
                "included": False,
                "reason": "not_available",
            },
            {
                "requirement": "independent_reproduction_receipt",
                "included": False,
                "reason": "not_performed",
            },
        ],
        "verifiable_now": {
            "public_file_integrity": True,
            "runtime_identity_manifest_consistency": True,
            "dependency_lock_and_sbom_consistency": True,
            "same_operator_private_rebuild_receipt_integrity": True,
            "candidate_lineage_rejected_as_exact": True,
            "public_runtime_data_byte_identity": True,
            "research_archive_exact_application_byte_identity": True,
        },
        "not_possible_from_public_packet": {
            "execute_exact_runtime_from_research_archive_alone": True,
            "retrain_exact_representation": True,
            "redistribute_exact_payload": True,
            "claim_independent_reproduction": True,
            "claim_official_rank_eligibility": True,
        },
        "release_gates": {
            "content_addressed_runtime_identity": True,
            "hash_locked_dependency_environment": True,
            "machine_readable_sbom": True,
            "same_operator_private_rebuild": True,
            "public_exact_data_byte_availability": True,
            "research_archive_exact_application_bytes": True,
            "public_executable_reconstruction": False,
            "training_lineage_recovered": False,
            "payload_rights_attested": False,
            "public_redistributable_payload": False,
            "immutable_oci_identity": False,
            "independent_reproduction": False,
        },
        "credential_material_included": False,
        "provider_calls_made": False,
        "epicure_network_calls_made": False,
        "rank_eligible": False,
        "redistributable": False,
        "status": "public_source_and_data_verifiable_release_governance_blocked",
        "interpretation_rule": (
            "The packet lets readers re-hash the published manifests, lock, SBOM, verifier, "
            "private rebuild receipt, all 28 exact study application files in a deterministic "
            "source-only archive, and the exact data files at an immutable public Git commit. "
            "The research archive does not mirror the data or wheelhouse. These technical byte "
            "records do not establish training lineage, payload rights, a clean signed release, "
            "an exact OCI image, public runtime provenance, independent reproduction, or an "
            "attested redistributable release."
        ),
    }


def _verify_packet_document(packet: Mapping[str, Any]) -> None:
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise PublicReconstructionError("public packet has the wrong schema")
    digest = packet.get("artifact_sha256")
    unhashed = {key: value for key, value in packet.items() if key != "artifact_sha256"}
    if not isinstance(digest, str) or digest != _canonical_sha256(unhashed):
        raise PublicReconstructionError("public packet content address is invalid")
    if packet.get("rank_eligible") is not False or packet.get("redistributable") is not False:
        raise PublicReconstructionError("public packet crosses the release boundary")
    if packet.get("credential_material_included") is not False:
        raise PublicReconstructionError("public packet reports credential material")

    expected_rights = {
        "application_code_license_observed": "MIT",
        "application_copyright_notice": "Copyright (c) 2026 KAIKAKU-AI",
        "application_license_file_sha256": APPLICATION_LICENSE_SHA256,
        "application_relicensed_for_archive": False,
        "exact_application_source_included": True,
        "payload_license": None,
        "payload_rights_attestation": None,
        "source_rights_matrix": None,
        "payload_redistribution_cleared": False,
        "status": "code_license_observed_payload_rights_unresolved",
    }
    if packet.get("rights_inventory") != expected_rights:
        raise PublicReconstructionError("public packet rights inventory is not fail-closed")
    if packet.get("public_runtime_data") != _expected_public_runtime_data():
        raise PublicReconstructionError("public runtime data register is not frozen")
    if packet.get("application_source_availability") != _expected_application_source_availability():
        raise PublicReconstructionError("application source availability register is not frozen")

    expected_gates = {
        "content_addressed_runtime_identity": True,
        "hash_locked_dependency_environment": True,
        "machine_readable_sbom": True,
        "same_operator_private_rebuild": True,
        "public_exact_data_byte_availability": True,
        "research_archive_exact_application_bytes": True,
        "public_executable_reconstruction": False,
        "training_lineage_recovered": False,
        "payload_rights_attested": False,
        "public_redistributable_payload": False,
        "immutable_oci_identity": False,
        "independent_reproduction": False,
    }
    if packet.get("release_gates") != expected_gates:
        raise PublicReconstructionError("public packet release gates are not fail-closed")
    unavailable = packet.get("not_possible_from_public_packet")
    if (
        not isinstance(unavailable, Mapping)
        or not unavailable
        or not all(value is True for value in unavailable.values())
    ):
        raise PublicReconstructionError("public packet understates unavailable capabilities")
    omitted = packet.get("publicly_omitted_requirements")
    if not isinstance(omitted, list) or len(omitted) != 9:
        raise PublicReconstructionError("public packet has an incomplete omission register")
    if any(
        not isinstance(item, Mapping)
        or item.get("included") is not False
        or not item.get("requirement")
        or not item.get("reason")
        for item in omitted
    ):
        raise PublicReconstructionError("public omission register is malformed")
    omissions_by_requirement = {
        item["requirement"]: item for item in omitted if isinstance(item, Mapping)
    }
    data_omission = omissions_by_requirement.get(
        "exact_runtime_data_payload_bytes_in_research_archive"
    )
    if not isinstance(data_omission, Mapping) or data_omission.get(
        "publicly_fetchable_elsewhere"
    ) is not True:
        raise PublicReconstructionError("public data availability is understated")


def verify_public_runtime_data(
    packet: Mapping[str, Any],
    *,
    data_dir: Path | None = None,
    online: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Optionally hash exact runtime data from a local directory or immutable raw URLs."""

    if data_dir is not None and online:
        raise PublicReconstructionError("choose either offline data-dir or online verification")
    if timeout <= 0:
        raise PublicReconstructionError("network timeout must be positive")
    register = packet.get("public_runtime_data")
    if register != _expected_public_runtime_data():
        raise PublicReconstructionError("public runtime data register is not frozen")
    if data_dir is None and not online:
        return {"mode": "not_requested", "files_verified": 0}

    files = register.get("files")
    if not isinstance(files, list):
        raise PublicReconstructionError("public runtime data file register is malformed")
    base = data_dir.resolve() if data_dir is not None else None
    for entry in files:
        if not isinstance(entry, Mapping):
            raise PublicReconstructionError("public runtime data file register is malformed")
        relative = _safe_relative_path(entry.get("path"), field="runtime_data_path")
        if base is not None:
            path = base / relative
            resolved = path.resolve()
            if not resolved.is_relative_to(base) or path.is_symlink() or not path.is_file():
                raise PublicReconstructionError(f"offline runtime data file is missing: {path}")
            observed_bytes = path.stat().st_size
            observed_sha256 = _file_sha256(path)
        else:
            raw_url = entry.get("raw_url")
            if not isinstance(raw_url, str) or raw_url != f"{PUBLIC_BUNDLE_RAW_ROOT}/{relative}":
                raise PublicReconstructionError("public runtime data URL is not frozen")
            observed_bytes, observed_sha256 = _download_sha256(raw_url, timeout=timeout)
        if observed_bytes != entry.get("bytes") or observed_sha256 != entry.get("sha256"):
            raise PublicReconstructionError(f"runtime data hash mismatch: {relative}")
    return {
        "mode": "online_immutable_urls" if online else "offline_directory",
        "files_verified": len(files),
        "bundle_sha256": BUNDLE_SHA256,
        "commit": PUBLIC_BUNDLE_COMMIT,
    }


def verify_public_packet(
    *,
    packet_path: Path,
    root: Path,
    layout: str,
    runtime_data_dir: Path | None = None,
    verify_runtime_data_online: bool = False,
    network_timeout: float = 30.0,
) -> dict[str, Any]:
    """Verify the packet and every included file in repository or archive layout."""

    if layout not in {"repository", "archive"}:
        raise PublicReconstructionError("layout must be repository or archive")
    packet = _read_json(packet_path)
    _verify_packet_document(packet)
    root = root.resolve()
    files = packet.get("public_files")
    if not isinstance(files, list) or len(files) != len(_evidence_specifications()):
        raise PublicReconstructionError("public file register is incomplete")

    observed_roles: set[str] = set()
    resolved: dict[str, Path] = {}
    specifications = {item["role"]: item for item in _evidence_specifications()}
    for entry in files:
        if not isinstance(entry, Mapping):
            raise PublicReconstructionError("public file register contains a non-object")
        role = entry.get("role")
        if not isinstance(role, str) or role in observed_roles:
            raise PublicReconstructionError("public file roles must be unique strings")
        observed_roles.add(role)
        specification = specifications.get(role)
        if not isinstance(specification, Mapping) or any(
            entry.get(field) != specification.get(field)
            for field in (
                "repository_path",
                "archive_path",
                "artifact_sha256",
                "canonical_json_sha256",
                "frozen_file_sha256",
            )
        ):
            raise PublicReconstructionError("public file register differs from frozen evidence")
        relative = _safe_relative_path(
            entry.get("repository_path" if layout == "repository" else "archive_path"),
            field=f"{layout}_path",
        )
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise PublicReconstructionError(f"public evidence is missing: {path}")
        if path.stat().st_size != entry.get("bytes") or _file_sha256(path) != entry.get(
            "file_sha256"
        ):
            raise PublicReconstructionError(f"public evidence hash mismatch: {path}")
        if (
            entry.get("included_in_public_research_archive") is not True
            or entry.get("contains_application_source")
            is not (role == "exact_study_application_source_archive")
            or entry.get("contains_model_or_data_payload") is not False
            or entry.get("contains_credential_material") is not False
        ):
            raise PublicReconstructionError("public file boundary flags are invalid")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(path.read_bytes()):
                raise PublicReconstructionError(f"credential-like material detected: {path}")
        artifact_sha256 = entry.get("artifact_sha256")
        if artifact_sha256 is not None:
            _content_addressed_json(path, artifact_sha256)
        canonical_json_sha256 = entry.get("canonical_json_sha256")
        if canonical_json_sha256 is not None:
            if _canonical_sha256(_read_json(path)) != canonical_json_sha256:
                raise PublicReconstructionError(f"canonical JSON evidence hash mismatch: {path}")
        resolved[role] = path

    expected_roles = {item["role"] for item in _evidence_specifications()}
    if observed_roles != expected_roles:
        raise PublicReconstructionError("public file roles do not match the frozen packet")
    inventory = _content_addressed_json(
        resolved["authoritative_recovered_runtime_inventory"], INVENTORY_ARTIFACT
    )
    manifest = _content_addressed_json(
        resolved["exact_source_data_dependency_manifest"], RUNTIME_MANIFEST_ARTIFACT
    )
    receipt = _content_addressed_json(
        resolved["same_operator_private_rebuild_receipt"], REBUILD_RECEIPT_ARTIFACT
    )
    _verify_evidence_chain(inventory=inventory, manifest=manifest, receipt=receipt)
    candidate = _content_addressed_json(
        resolved["candidate_training_lineage_falsification"],
        CANDIDATE_LINEAGE_ARTIFACT,
    )
    _verify_candidate_lineage(candidate)
    if (
        inventory.get("runtime_id") != RUNTIME_ID
        or manifest.get("data", {}).get("sha256") != BUNDLE_SHA256
        or manifest.get("source", {}).get("sha256") != APPLICATION_SHA256
    ):
        raise PublicReconstructionError("public evidence does not bind the frozen runtime")
    _verify_manifest_matches_frozen_public_data(manifest)
    _verify_manifest_matches_application_gap(manifest)
    verify_application_source_archive(
        archive_path=resolved["exact_study_application_source_archive"],
        runtime_manifest=manifest,
        expected_sha256=APPLICATION_SOURCE_ARCHIVE_SHA256,
    )
    verify_public_runtime_data(
        packet,
        data_dir=runtime_data_dir,
        online=verify_runtime_data_online,
        timeout=network_timeout,
    )
    return packet


def write_public_packet(*, root: Path, output_dir: Path) -> Path:
    payload = build_public_packet(root)
    digest = _canonical_sha256(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"epicure-public-reconstruction-packet-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise PublicReconstructionError("content-addressed packet conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source_archive = subparsers.add_parser("build-source-archive")
    source_archive.add_argument("--root", type=Path, required=True)
    source_archive.add_argument("--mcp-root", type=Path, required=True)
    source_archive.add_argument("--output-dir", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--packet", type=Path, required=True)
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--layout", choices=("repository", "archive"), required=True)
    data_mode = verify.add_mutually_exclusive_group()
    data_mode.add_argument(
        "--runtime-data-dir",
        type=Path,
        help="offline directory containing the 11 exact runtime data files",
    )
    data_mode.add_argument(
        "--verify-runtime-data-online",
        action="store_true",
        help="stream and hash the immutable public raw URLs; disabled by default",
    )
    verify.add_argument("--network-timeout", type=float, default=30.0)
    arguments = parser.parse_args(argv)
    if arguments.command == "build-source-archive":
        path = build_application_source_archive(
            root=arguments.root,
            mcp_root=arguments.mcp_root,
            output_dir=arguments.output_dir,
        )
        print(
            json.dumps(
                {
                    "archive_sha256": _file_sha256(path),
                    "credential_material_included": False,
                    "model_or_data_payload_included": False,
                    "output": str(path.resolve()),
                    "status": "exact_study_application_source_archived",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "build":
        path = write_public_packet(root=arguments.root, output_dir=arguments.output_dir)
        packet = _read_json(path)
        runtime_data_verification = "not_applicable"
    else:
        path = arguments.packet
        packet = verify_public_packet(
            packet_path=path,
            root=arguments.root,
            layout=arguments.layout,
            runtime_data_dir=arguments.runtime_data_dir,
            verify_runtime_data_online=arguments.verify_runtime_data_online,
            network_timeout=arguments.network_timeout,
        )
        if arguments.runtime_data_dir is not None:
            runtime_data_verification = "offline_directory"
        elif arguments.verify_runtime_data_online:
            runtime_data_verification = "online_immutable_urls"
        else:
            runtime_data_verification = "not_requested"
    print(
        json.dumps(
            {
                "artifact_sha256": packet["artifact_sha256"],
                "output": str(path.resolve()),
                "status": packet["status"],
                "rank_eligible": False,
                "redistributable": False,
                "provider_calls_made": False,
                "epicure_network_calls_made": False,
                "runtime_data_verification": runtime_data_verification,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
