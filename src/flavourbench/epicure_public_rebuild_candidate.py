"""Build the deterministic Epicure public-input reconstruction candidate.

The candidate packages only source, manifests, verification code, and rights
metadata. It does not package runtime data, wheels, credentials, or training
material and never turns same-operator execution into independent reproduction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from flavourbench.epicure_public_reconstruction import (
    APPLICATION_SHA256,
    APPLICATION_SOURCE_ARCHIVE_FILENAME,
    APPLICATION_SOURCE_ARCHIVE_SHA256,
    BUNDLE_SHA256,
    PUBLIC_BUNDLE_COMMIT,
    PUBLIC_BUNDLE_FILES,
    PUBLIC_BUNDLE_RAW_ROOT,
    RUNTIME_ID,
    RUNTIME_MANIFEST_ARTIFACT,
    TOOL_SCHEMA_SHA256,
    verify_public_packet,
)

KIT_MANIFEST_SCHEMA = "epicure-public-reconstruction-kit-manifest-v1"
DATA_SOURCES_SCHEMA = "epicure-public-runtime-data-sources-v1"
RIGHTS_SCHEMA = "epicure-reconstruction-rights-boundary-v1"
FIXTURE_SCHEMA = "epicure-same-operator-functional-fixture-v1"
CANDIDATE_SCHEMA = "epicure-public-reconstruction-release-candidate-v1"
RECEIPT_SCHEMA = "epicure-public-input-reconstruction-receipt-v1"
RELEASE_ID = "exploratory-unmatched-1790-runtime"
PACKET_V3_SHA256 = "8defca7356994d32126ad8b328d0727cd9cc20af4c4c5497c45d279a096706ad"
DEPENDENCY_LOCK_SHA256 = (
    "86fce704f665270d18a48812b489c651efc8a5688637fa7e89fcd641b9b8d5f1"
)
SBOM_SHA256 = "ec689f51124f307dcaeb1de33007dd6065b97275a9f561a1a84b1bcad97fc25b"
RUNTIME_PAYLOAD_ENVIRONMENT_SHA256 = (
    "f715e57d4c8879916d56aeb6c0983c75ed9f6ad7a96187906cfbb375e8e49fd0"
)
APPLICATION_LICENSE_SHA256 = (
    "65baefcdc9727c4db42391a219389e97970496fc2d252dff32aae7c5c76319fb"
)

_SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"cohere_[A-Za-z0-9._-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)

KIT_MEMBER_SPECS: tuple[tuple[str, str], ...] = (
    ("README.md", "public_reconstruction_readme"),
    ("application-source.tar.gz", "exact_study_application_source"),
    ("data-sources.json", "immutable_public_data_source_map"),
    ("functional-fixtures.json", "same_operator_functional_fixture"),
    ("reconstruct.py", "standalone_public_reconstruction_program"),
    ("rights-boundary.json", "rights_and_lineage_boundary"),
    ("runtime-dependency-lock.txt", "hash_locked_runtime_dependencies"),
    ("runtime-manifest.json", "exact_source_data_dependency_manifest"),
    ("runtime-sbom.cdx.json", "cyclonedx_runtime_sbom"),
    ("tool-catalog.json", "semantic_tool_catalog"),
)


class CandidateError(RuntimeError):
    """The candidate would be incomplete, inconsistent, or overclaim evidence."""


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateError(f"invalid JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise CandidateError(f"JSON evidence is not an object: {path}")
    return value


def _content_address(document: Mapping[str, Any], schema: str) -> None:
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("schema_version") != schema
        or document.get("artifact_sha256") != _canonical_sha256(unhashed)
    ):
        raise CandidateError(f"invalid content-addressed {schema}")


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "artifact_sha256": _canonical_sha256(payload)}


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_exact(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise CandidateError(f"content-addressed output conflict: {path}")
        return path
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)
    return path


def _data_sources_document() -> dict[str, Any]:
    return _with_digest(
        {
            "schema_version": DATA_SOURCES_SCHEMA,
            "record_role": "immutable_public_byte_sources_not_payload_license",
            "runtime_id": RUNTIME_ID,
            "bundle_sha256": BUNDLE_SHA256,
            "repository": "https://github.com/KAIKAKU-AI/epicure-mcp",
            "commit": PUBLIC_BUNDLE_COMMIT,
            "files": [
                {
                    **entry,
                    "url": f"{PUBLIC_BUNDLE_RAW_ROOT}/{entry['path']}",
                }
                for entry in PUBLIC_BUNDLE_FILES
            ],
            "payload_license": None,
            "payload_rights_attestation": None,
            "source_rights_matrix": None,
            "redistribution_cleared": False,
            "interpretation_rule": (
                "The URLs and hashes establish public technical byte availability. They do "
                "not establish a payload licence, upstream-source rights, or authority to "
                "redistribute the bytes in this kit."
            ),
        }
    )


def _rights_document() -> dict[str, Any]:
    return _with_digest(
        {
            "schema_version": RIGHTS_SCHEMA,
            "record_role": "fail_closed_reconstruction_rights_and_lineage_boundary",
            "runtime_id": RUNTIME_ID,
            "application_code": {
                "license": "MIT",
                "license_observed": True,
                "license_sha256": APPLICATION_LICENSE_SHA256,
                "copyright_notice": "Copyright (c) 2026 KAIKAKU-AI",
                "relicensed_for_kit": False,
            },
            "runtime_data": {
                "public_exact_bytes_observed": True,
                "license": None,
                "rights_attestation": None,
                "source_rights_matrix": None,
                "included_in_kit": False,
            },
            "dependency_wheels": {
                "public_pypi_hash_sources_observed": True,
                "included_in_kit": False,
                "aggregate_licence_claim": None,
            },
            "unresolved": [
                "immutable_oci_identity",
                "independent_reproduction",
                "payload_license",
                "payload_rights_attestation",
                "signed_release",
                "source_rights_matrix",
                "training_lineage",
            ],
            "payload_redistribution_cleared": False,
            "rank_eligible": False,
            "official_release": False,
            "interpretation_rule": (
                "A successful technical reconstruction cannot close a legal, training-lineage, "
                "independence, signed-release, image-identity, or governance gate."
            ),
        }
    )


def _fixture_document() -> dict[str, Any]:
    return _with_digest(
        {
            "schema_version": FIXTURE_SCHEMA,
            "record_role": "small_deterministic_same_operator_runtime_parity_fixture",
            "runtime_id": RUNTIME_ID,
            "origin": "same_operator_exact_byte_runtime",
            "golden_fixture_status": "operator_generated_parity_fixture",
            "independent_reproduction": False,
            "provider_calls_made": 0,
            "epicure_network_calls_made": 0,
            "expected": {
                "provenance": {
                    "application_sha256": APPLICATION_SHA256,
                    "bundle_sha256": BUNDLE_SHA256,
                    "embedding_dimensions": 300,
                    "ingredient_count": 1790,
                    "release_id": RELEASE_ID,
                },
                "pairing_score_tomato_basil": {
                    "all_pairs_range": {
                        "median": 0.092,
                        "p10": 0.0197,
                        "p25": 0.0521,
                        "p75": 0.1392,
                        "p90": 0.1899,
                    },
                    "pairing_score": 0.2707,
                    "percentile_label": "very high (>=p90)",
                    "resolved_a": "tomato",
                    "resolved_b": "basil",
                },
                "neighbors_tomato_3": {
                    "ingredient": "tomato",
                    "neighbors": [
                        {"name": "bell_pepper", "rank": 1, "sim": 0.4799},
                        {"name": "olive_oil", "rank": 2, "sim": 0.4786},
                        {"name": "oregano", "rank": 3, "sim": 0.4646},
                    ],
                },
            },
            "interpretation_rule": (
                "The fixture detects reconstruction drift. Because the study operator generated "
                "it, a passing fixture is not independent reproduction or semantic validation."
            ),
        }
    )


def _deterministic_tar_gz(members: Mapping[str, bytes]) -> bytes:
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


def _member_bytes(root: Path) -> dict[str, bytes]:
    sources = {
        "README.md": root / "contracts/epicure/reproducibility/PUBLIC-RECONSTRUCTION-RC.md",
        "application-source.tar.gz": root
        / "artifacts/season1/epicure-lineage/public-reconstruction"
        / APPLICATION_SOURCE_ARCHIVE_FILENAME,
        "reconstruct.py": root / "contracts/epicure/reproducibility/public_rebuild.py",
        "runtime-dependency-lock.txt": root
        / "contracts/epicure/reproducibility/runtime-linux-x86_64-cp312.lock",
        "runtime-manifest.json": root
        / "artifacts/season1/epicure-lineage/reproducibility"
        / f"epicure-exact-runtime-manifest-{RUNTIME_MANIFEST_ARTIFACT}.json",
        "runtime-sbom.cdx.json": root
        / "contracts/epicure/reproducibility/runtime-linux-x86_64-cp312.cdx.json",
        "tool-catalog.json": root / f"contracts/epicure/tool-catalog-{TOOL_SCHEMA_SHA256}.json",
    }
    members: dict[str, bytes] = {}
    for name, path in sources.items():
        if path.is_symlink() or not path.is_file():
            raise CandidateError(f"kit input is not a regular file: {path}")
        members[name] = path.read_bytes()
    members["data-sources.json"] = _json_bytes(_data_sources_document())
    members["rights-boundary.json"] = _json_bytes(_rights_document())
    members["functional-fixtures.json"] = _json_bytes(_fixture_document())
    if set(members) != {name for name, _role in KIT_MEMBER_SPECS}:
        raise CandidateError("kit input register is incomplete")
    if hashlib.sha256(members["application-source.tar.gz"]).hexdigest() != (
        APPLICATION_SOURCE_ARCHIVE_SHA256
    ):
        raise CandidateError("application source archive identity changed")
    if hashlib.sha256(members["runtime-dependency-lock.txt"]).hexdigest() != (
        DEPENDENCY_LOCK_SHA256
    ):
        raise CandidateError("dependency lock identity changed")
    if hashlib.sha256(members["runtime-sbom.cdx.json"]).hexdigest() != SBOM_SHA256:
        raise CandidateError("runtime SBOM identity changed")
    if _canonical_sha256(json.loads(members["tool-catalog.json"])) != TOOL_SCHEMA_SHA256:
        raise CandidateError("tool catalog identity changed")
    for name, data in members.items():
        if any(pattern.search(data) for pattern in _SECRET_PATTERNS):
            raise CandidateError(f"credential-like material detected in kit member: {name}")
    return members


def build_kit(root: Path, output_dir: Path) -> Path:
    """Build the deterministic source/manifest/verifier kit and verify it offline."""

    root = root.resolve()
    packet = root / (
        "artifacts/season1/epicure-lineage/public-reconstruction/"
        f"epicure-public-reconstruction-packet-{PACKET_V3_SHA256}.json"
    )
    verified_packet = verify_public_packet(
        packet_path=packet,
        root=root,
        layout="repository",
    )
    if verified_packet.get("status") != (
        "public_source_and_data_verifiable_release_governance_blocked"
    ):
        raise CandidateError("packet v3 is not the frozen release-blocked predecessor")
    members = _member_bytes(root)
    roles = dict(KIT_MEMBER_SPECS)
    manifest_payload = {
        "schema_version": KIT_MANIFEST_SCHEMA,
        "record_role": "portable_public_input_reconstruction_candidate_not_payload_release",
        "runtime_id": RUNTIME_ID,
        "release_id": RELEASE_ID,
        "runtime_identities": {
            "application_sha256": APPLICATION_SHA256,
            "bundle_sha256": BUNDLE_SHA256,
            "tool_schema_sha256": TOOL_SCHEMA_SHA256,
            "source_archive_sha256": APPLICATION_SOURCE_ARCHIVE_SHA256,
            "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
            "sbom_sha256": SBOM_SHA256,
        },
        "target": "CPython 3.12 / Linux x86_64 / manylinux-compatible glibc",
        "members": [
            {
                "path": name,
                "role": roles[name],
                "bytes": len(members[name]),
                "sha256": hashlib.sha256(members[name]).hexdigest(),
                "contains_runtime_data": False,
                "contains_dependency_wheel": False,
                "contains_credential_material": False,
            }
            for name in sorted(members)
        ],
        "technical_reconstruction_candidate": True,
        "contains_runtime_data": False,
        "contains_dependency_wheels": False,
        "contains_training_material": False,
        "contains_credentials": False,
        "payload_redistribution_cleared": False,
        "independent_reproduction": False,
        "rank_eligible": False,
        "interpretation_rule": (
            "The kit can fetch and verify exact public inputs on its frozen target. It is not "
            "a model payload release, licence grant, training-lineage record, OCI identity, "
            "independent reproduction, or official-ranking authorization."
        ),
    }
    kit_manifest = _with_digest(manifest_payload)
    archive_members = {**members, "KIT-MANIFEST.json": _json_bytes(kit_manifest)}
    rendered = _deterministic_tar_gz(archive_members)
    digest = hashlib.sha256(rendered).hexdigest()
    output = output_dir / f"epicure-public-reconstruction-kit-{digest}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="epicure-public-kit-build-") as temporary:
        staged = Path(temporary) / output.name
        _write_exact(staged, rendered)
        verify_kit_archive(staged, expected_sha256=digest)
    _write_exact(output, rendered)
    return output


def verify_kit_archive(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    """Extract safely and run the packaged offline verifier."""

    if path.is_symlink() or not path.is_file() or _file_sha256(path) != expected_sha256:
        raise CandidateError("reconstruction kit archive hash mismatch")
    raw = path.read_bytes()
    if len(raw) < 10 or raw[:2] != b"\x1f\x8b" or int.from_bytes(raw[4:8], "little") != 0:
        raise CandidateError("reconstruction kit archive has a non-deterministic header")
    with tempfile.TemporaryDirectory(prefix="epicure-public-kit-") as temporary:
        root = Path(temporary)
        try:
            with tarfile.open(path, mode="r:gz") as archive:
                members = archive.getmembers()
                expected_names = {"KIT-MANIFEST.json", *{name for name, _ in KIT_MEMBER_SPECS}}
                names = [member.name for member in members]
                if names != sorted(expected_names) or len(names) != len(set(names)):
                    raise CandidateError("reconstruction kit member set is not exact")
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
                        raise CandidateError(
                            f"reconstruction kit metadata is not deterministic: {member.name}"
                        )
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise CandidateError(f"unreadable reconstruction kit member: {member.name}")
                    _write_exact(root / member.name, handle.read())
        except (OSError, tarfile.TarError) as error:
            raise CandidateError("reconstruction kit archive is invalid") from error
        process = subprocess.run(
            [
                sys.executable,
                str(root / "reconstruct.py"),
                "verify-kit",
                "--kit-root",
                str(root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise CandidateError(
                f"packaged offline verifier failed: {process.stderr.strip()}"
            )
        try:
            report = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise CandidateError("packaged offline verifier returned invalid JSON") from error
        if report.get("status") != "public_input_reconstruction_kit_verified_offline":
            raise CandidateError("packaged offline verifier returned the wrong status")
        return report


def _verify_execution_receipt(path: Path, kit_manifest_sha256: str) -> dict[str, Any]:
    receipt = _read_json(path)
    _content_address(receipt, RECEIPT_SCHEMA)
    if (
        receipt.get("kit_manifest_sha256") != kit_manifest_sha256
        or receipt.get("runtime_id") != RUNTIME_ID
        or receipt.get("source_sha256") != APPLICATION_SHA256
        or receipt.get("data_sha256") != BUNDLE_SHA256
        or receipt.get("dependency_lock_sha256") != DEPENDENCY_LOCK_SHA256
        or receipt.get("sbom_sha256") != SBOM_SHA256
        or receipt.get("dependency_wheels_verified") != 41
        or receipt.get("runtime_payload_environment_sha256")
        != RUNTIME_PAYLOAD_ENVIRONMENT_SHA256
        or receipt.get("installed_distribution_integrity", {}).get(
            "all_declared_record_hashes_match_physical_files"
        )
        is not True
        or receipt.get("functional_fixture_passed") is not True
        or receipt.get("independent_reproduction") is not False
        or receipt.get("payload_license") is not None
        or receipt.get("payload_rights_attestation") is not None
        or receipt.get("payload_redistribution_cleared") is not False
        or receipt.get("training_lineage_recovered") is not False
        or receipt.get("immutable_oci_identity") is not None
        or receipt.get("rank_eligible") is not False
        or receipt.get("operator_independence") != "not_adjudicated"
    ):
        raise CandidateError("execution receipt crosses the evidence boundary")
    return receipt


def write_candidate(
    *,
    kit_archive: Path,
    execution_receipt: Path,
    output_dir: Path,
) -> Path:
    """Bind the kit and a same-operator/unadjudicated public-input execution receipt."""

    kit_sha256 = _file_sha256(kit_archive)
    kit_report = verify_kit_archive(kit_archive, expected_sha256=kit_sha256)
    receipt = _verify_execution_receipt(
        execution_receipt,
        str(kit_report["kit_manifest_sha256"]),
    )
    archived_receipt = _write_exact(
        output_dir / execution_receipt.name,
        execution_receipt.read_bytes(),
    )
    payload = {
        "schema_version": CANDIDATE_SCHEMA,
        "record_role": "public_input_executable_reconstruction_candidate_not_official_release",
        "runtime_id": RUNTIME_ID,
        "release_id": RELEASE_ID,
        "predecessor_packet_v3_sha256": PACKET_V3_SHA256,
        "kit": {
            "filename": kit_archive.name,
            "bytes": kit_archive.stat().st_size,
            "sha256": kit_sha256,
            "manifest_sha256": kit_report["kit_manifest_sha256"],
        },
        "same_operator_or_unadjudicated_execution_receipt": {
            "filename": archived_receipt.name,
            "bytes": archived_receipt.stat().st_size,
            "sha256": _file_sha256(archived_receipt),
            "artifact_sha256": receipt["artifact_sha256"],
            "functional_fixture_passed": True,
            "dependency_wheels_verified": 41,
        },
        "fixed_technical_gaps": [
            "portable_exact_application_source_bytes",
            "immutable_exact_runtime_data_source_map",
            "public_pypi_acquisition_of_all_41_hash_locked_wheels",
            "offline_exact_wheel_installation",
            "installed_distribution_record_and_runtime_payload_verification",
            "cross_file_runtime_manifest_and_sbom_verification",
            "operator_generated_deterministic_tool_parity_fixture",
            "public_input_runtime_materialization_and_functional_probe",
        ],
        "remaining_governance_blockers": [
            {
                "gate": "payload_license_and_rights_attestation",
                "status": "unresolved",
                "required_evidence": "signed_payload_attestation_and_source_rights_matrix",
            },
            {
                "gate": "training_lineage",
                "status": "unresolved",
                "required_evidence": (
                    "training_run_seed_code_environment_and_inputs_or_approved_opaque_boundary"
                ),
            },
            {
                "gate": "clean_signed_release",
                "status": "unresolved",
                "required_evidence": "clean_signed_source_tag_binding_exact_study_bytes",
            },
            {
                "gate": "immutable_oci_identity",
                "status": "unresolved",
                "required_evidence": "published_content_addressed_image_digest",
            },
            {
                "gate": "independent_reproduction",
                "status": "unresolved",
                "required_evidence": (
                    "receipt_and_fixture_parity_from_operator_independent_of_study_authors"
                ),
            },
            {
                "gate": "public_runtime_provenance",
                "status": "unresolved",
                "required_evidence": "public_provenance_endpoint_serving_exact_studied_identities",
            },
            {
                "gate": "official_benchmark_governance",
                "status": "unresolved",
                "required_evidence": "documented_release_and_ranking_approval",
            },
        ],
        "technical_public_input_reconstruction_verified": True,
        "operator_independence": "not_adjudicated",
        "independent_reproduction": False,
        "payload_license": None,
        "payload_rights_attestation": None,
        "source_rights_matrix": None,
        "payload_redistribution_cleared": False,
        "training_lineage_recovered": False,
        "clean_signed_release": False,
        "immutable_oci_identity": None,
        "public_runtime_provenance_verified": False,
        "official_release": False,
        "rank_eligible": False,
        "provider_calls_made": 0,
        "epicure_network_calls_made": 0,
        "status": "public_input_reconstruction_verified_release_governance_blocked",
        "interpretation_rule": (
            "The exact study runtime can now be materialized and probed from the kit and public "
            "inputs on the frozen target. The execution remains same-operator or unadjudicated. "
            "It does not establish rights, training lineage, a signed release, OCI identity, "
            "independent reproduction, public-service provenance, or ranking authority."
        ),
    }
    document = _with_digest(payload)
    output = output_dir / (
        f"epicure-public-reconstruction-release-candidate-{document['artifact_sha256']}.json"
    )
    return _write_exact(output, _json_bytes(document))


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-kit")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-kit")
    verify.add_argument("--kit", type=Path, required=True)
    candidate = subparsers.add_parser("write-candidate")
    candidate.add_argument("--kit", type=Path, required=True)
    candidate.add_argument("--execution-receipt", type=Path, required=True)
    candidate.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "build-kit":
        output = build_kit(arguments.root, arguments.output_dir)
        report = verify_kit_archive(output, expected_sha256=_file_sha256(output))
        report.update({"kit": str(output), "kit_sha256": _file_sha256(output)})
    elif arguments.command == "verify-kit":
        output = arguments.kit
        report = verify_kit_archive(output, expected_sha256=_file_sha256(output))
        report.update({"kit": str(output), "kit_sha256": _file_sha256(output)})
    else:
        output = write_candidate(
            kit_archive=arguments.kit,
            execution_receipt=arguments.execution_receipt,
            output_dir=arguments.output_dir,
        )
        report = {
            "candidate": str(output),
            "artifact_sha256": _read_json(output)["artifact_sha256"],
            "status": "public_input_reconstruction_verified_release_governance_blocked",
            "rank_eligible": False,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
