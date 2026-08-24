"""Verify and materialize the Epicure public-input reconstruction candidate.

This program uses only the Python standard library. Network access and payload
materialization are explicit. A successful run verifies technical byte parity;
it does not establish payload rights, training lineage, or independent review.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import venv
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

KIT_SCHEMA = "epicure-public-reconstruction-kit-manifest-v1"
RUNTIME_MANIFEST_SCHEMA = "epicure-exact-runtime-manifest-v1"
FIXTURE_SCHEMA = "epicure-same-operator-functional-fixture-v1"
RIGHTS_SCHEMA = "epicure-reconstruction-rights-boundary-v1"
DATA_SOURCES_SCHEMA = "epicure-public-runtime-data-sources-v1"
RECEIPT_SCHEMA = "epicure-public-input-reconstruction-receipt-v1"

RUNTIME_ID = "epicure-mcp-1790-r1+bundle.98d0403115bf.app.be4216ae799f"
RELEASE_ID = "exploratory-unmatched-1790-runtime"
BUNDLE_SHA256 = "98d0403115bf8eb4fb71dbb89a53362e9b9acafda7494c464763df7d764174d1"
APPLICATION_SHA256 = "be4216ae799f330b76f1f8c009ad2e79ce891d5dec452854dbc449e66e2eb313"
TOOL_SCHEMA_SHA256 = "666a9a44b3534d7c8321f179e4513f71e0acf0d281a01fb3f41be5f8c0dfc8dd"
SOURCE_ARCHIVE_SHA256 = (
    "d08fb475e9c325a8c41daf5b789e6b4bca547228139eece4578f9b06c324703c"
)
DEPENDENCY_LOCK_SHA256 = (
    "86fce704f665270d18a48812b489c651efc8a5688637fa7e89fcd641b9b8d5f1"
)
SBOM_SHA256 = "ec689f51124f307dcaeb1de33007dd6065b97275a9f561a1a84b1bcad97fc25b"
RUNTIME_PAYLOAD_ENVIRONMENT_SHA256 = (
    "f715e57d4c8879916d56aeb6c0983c75ed9f6ad7a96187906cfbb375e8e49fd0"
)

PYPI_INDEX = "https://pypi.org/simple"
PUBLIC_DATA_COMMIT = "14ddf04aba81a76b75efa6554041f6bff48992c6"
PUBLIC_DATA_RAW_ROOT = (
    "https://raw.githubusercontent.com/KAIKAKU-AI/epicure-mcp/"
    f"{PUBLIC_DATA_COMMIT}/data"
)

KIT_FILES = {
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

_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)\s+(.*)$")
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})(?:\s|$)")


class ReconstructionError(RuntimeError):
    """The kit, an input, or a reconstruction result failed closed."""


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
        raise ReconstructionError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReconstructionError(f"JSON root is not an object: {path}")
    return value


def _read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReconstructionError(f"invalid JSON: {path}") from error


def _verify_content_address(document: Mapping[str, Any], *, schema: str) -> None:
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    if (
        document.get("schema_version") != schema
        or not isinstance(digest, str)
        or digest != _canonical_sha256(unhashed)
    ):
        raise ReconstructionError(f"invalid content-addressed {schema}")


def _safe_relative(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReconstructionError(f"{field} is not a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ReconstructionError(f"unsafe {field}: {value}")
    return value


def _regular_file(root: Path, relative: str) -> Path:
    path = root / relative
    resolved = path.resolve()
    if (
        not resolved.is_relative_to(root.resolve())
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ReconstructionError(f"registered kit member is not a regular file: {relative}")
    return path


def _normalise_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _parse_lock(path: Path) -> list[dict[str, Any]]:
    logical: list[str] = []
    pending = ""
    binary_only = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "--only-binary=:all:":
            if pending:
                raise ReconstructionError("binary-only directive interrupts a requirement")
            binary_only = True
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    if pending or not binary_only:
        raise ReconstructionError("dependency lock is incomplete or not binary-only")

    packages: list[dict[str, Any]] = []
    names: set[str] = set()
    for line in logical:
        match = _PIN.fullmatch(line)
        if not match:
            raise ReconstructionError(f"dependency is not exactly pinned: {line}")
        name = _normalise_name(match.group(1))
        hashes = sorted(set(_HASH.findall(match.group(3))))
        residue = _HASH.sub("", match.group(3)).strip()
        if name in names or len(hashes) != 1 or residue:
            raise ReconstructionError(f"dependency lock entry is not exact: {name}")
        packages.append({"name": name, "version": match.group(2), "sha256": hashes[0]})
        names.add(name)
    if len(packages) != 41 or [item["name"] for item in packages] != sorted(names):
        raise ReconstructionError("dependency lock does not contain the frozen 41-package set")
    return packages


def _verify_sbom(sbom: Mapping[str, Any], packages: Sequence[Mapping[str, Any]]) -> None:
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.5":
        raise ReconstructionError("runtime SBOM is not CycloneDX 1.5")
    components = sbom.get("components")
    if not isinstance(components, list):
        raise ReconstructionError("runtime SBOM has no component register")
    observed: list[dict[str, Any]] = []
    for component in components:
        if not isinstance(component, Mapping):
            raise ReconstructionError("runtime SBOM component is malformed")
        hashes = component.get("hashes")
        if not isinstance(hashes, list) or len(hashes) != 1:
            raise ReconstructionError("runtime SBOM component hash is malformed")
        hash_entry = hashes[0]
        if not isinstance(hash_entry, Mapping) or hash_entry.get("alg") != "SHA-256":
            raise ReconstructionError("runtime SBOM component is not SHA-256 bound")
        observed.append(
            {
                "name": _normalise_name(str(component.get("name"))),
                "version": component.get("version"),
                "sha256": hash_entry.get("content"),
            }
        )
    if observed != list(packages):
        raise ReconstructionError("runtime SBOM differs from the dependency lock")


def _verify_source_archive(path: Path, runtime: Mapping[str, Any]) -> None:
    if _file_sha256(path) != SOURCE_ARCHIVE_SHA256:
        raise ReconstructionError("application source archive hash mismatch")
    source = runtime.get("source")
    if not isinstance(source, Mapping) or source.get("sha256") != APPLICATION_SHA256:
        raise ReconstructionError("runtime source identity is not frozen")
    entries = source.get("entries")
    if not isinstance(entries, list) or len(entries) != 28:
        raise ReconstructionError("runtime source register is incomplete")
    expected = {
        f"src/epicure_mcp/{_safe_relative(item.get('path'), field='source path')}": (
            item.get("bytes"),
            item.get("sha256"),
        )
        for item in entries
        if isinstance(item, Mapping)
    }
    expected_names = {"LICENSE", "MANIFEST.json", *expected}
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if names != sorted(expected_names) or len(names) != len(set(names)):
                raise ReconstructionError("application source archive member set is not exact")
            for member in members:
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.mode != 0o644
                    or member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                ):
                    raise ReconstructionError(
                        f"application source metadata is not deterministic: {member.name}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise ReconstructionError(f"unreadable source member: {member.name}")
                data = handle.read()
                if member.name in expected:
                    size, digest = expected[member.name]
                    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                        raise ReconstructionError(
                            f"application source member hash mismatch: {member.name}"
                        )
    except (OSError, tarfile.TarError) as error:
        raise ReconstructionError("application source archive is invalid") from error


def _verify_data_sources(
    sources: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    _verify_content_address(sources, schema=DATA_SOURCES_SCHEMA)
    if (
        sources.get("commit") != PUBLIC_DATA_COMMIT
        or sources.get("bundle_sha256") != BUNDLE_SHA256
        or sources.get("payload_license") is not None
        or sources.get("payload_rights_attestation") is not None
        or sources.get("redistribution_cleared") is not False
    ):
        raise ReconstructionError("data-source map crosses the technical or rights boundary")
    runtime_data = runtime.get("data")
    if not isinstance(runtime_data, Mapping):
        raise ReconstructionError("runtime data register is missing")
    entries = runtime_data.get("entries")
    mapped = sources.get("files")
    if not isinstance(entries, list) or not isinstance(mapped, list) or len(mapped) != 11:
        raise ReconstructionError("data-source map is incomplete")
    expected = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ReconstructionError("runtime data register is malformed")
        relative = _safe_relative(entry.get("path"), field="runtime data path")
        expected.append(
            {
                "path": relative,
                "bytes": entry.get("bytes"),
                "sha256": entry.get("sha256"),
                "url": f"{PUBLIC_DATA_RAW_ROOT}/{relative}",
            }
        )
    if mapped != expected:
        raise ReconstructionError("data-source map differs from the runtime manifest")
    return expected


def _verify_rights_boundary(document: Mapping[str, Any]) -> None:
    _verify_content_address(document, schema=RIGHTS_SCHEMA)
    application_code = document.get("application_code")
    if not isinstance(application_code, Mapping) or {
        key: application_code.get(key)
        for key in ("license", "license_observed", "relicensed_for_kit")
    } != {
        "license": "MIT",
        "license_observed": True,
        "relicensed_for_kit": False,
    }:
        raise ReconstructionError("application-code rights record is not frozen")
    unresolved = document.get("unresolved")
    if not isinstance(unresolved, list) or {
        "payload_license",
        "payload_rights_attestation",
        "source_rights_matrix",
        "training_lineage",
        "independent_reproduction",
        "immutable_oci_identity",
        "signed_release",
    } != set(unresolved):
        raise ReconstructionError("rights and lineage omission register is incomplete")
    if (
        document.get("payload_redistribution_cleared") is not False
        or document.get("rank_eligible") is not False
        or document.get("official_release") is not False
    ):
        raise ReconstructionError("rights record crosses an unresolved release gate")


def _verify_fixture(document: Mapping[str, Any]) -> None:
    _verify_content_address(document, schema=FIXTURE_SCHEMA)
    if (
        document.get("runtime_id") != RUNTIME_ID
        or document.get("origin") != "same_operator_exact_byte_runtime"
        or document.get("independent_reproduction") is not False
        or document.get("golden_fixture_status") != "operator_generated_parity_fixture"
    ):
        raise ReconstructionError("functional fixture overstates its provenance")
    expected = document.get("expected")
    if not isinstance(expected, Mapping) or expected.get("provenance") != {
        "application_sha256": APPLICATION_SHA256,
        "bundle_sha256": BUNDLE_SHA256,
        "embedding_dimensions": 300,
        "ingredient_count": 1790,
        "release_id": RELEASE_ID,
    }:
        raise ReconstructionError("functional fixture has the wrong runtime identity")


def verify_kit(root: Path) -> dict[str, Any]:
    """Verify every kit byte and all cross-file invariants without network access."""

    root = root.resolve()
    if not root.is_dir():
        raise ReconstructionError("kit root is not a directory")
    manifest = _read_json(_regular_file(root, "KIT-MANIFEST.json"))
    _verify_content_address(manifest, schema=KIT_SCHEMA)
    if manifest.get("runtime_id") != RUNTIME_ID or manifest.get("release_id") != RELEASE_ID:
        raise ReconstructionError("kit manifest has the wrong runtime identity")
    if manifest.get("technical_reconstruction_candidate") is not True:
        raise ReconstructionError("kit is not marked as a technical reconstruction candidate")
    if (
        manifest.get("independent_reproduction") is not False
        or manifest.get("payload_redistribution_cleared") is not False
        or manifest.get("rank_eligible") is not False
    ):
        raise ReconstructionError("kit manifest crosses an unresolved release gate")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != len(KIT_FILES):
        raise ReconstructionError("kit member register is incomplete")
    by_path: dict[str, Mapping[str, Any]] = {}
    for entry in members:
        if not isinstance(entry, Mapping):
            raise ReconstructionError("kit member register is malformed")
        relative = _safe_relative(entry.get("path"), field="kit member path")
        if relative in by_path:
            raise ReconstructionError("kit member paths are not unique")
        path = _regular_file(root, relative)
        if path.stat().st_size != entry.get("bytes") or _file_sha256(path) != entry.get(
            "sha256"
        ):
            raise ReconstructionError(f"kit member hash mismatch: {relative}")
        by_path[relative] = entry
    if set(by_path) != KIT_FILES:
        raise ReconstructionError("kit member paths differ from the frozen layout")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if observed != {"KIT-MANIFEST.json", *KIT_FILES}:
        raise ReconstructionError("kit directory contains unregistered files")

    runtime = _read_json(root / "runtime-manifest.json")
    _verify_content_address(runtime, schema=RUNTIME_MANIFEST_SCHEMA)
    if (
        runtime.get("runtime_id") != RUNTIME_ID
        or runtime.get("data", {}).get("sha256") != BUNDLE_SHA256
        or runtime.get("source", {}).get("sha256") != APPLICATION_SHA256
        or runtime.get("dependency_lock", {}).get("sha256") != DEPENDENCY_LOCK_SHA256
        or runtime.get("sbom", {}).get("sha256") != SBOM_SHA256
    ):
        raise ReconstructionError("runtime manifest identities are inconsistent")

    lock = root / "runtime-dependency-lock.txt"
    if _file_sha256(lock) != DEPENDENCY_LOCK_SHA256:
        raise ReconstructionError("dependency lock hash mismatch")
    packages = _parse_lock(lock)
    sbom_path = root / "runtime-sbom.cdx.json"
    if _file_sha256(sbom_path) != SBOM_SHA256:
        raise ReconstructionError("runtime SBOM hash mismatch")
    _verify_sbom(_read_json(sbom_path), packages)
    _verify_source_archive(root / "application-source.tar.gz", runtime)
    data_files = _verify_data_sources(_read_json(root / "data-sources.json"), runtime)
    _verify_rights_boundary(_read_json(root / "rights-boundary.json"))
    _verify_fixture(_read_json(root / "functional-fixtures.json"))
    tool_catalog = _read_json_value(root / "tool-catalog.json")
    if _canonical_sha256(tool_catalog) != TOOL_SCHEMA_SHA256:
        raise ReconstructionError("tool catalog identity mismatch")
    return {
        "kit_manifest_sha256": manifest["artifact_sha256"],
        "runtime_id": RUNTIME_ID,
        "source_files_verified": 28,
        "data_files_registered": len(data_files),
        "dependency_wheels_registered": len(packages),
        "network_calls_made": 0,
        "payload_redistribution_cleared": False,
        "independent_reproduction": False,
        "rank_eligible": False,
        "status": "public_input_reconstruction_kit_verified_offline",
    }


def _write_exact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ReconstructionError(f"existing output differs: {path}")
        return
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    path.chmod(0o644)


def _download_exact(url: str, *, size: int, digest: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Epicure-Public-Reconstruction/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60.0) as response:
            data = response.read()
    except (OSError, urllib.error.URLError) as error:
        raise ReconstructionError(f"could not fetch frozen input: {url}") from error
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ReconstructionError(f"downloaded input hash mismatch: {url}")
    _write_exact(destination, data)


def _verify_data_directory(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    root = path.resolve()
    if path.is_symlink() or not root.is_dir():
        raise ReconstructionError("runtime data directory is missing")
    expected: set[str] = set()
    for entry in entries:
        relative = _safe_relative(entry.get("path"), field="runtime data path")
        expected.add(relative)
        source = _regular_file(root, relative)
        if source.stat().st_size != entry.get("bytes") or _file_sha256(source) != entry.get(
            "sha256"
        ):
            raise ReconstructionError(f"runtime data hash mismatch: {relative}")
    observed = {
        file.relative_to(root).as_posix()
        for file in root.rglob("*")
        if file.is_file() and not file.is_symlink()
    }
    if observed != expected:
        raise ReconstructionError("runtime data directory contains missing or extra files")


def _extract_source(source_archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_archive, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.startswith("src/epicure_mcp/"):
                continue
            relative = _safe_relative(member.name, field="application source member")
            handle = archive.extractfile(member)
            if handle is None:
                raise ReconstructionError(f"unreadable source member: {relative}")
            _write_exact(destination / relative, handle.read())


def _verify_source_directory(path: Path, runtime: Mapping[str, Any]) -> None:
    root = path.resolve() / "src/epicure_mcp"
    source = runtime.get("source")
    entries = source.get("entries") if isinstance(source, Mapping) else None
    if not isinstance(entries, list):
        raise ReconstructionError("runtime source register is unavailable")
    expected: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ReconstructionError("runtime source register is malformed")
        relative = _safe_relative(entry.get("path"), field="runtime source path")
        expected.add(relative)
        source_path = _regular_file(root, relative)
        if source_path.stat().st_size != entry.get("bytes") or _file_sha256(
            source_path
        ) != entry.get("sha256"):
            raise ReconstructionError(f"runtime source hash mismatch: {relative}")
    observed = {
        file.relative_to(root).as_posix()
        for file in root.rglob("*.py")
        if file.is_file() and not file.is_symlink()
    }
    if observed != expected:
        raise ReconstructionError("runtime source directory contains missing or extra Python files")


def _verify_wheelhouse(path: Path, packages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    root = path.resolve()
    if path.is_symlink() or not root.is_dir():
        raise ReconstructionError("wheelhouse is missing")
    expected = {str(package["sha256"]) for package in packages}
    entries: list[dict[str, Any]] = []
    for wheel in sorted(root.iterdir(), key=lambda item: item.name):
        if wheel.is_symlink() or not wheel.is_file() or wheel.suffix != ".whl":
            raise ReconstructionError("wheelhouse must contain only regular wheel files")
        entries.append(
            {
                "filename": wheel.name,
                "bytes": wheel.stat().st_size,
                "sha256": _file_sha256(wheel),
            }
        )
    observed = {entry["sha256"] for entry in entries}
    if len(entries) != 41 or observed != expected:
        raise ReconstructionError("wheelhouse is not an exact cover of the dependency lock")
    return entries


def _assert_target_python(python: Path) -> None:
    command = [
        str(python),
        "-c",
        (
            "import json,platform,sys;"
            "print(json.dumps({'implementation':platform.python_implementation(),"
            "'major':sys.version_info.major,'minor':sys.version_info.minor,"
            "'system':platform.system(),'machine':platform.machine()}))"
        ),
    ]
    try:
        observed = json.loads(subprocess.check_output(command, text=True))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ReconstructionError("could not inspect the requested Python interpreter") from error
    if (
        observed.get("implementation") != "CPython"
        or (observed.get("major"), observed.get("minor")) != (3, 12)
        or observed.get("system") != "Linux"
        or observed.get("machine") not in {"x86_64", "AMD64"}
    ):
        raise ReconstructionError(
            "reconstruction requires CPython 3.12 on Linux x86-64"
        )


def _download_wheels(python: Path, lock: Path, wheelhouse: Path) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-m",
        "pip",
        "download",
        "--no-deps",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--require-hashes",
        "--index-url",
        PYPI_INDEX,
        "--dest",
        str(wheelhouse),
        "-r",
        str(lock),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReconstructionError("public PyPI wheel acquisition failed") from error


def _install_dependencies(python: Path, lock: Path, wheelhouse: Path, venv_root: Path) -> Path:
    if venv_root.exists():
        raise ReconstructionError("clean dependency installation requires an absent .venv")
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=False)
    builder.create(venv_root)
    installed_python = venv_root / "bin/python"
    command = [
        str(installed_python),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--require-hashes",
        "--find-links",
        str(wheelhouse),
        "-r",
        str(lock),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReconstructionError("offline dependency installation failed") from error
    return installed_python


def _distribution_integrity(lock: Path) -> dict[str, Any]:
    """Verify installed distributions against wheel RECORD hashes and bind runtime bytes."""

    packages = _parse_lock(lock)
    distributions: list[dict[str, Any]] = []
    runtime_payloads: list[dict[str, Any]] = []
    for package in packages:
        try:
            distribution = importlib.metadata.distribution(str(package["name"]))
        except importlib.metadata.PackageNotFoundError as error:
            raise ReconstructionError(
                f"locked dependency is not installed: {package['name']}"
            ) from error
        if distribution.version != package["version"]:
            raise ReconstructionError(
                f"installed version differs from lock: {package['name']}"
            )
        entries: list[dict[str, Any]] = []
        unhashed_paths: list[str] = []
        record_verified = 0
        installation_specific = 0
        for item in sorted(distribution.files or (), key=lambda value: str(value)):
            path = Path(distribution.locate_file(item))
            if path.is_symlink() or not path.is_file():
                raise ReconstructionError(
                    f"installed distribution file is missing or symbolic: "
                    f"{package['name']}:{item}"
                )
            digest = _file_sha256(path)
            declared = item.hash
            if declared is None:
                unhashed_paths.append(str(item).replace(os.sep, "/"))
                continue
            if declared.mode != "sha256":
                raise ReconstructionError(
                    f"unsupported RECORD hash for {package['name']}:{item}"
                )
            observed = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
            if observed != declared.value:
                raise ReconstructionError(
                    f"installed file differs from RECORD: {package['name']}:{item}"
                )
            record_verified += 1
            relative = str(item).replace(os.sep, "/")
            if relative.startswith("../") or ".dist-info/" in relative:
                installation_specific += 1
                continue
            entries.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
        runtime_payload = {
            "name": package["name"],
            "version": distribution.version,
            "file_count": len(entries),
            "bytes": sum(entry["bytes"] for entry in entries),
            "physical_files_manifest_sha256": _canonical_sha256(entries),
        }
        runtime_payloads.append(runtime_payload)
        distributions.append(
            {
                **runtime_payload,
                "record_entry_count": (
                    len(entries) + installation_specific + len(unhashed_paths)
                ),
                "physically_verified_file_count": len(entries),
                "record_verified_file_count": record_verified,
                "installation_specific_verified_file_count": installation_specific,
                "unhashed_record_file_count": len(unhashed_paths),
                "unhashed_record_paths_sha256": _canonical_sha256(unhashed_paths),
            }
        )
    report = {
        "schema_version": "epicure-installed-distribution-integrity-v1",
        "package_count": len(distributions),
        "all_versions_match_lock": True,
        "all_declared_record_hashes_match_physical_files": True,
        "distributions": distributions,
        "environment_sha256": _canonical_sha256(distributions),
        "runtime_payload_environment_sha256": _canonical_sha256(runtime_payloads),
    }
    if report["runtime_payload_environment_sha256"] != RUNTIME_PAYLOAD_ENVIRONMENT_SHA256:
        raise ReconstructionError(
            "installed runtime payload differs from the frozen observed environment"
        )
    return report


def _inspect_installed(python: Path, lock: Path) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                str(python),
                str(Path(__file__).resolve()),
                "inspect-installed",
                "--lock",
                str(lock.resolve()),
            ],
            text=True,
        )
        value = json.loads(output)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ReconstructionError("installed-distribution integrity probe failed") from error
    if not isinstance(value, dict):
        raise ReconstructionError("installed-distribution probe returned no object")
    return value


def _run_probe(
    python: Path,
    *,
    application_root: Path,
    data_root: Path,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    program = """
import json
from epicure_mcp.provenance import build_provenance_payload
from epicure_mcp.tools import neighbors, pairing_score

payload = build_provenance_payload(
    __DATA_ROOT__,
    "exploratory-unmatched-1790-runtime",
    None,
    __SOURCE_ROOT__,
)
print(json.dumps({
    "provenance": {
        key: payload[key]
        for key in (
            "release_id",
            "bundle_sha256",
            "application_sha256",
            "ingredient_count",
            "embedding_dimensions",
        )
    },
    "pairing_score_tomato_basil": pairing_score.run("tomato", "basil"),
    "neighbors_tomato_3": neighbors.run("tomato", 3),
}, ensure_ascii=False, sort_keys=True))
""".replace("__DATA_ROOT__", repr(str(data_root.resolve()))).replace(
        "__SOURCE_ROOT__", repr(str((application_root / "src/epicure_mcp").resolve()))
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str((application_root / "src").resolve())
    env["EPICURE_DATA_DIR"] = str(data_root.resolve())
    try:
        output = subprocess.check_output([str(python), "-c", program], env=env, text=True)
        observed = json.loads(output)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ReconstructionError("functional reconstruction probe failed") from error
    expected = fixture.get("expected")
    if observed != expected:
        raise ReconstructionError("functional fixture output mismatch")
    return observed


def _copy_data(source: Path, destination: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        relative = _safe_relative(entry.get("path"), field="runtime data path")
        _write_exact(destination / relative, (source / relative).read_bytes())


def _write_receipt(workspace: Path, payload: Mapping[str, Any]) -> Path:
    digest = _canonical_sha256(payload)
    document = {**payload, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = workspace / f"epicure-public-input-reconstruction-receipt-{digest}.json"
    _write_exact(path, rendered.encode("utf-8"))
    return path


def materialize(arguments: argparse.Namespace) -> dict[str, Any]:
    """Materialize exact public inputs and optionally install and probe the runtime."""

    if not arguments.acknowledge_unattested_payload_rights:
        raise ReconstructionError(
            "materialization requires --acknowledge-unattested-payload-rights"
        )
    kit_root = arguments.kit_root.resolve()
    kit_report = verify_kit(kit_root)
    runtime = _read_json(kit_root / "runtime-manifest.json")
    sources = _read_json(kit_root / "data-sources.json")
    data_entries = _verify_data_sources(sources, runtime)
    fixture = _read_json(kit_root / "functional-fixtures.json")
    packages = _parse_lock(kit_root / "runtime-dependency-lock.txt")
    python = arguments.python.resolve()
    _assert_target_python(python)

    workspace = arguments.workspace.resolve()
    if workspace.exists() and (workspace.is_symlink() or any(workspace.iterdir())):
        raise ReconstructionError("workspace must be absent or an empty regular directory")
    workspace.mkdir(parents=True, exist_ok=True)
    application_root = workspace / "application"
    _extract_source(kit_root / "application-source.tar.gz", application_root)
    _verify_source_directory(application_root, runtime)

    data_root = workspace / "data"
    if arguments.data_dir is not None:
        _verify_data_directory(arguments.data_dir, data_entries)
        _copy_data(arguments.data_dir.resolve(), data_root, data_entries)
        data_mode = "operator_supplied_exact_directory"
    elif arguments.online:
        data_root.mkdir(parents=True, exist_ok=True)
        for entry in data_entries:
            _download_exact(
                str(entry["url"]),
                size=int(entry["bytes"]),
                digest=str(entry["sha256"]),
                destination=data_root / str(entry["path"]),
            )
        data_mode = "immutable_public_git_urls"
    else:
        raise ReconstructionError("choose --online or provide --data-dir")
    _verify_data_directory(data_root, data_entries)

    wheelhouse = workspace / "wheelhouse"
    wheel_entries: list[dict[str, Any]] = []
    dependency_python: Path | None = None
    installed_integrity: dict[str, Any] | None = None
    if arguments.with_dependencies or arguments.run_probe:
        if arguments.wheelhouse is not None:
            wheel_entries = _verify_wheelhouse(arguments.wheelhouse, packages)
            wheelhouse.mkdir(parents=True, exist_ok=True)
            for wheel in arguments.wheelhouse.iterdir():
                _write_exact(wheelhouse / wheel.name, wheel.read_bytes())
            wheel_mode = "operator_supplied_exact_wheelhouse"
        elif arguments.online:
            _download_wheels(python, kit_root / "runtime-dependency-lock.txt", wheelhouse)
            wheel_entries = _verify_wheelhouse(wheelhouse, packages)
            wheel_mode = "public_pypi_exact_hashes"
        else:
            raise ReconstructionError("dependency reconstruction needs --online or --wheelhouse")
        dependency_python = _install_dependencies(
            python,
            kit_root / "runtime-dependency-lock.txt",
            wheelhouse,
            workspace / ".venv",
        )
        installed_integrity = _inspect_installed(
            dependency_python,
            kit_root / "runtime-dependency-lock.txt",
        )
    else:
        wheel_mode = "not_requested"

    observed_fixture: dict[str, Any] | None = None
    if arguments.run_probe:
        if dependency_python is None:
            raise ReconstructionError("functional probe requires installed dependencies")
        observed_fixture = _run_probe(
            dependency_python,
            application_root=application_root,
            data_root=data_root,
            fixture=fixture,
        )

    receipt_payload = {
        "schema_version": RECEIPT_SCHEMA,
        "record_role": "same_operator_or_unadjudicated_public_input_reconstruction_receipt",
        "generated_at": datetime.now(UTC).isoformat(),
        "kit_manifest_sha256": kit_report["kit_manifest_sha256"],
        "runtime_id": RUNTIME_ID,
        "release_id": RELEASE_ID,
        "source_sha256": APPLICATION_SHA256,
        "data_sha256": BUNDLE_SHA256,
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "sbom_sha256": SBOM_SHA256,
        "data_input_mode": data_mode,
        "dependency_input_mode": wheel_mode,
        "dependency_wheels_verified": len(wheel_entries),
        "installed_distribution_integrity": installed_integrity,
        "runtime_payload_environment_sha256": (
            installed_integrity["runtime_payload_environment_sha256"]
            if installed_integrity is not None
            else None
        ),
        "functional_fixture_passed": observed_fixture is not None,
        "functional_fixture": observed_fixture,
        "target": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "operator_independence": "not_adjudicated",
        "independent_reproduction": False,
        "payload_license": None,
        "payload_rights_attestation": None,
        "payload_redistribution_cleared": False,
        "training_lineage_recovered": False,
        "immutable_oci_identity": None,
        "rank_eligible": False,
        "provider_calls_made": 0,
        "epicure_network_calls_made": 0,
        "status": (
            "public_inputs_materialized_and_functionally_verified_release_governance_blocked"
            if observed_fixture is not None
            else "public_inputs_materialized_release_governance_blocked"
        ),
    }
    receipt = _write_receipt(workspace, receipt_payload)
    return {
        **receipt_payload,
        "artifact_sha256": _read_json(receipt)["artifact_sha256"],
        "receipt": str(receipt),
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-kit")
    verify.add_argument("--kit-root", type=Path, required=True)
    build = subparsers.add_parser("materialize")
    build.add_argument("--kit-root", type=Path, required=True)
    build.add_argument("--workspace", type=Path, required=True)
    build.add_argument("--python", type=Path, default=Path(sys.executable))
    source = build.add_mutually_exclusive_group()
    source.add_argument("--online", action="store_true")
    source.add_argument("--data-dir", type=Path)
    build.add_argument("--wheelhouse", type=Path)
    build.add_argument("--with-dependencies", action="store_true")
    build.add_argument("--run-probe", action="store_true")
    build.add_argument("--acknowledge-unattested-payload-rights", action="store_true")
    inspect_installed = subparsers.add_parser("inspect-installed")
    inspect_installed.add_argument("--lock", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "verify-kit":
        report = verify_kit(arguments.kit_root)
    elif arguments.command == "inspect-installed":
        report = _distribution_integrity(arguments.lock)
    else:
        report = materialize(arguments)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
