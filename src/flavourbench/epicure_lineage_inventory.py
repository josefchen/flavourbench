"""Recover a content-addressed Epicure runtime inventory without inventing lineage.

The inventory is deliberately a *recovery* record.  It proves which application
sources and data files are present and which runtime identity they produce.  It
does not infer the unrecovered embedding-training run or grant redistribution
rights that have not been attested by the data steward.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "epicure-recovered-runtime-inventory-v2"
DATA_MANIFEST_VERSION = "epicure-data-bundle-manifest-v1"
SOURCE_MANIFEST_VERSION = "epicure-python-source-manifest-v1"
RELEASE_ID = "epicure-mcp-1790-r1"
RUNTIME_ID_PREFIX = "epicure-mcp-1790-r1"


class LineageInventoryError(RuntimeError):
    """The local runtime could not be inventoried without ambiguity."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(root: Path, paths: Sequence[Path], version: str) -> tuple[list[dict[str, Any]], str]:
    if root.is_symlink() or not root.is_dir():
        raise LineageInventoryError(f"manifest root must be a regular directory: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise LineageInventoryError(f"manifest input is a symbolic link: {path}")
        if not path.is_file():
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not entries:
        raise LineageInventoryError(f"{version} contains no files")
    digest = _sha256({"manifest_version": version, "entries": entries})
    return entries, digest


def _embedding_shape(path: Path) -> tuple[int, int]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            dimensions = sum(column.startswith("dim_") for column in header)
            ingredients = sum(1 for row in reader if row)
    except (OSError, StopIteration, UnicodeError) as error:
        raise LineageInventoryError("embeddings.csv shape could not be recovered") from error
    if ingredients <= 0 or dimensions <= 0:
        raise LineageInventoryError("embeddings.csv has an invalid shape")
    return ingredients, dimensions


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and process.returncode != 0:
        raise LineageInventoryError(f"git {' '.join(arguments)} failed")
    return process.stdout.strip()


def _dirty_files(root: Path) -> list[dict[str, Any]]:
    process = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise LineageInventoryError("git status --porcelain=v1 -z failed")
    fields = process.stdout.split(b"\0")
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if len(field) < 4 or field[2:3] != b" ":
            raise LineageInventoryError("git porcelain record is malformed")
        status = field[:2].decode("ascii", errors="strict")
        target = os.fsdecode(field[3:])
        rendered_path = target
        if "R" in status or "C" in status:
            if index >= len(fields) or not fields[index]:
                raise LineageInventoryError("git porcelain rename record is incomplete")
            original = os.fsdecode(fields[index])
            index += 1
            # With ``-z``, Git emits the destination before the source and
            # omits the textual arrow. Render the conventional source -> target
            # form while hashing the working-tree destination.
            rendered_path = f"{original} -> {target}"
        path = root / target
        record: dict[str, Any] = {"git_status": status, "path": rendered_path}
        if path.is_file() and not path.is_symlink():
            record.update({"bytes": path.stat().st_size, "sha256": _sha256_file(path)})
        else:
            record.update({"bytes": None, "sha256": None})
        records.append(record)
    return records


def _environment_inventory(root: Path) -> dict[str, Any]:
    python = root / ".venv/bin/python"
    if not python.is_file():
        return {
            "status": "unavailable",
            "python": None,
            "packages": [],
            "packages_sha256": None,
            "is_lockfile": False,
        }
    version = subprocess.run(
        [str(python), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    process = subprocess.run(
        [str(python), "-m", "pip", "list", "--format=json"],
        check=True,
        capture_output=True,
        text=True,
    )
    packages_raw = json.loads(process.stdout)
    packages = sorted(
        (
            {"name": str(item["name"]), "version": str(item["version"])}
            for item in packages_raw
            if isinstance(item, Mapping) and item.get("name") and item.get("version")
        ),
        key=lambda item: item["name"].casefold(),
    )
    return {
        "status": "observed_local_environment",
        "python": version,
        "packages": packages,
        "packages_sha256": _sha256(packages),
        "is_lockfile": False,
        "limitation": "Observed versions have no wheel hashes and are not a dependency lock.",
    }


def _verified_attestation(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise LineageInventoryError("runtime attestation must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "release_id",
        "bundle_sha256",
        "application_sha256",
        "ingredient_count",
        "embedding_dimensions",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise LineageInventoryError("runtime attestation is incomplete")
    # Authorization headers are never part of the response payload.  Rejecting
    # common secret-like keys prevents an operator from accidentally archiving
    # a request wrapper instead of the provenance response.
    forbidden = {"authorization", "api_key", "token", "secret", "password"}
    if any(str(key).casefold() in forbidden for key in value):
        raise LineageInventoryError("runtime attestation contains a forbidden secret field")
    return value


def capture_local_runtime_attestation(
    *,
    provenance_url: str,
    bearer_token: str | None,
    output_dir: Path,
) -> Path:
    """Capture a loopback runtime's provenance response without serializing its token."""

    parsed = urllib.parse.urlsplit(provenance_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.path.rstrip("/") != "/provenance"
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise LineageInventoryError(
            "runtime provenance URL must be a loopback http(s) /provenance endpoint"
        )
    headers = {"Accept": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = urllib.request.Request(provenance_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            body = response.read(1024 * 1024 + 1)
    except (OSError, urllib.error.HTTPError, urllib.error.URLError) as error:
        raise LineageInventoryError("local runtime provenance request failed") from error
    if len(body) > 1024 * 1024:
        raise LineageInventoryError("local runtime provenance response exceeds 1 MiB")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LineageInventoryError("local runtime provenance response is not JSON") from error
    if not isinstance(value, dict):
        raise LineageInventoryError("local runtime provenance response is not an object")
    forbidden = {"authorization", "api_key", "token", "secret", "password"}
    if any(str(key).casefold() in forbidden for key in value):
        raise LineageInventoryError("runtime attestation contains a forbidden secret field")
    digest = _sha256(value)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"runtime-provenance-attestation-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise LineageInventoryError("content-addressed runtime attestation conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as file:
        temporary = Path(file.name)
        file.write(rendered)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def build_inventory(
    *,
    mcp_root: Path,
    tool_contract_path: Path,
    runtime_attestation_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic recovery inventory from an Epicure MCP checkout."""

    root = mcp_root.resolve()
    data_root = root / "data"
    source_root = root / "src/epicure_mcp"
    data_entries, bundle_sha256 = _manifest(
        data_root,
        tuple(data_root.rglob("*")),
        DATA_MANIFEST_VERSION,
    )
    source_entries, application_sha256 = _manifest(
        source_root,
        tuple(source_root.rglob("*.py")),
        SOURCE_MANIFEST_VERSION,
    )
    ingredient_count, dimensions = _embedding_shape(data_root / "embeddings.csv")

    if tool_contract_path.is_symlink() or not tool_contract_path.is_file():
        raise LineageInventoryError("tool contract must be a regular file")
    tool_contract = json.loads(tool_contract_path.read_text(encoding="utf-8"))
    semantic_tool_sha256 = (
        str(tool_contract.get("semantic_sha256") or "")
        if isinstance(tool_contract, Mapping)
        else ""
    )
    if len(semantic_tool_sha256) != 64:
        # The frozen catalog filename itself is the semantic content address.
        stem_tokens = tool_contract_path.stem.split("-")
        semantic_tool_sha256 = stem_tokens[-1] if stem_tokens else ""
    if len(semantic_tool_sha256) != 64:
        raise LineageInventoryError("tool contract has no recoverable semantic SHA-256")

    build_paths = [
        root / name
        for name in (
            "Dockerfile",
            "LICENSE",
            "PRIVACY.md",
            "README.md",
            "SECURITY.md",
            "SUPPORT.md",
            "TERMS.md",
            "pyproject.toml",
            "scripts/build_data.py",
            "scripts/verify_data.py",
        )
        if (root / name).is_file()
    ]
    build_entries, build_context_sha256 = _manifest(
        root,
        tuple(build_paths),
        "epicure-build-and-policy-manifest-v1",
    )
    attestation = _verified_attestation(runtime_attestation_path)
    attestation_matches = None
    if attestation is not None:
        attestation_matches = (
            attestation.get("release_id") == "exploratory-unmatched-1790-runtime"
            and attestation.get("bundle_sha256") == bundle_sha256
            and attestation.get("application_sha256") == application_sha256
            and attestation.get("ingredient_count") == ingredient_count
            and attestation.get("embedding_dimensions") == dimensions
        )
        if not attestation_matches:
            raise LineageInventoryError("runtime attestation differs from the recovered checkout")

    head = _git(root, "rev-parse", "HEAD")
    dirty = _dirty_files(root)
    tag = _git(root, "tag", "--points-at", "HEAD", check=False).splitlines()
    runtime_id = (
        f"{RUNTIME_ID_PREFIX}+bundle.{bundle_sha256[:12]}.app.{application_sha256[:12]}"
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_role": "append_only_recovered_runtime_identity",
        "release_id": RELEASE_ID,
        "runtime_id": runtime_id,
        "canonical_urn": (
            f"urn:epicure:runtime:{RELEASE_ID}:bundle:{bundle_sha256}:app:{application_sha256}"
        ),
        "identity_boundary": {
            "representation_class": "opaque_recovered_artifact",
            "public_sibling_match": "none_exact",
            "prohibited_claims": [
                "cooc_exact",
                "core_exact",
                "chem_exact",
                "paper_ii_exact_model",
                "recovered_training_run",
            ],
        },
        "bundle": {
            "manifest_version": DATA_MANIFEST_VERSION,
            "sha256": bundle_sha256,
            "ingredient_count": ingredient_count,
            "embedding_dimensions": dimensions,
            "manifest": data_entries,
        },
        "application": {
            "identity": "development_source_manifest",
            "development": True,
            "manifest_version": SOURCE_MANIFEST_VERSION,
            "sha256": application_sha256,
            "manifest": source_entries,
            "git": {
                "head_commit": head,
                "clean": not dirty,
                "signed_release_tag": tag[0] if len(tag) == 1 else None,
                "tags_at_head": tag,
                "dirty_files": dirty,
            },
            "build_and_policy_manifest": {
                "manifest_version": "epicure-build-and-policy-manifest-v1",
                "sha256": build_context_sha256,
                "manifest": build_entries,
            },
            "environment": _environment_inventory(root),
            "oci_image_digest": None,
            "sbom_sha256": None,
            "dependency_lock_sha256": None,
        },
        "tool_contract": {
            "semantic_sha256": semantic_tool_sha256,
            "source_file_sha256": _sha256_file(tool_contract_path),
            "source_filename": tool_contract_path.name,
        },
        "runtime_attestation": {
            "provided": attestation is not None,
            "matches_recovered_checkout": attestation_matches,
            "response_sha256": _sha256(attestation) if attestation is not None else None,
            "response": attestation,
            "public_provenance_route_status": "not_attested_by_this_inventory",
        },
        "training_lineage": {
            "status": "not_recovered",
            "training_run_id": None,
            "training_seed": None,
            "training_code_revision": None,
            "training_environment": None,
            "recipe_corpus_assertion": "4.14M recipes; source assertion not a recovered run record",
        },
        "rights": {
            "code_license": "MIT",
            "license_file_sha256": _sha256_file(root / "LICENSE"),
            "payload_license": None,
            "artifact_rights_attestation": None,
            "source_rights_matrix": None,
            "redistributable_payload": False,
            "status": "blocked_pending_data_steward_attestation",
        },
        "release_gates": {
            "content_addressed_bundle": True,
            "content_addressed_application": True,
            "runtime_attestation_matches": attestation_matches is True,
            "clean_signed_application_release": False,
            "immutable_oci_identity": False,
            "dependency_lock": False,
            "sbom": False,
            "training_lineage_recovered": False,
            "payload_rights_attested": False,
            "independent_reproduction": False,
        },
        "status": "recovered_runtime_identity_release_blocked",
        "rank_eligible": False,
        "redistributable": False,
        "supersedes": None,
        "does_not_supersede": [
            "exploratory-unmatched-1790-runtime",
            "epicure-mcp-1790-r1.release-candidate.json",
        ],
        "official_use_rule": (
            "This record permits exact retrospective attribution to the opaque runtime only. "
            "It cannot authorize official ranking or payload redistribution. A new immutable "
            "release record must supersede it after all false gates are independently evidenced."
        ),
    }
    return payload


def verify_inventory(document: object) -> bool:
    if not isinstance(document, Mapping):
        return False
    digest = document.get("artifact_sha256")
    unhashed = {key: value for key, value in document.items() if key != "artifact_sha256"}
    return (
        document.get("schema_version") == SCHEMA_VERSION
        and isinstance(digest, str)
        and len(digest) == 64
        and _sha256(unhashed) == digest
    )


def write_inventory(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    unhashed = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    digest = _sha256(unhashed)
    document = {**unhashed, "artifact_sha256": digest}
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"epicure-recovered-runtime-inventory-{digest}.json"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise LineageInventoryError("content-addressed inventory conflict")
        return destination
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, delete=False) as file:
        temporary = Path(file.name)
        file.write(rendered)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-root", type=Path, required=True)
    parser.add_argument("--tool-contract", type=Path, required=True)
    attestation = parser.add_mutually_exclusive_group()
    attestation.add_argument("--runtime-attestation", type=Path)
    attestation.add_argument("--runtime-provenance-url")
    parser.add_argument(
        "--runtime-token-env",
        default="MCP_API_TOKEN",
        help="Environment variable containing the bearer token; the value is never serialized",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    runtime_attestation_path = arguments.runtime_attestation
    if arguments.runtime_provenance_url:
        runtime_attestation_path = capture_local_runtime_attestation(
            provenance_url=arguments.runtime_provenance_url,
            bearer_token=os.environ.get(arguments.runtime_token_env),
            output_dir=arguments.output_dir / "attestations",
        )
    payload = build_inventory(
        mcp_root=arguments.mcp_root,
        tool_contract_path=arguments.tool_contract,
        runtime_attestation_path=runtime_attestation_path,
    )
    path = write_inventory(arguments.output_dir, payload)
    written = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(path.resolve()),
                "artifact_sha256": written["artifact_sha256"],
                "runtime_id": written["runtime_id"],
                "bundle_sha256": written["bundle"]["sha256"],
                "application_sha256": written["application"]["sha256"],
                "runtime_attestation": (
                    str(runtime_attestation_path.resolve())
                    if runtime_attestation_path is not None
                    else None
                ),
                "rank_eligible": written["rank_eligible"],
                "redistributable": written["redistributable"],
                "provider_calls_made": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
