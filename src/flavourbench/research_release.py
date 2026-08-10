"""Build and verify a deterministic Season 1 research-evidence archive.

The archive is derived from four published, snapshot-bound analysis cells.  It
contains the exact database records needed to audit those cells, the frozen
contracts, and the byte-exact analysis implementation.  It deliberately omits
operational credentials and replaces persistent rater identifiers with an
archive-local namespace.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
    load_pem_public_key,
)
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .config import get_settings
from .database import SessionLocal, database_readiness
from .models import (
    RESEARCH_ARCHIVE_SIGNATURE_CONTEXT,
    AdmissionEvent,
    Battle,
    BedrockBillingCrosscheck,
    BedrockBillingCrosscheckArm,
    CatalogModel,
    ControlledRun,
    ControlledRunAssignment,
    ControlledRunReviewer,
    CostEvent,
    EpicureRelease,
    ExpertReviewer,
    GenerationAttempt,
    Incident,
    LeaderboardSnapshot,
    ResearchReleaseArchive,
    ResponseArm,
    RunEvent,
    Season,
    SeasonModel,
    Task,
    TaskEvidenceArtifact,
    ToolCall,
    ValidatorResult,
    Vote,
)

SCHEMA_VERSION = "flavourbench-research-release-v1"
SOURCE_DATE_EPOCH = 0
REQUIRED_CELLS = frozenset(
    {
        ("model_arena", "public"),
        ("model_arena", "expert_independent"),
        ("epicure_uplift", "public"),
        ("epicure_uplift", "expert_independent"),
    }
)
def _project_root() -> Path:
    candidates = (Path.cwd().resolve(), Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (candidate / "requirements.lock").is_file() and (candidate / "contracts").is_dir():
            return candidate
    return candidates[-1]


ROOT = _project_root()
REQUIRED_IMPLEMENTATION_FILES = (
    ROOT / "src/flavourbench/service_ranking.py",
    ROOT / "src/flavourbench/season1_statistics.py",
    ROOT / "src/flavourbench/season1_arena_acceptance.py",
    ROOT / "src/flavourbench/season1_arena_monte_carlo.py",
    ROOT / "src/flavourbench/season1_arena_distributed.py",
    ROOT / "src/flavourbench/season1_arena_modal.py",
    ROOT / "src/flavourbench/season1_method_validation.py",
    ROOT / "src/flavourbench/construct_blueprint.py",
    ROOT / "src/flavourbench/research_release.py",
    ROOT / "contracts/season1/season1-study-design-v5.json",
    ROOT / "contracts/season1/season1-construct-blueprint-v1.json",
    ROOT / "contracts/season1/season1-validity-robustness-evidence-v1.md",
    ROOT / "contracts/season1/season1-arena-inference-acceptance-v1.json",
    ROOT
    / "contracts/season1/method-validation/season1-arena-production-monte-carlo-v1.json",
    ROOT
    / "contracts/season1/method-validation/season1-arena-distributed-execution-v2.json",
    ROOT
    / "contracts/season1/method-validation/season1-statistical-method-validation-"
    "0b4345e523fdaa97d1b406cd1f2165540d0f9ad338bb49f3ac656da73e3c1933.json",
    ROOT / "requirements.lock",
    ROOT / "Dockerfile",
    ROOT / "pyproject.toml",
)
ROBUSTNESS_EVIDENCE_SCHEMAS = {
    "post_collection_item_audit": "flavourbench-season1-post-collection-item-audit-v1",
    "generation_reliability_panel": "flavourbench-season1-generation-reliability-panel-v1",
    "prompt_sensitivity_audit": "flavourbench-season1-prompt-sensitivity-audit-v1",
    "practical_cookability_execution": (
        "flavourbench-season1-practical-cookability-execution-v1"
    ),
}
TEXT_SUFFIXES = frozenset(
    {".json", ".jsonl", ".lock", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
)
SECRET_PATTERNS = {
    "aws_access_key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "openrouter_api_key": re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----[ \t]*\r?\n"),
    "authorization_header": re.compile(
        rb"(?i)authorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._~+/-]{20,}"
    ),
    "secret_assignment": re.compile(
        rb"(?i)[\"']?(?:aws_secret_access_key|openrouter_api_key|secret_access_key)"
        rb"[\"']?\s*[:=]\s*[\"'][^\"']{12,}[\"']"
    ),
}


class ResearchReleaseError(RuntimeError):
    """Official research evidence cannot be sealed without inventing or omitting data."""


@dataclass(frozen=True)
class ArchiveMember:
    path: str
    data: bytes
    row_count: int | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class SealedRelease:
    manifest: dict[str, Any]
    manifest_sha256: str
    archive_path: Path
    archive_sha256: str
    archive_size_bytes: int
    signature_base64: str
    public_key_pem: str
    public_key_sha256: str
    signature_path: Path
    public_key_path: Path


def _canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + suffix
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, newline=False)).hexdigest()


def _utc_iso(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat()


def _normalize(value: object) -> object:
    if isinstance(value, datetime):
        return _utc_iso(value)
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ResearchReleaseError(f"unsafe archive member path: {value}")
    normalized = path.as_posix()
    if normalized in {".", "release/MANIFEST.json"}:
        raise ResearchReleaseError(f"reserved archive member path: {value}")
    return normalized


def _scan_member(member: ArchiveMember) -> None:
    if PurePosixPath(member.path).suffix.lower() not in TEXT_SUFFIXES:
        return
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(member.data):
            raise ResearchReleaseError(f"secret scan matched {label} in {member.path}")


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> tuple[bytes, int]:
    materialized = [dict(_normalize(row)) for row in rows]
    ordered = sorted(
        materialized,
        key=lambda row: (
            str(row.get("id", "")),
            _canonical_json_bytes(row, newline=False),
        ),
    )
    return b"".join(_canonical_json_bytes(row) for row in ordered), len(ordered)


def _tar_info(path: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(path)
    info.size = size
    info.mtime = SOURCE_DATE_EPOCH
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _archive_bytes(members: Sequence[ArchiveMember], manifest_bytes: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(
        fileobj=output,
        mode="wb",
        filename="",
        mtime=SOURCE_DATE_EPOCH,
        compresslevel=9,
    ) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            archive.addfile(
                _tar_info("release/MANIFEST.json", len(manifest_bytes)),
                io.BytesIO(manifest_bytes),
            )
            for member in sorted(members, key=lambda item: item.path):
                archive.addfile(_tar_info(member.path, len(member.data)), io.BytesIO(member.data))
    return output.getvalue()


def _atomic_write(path: Path, data: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _public_key_pem(private_key: Ed25519PrivateKey) -> str:
    return (
        private_key.public_key()
        .public_bytes(
            Encoding.PEM,
            PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )


def load_signing_key(path: Path) -> Ed25519PrivateKey:
    if not path.is_file() or path.is_symlink():
        raise ResearchReleaseError("research-release signing key is not a regular file")
    permissions = stat.S_IMODE(path.stat().st_mode)
    if permissions & 0o077:
        raise ResearchReleaseError("research-release signing key must not be group/world readable")
    try:
        key = load_pem_private_key(path.read_bytes(), password=None)
    except (TypeError, ValueError) as exc:
        raise ResearchReleaseError("research-release signing key is invalid PEM") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ResearchReleaseError("research-release signing key must be Ed25519")
    return key


def seal_members(
    *,
    members: Sequence[ArchiveMember],
    manifest_metadata: Mapping[str, Any],
    output_dir: Path,
    private_key: Ed25519PrivateKey,
    signing_key_id: str,
) -> SealedRelease:
    """Seal canonical members; identical inputs and key produce identical bytes."""

    if not signing_key_id or len(signing_key_id) > 160:
        raise ResearchReleaseError("a stable signing-key identifier is required")
    public_key_pem = _public_key_pem(private_key)
    public_key_sha256 = hashlib.sha256(public_key_pem.encode("ascii")).hexdigest()
    all_members = [
        *members,
        ArchiveMember("release/PUBLIC_KEY.pem", public_key_pem.encode("ascii"), None),
    ]
    paths = [_safe_path(member.path) for member in all_members]
    if len(paths) != len(set(paths)):
        raise ResearchReleaseError("archive contains duplicate member paths")
    for member in all_members:
        _scan_member(member)
    member_manifest = [
        {
            "path": member.path,
            "sha256": member.sha256,
            "size_bytes": len(member.data),
            **({"row_count": member.row_count} if member.row_count is not None else {}),
        }
        for member in sorted(all_members, key=lambda item: item.path)
    ]
    manifest = {
        **dict(_normalize(manifest_metadata)),
        "schema_version": SCHEMA_VERSION,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "archive_member_count_including_manifest": len(all_members) + 1,
        "signing": {
            "algorithm": "Ed25519",
            "key_id": signing_key_id,
            "public_key_sha256": public_key_sha256,
            "signed_message": "context || raw archive SHA-256 bytes",
        },
        "members": member_manifest,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_sha256 = _canonical_sha256(manifest)
    archive_bytes = _archive_bytes(all_members, manifest_bytes)
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    signature = private_key.sign(RESEARCH_ARCHIVE_SIGNATURE_CONTEXT + bytes.fromhex(archive_sha256))
    signature_base64 = base64.b64encode(signature).decode("ascii")
    release_name = str(manifest.get("release_slug") or "flavourbench-research-release")
    safe_release_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", release_name).strip("-")
    archive_path = output_dir / f"{safe_release_name}-{archive_sha256}.tar.gz"
    signature_path = Path(f"{archive_path}.sig")
    public_key_path = Path(f"{archive_path}.pub.pem")
    _atomic_write(archive_path, archive_bytes, mode=0o600)
    _atomic_write(signature_path, f"{signature_base64}\n".encode("ascii"), mode=0o644)
    _atomic_write(public_key_path, public_key_pem.encode("ascii"), mode=0o644)
    verification = verify_archive(
        archive_path=archive_path,
        signature_base64=signature_base64,
        public_key_pem=public_key_pem,
        expected_archive_sha256=archive_sha256,
    )
    if verification["manifest_sha256"] != manifest_sha256:
        raise ResearchReleaseError("sealed manifest changed during archive verification")
    return SealedRelease(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_size_bytes=len(archive_bytes),
        signature_base64=signature_base64,
        public_key_pem=public_key_pem,
        public_key_sha256=public_key_sha256,
        signature_path=signature_path,
        public_key_path=public_key_path,
    )


def verify_archive(
    *,
    archive_path: Path,
    signature_base64: str,
    public_key_pem: str,
    expected_archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify signature, canonical manifest, member inventory, and tar metadata."""

    archive_bytes = archive_path.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if expected_archive_sha256 is not None and archive_sha256 != expected_archive_sha256:
        raise ResearchReleaseError("research archive byte digest mismatch")
    try:
        signature = base64.b64decode(signature_base64.strip(), validate=True)
        key = load_pem_public_key(public_key_pem.encode("ascii"))
    except (TypeError, ValueError) as exc:
        raise ResearchReleaseError("research archive verification material is malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ResearchReleaseError("research archive verification key is not Ed25519")
    try:
        key.verify(
            signature,
            RESEARCH_ARCHIVE_SIGNATURE_CONTEXT + bytes.fromhex(archive_sha256),
        )
    except InvalidSignature as exc:
        raise ResearchReleaseError("research archive signature is invalid") from exc
    if len(archive_bytes) < 10 or int.from_bytes(archive_bytes[4:8], "little") != 0:
        raise ResearchReleaseError("research archive gzip header is not reproducible")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        infos = archive.getmembers()
        names = [item.name for item in infos]
        if not names or names[0] != "release/MANIFEST.json" or len(names) != len(set(names)):
            raise ResearchReleaseError("research archive has an invalid member order or duplicate")
        if names[1:] != sorted(names[1:]):
            raise ResearchReleaseError("research archive members are not canonically ordered")
        extracted: dict[str, bytes] = {}
        for info in infos:
            unsafe_path = info.name.startswith("/") or ".." in PurePosixPath(info.name).parts
            if not info.isfile() or unsafe_path:
                raise ResearchReleaseError("research archive contains an unsafe member")
            if not (
                info.mtime == SOURCE_DATE_EPOCH
                and info.mode == 0o644
                and info.uid == 0
                and info.gid == 0
                and info.uname == ""
                and info.gname == ""
            ):
                raise ResearchReleaseError("research archive metadata is not reproducible")
            handle = archive.extractfile(info)
            if handle is None:
                raise ResearchReleaseError("research archive member cannot be read")
            extracted[info.name] = handle.read()
    try:
        manifest = json.loads(extracted["release/MANIFEST.json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ResearchReleaseError("research archive manifest is invalid") from exc
    canonical_manifest = _canonical_json_bytes(manifest) if isinstance(manifest, dict) else None
    if canonical_manifest is None or extracted["release/MANIFEST.json"] != canonical_manifest:
        raise ResearchReleaseError("research archive manifest is not canonical JSON")
    members = manifest.get("members")
    if not isinstance(members, list):
        raise ResearchReleaseError("research archive manifest has no member inventory")
    expected_names = [str(item.get("path")) for item in members if isinstance(item, dict)]
    if len(expected_names) != len(members) or sorted(expected_names) != names[1:]:
        raise ResearchReleaseError("research archive inventory differs from its members")
    for item in members:
        path = str(item["path"])
        data = extracted[path]
        if item.get("sha256") != hashlib.sha256(data).hexdigest() or item.get("size_bytes") != len(
            data
        ):
            raise ResearchReleaseError(f"research archive member digest mismatch: {path}")
        if "row_count" in item:
            observed_rows = len(data.splitlines()) if data else 0
            if item["row_count"] != observed_rows:
                raise ResearchReleaseError(f"research archive row count mismatch: {path}")
    if manifest.get("archive_member_count_including_manifest") != len(names):
        raise ResearchReleaseError("research archive member count does not reconcile")
    embedded_public_key = extracted.get("release/PUBLIC_KEY.pem")
    if embedded_public_key != public_key_pem.encode("ascii"):
        raise ResearchReleaseError("research archive public key differs from verification key")
    return {
        "schema_version": "flavourbench-research-release-verification-v1",
        "archive_sha256": archive_sha256,
        "manifest_sha256": _canonical_sha256(manifest),
        "member_count": len(names),
        "signature_valid": True,
        "inventory_valid": True,
        "reproducible_metadata_valid": True,
    }


def _serialize(
    value: object,
    *,
    exclude: frozenset[str] = frozenset(),
    replacements: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    mapper = sqlalchemy_inspect(value).mapper
    output = {
        attribute.key: _normalize(getattr(value, attribute.key))
        for attribute in mapper.column_attrs
        if attribute.key not in exclude
    }
    if replacements:
        output.update({key: _normalize(item) for key, item in replacements.items()})
    return output


def _archive_local_identifier(snapshot_set_sha256: str, value: str) -> str:
    return hashlib.sha256(
        f"flavourbench-archive-identity-v1:{snapshot_set_sha256}:{value}".encode()
    ).hexdigest()


def _privacy_filter_payload(value: object, snapshot_set_sha256: str) -> object:
    if isinstance(value, list):
        return [_privacy_filter_payload(item, snapshot_set_sha256) for item in value]
    if not isinstance(value, Mapping):
        return _normalize(value)
    output: dict[str, object] = {}
    for raw_key, raw_value in sorted(value.items(), key=lambda pair: str(pair[0])):
        key = str(raw_key)
        lowered = key.casefold()
        if any(marker in lowered for marker in ("secret", "plaintext_token", "access_token")):
            if not lowered.endswith("sha256"):
                continue
        if any(
            marker in lowered
            for marker in (
                "pseudonym",
                "reviewer_code",
                "reviewer_id",
                "invitation",
                "browser_identifier",
                "network_signal",
                "rater_id",
            )
        ):
            if raw_value is None:
                output[key] = None
            elif isinstance(raw_value, str):
                output[f"{key}_archive_local_sha256"] = _archive_local_identifier(
                    snapshot_set_sha256, raw_value
                )
            continue
        output[key] = _privacy_filter_payload(raw_value, snapshot_set_sha256)
    return output


def _records_member(path: str, rows: Iterable[Mapping[str, Any]]) -> ArchiveMember:
    data, count = _jsonl(rows)
    return ArchiveMember(path, data, count)


def _ids(evidence: Sequence[Mapping[str, Any]], section: str) -> set[str]:
    output: set[str] = set()
    for manifest in evidence:
        rows = manifest.get(section, [])
        if not isinstance(rows, list):
            raise ResearchReleaseError(f"snapshot evidence section {section} is malformed")
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
                if section == "bedrock_billing_memberships":
                    continue
                raise ResearchReleaseError(f"snapshot evidence section {section} lacks row IDs")
            output.add(str(row["id"]))
    return output


def _query_ids(session: Session, model: Any, ids: set[str]) -> list[Any]:
    if not ids:
        return []
    primary_key = sqlalchemy_inspect(model).primary_key[0]
    rows = session.scalars(select(model).where(primary_key.in_(sorted(ids)))).all()
    observed = {str(getattr(row, primary_key.key)) for row in rows}
    if observed != ids:
        raise ResearchReleaseError(
            f"archive source lost {model.__tablename__} rows: {sorted(ids - observed)}"
        )
    return rows


def _validate_snapshot_set(
    session: Session,
    *,
    season: Season,
    snapshot_ids: Sequence[str],
) -> tuple[list[LeaderboardSnapshot], ControlledRun]:
    normalized_ids = sorted(set(snapshot_ids))
    if len(normalized_ids) != 4 or len(normalized_ids) != len(snapshot_ids):
        raise ResearchReleaseError(
            "official Season 1 archive requires exactly four unique snapshots"
        )
    snapshots = _query_ids(session, LeaderboardSnapshot, set(normalized_ids))
    snapshots.sort(key=lambda item: (item.track, item.cohort, item.id))
    if (
        season.slug != "season-1"
        or not season.official
        or season.status != "active"
        or season.frozen_at is None
    ):
        raise ResearchReleaseError("research archives require the frozen official Season 1")
    if any(
        value in {"", "unfrozen", "unresolved"}
        for value in (
            season.manifest_sha256,
            season.prompt_registry_sha256,
            season.tool_registry_sha256,
            season.analysis_plan_sha256,
            season.protocol_bundle_sha256,
        )
    ):
        raise ResearchReleaseError("official Season 1 protocol hashes are not frozen")
    cells = {(item.track, item.cohort) for item in snapshots}
    run_ids = {item.controlled_run_id for item in snapshots}
    if cells != REQUIRED_CELLS:
        raise ResearchReleaseError("research archive does not cover the four canonical cells")
    if any(
        item.season_id != season.id
        or item.category != "all"
        or item.data_stratum != "controlled"
        or item.publication_status != "published"
        or item.payload_sha256 != _canonical_sha256(item.payload_json)
        or item.input_sha256 != item.payload_sha256
        or not isinstance(item.input_evidence_json, dict)
        or item.input_evidence_sha256 != _canonical_sha256(item.input_evidence_json)
        for item in snapshots
    ):
        raise ResearchReleaseError("a canonical snapshot is unpublished, mutable, or out of scope")
    if len(run_ids) != 1 or None in run_ids:
        raise ResearchReleaseError("the four canonical snapshots must share one controlled run")
    run_row = session.get(ControlledRun, next(iter(run_ids)))
    if run_row is None or run_row.season_id != season.id:
        raise ResearchReleaseError("snapshot-bound controlled run is unavailable")
    if (
        run_row.organization_id is not None
        or run_row.evaluation_order_id is not None
        or run_row.status not in {"collection_complete", "closed"}
    ):
        raise ResearchReleaseError("commercial or incomplete runs cannot enter a research archive")

    # Rebuild each snapshot evidence view at its immutable cutoff.  This catches
    # backdated inserts and post-publication eligibility changes.
    from .main import (  # Local import avoids an application/module import cycle.
        _require_season1_statistical_acceptance,
        _verified_current_snapshot_payload,
    )

    for snapshot in snapshots:
        try:
            payload = _verified_current_snapshot_payload(
                session,
                season=season,
                snapshot=snapshot,
            )
            _require_season1_statistical_acceptance(season, snapshot, payload)
        except Exception as exc:
            raise ResearchReleaseError(
                f"snapshot {snapshot.id} failed current-evidence verification"
            ) from exc
    return snapshots, run_row


def _static_members(evidence: Sequence[Mapping[str, Any]]) -> list[ArchiveMember]:
    missing = [str(path) for path in REQUIRED_IMPLEMENTATION_FILES if not path.is_file()]
    if missing:
        raise ResearchReleaseError(f"required implementation files are missing: {missing}")
    analysis_sha = hashlib.sha256(REQUIRED_IMPLEMENTATION_FILES[0].read_bytes()).hexdigest()
    statistics_sha = hashlib.sha256(REQUIRED_IMPLEMENTATION_FILES[1].read_bytes()).hexdigest()
    acceptance_sha = hashlib.sha256(REQUIRED_IMPLEMENTATION_FILES[2].read_bytes()).hexdigest()
    policy_sha = hashlib.sha256(
        (
            ROOT / "contracts/season1/season1-arena-inference-acceptance-v1.json"
        ).read_bytes()
    ).hexdigest()
    if any(
        item.get("analysis_source_sha256") != analysis_sha
        or item.get("season1_statistics_source_sha256") != statistics_sha
        or item.get("arena_acceptance_source_sha256") != acceptance_sha
        or item.get("arena_acceptance_policy_file_sha256") != policy_sha
        for item in evidence
    ):
        raise ResearchReleaseError("snapshot evidence is bound to different analysis source bytes")
    members = []
    for path in REQUIRED_IMPLEMENTATION_FILES:
        logical = path.relative_to(ROOT).as_posix()
        members.append(ArchiveMember(f"implementation/{logical}", path.read_bytes()))
    return members


def _robustness_members(
    paths: Mapping[str, Path] | None,
) -> tuple[list[ArchiveMember], dict[str, str]]:
    if paths is None:
        return [], {}
    if set(paths) != set(ROBUSTNESS_EVIDENCE_SCHEMAS):
        raise ResearchReleaseError(
            "all four validity-and-robustness artifacts are required together"
        )
    design = json.loads(
        (ROOT / "contracts/season1/season1-study-design-v5.json").read_text(encoding="utf-8")
    )
    design_sha256 = str(design.get("artifact_sha256", ""))
    members: list[ArchiveMember] = []
    digests: dict[str, str] = {}
    for name, schema_version in ROBUSTNESS_EVIDENCE_SCHEMAS.items():
        path = paths[name]
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchReleaseError(f"invalid robustness artifact: {name}") from exc
        if not isinstance(value, dict):
            raise ResearchReleaseError(f"robustness artifact is not an object: {name}")
        embedded = value.get("artifact_sha256")
        payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
        if not (
            isinstance(embedded, str)
            and embedded == _canonical_sha256(payload)
            and value.get("schema_version") == schema_version
            and value.get("status") == "complete"
            and value.get("study_design_artifact_sha256") == design_sha256
            and value.get("synthetic_observations") == 0
        ):
            raise ResearchReleaseError(f"robustness artifact failed its common contract: {name}")
        data = _canonical_json_bytes(value)
        members.append(ArchiveMember(f"validity-and-robustness/{name}.json", data))
        digests[name] = embedded
    return members, dict(sorted(digests.items()))


def _readme() -> bytes:
    return (
        b"# FlavourBench research release\n\n"
        b"This internal archive is the immutable evidence source for the four published "
        b"Season 1 analysis cells. Verify it with `flavourbench-verify-research-release`. "
        b"The manifest binds every record and the exact analysis source. Persistent reviewer "
        b"identifiers and operational credentials are intentionally absent.\n\n"
        b"A public dataset must be derived from this archive under a separately hashed privacy "
        b"review; this internal artifact is not itself a public-data authorization.\n"
    )


def build_research_release(
    *,
    session: Session,
    season_slug: str,
    snapshot_ids: Sequence[str],
    output_dir: Path,
    private_key: Ed25519PrivateKey,
    signing_key_id: str,
    build_image_digest: str,
    robustness_evidence_paths: Mapping[str, Path] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Build the sole internal official archive for an exact snapshot set."""

    if not re.fullmatch(r"sha256:[0-9a-f]{64}", build_image_digest):
        raise ResearchReleaseError("a content-addressed build image digest is required")
    season = session.scalar(select(Season).where(Season.slug == season_slug))
    if season is None:
        raise ResearchReleaseError(f"season not found: {season_slug}")
    snapshots, controlled_run = _validate_snapshot_set(
        session,
        season=season,
        snapshot_ids=snapshot_ids,
    )
    normalized_snapshot_ids = sorted(item.id for item in snapshots)
    snapshot_set_sha256 = _canonical_sha256({"snapshot_ids": normalized_snapshot_ids})
    existing = session.scalar(
        select(ResearchReleaseArchive).where(
            ResearchReleaseArchive.snapshot_set_sha256 == snapshot_set_sha256
        )
    )
    if existing is not None:
        raise ResearchReleaseError("this exact snapshot set already has an immutable archive")
    evidence = [dict(item.input_evidence_json or {}) for item in snapshots]
    battle_ids = _ids(evidence, "battles")
    battles = _query_ids(session, Battle, battle_ids)
    if any(
        battle.retention_basis != "official_research"
        or battle.controlled_run_id != controlled_run.id
        or battle.run_class != "official"
        or not battle.rank_eligible
        for battle in battles
    ):
        raise ResearchReleaseError("snapshot evidence includes non-official retention or run scope")
    linked_arm_ids = {
        arm_id
        for battle in battles
        for arm_id in (battle.left_arm_id, battle.right_arm_id)
        if arm_id is not None
    }
    arm_ids = _ids(evidence, "arms") | linked_arm_ids
    arms = _query_ids(session, ResponseArm, arm_ids)
    if any(
        arm.provider_slug == "mock" or arm.model_id.startswith("flavourbench/mock-") for arm in arms
    ):
        raise ResearchReleaseError("official research archive refuses mock or placeholder arms")
    vote_ids = _ids(evidence, "votes")
    votes = _query_ids(session, Vote, vote_ids)
    raw_rater_ids = {vote.rater_pseudonym for vote in votes}
    assignments = _query_ids(session, ControlledRunAssignment, _ids(evidence, "assignments"))
    if any(item.controlled_run_id != controlled_run.id for item in assignments):
        raise ResearchReleaseError("snapshot assignment escaped its controlled run")
    task_ids = {
        str(task_id)
        for task_id in (
            *[battle.task_id for battle in battles],
            *[assignment.task_id for assignment in assignments],
        )
        if task_id
    }
    tasks = _query_ids(session, Task, task_ids)
    model_ids = {arm.model_id for arm in arms}
    season_model_ids = _ids(evidence, "season_models")
    season_models = _query_ids(session, SeasonModel, season_model_ids)
    if any(item.season_id != season.id for item in season_models):
        raise ResearchReleaseError("snapshot endpoint contract escaped its season")
    catalog_models = session.scalars(
        select(CatalogModel).where(CatalogModel.model_id.in_(sorted(model_ids)))
    ).all()
    if {item.model_id for item in catalog_models} != model_ids:
        raise ResearchReleaseError("catalog identity is incomplete for archived response arms")
    attempts = _query_ids(session, GenerationAttempt, _ids(evidence, "generation_attempts"))
    tools = _query_ids(session, ToolCall, _ids(evidence, "tool_calls"))
    validators = _query_ids(session, ValidatorResult, _ids(evidence, "validators"))
    costs = _query_ids(session, CostEvent, _ids(evidence, "cost_events"))
    crosschecks = _query_ids(
        session,
        BedrockBillingCrosscheck,
        _ids(evidence, "bedrock_billing_crosschecks"),
    )
    crosscheck_ids = {item.id for item in crosschecks}
    memberships = (
        session.scalars(
            select(BedrockBillingCrosscheckArm).where(
                BedrockBillingCrosscheckArm.crosscheck_id.in_(sorted(crosscheck_ids)),
                BedrockBillingCrosscheckArm.arm_id.in_(sorted(arm_ids)),
            )
        ).all()
        if crosscheck_ids
        else []
    )
    max_cutoff = max(item.evidence_cutoff_at for item in snapshots)
    task_evidence = (
        session.scalars(
            select(TaskEvidenceArtifact).where(
                TaskEvidenceArtifact.task_id.in_(sorted(task_ids)),
                TaskEvidenceArtifact.created_at <= max_cutoff,
            )
        ).all()
        if task_ids
        else []
    )
    relevant_entity_ids = {
        season.id,
        controlled_run.id,
        *normalized_snapshot_ids,
        *battle_ids,
        *arm_ids,
        *vote_ids,
        *task_ids,
    }
    run_events = session.scalars(
        select(RunEvent).where(
            RunEvent.entity_id.in_(sorted(relevant_entity_ids)),
            RunEvent.created_at <= max_cutoff,
        )
    ).all()
    if any("withdraw" in item.event_type.casefold() for item in run_events):
        raise ResearchReleaseError("withdrawn evidence cannot enter a research archive")
    incidents = (
        session.scalars(
            select(Incident).where(
                Incident.battle_id.in_(sorted(battle_ids)),
                Incident.created_at <= max_cutoff,
            )
        ).all()
        if battle_ids
        else []
    )
    reviewer_links = session.scalars(
        select(ControlledRunReviewer).where(
            ControlledRunReviewer.controlled_run_id == controlled_run.id
        )
    ).all()
    linked_reviewer_ids = {item.reviewer_id for item in reviewer_links}
    reviewers = (
        session.scalars(
            select(ExpertReviewer).where(
                or_(
                    ExpertReviewer.reviewer_code.in_(sorted(raw_rater_ids)),
                    ExpertReviewer.id.in_(sorted(linked_reviewer_ids)),
                )
            )
        ).all()
        if raw_rater_ids or linked_reviewer_ids
        else []
    )
    reviewers_by_alias = {
        alias: reviewer for reviewer in reviewers for alias in (reviewer.id, reviewer.reviewer_code)
    }
    unmapped_expert_raters = {
        vote.rater_pseudonym
        for vote in votes
        if vote.cohort == "expert_independent" and vote.rater_pseudonym not in reviewers_by_alias
    }
    if unmapped_expert_raters:
        raise ResearchReleaseError("expert judgments lack qualification-bound reviewer identities")

    def local_rater_id(raw: str) -> str:
        reviewer = reviewers_by_alias.get(raw)
        canonical = reviewer.id if reviewer is not None else raw
        return _archive_local_identifier(snapshot_set_sha256, canonical)

    admission_events = (
        session.scalars(
            select(AdmissionEvent).where(
                AdmissionEvent.pseudonym.in_(sorted(raw_rater_ids)),
                AdmissionEvent.created_at <= max_cutoff,
            )
        ).all()
        if raw_rater_ids
        else []
    )
    epicure_release = session.get(EpicureRelease, season.epicure_release_id)
    if (
        epicure_release is None
        or not epicure_release.official_eligible
        or not epicure_release.reproducibility_verified
        or epicure_release.bundle_sha256 != season.epicure_bundle_sha256
        or epicure_release.application_sha256 != season.epicure_application_sha256
    ):
        raise ResearchReleaseError("official Epicure release is missing or mismatched")
    requirements_lock_sha256 = hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest()
    robustness_members, robustness_evidence_sha256 = _robustness_members(
        robustness_evidence_paths
    )

    members: list[ArchiveMember] = [
        ArchiveMember("README.md", _readme()),
        ArchiveMember("records/season.json", _canonical_json_bytes(_serialize(season))),
        ArchiveMember(
            "records/epicure_release.json",
            _canonical_json_bytes(_serialize(epicure_release)),
        ),
        ArchiveMember(
            "records/controlled_run.json",
            _canonical_json_bytes(
                _serialize(
                    controlled_run,
                    exclude=frozenset(
                        {
                            "access_token_sha256",
                            "organization_id",
                            "organization_reference_sha256",
                            "evaluation_order_id",
                            "route_revision_id",
                            "spend_authorization_id",
                            "spend_authorization_binding_sha256",
                            "publication_authorization_id",
                            "publication_authorization_binding_sha256",
                        }
                    ),
                )
            ),
        ),
        *[
            ArchiveMember(
                f"snapshots/{item.track}-{item.cohort}-{item.id}.json",
                _canonical_json_bytes(_serialize(item)),
            )
            for item in snapshots
        ],
        _records_member("records/tasks.jsonl", (_serialize(item) for item in tasks)),
        _records_member(
            "records/task_evidence_artifacts.jsonl",
            (_serialize(item) for item in task_evidence),
        ),
        _records_member(
            "records/controlled_run_assignments.jsonl",
            (_serialize(item) for item in assignments),
        ),
        _records_member(
            "records/catalog_models.jsonl",
            (_serialize(item) for item in catalog_models),
        ),
        _records_member(
            "records/season_models.jsonl",
            (_serialize(item) for item in season_models),
        ),
        _records_member(
            "records/battles.jsonl",
            (
                _serialize(
                    item,
                    exclude=frozenset(
                        {"requester_pseudonym", "client_nonce_sha256", "provider_reservations_json"}
                    ),
                )
                for item in battles
            ),
        ),
        _records_member("records/response_arms.jsonl", (_serialize(item) for item in arms)),
        _records_member(
            "records/generation_attempts.jsonl",
            (_serialize(item) for item in attempts),
        ),
        _records_member("records/tool_calls.jsonl", (_serialize(item) for item in tools)),
        _records_member(
            "records/validator_results.jsonl",
            (_serialize(item) for item in validators),
        ),
        _records_member(
            "records/votes.jsonl",
            (
                _serialize(
                    item,
                    exclude=frozenset({"rater_pseudonym", "idempotency_key"}),
                    replacements={
                        "rater_archive_local_sha256": local_rater_id(item.rater_pseudonym)
                    },
                )
                for item in votes
            ),
        ),
        _records_member("records/cost_events.jsonl", (_serialize(item) for item in costs)),
        _records_member(
            "records/bedrock_billing_crosschecks.jsonl",
            (_serialize(item) for item in crosschecks),
        ),
        _records_member(
            "records/bedrock_billing_memberships.jsonl",
            (_serialize(item) for item in memberships),
        ),
        _records_member(
            "records/run_events.jsonl",
            (
                _serialize(
                    item,
                    exclude=frozenset({"payload_json"}),
                    replacements={
                        "payload_json": _privacy_filter_payload(
                            item.payload_json, snapshot_set_sha256
                        )
                    },
                )
                for item in run_events
            ),
        ),
        _records_member("records/incidents.jsonl", (_serialize(item) for item in incidents)),
        _records_member(
            "records/expert_reviewer_qualifications.jsonl",
            (
                {
                    "reviewer_archive_local_sha256": local_rater_id(item.reviewer_code),
                    "qualification_json": _normalize(item.qualification_json),
                    "qualification_verified": item.qualification_verified,
                    "cohort": item.cohort,
                    "active_at_cutoff": item.revoked_at is None or item.revoked_at > max_cutoff,
                    "created_at": _utc_iso(item.created_at),
                    "revoked_at": _utc_iso(item.revoked_at) if item.revoked_at else None,
                }
                for item in reviewers
            ),
        ),
        _records_member(
            "records/controlled_run_reviewers.jsonl",
            (
                _serialize(
                    item,
                    exclude=frozenset({"reviewer_id", "authorization_reference_sha256"}),
                    replacements={
                        "reviewer_archive_local_sha256": local_rater_id(item.reviewer_id)
                    },
                )
                for item in reviewer_links
            ),
        ),
        _records_member(
            "records/admission_events.jsonl",
            (
                _serialize(
                    item,
                    exclude=frozenset({"pseudonym"}),
                    replacements={"rater_archive_local_sha256": local_rater_id(item.pseudonym)},
                )
                for item in admission_events
            ),
        ),
        *_static_members(evidence),
        *robustness_members,
    ]
    source_counts = {
        "snapshots": len(snapshots),
        "tasks": len(tasks),
        "task_evidence_artifacts": len(task_evidence),
        "assignments": len(assignments),
        "models": len(catalog_models),
        "battles": len(battles),
        "response_arms": len(arms),
        "generation_attempts": len(attempts),
        "tool_calls": len(tools),
        "validator_results": len(validators),
        "votes": len(votes),
        "cost_events": len(costs),
        "incidents": len(incidents),
        "synthetic_arms": 0,
    }
    sealed = seal_members(
        members=members,
        manifest_metadata={
            "release_slug": f"flavourbench-{season.slug}-internal-official",
            "archive_class": "internal_official",
            "season": {
                "id": season.id,
                "slug": season.slug,
                "manifest_sha256": season.manifest_sha256,
                "protocol_bundle_sha256": season.protocol_bundle_sha256,
                "epicure_release_id": season.epicure_release_id,
                "epicure_bundle_sha256": season.epicure_bundle_sha256,
                "epicure_application_sha256": season.epicure_application_sha256,
            },
            "snapshots": [
                {
                    "id": item.id,
                    "track": item.track,
                    "cohort": item.cohort,
                    "payload_sha256": item.payload_sha256,
                    "input_evidence_sha256": item.input_evidence_sha256,
                    "evidence_cutoff_at": _utc_iso(item.evidence_cutoff_at),
                }
                for item in snapshots
            ],
            "snapshot_set_sha256": snapshot_set_sha256,
            "requirements_lock_sha256": requirements_lock_sha256,
            "build_image_digest": build_image_digest,
            "robustness_evidence_sha256": robustness_evidence_sha256,
            "counts": source_counts,
            "privacy": {
                "raw_ip_addresses": False,
                "operational_credentials": False,
                "persistent_rater_identifiers": False,
                "archive_local_rater_namespace": True,
                "public_release_authorized": False,
            },
        },
        output_dir=output_dir,
        private_key=private_key,
        signing_key_id=signing_key_id,
    )
    record = ResearchReleaseArchive(
        season_id=season.id,
        archive_class="internal_official",
        schema_version=SCHEMA_VERSION,
        snapshot_ids_json=normalized_snapshot_ids,
        snapshot_set_sha256=snapshot_set_sha256,
        manifest_json=sealed.manifest,
        manifest_sha256=sealed.manifest_sha256,
        archive_sha256=sealed.archive_sha256,
        storage_object_key=str(sealed.archive_path.resolve()),
        size_bytes=sealed.archive_size_bytes,
        member_count=int(sealed.manifest["archive_member_count_including_manifest"]),
        source_date_epoch=SOURCE_DATE_EPOCH,
        requirements_lock_sha256=requirements_lock_sha256,
        build_image_digest=build_image_digest,
        signature_algorithm="Ed25519",
        signing_key_id=signing_key_id,
        public_key_pem=sealed.public_key_pem,
        public_key_sha256=sealed.public_key_sha256,
        signature_base64=sealed.signature_base64,
    )
    if persist:
        session.add(record)
        session.commit()
    return {
        "archive_id": record.id,
        "archive_path": str(sealed.archive_path),
        "archive_sha256": sealed.archive_sha256,
        "manifest_sha256": sealed.manifest_sha256,
        "snapshot_set_sha256": snapshot_set_sha256,
        "signature_path": str(sealed.signature_path),
        "public_key_path": str(sealed.public_key_path),
        "member_count": record.member_count,
        "counts": source_counts,
        "robustness_evidence_sha256": robustness_evidence_sha256,
        "persisted": persist,
    }


def run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="season-1")
    parser.add_argument("--snapshot-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--private-key", type=Path)
    parser.add_argument("--signing-key-id")
    parser.add_argument("--build-image-digest")
    parser.add_argument("--post-collection-item-audit", type=Path)
    parser.add_argument("--generation-reliability-panel", type=Path)
    parser.add_argument("--prompt-sensitivity-audit", type=Path)
    parser.add_argument("--practical-cookability-execution", type=Path)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()
    private_key_path = args.private_key or Path(settings.research_archive_signing_private_key_path)
    if not str(private_key_path):
        raise SystemExit(
            "--private-key or FLAVOURBENCH_RESEARCH_ARCHIVE_SIGNING_PRIVATE_KEY_PATH is required"
        )
    output_dir = args.output_dir or Path(settings.research_archive_directory)
    signing_key_id = args.signing_key_id or settings.research_archive_signing_key_id
    build_image_digest = args.build_image_digest or settings.build_image_digest
    robustness_paths = {
        "post_collection_item_audit": args.post_collection_item_audit,
        "generation_reliability_panel": args.generation_reliability_panel,
        "prompt_sensitivity_audit": args.prompt_sensitivity_audit,
        "practical_cookability_execution": args.practical_cookability_execution,
    }
    supplied_robustness_paths = {
        name: path.resolve() for name, path in robustness_paths.items() if path is not None
    }
    with SessionLocal() as session:
        database_readiness(session, expected_role="flavourbench_api")
        result = build_research_release(
            session=session,
            season_slug=args.season,
            snapshot_ids=args.snapshot_id,
            output_dir=output_dir,
            private_key=load_signing_key(private_key_path),
            signing_key_id=signing_key_id,
            build_image_digest=build_image_digest,
            robustness_evidence_paths=(supplied_robustness_paths or None),
            persist=not args.no_persist,
        )
    print(json.dumps(result, sort_keys=True))


def verify_run(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify a sealed FlavourBench research archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--signature", type=Path)
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    signature_path = args.signature or Path(f"{args.archive}.sig")
    public_key_path = args.public_key or Path(f"{args.archive}.pub.pem")
    result = verify_archive(
        archive_path=args.archive,
        signature_base64=signature_path.read_text(encoding="ascii").strip(),
        public_key_pem=public_key_path.read_text(encoding="ascii"),
        expected_archive_sha256=args.expected_sha256,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    run()
