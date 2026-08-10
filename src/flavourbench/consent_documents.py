"""Resolve reviewer-visible consent documents from a read-only hash registry."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings
from .human_study_activation import resolve_human_study_activation

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ACTIVE_STATUS_PATTERN = re.compile(r"(?im)^status:\s*active\s*$")
MAX_CONSENT_DOCUMENT_BYTES = 256 * 1024


@dataclass(frozen=True)
class ConsentDocumentResolution:
    sha256: str | None
    status: str
    text: str | None


def resolve_expert_consent_document(
    consent_sha256: object,
    *,
    settings: Settings | None = None,
) -> ConsentDocumentResolution:
    """Fail closed unless an active digest resolves to the exact visible text."""

    configured = settings or get_settings()
    if not isinstance(consent_sha256, str) or SHA256_PATTERN.fullmatch(consent_sha256) is None:
        return ConsentDocumentResolution(None, "invalid_digest", None)
    if consent_sha256 not in configured.active_expert_consent_sha256s:
        return ConsentDocumentResolution(consent_sha256, "inactive", None)
    directory_value = configured.expert_consent_documents_dir.strip()
    if not directory_value:
        return ConsentDocumentResolution(consent_sha256, "documents_directory_unconfigured", None)
    directory = Path(directory_value)
    try:
        resolved_directory = directory.resolve(strict=True)
    except OSError:
        return ConsentDocumentResolution(consent_sha256, "documents_directory_unavailable", None)
    if not resolved_directory.is_dir():
        return ConsentDocumentResolution(consent_sha256, "documents_directory_unavailable", None)
    candidate = resolved_directory / f"{consent_sha256}.md"
    if candidate.is_symlink():
        return ConsentDocumentResolution(consent_sha256, "document_path_invalid", None)
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except OSError:
        return ConsentDocumentResolution(consent_sha256, "document_missing", None)
    if resolved_candidate.parent != resolved_directory or not resolved_candidate.is_file():
        return ConsentDocumentResolution(consent_sha256, "document_path_invalid", None)
    try:
        size = resolved_candidate.stat().st_size
    except OSError:
        return ConsentDocumentResolution(consent_sha256, "document_unreadable", None)
    if size < 1 or size > MAX_CONSENT_DOCUMENT_BYTES:
        return ConsentDocumentResolution(consent_sha256, "document_size_invalid", None)
    try:
        content = resolved_candidate.read_bytes()
    except OSError:
        return ConsentDocumentResolution(consent_sha256, "document_unreadable", None)
    if len(content) != size:
        return ConsentDocumentResolution(consent_sha256, "document_size_invalid", None)
    if hashlib.sha256(content).hexdigest() != consent_sha256:
        return ConsentDocumentResolution(consent_sha256, "document_hash_mismatch", None)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return ConsentDocumentResolution(consent_sha256, "document_encoding_invalid", None)
    if ACTIVE_STATUS_PATTERN.search(text) is None:
        return ConsentDocumentResolution(consent_sha256, "document_not_marked_active", None)
    if configured.environment == "production":
        activation = resolve_human_study_activation(
            configured.human_study_activation_manifest_path,
            configured.human_study_activation_manifest_sha256,
            consent_sha256=consent_sha256,
        )
        if not activation.ready:
            return ConsentDocumentResolution(
                consent_sha256,
                f"human_study_governance_{activation.status}",
                None,
            )
    return ConsentDocumentResolution(consent_sha256, "active", text)
