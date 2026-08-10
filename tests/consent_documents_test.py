from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi import HTTPException

from flavourbench.config import Settings
from flavourbench.consent_documents import (
    ConsentDocumentResolution,
    resolve_expert_consent_document,
)
from flavourbench.main import _expert_consent_document_active, _expert_identity
from flavourbench.models import ExpertReviewer


def _settings(tmp_path: Path, digest: str) -> Settings:
    return Settings(
        active_expert_consent_sha256s=[digest],
        expert_consent_documents_dir=str(tmp_path),
    )


def test_active_consent_resolves_to_the_exact_reviewer_visible_text(tmp_path) -> None:
    text = "# Review consent\n\nStatus: active\n\nParticipation is voluntary.\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (tmp_path / f"{digest}.md").write_text(text, encoding="utf-8")

    resolved = resolve_expert_consent_document(
        digest,
        settings=_settings(tmp_path, digest),
    )

    assert resolved.status == "active"
    assert resolved.sha256 == digest
    assert resolved.text == text


def test_external_volunteer_candidate_cannot_activate_even_if_registered(tmp_path) -> None:
    candidate_path = (
        Path(__file__).resolve().parents[2]
        / "protocol"
        / "consent"
        / "EXPERT-CONSENT-v1-DRAFT.md"
    )
    content = candidate_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    (tmp_path / f"{digest}.md").write_bytes(content)

    resolved = resolve_expert_consent_document(
        digest,
        settings=_settings(tmp_path, digest),
    )

    assert resolved.status == "document_not_marked_active"
    assert resolved.text is None


@pytest.mark.parametrize(
    ("write_document", "expected_status"),
    [(False, "document_missing"), (True, "document_hash_mismatch")],
)
def test_missing_or_hash_mismatched_text_is_not_active(
    tmp_path,
    write_document: bool,
    expected_status: str,
) -> None:
    expected_digest = hashlib.sha256(b"expected consent text").hexdigest()
    if write_document:
        (tmp_path / f"{expected_digest}.md").write_text(
            "# Different text\n\nStatus: active\n",
            encoding="utf-8",
        )

    resolved = resolve_expert_consent_document(
        expected_digest,
        settings=_settings(tmp_path, expected_digest),
    )

    assert resolved.status == expected_status
    assert resolved.text is None


@pytest.mark.parametrize("resolution_status", ["document_missing", "document_hash_mismatch"])
def test_expert_admission_fails_closed_when_visible_consent_text_is_unresolved(
    monkeypatch: pytest.MonkeyPatch,
    resolution_status: str,
) -> None:
    reviewer = ExpertReviewer(
        reviewer_code=f"unresolved-consent-{resolution_status}",
        invitation_sha256="a" * 64,
        qualification_json=["evidence"],
        qualification_verified=False,
        cohort="expert_independent",
        profile_json={"consent_document_sha256": "b" * 64},
        batch_reveal_only=True,
    )

    monkeypatch.setattr(
        "flavourbench.main.resolve_expert_consent_document",
        lambda _: ConsentDocumentResolution("b" * 64, resolution_status, None),
    )

    assert _expert_consent_document_active(reviewer) is False


@pytest.mark.parametrize("resolution_status", ["document_missing", "document_hash_mismatch"])
def test_verified_credential_cannot_bypass_unresolved_visible_consent(
    monkeypatch: pytest.MonkeyPatch,
    resolution_status: str,
) -> None:
    reviewer = ExpertReviewer(
        reviewer_code=f"verified-unresolved-{resolution_status}",
        invitation_sha256="a" * 64,
        qualification_json=["evidence"],
        qualification_verified=True,
        cohort="expert_independent",
        profile_json={"consent_document_sha256": "b" * 64},
        batch_reveal_only=True,
    )
    reviewer._flavourbench_verified_credential_id = "verified-credential"

    monkeypatch.setattr(
        "flavourbench.main._invited_expert_identity",
        lambda _session, _authorization: ("pseudonymous-rater", reviewer),
    )
    monkeypatch.setattr(
        "flavourbench.main.resolve_expert_consent_document",
        lambda _: ConsentDocumentResolution("b" * 64, resolution_status, None),
    )

    with pytest.raises(HTTPException, match="bound consent document is active") as exc_info:
        _expert_identity(None, "Bearer verified")  # type: ignore[arg-type]

    assert exc_info.value.status_code == 403
